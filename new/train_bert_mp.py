from torch.utils.data import DataLoader
import random
import wandb
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import numpy as np

from Tokenizer import SignalTokenizer
from wrappers import MaskedWrapper
from models import BERT
from utils import SignalDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os

def setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def sync_tensor_across_gpus(t):
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t) 
    return torch.cat(gather_t_tensor, dim=0)

def cleanup():
    destroy_process_group()

def train(args: argparse,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          loader: DataLoader,
          tokenizer: SignalTokenizer,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
    Returns:
        None
    """
    model.train()
    batch_size = args.batch_size

    losses = []
    grad_norms = []
    loader.sampler.set_epoch(epoch)
    for x in loader:
        # Reset gradients
        optimizer.zero_grad()

        # Tokenize data to shape (batch_size, num_tokens, d_in)
        tokens = tokenizer.forward(x).to(device)

        # Forward pass
        loss = model(tokens)

        # Backward pass
        loss.backward()
        losses.append(loss.detach() / batch_size)

        optimizer.step()

        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        grad_norm = torch.cat(grads).norm()
        grad_norms.append(grad_norm / batch_size)

        scheduler.step()

    losses = torch.stack(losses)
    grad_norms = torch.stack(grad_norms)
    losses_out, grad_norms_out = torch.mean(sync_tensor_across_gpus(losses.unsqueeze(0))), torch.mean(sync_tensor_across_gpus(grad_norms.unsqueeze(0)))
    print(f'Training Loss: {losses_out}')
    print(f'Grad Norm: {grad_norms_out}')
    #if device == 0:
    ##    print(f'Training Loss: {losses_out}')
    #    wandb.log({"train/loss": losses_out,
    #                "metrics/grad_norm": grad_norms_out})
  
def prepare_data(rank, world_size, args, data_path, label_path = None, mode='train'):
    dataset = SignalDataset(data_path, label_path)
    shuffle = False
    if(mode == 'train'):
         shuffle = True
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False, num_workers=0, shuffle=False, sampler=sampler)
    return dataloader

def main(rank, world_size, args: argparse):
    setup(rank, world_size)
    #if rank == 0:
    #    run = wandb.init(project="bert-pde",
    #            config=vars(args))
        
    train_string = "/home/cmu/anthony/FaultFormer/FaultFormer/NewAndImprovedISwearItWorks/data_uwu/pretraining_data_0124578.pt"
    train_loader = prepare_data(rank, world_size, args, train_string)

    bert = BERT(d_in=args.d_in,
             d_model = args.d_model,
             nhead = args.nhead,
             num_layers= args.num_layers,
             dropout = args.dropout,
             )
    reconstruction_net = nn.Linear(in_features=args.d_model, out_features=args.d_in)
    
    model = MaskedWrapper(net=bert,
                          reconstruction_net=reconstruction_net,
                            mask_prob = args.mask_prob,
                            replace_prob = args.replace_prob,
                            random_token_prob = args.random_token_prob).to(rank)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of BERT parameters: {params}')

    tokenizer = SignalTokenizer(num_tokens=args.num_tokens, mode=args.mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.min_lr, betas=(args.beta1, args.beta2), fused=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=args.max_lr, 
                                                        steps_per_epoch= len(train_loader), 
                                                        epochs=args.num_epochs, 
                                                        pct_start=args.pct_start, 
                                                        anneal_strategy='cos', 
                                                        final_div_factor=args.max_lr/args.min_lr)

    # Multiprocessing
    model = DDP(model, device_ids=[rank])

    ## Training
    num_epochs = args.num_epochs

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    args.save_path= f'models/BERT_{args.experiment}_time{timestring}.pt'

    for epoch in range(num_epochs):
        train(args, epoch = epoch, model=model, optimizer=optimizer, scheduler=scheduler, loader=train_loader, tokenizer=tokenizer, device=rank)
        if rank == 0:
            torch.save(model.module.state_dict(), args.save_path)
        
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain FaultFormer')

    # BERT parameters
    parser.add_argument('--batch_size', type=int, default=16,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=100,
            help='Number of training epochs')
    parser.add_argument('--d_in', type=int, default=8,
            help='Input dimension of BERT')
    parser.add_argument('--d_model', type=int, default=128,
            help='Model dimension of BERT')
    parser.add_argument('--nhead', type=int, default=4,
            help='Number of heads in BERT')
    parser.add_argument('--num_layers', type=int, default=2,
            help='Number of layers in BERT')
    parser.add_argument('--dropout', type=float, default=0,
                help='Dropout probability in BERT')
    
    # Reconstruction parameters
    parser.add_argument('--mask_prob', type=float, default=0.15,
            help='Probability to mask out a token')
    parser.add_argument('--replace_prob', type=float, default=0.9,
            help='Probability to replace a masked token with 0')
    parser.add_argument('--random_token_prob', type=float, default=0.1,
            help='Probability to replace a masked token with a random token')
    
    # Optimizer parameters
    parser.add_argument('--min_lr', type=float, default=1e-4,
            help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
            help='Maximum learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
            help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98,
            help='Adam beta2')
    
    # Scheduler parameters
    parser.add_argument('--pct_start', type=float, default=0.1,
            help='Percentage of training to increase learning rate')
    
    # Tokenizer parameters
    parser.add_argument('--num_tokens', type=int, default=200,
            help='Number of tokens to generate from a PDE time/spatial sequence')
    parser.add_argument('--mode', type=str, default='constant',
                help='Mode for tokenization: [constant, bicubic, fourier]')

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size
    ) 