from torch.utils.data import DataLoader
import random
import wandb
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import numpy as np

from Tokenizer import SignalTokenizer
from wrappers import MaskedWrapper, FinetuneWrapper
from models import BERT, MLP, CNN
from utils import SignalDataset, process_dict

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
          criterion: torch.nn.Module,
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
    accs = []
    for x, y in loader:
        # Reset gradients
        optimizer.zero_grad()

        # Tokenize data to shape (batch_size, num_tokens, d_in)
        x, y = x.to(device), y.to(device)

        # Forward pass
        pred = model(x)
        loss = criterion(pred, y)

        # Backward pass
        loss.backward()
        losses.append(loss.detach() / batch_size)

        # Accuracy
        pred_labels = torch.argmax(pred, dim=1)
        num_correct = (pred_labels == y).sum()
        accs.append(num_correct.detach() / batch_size)

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
    accs = torch.stack(accs)
    grad_norms = torch.stack(grad_norms)
    losses_out, grad_norms_out = torch.mean(sync_tensor_across_gpus(losses.unsqueeze(0))), torch.mean(sync_tensor_across_gpus(grad_norms.unsqueeze(0)))
    accs_out = torch.mean(sync_tensor_across_gpus(accs.unsqueeze(0)))
    if device == 0:
        print(f'Training Loss: {losses_out}, Training Accuracy: {accs_out}')
        wandb.log({"finetune_train/loss": losses_out,
                   "finetune_train/accuracy": accs_out,
                    "metrics/grad_norm": grad_norms_out})

def evaluate(args: argparse,
             epoch: int,
          model: torch.nn.Module,
          loader: DataLoader,
          criterion: torch.nn.Module,
          device: torch.cuda.device="cpu") -> None:
    
    model.eval()
    batch_size = args.batch_size
    losses = []
    accs = []
    loader.sampler.set_epoch(epoch)
    with torch.no_grad():
        for x, y in loader:
                # Tokenize data to shape (batch_size, num_tokens, d_in)
                x, y = x.to(device), y.to(device)

                # Forward pass
                pred = model(x, mode='eval')
                loss = criterion(pred, y)

                losses.append(loss.detach() / batch_size)

                pred_labels = torch.argmax(pred, dim=1)
                num_correct = (pred_labels == y).sum()
                accs.append(num_correct.detach() / batch_size)

    losses = torch.stack(losses)
    accs = torch.stack(accs)
    losses_out = torch.mean(sync_tensor_across_gpus(losses.unsqueeze(0)))
    accs_out = torch.mean(sync_tensor_across_gpus(accs.unsqueeze(0)))
    if device == 0:
        print(f'Evaluation Loss: {losses_out}, Evaluation Accuracy: {accs_out}')
        wandb.log({"finetune_eval/loss": losses_out,
                   "finetune_eval/accuracy": accs_out})
  
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
    #generator = torch.Generator().manual_seed(42)
    train_data_path = "/home/ayz2/FaultFormer/new/data/CWRU/Finetuning/training_data_1"
    train_labels_path = "/home/ayz2/FaultFormer/new/data/CWRU/Finetuning/training_labels_1"

    valid_data_path = "/home/ayz2/FaultFormer/new/data/CWRU/Finetuning/test_data_1"
    valid_labels_path = "/home/ayz2/FaultFormer/new/data/CWRU/Finetuning/test_labels_1"

    #dataset = SignalDataset(train_data_path, train_labels_path)
    #train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)

    train_dataset = SignalDataset(train_data_path, train_labels_path)
    valid_dataset = SignalDataset(valid_data_path, valid_labels_path)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False, num_workers=0, shuffle=False, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=False, num_workers=0, shuffle=False, sampler=valid_sampler)
    
    tokenizer_model = None
    if args.mode == "CNN":
        tokenizer_model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=args.d_cnn, kernel_size=args.kernel_size, stride=args.stride, padding="valid"),
        nn.GELU(),
        nn.Conv1d(in_channels=args.d_cnn, out_channels=args.d_in, kernel_size=args.kernel_size, stride=args.stride, padding="valid"),
        )

    tokenizer = SignalTokenizer(num_tokens=args.num_tokens, mode=args.mode, tokenizer_model=tokenizer_model, fourier_k=args.fourier_k)

    embedding_net = None
    if args.model == "BERT":
        model_inner = BERT(d_in=args.d_in,
                d_model = args.d_model,
                nhead = args.nhead,
                num_layers= args.num_layers,
                dropout = args.dropout,
                )
    
        if args.load_bert:
                bert_path = "/home/ayz2/FaultFormer/NewAndImprovedISwearItWorks/models/BERT_pretrain_CWRU_full_time122176.pt"
                checkpoint = process_dict(torch.load(bert_path, map_location=torch.device('cpu')))
                
                reconstruction_net = nn.Linear(in_features=args.d_model, out_features=args.d_in)
        
                model_checkpoint = MaskedWrapper(net=model_inner,
                                reconstruction_net=reconstruction_net,
                                mask_prob = args.mask_prob,
                                replace_prob = args.replace_prob,
                                random_token_prob = args.random_token_prob)
                model_checkpoint.load_state_dict(checkpoint)
                model_inner = model_checkpoint.net

        embedding_net = nn.Linear(in_features=args.d_model, out_features=args.d_out)

    elif args.model == "MLP":
        model_inner = MLP(d_in=args.d_in,
                           d_model = args.d_model,
                           d_out = args.d_out,)
    elif args.model == "CNN":
        model_inner = CNN(d_in=args.d_in,
                           d_model = args.d_model,
                           d_out = args.d_out,
                           dropout = args.dropout,)
    else: 
        raise Exception("Model not supported")

    model = FinetuneWrapper(net = model_inner,
                                embedding_net=embedding_net,
                                tokenizer=tokenizer,
                                p_aug = args.p_aug,
                                model = args.model).to(rank)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of model parameters: {params}')

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
    criterion = nn.CrossEntropyLoss().to(rank)

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    args.save_path= f'models/BERT_{args.experiment}_time{timestring}.pt'
    if rank == 0:
        run = wandb.init(project="faultformer",
                config=vars(args),
                name = f'BERT_{args.experiment}_time{timestring}')
    for epoch in range(num_epochs):
        train(args, epoch, model, optimizer, scheduler, train_loader, criterion, device=rank)
        evaluate(args, epoch, model, valid_loader, criterion, rank)
        if epoch % args.save_interval == 0 and epoch != 0:
            if rank == 0:
                torch.save(model.module.state_dict(), args.save_path)
        
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FaultFormer')
    parser.add_argument('--experiment', type=str, default='Train_End2End_CWRU',
            help='Experiment name')
    parser.add_argument('--save_interval', type=int, default=50,
            help='Number of epochs between saving models')
    parser.add_argument("--load_bert", type=eval, default=False,
                        help='Whether to load bert')
    parser.add_argument("--model", type=str, default="BERT",
                        help='Model type')

    # BERT parameters
    parser.add_argument('--batch_size', type=int, default=8,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=1000,
            help='Number of training epochs')
    parser.add_argument('--d_in', type=int, default=8,
            help='Input dimension of BERT')
    parser.add_argument('--d_out', type=int, default=10,
                        help='Output dimension of BERT')
    parser.add_argument('--d_model', type=int, default=256,
            help='Model dimension of BERT')
    parser.add_argument('--nhead', type=int, default=32,
            help='Number of heads in BERT')
    parser.add_argument('--num_layers', type=int, default=4,
            help='Number of layers in BERT')
    parser.add_argument('--dropout', type=float, default=0.3,
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
    parser.add_argument('--mode', type=str, default='CNN',
                help='Mode for tokenization: [constant, CNN, fourier]')
    parser.add_argument('--d_cnn', type=int, default=4,
                help='Number of channels in CNN')
    parser.add_argument('--kernel_size', type=int, default=4,
                help='Kernel size in CNN')
    parser.add_argument('--stride', type=int, default=2,
                help='Stride in CNN')
    parser.add_argument('--fourier_k', type=int, default=40,
                help='Number of fourier modes to keep')
    parser.add_argument('--p_aug', type=float, default=0.2,
                        help='Probability of augmentation')

    args = parser.parse_args()
    #world_size = torch.cuda.device_count()
    world_size = 8
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size
    ) 