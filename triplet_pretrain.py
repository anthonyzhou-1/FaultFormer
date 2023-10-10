import math
import os
from typing import Tuple
import numpy as np
import time
from tqdm import tqdm
import json

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import collate_fn_transpose, NoamOpt
from online_triplet_loss.losses import *

def pretrain_model(model, params, dataloader=None, optimizer=None):

    def train(model: nn.Module, dataloader):
        """
        Arguments:
            model: model to be trained
            dataloader: passes data to model

        Returns:
            loss: training loss

        Embeds each sample in the batch. Calculates triplet loss on embeddings based on labels
        """
        model.train() 
        total_loss = 0.

        for (batch, labels) in tqdm(dataloader, desc=f"Training Epoch [{epoch}/{epochs}]"):
            batch, labels = batch.cuda(), labels.cuda()
            
            embeddings = model(batch)
            loss = batch_hard_triplet_loss(labels, embeddings, margin=margin)

            optimizer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        return total_loss
        
    batch_size = params["batch_size"]
    warmup = params["warmup"]
    d_model = params["d_model"]
    epochs = params["epochs"]
    seq_len = params["seq_len"]
    params["model"] = model.model_type
    margin = params["margin"]

    model.cuda()
    if(optimizer == None):
        optimizer = NoamOpt(d_model, warmup,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    best_loss = float('inf')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    best_model_params_path = "models/" + timestr
    writer = SummaryWriter()
    
    log_path = "logs/" + timestr + ".txt"
    with open(log_path, 'w') as file:
        file.write(json.dumps(params))

    for epoch in range(1, epochs + 1):
        loss = train(model = model, 
                    dataloader = dataloader)

        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,}, best_model_params_path)

        writer.add_scalar("Loss/train", loss, epoch)

        print('-' * 89)
        print(f'training loss {loss:5.4f}')
        print('-' * 89)

    writer.flush()
    writer.close()