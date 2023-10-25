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

def train_model(model, params, train_dataloader=None, test_dataloader=None, optimizer=None, scheduler=None):

    def train(model: nn.Module, dataloader):
        """
        Arguments:
            model: model to be trained
            dataloader: passes data to model

        Returns:
            loss: training loss
            accuracy: training accuracy
        """ 

        n_samples = len(dataloader.dataset)
        model.train() 
        total_loss = 0.
        num_correct = 0

        for (batch, labels) in tqdm(dataloader, desc=f"Training Epoch [{epoch}/{epochs}]"):
            batch, labels = batch.cuda(), labels.cuda()

            output = model(batch)
            loss = criterion(output, labels)

            optimizer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            y_hat= torch.argmax(s(output), dim = 1)
            num_correct += (y_hat == labels).sum()

        if(scheduler != None):
            scheduler.step()

        return total_loss/(n_samples-1), num_correct/n_samples

    def evaluate(model: nn.Module, dataloader):
        """
        Arguments:
            model: model to be trained
            dataloader: passes data to model

        Returns:
            loss: test loss
            accuracy: test accuracy
        """

        n_samples = len(dataloader.dataset)
        model.eval() 
        total_loss = 0.
        num_correct = 0

        with torch.no_grad():
            for (batch, labels) in tqdm(dataloader, desc=f"Validation Epoch [{epoch}/{epochs}]"):
                batch, labels = batch.cuda(), labels.cuda()

                output = model(batch)
                loss = criterion(output, labels)

                total_loss += loss.item()
                y_hat= torch.argmax(s(output), dim = 1)
                num_correct += (y_hat == labels).sum()

        return total_loss/(n_samples- 1), num_correct/n_samples
        
    batch_size = params["batch_size"]
    warmup = params["warmup"]
    d_model = params["d_model"]
    epochs = params["epochs"]
    params["model"] = model.model_type
    fourier = params["fourier"]
    p_no_aug = params["p_no_aug"]
    p_two_aug = params["p_two_aug"]

    if(train_dataloader == None):
        raise Exception("Must specify training dataloader")

    if(test_dataloader == None):
        raise Exception("Must specify testing dataloader")

    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    s = nn.Softmax(dim = 1).cuda()
    if(optimizer == None):
        optimizer = NoamOpt(d_model, warmup,
                        torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))

    best_val_loss = float('inf')
    best_val_acc = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    best_model_params_path = "models/" + timestr
    writer = SummaryWriter()
    
    log_path = "logs/" + timestr + ".txt"
    with open(log_path, 'w') as file:
     file.write(json.dumps(params))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model = model, 
                            dataloader = train_dataloader)

        val_loss, val_acc = evaluate(model = model,
                            dataloader = test_dataloader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,}, best_model_params_path)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/valid", val_acc, epoch)

        print('-' * 89)
        print(f'training loss {train_loss:5.4f} | valid loss {val_loss:5.4f}')
        print(f'training accuracy {train_acc:5.4f} | validation accuracy {val_acc:5.4f}')
        print('-' * 89)

    writer.flush()
    writer.close()

    f = open(log_path, 'a')
    f.write(str(best_val_acc))
    f.close()
