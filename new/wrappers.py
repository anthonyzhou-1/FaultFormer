import math
from random import random

import torch
from torch import nn

from utils import random_augmentation


"Masking functions from: https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py"

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_seq(seq, mask_prob=0.15, replace_prob=0.9, random_token_prob=0.1):

    mask_token_id = 0

    mask = prob_mask_like(seq, mask_prob)
    mask_out = mask.clone().detach()

    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)

    masked_seq = seq.clone().detach()

    idx = torch.randperm(seq.numel())
    random_tokens = torch.flatten(seq)[idx].view(seq.size())

    random_token_prob = prob_mask_like(seq, random_token_prob) & mask
    masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

    # remove tokens that were substituted randomly from being [mask]ed later
    mask = mask & ~random_token_prob

    # [mask] input

    replace_prob = prob_mask_like(seq, replace_prob)
    masked_seq = masked_seq.masked_fill(mask * replace_prob, mask_token_id)

    return masked_seq, mask_out

class MaskedWrapper(nn.Module):
    '''
    Wrapper around a nn.Module that:
        - masks the input
        - computes a forward pass with the model
        - calculates a MSE loss 
    '''
    

    def __init__(
        self,
        net,
        reconstruction_net,
        tokenizer,
        mask_prob = 0.15, # Probability to mask out a token
        replace_prob = 0.9, # Probability to replace a masked token with 0
        random_token_prob = 0.1 # Probability to replace a masked token with a random token
    ):
        super().__init__()
        self.net = net
        self.reconstruction_net = reconstruction_net
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_token_prob = random_token_prob
        self.mse = nn.MSELoss()

    def forward(
        self,
        x,
        mode = 'train',
    ):
        '''
        Inputs:
            x: input sequence in shape [batch_size, num_tokens, d_in]
            mode: train or eval
            variables: PDE variables in shape [batch_size, num_vars]
        '''
        x = self.tokenizer(x)
        orig_seq = x.clone()

        # Mask input
        x_masked, mask = mask_seq(x, self.mask_prob, self.replace_prob, self.random_token_prob)

        # Forward pass
        x_pred = self.net(x_masked)
        x_pred = x_pred[:, 1:, :] # remove cls token
        x_pred = self.reconstruction_net(x_pred)

        if mode == "train":
            # Reconstruction Loss
            loss = self.mse(x_pred[mask], orig_seq[mask])
            return loss
        
        else:
            return x_pred, orig_seq, x_masked

        
class FinetuneWrapper(nn.Module):
    '''
    Wrapper around a nn.Module that:
        - Selects CLS token from output
        - Projects CLS token to embedding dimension
    '''

    def __init__(
        self,
        net,
        embedding_net,
        tokenizer,
        p_aug = 0,
        model = "BERT"
    ):
        super().__init__()
        self.net = net
        self.embedding_net = embedding_net
        self.tokenizer = tokenizer
        self.p_aug = p_aug
        self.model = model

    def augment(self, x):
        '''
        Augments input sequence
        '''
        for i in range(x.shape[0]):
            # Augment with probability p_aug
            if random() < self.p_aug:
                x[i] = random_augmentation(x[i])
        return x

    def forward(
        self,
        x,
        mode = 'train',
    ):  
        if self.p_aug > 0 and mode == 'train':
            x = self.augment(x)

        x_tokens = self.tokenizer(x)
        y = self.net(x_tokens)

        if self.model == "BERT":
            cls_token = y[:, 0, :]
            y = self.embedding_net(cls_token)
        return y