import torch
import torch.nn as nn
import numpy as np
from utils import get_fourier_features

class SignalTokenizer(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 mode = 'constant',
                 tokenizer_model = None,
                 fourier_k = 40):
        super().__init__()
        self.num_tokens = num_tokens
        self.mode = mode
        if mode == "CNN":
            self.CNN = tokenizer_model
        if mode == "fourier":
            self.fourier_k = fourier_k

    def forward(self, sequence):
        '''
        Tokenizes sequence based on mode
        Sequence is given in shape (batch_size, nx)
        Returns sequence in shape (batch_size, num_tokens, d_in)
        '''
        if self.mode == "constant":
            batch_size, nx = sequence.shape
            d_in = int(nx/self.num_tokens)
            return sequence.view(batch_size, self.num_tokens, d_in)
        
        if self.mode == "CNN":

            # Reshape to batch, channels, sequence_len
            sequence_CNN = sequence.unsqueeze(1)
            out = self.CNN(sequence_CNN)

            # Reshape to batch, sequence_len, channels
            return torch.transpose(out, 1, 2)
        
        elif self.mode == "fourier":
            features = get_fourier_features(sequence, self.fourier_k)
            return features
        else:
            assert False, f'Mode {self.mode} not recognized'
