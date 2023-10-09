import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerModel_triplet(nn.Module):

    def __init__(self, d_in: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, d_lin:int, d_out: int, seq_len, dropout: float = 0.5):
        '''
        Arguments:
            d_in: input dimension of signal data.
            d_model: model dimension 
            nhead: number of heads in each self-attention layer
            d_hid: hidden dimension in self-attention layer
            nlayers: number of transformer layers
            d_lin: dimension to project model embedding down to
            d_out: final output dimension of model embedding
            seq_len: length of input sequence
            dropout: probability of dropout
        
        Transformer model that encodes embeddings for contrastive pretraining
        '''

        super().__init__()
        self.model_type = 'Transformer_triplet'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Linear(d_in, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model*seq_len, d_lin)
        self.relu = nn.ReLU()
        self.project = nn.Linear(d_lin, d_out)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.project.bias.data.zero_()
        self.project.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, d_in]``

        Returns:
            output: Tensor of shape ``[batch_size, d_out]``
        
        Embeds signal to higher dimension, adds positional encoding, and generates output encodings.
        Output encodings are flattened together, and passed through MLP + ReLU + MLP to make output embedding.
        """

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)

        output = torch.transpose(output, 0, 1)
        output = torch.flatten(output, start_dim=1, end_dim=2)
        output = self.linear(output)
        output = self.relu(output)
        output = self.project(output)

        return output

class TransformerModel_mask(nn.Module):

    def __init__(self, d_in: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, seq_len: int, dropout: float = 0.5):

        '''
        Arguments:
            d_in: input dimension of signal data.
            d_model: model dimension 
            nhead: number of heads in each self-attention layer
            d_hid: hidden dimension in self-attention layer
            nlayers: number of transformer layers
            seq_len: length of input sequence
            dropout: probability of dropout
        
        Transformer model that outputs encodings projected down to input dimension for a reconstruction task
        '''

        super().__init__()
        self.model_type = 'Transformer_mask'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Linear(d_in, d_model)
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.rand(1, d_model))
        self.linear = nn.Linear(d_model, d_in)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def cls(self, src: Tensor):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, d_in]``

        Returns:
            output: Tensor of shape ``[1 + seq_len, batch_size, d_in]``
        """
        # What a monstrosity
        return torch.stack([torch.cat((self.cls_token, src[:,i,:]), dim=0) for i in range(src.shape[1])], dim = 1)

    def forward(self, src: Tensor, src_mask: Tensor = None, fine_tune = False) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, d_in]``
            fine_tune: flag to shift to fine-tuning mode (model encodings are not projected down to d_in)

        Returns:
            output: Tensor of shape ``[seq_len, batch_size, d_in]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.cls(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)

        if(fine_tune):
            return output

        output = self.linear(output)
        return output


class TransformerModel_cls(nn.Module):

    def __init__(self, d_in: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, seq_len: int, dropout: float = 0.5, n_classes: int = 10):

        '''
        Arguments:
            d_in: input dimension of signal data.
            d_model: model dimension 
            nhead: number of heads in each self-attention layer
            d_hid: hidden dimension in self-attention layer
            nlayers: number of transformer layers
            seq_len: length of input sequence
            dropout: probability of dropout
            n_classes: number of classes to perform classification
        
        Transformer model that outputs logits based on extracting the cls token encoding.
        '''

        super().__init__()
        self.model_type = 'Transformer_cls'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Linear(d_in, d_model)
        self.d_model = d_model
        self.classification_layer = nn.Linear(d_model, n_classes)
        self.cls_token = nn.Parameter(torch.rand(1, d_model))

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()
        self.classification_layer.bias.data.zero_()
        self.classification_layer.weight.data.uniform_(-initrange, initrange)
    
    def cls(self, src: Tensor):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, d_in]``

        Returns:
            output: Tensor of shape ``[1 + seq_len, batch_size, d_in]``
        """
        # What a monstrosity
        return torch.stack([torch.cat((self.cls_token, src[:,i,:]), dim=0) for i in range(src.shape[1])], dim = 1)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, d_in]``

        Returns:
            output: Tensor of shape ``[batch_size, num_classes]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.cls(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output[0, :, :]
        output = self.classification_layer(output)
        return output
            
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        '''
        Positional encoding module based on: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        '''
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)