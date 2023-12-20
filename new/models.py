import math
import torch
from torch import nn, Tensor
from x_transformers import Encoder, ContinuousTransformerWrapper

class BERT(nn.Module):

    def __init__(self, 
                 d_in: int, 
                 d_model: int, 
                 nhead: int, 
                 num_layers: int, 
                 dropout: float = 0.1):

        '''
        Arguments:
            d_in: input dimension of signal data.
            d_model: model dimension 
            nhead: number of heads in each self-attention layer
            d_hid: hidden dimension in self-attention layer
            nlayers: number of transformer layers
            dropout: probability of dropout
        
        Transformer model that outputs encodings projected down to input dimension for a reconstruction task
        '''

        super().__init__()
        self.model_type = 'BERT'
        self.encoder = ContinuousTransformerWrapper(
            max_seq_len = 500,
            use_abs_pos_emb = False,
            attn_layers = Encoder(
                dim = d_model,
                depth = num_layers,
                heads = nhead,
                rotary_pos_emb = True,
                attn_dropout = dropout,
                ff_dropout = dropout,
                attn_flash = True,
                ff_glu = True
                )
            )
        self.embedding = nn.Linear(d_in, d_model)
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, d_model))
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()

    def token_embedding(self, src: Tensor) -> Tensor:
        '''
        Embeds input data w/ and MLP and scale by model dimension
        Adds CLS and SEP tokens
        '''

        batch_size, seq_len, d_in = src.shape

        # Embeds input data w/ and MLP and scale by model dimension
        src = self.embedding(src) * math.sqrt(self.d_model)

        # Copies cls token across batch dimension and prepends to sequence
        src = torch.column_stack([self.cls_token.expand(batch_size, -1, -1), src])

        return src

    
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_in]``

        Returns:
            output: Tensor of shape ``[batch_size, seq_len, d_out]``
        """
        # Token embedding projects input to model dimension, adds cls token, and adds sep tokens
        src = self.token_embedding(src)

        # Forward pass through transformer encoder. Automatically adds positional embeddings
        output = self.encoder(src)

        return output
    
class MLP(nn.Module):

    def __init__(self, 
                 d_in: int, 
                 d_model: int,
                 d_out: int,
                 dropout: float = 0.3):
        '''
        MLP class
        '''
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(d_model)
        self.model = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=d_in*d_model, out_features=4*d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_features=4*d_model, out_features=4*d_model),
                nn.GELU(),
                nn.Linear(in_features=4*d_model, out_features=2*d_model),
                nn.GELU(),
                nn.Linear(in_features=2*d_model, out_features= d_model),
                nn.GELU(),
                nn.Linear(in_features=d_model, out_features=d_out),
         )
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.model(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_model: int,
                 d_out: int,
                 dropout: float = 0.1):
        '''
        CNN Class
        '''
        super().__init__()
        self.CNN_layers = nn.Sequential(
                nn.Conv1d(in_channels = d_in, out_channels = 4*d_in, kernel_size = 10, stride=2, padding="valid"),
                nn.GELU(),
                nn.Conv1d(in_channels=4*d_in, out_channels=d_model, kernel_size = 5, stride=1, padding="valid"),
                nn.GELU(),
                nn.Conv1d(in_channels=d_model, out_channels=2*d_model, kernel_size = 3, stride=1, padding="valid"),
                nn.GELU(),
                nn.Conv1d(in_channels=2*d_model, out_channels= d_model, kernel_size = 3, stride=1, padding="valid"),
                nn.GELU(),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size = 3, stride=2, padding="valid"),
         )
        
        self.avg_pool = nn.AdaptiveAvgPool1d(d_in)

        self.linear_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=d_in*d_model, out_features=2*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=2*d_model, out_features=d_model),
            nn.GELU(),
            nn.Linear(in_features=d_model, out_features=d_out),
        )
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.CNN_layers(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layers(x)

        return x