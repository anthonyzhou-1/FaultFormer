import math
from functools import reduce
import torch
import numpy as np
from torch import nn, Tensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def collate_fn_transpose(data):        
    """
    Arguments:
        data: List of tuples of (data, label)

    Returns:
        features: Design matrix of shape [seq_len, batch_size, d] 
        labels: labels of shape [batch_size, 1]
    """

    if(type(data[0]) is tuple):
        batch_size = len(data)
        seq_len, d = data[0][0].shape

        features = torch.zeros(seq_len, batch_size, d)
        labels = torch.zeros(batch_size, dtype=torch.long)

        for idx, (x, y) in enumerate(data):
            features[:, idx, :] = x
            labels[idx] = y
        
        return features, labels

    else:
        batch_size = len(data)
        seq_len, d = data[0].shape

        features = torch.zeros(seq_len, batch_size, d)

        for idx, x in enumerate(data):
            features[:, idx, :] = x
        
        return features

def train_val_split(data, labels, val_split=0.2):
    """
    Arguments:
        data: torch tensor of shape (n, seq_len)
        labels: torch tensor of shape (n)
        val_split: percent of data to form validation split

    Returns:
        x_train: training data
        y_train: training labels
        x_valid: validation data
        y_valid: validation labels
    """

    train_idx, val_idx = train_test_split(list(range(len(labels))), test_size=val_split, random_state=42)
    x_train = data[train_idx]
    y_train = labels[train_idx]

    x_valid = data[val_idx]
    y_valid = labels[val_idx]

    return x_train, y_train, x_valid, y_valid

class NoamOpt:
    "Optim wrapper that implements rate. Citation: https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch"

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 

"Masking functions from: https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py"

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

def mask(seq, mask_prob=0.15, replace_prob=0.9, random_token_prob=0.1):

    mask_token_id = 0

    mask = prob_mask_like(seq, mask_prob)
    mask_out = mask.clone().detach()

    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)

    masked_seq = seq.clone().detach()

    idx = torch.randperm(seq.nelement())
    random_tokens = seq.view(-1)[idx].view(seq.size())

    random_token_prob = prob_mask_like(seq, random_token_prob) & mask
    masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

    # remove tokens that were substituted randomly from being [mask]ed later
    mask = mask & ~random_token_prob

    # [mask] input

    replace_prob = prob_mask_like(seq, replace_prob)
    masked_seq = masked_seq.masked_fill(mask * replace_prob, mask_token_id)

    return masked_seq, mask_out
    
def complex_mag(data):
    '''
    Arguments:
        data: Tensor of shape (seq_len)

    Returns:
        mag: Complex magnitude of input sequence
    '''

    data_r = data.real
    data_i = data.imag

    mag = torch.sqrt(data_r*data_r + data_i*data_i)

    return mag

def get_fourier_features(data, k=40):
    '''
    Arguments:
        data: Tensor of shape (seq_len)
        k: Top k fourier modes to truncate input to
    
    Returns:
        features: Tensor of shape (k, 3)
    '''

    data_spectral = torch.fft.rfft(data)
    freqs = torch.fft.rfftfreq(len(data))
    mag = complex_mag(data_spectral)

    max_amp, max_idx = torch.topk(mag, k)

    data_r = data_spectral.real
    data_i = data_spectral.imag

    top_r = data_r[max_idx]
    top_i = data_i[max_idx]
    top_f = freqs[max_idx]

    features = torch.stack((top_r, top_i, top_f), dim=1)
    
    return features

def crop(x, start = None):
    '''
    Arguments:
        x: Tensor of shape (seq_len)
        start: start of cropping window
    
    Returns: 
        aug: Tensor of shape (seq_len)

    Crops signal by half beginning at start, then resizes to original length
    '''
    cropped_size = int(len(x)/2)
    if start == None:
        start = torch.randint(0, cropped_size, (1,)).item()

    cropped = x[start:start+cropped_size].unsqueeze(0).unsqueeze(2)
    m = torch.nn.Upsample(scale_factor=2)
    aug = m(cropped)
    aug = torch.flatten(aug)
    return aug

def cutout(x, window=None):
    '''
    Arguments:
        x: Tensor of shape (seq_len)
        window: Size of window to cut out
    
    Returns: 
        aug: Tensor of shape (seq_len)

    Zeros out a window in x. Window values from 100 to 400
    '''

    if window == None:
        window = torch.randint(100,500, (1,)).item()

    l = len(x)
    start = torch.randint(0,l-window, (1,))
    zeros = torch.zeros(window)

    aug = x.detach().clone()
    aug[start:start+window] = zeros

    return aug

def gaussian_noise(x, std=None):
    '''
    Arguments:
        x: Tensor of shape (seq_len)
        std: standard deviation of Gaussian Noise to add
    
    Returns: 
        aug: Tensor of shape (seq_len)

    Adds gaussian noise to signal. std values between 0 and .1
    '''

    if std== None:
        std = .1*torch.rand((1,))

    noise = std*torch.randn(x.shape)
    return x + noise

def shift(x, k=None):
    '''
    Arguments:
        x: Tensor of shape (seq_len)
        k: Time scale to shift signals
    
    Returns: 
        aug: Tensor of shape (seq_len)

    Shifts signal across time. k values between -800 and 800
    '''
    shift_len = int(len(x)/2)
    dims = -1
    if(k == None):
        k = torch.randint(-1*shift_len, shift_len, (1,)).tolist()

    return torch.roll(x, k, dims)

def signal_power(x):
    '''
    Arguments:
        x: tensor of shape (seq_len)
    
    Returns:
        power of signal, determined by sum of squares
    '''

    n = len(x)
    return torch.sum(torch.square(x))/n

def noisify(x, SNR):
    '''
    Arguments:
        x: tensor of shape (batch size, seq_len)
        SNR: signal to noise ratio to determine how much noise to add
    
    Returns:
        out: tensor of shape (batch size, seq_len)
    '''

    out = torch.zeros_like(x)

    L, n = x.shape
    den = 10**(SNR/10)

    for i in range(L):
        signal = x[i]
        Pn = signal_power(signal)/den
        out[i] = gaussian_noise(signal, std=Pn)

    return out

def fourier_iterate(x):
    x_fourier = torch.zeros((560, 40, 3))

    for i in range(len(x)):
        x_fourier[i] = get_fourier_features(x[i], k=40)
    return x_fourier

def encoder_forward(model, x):
    '''
    Arguments:
        model: Model to compute forward pass with
        x: tensor of shape (seq_len, batch size, d_in)
    
    Returns:
        out: tensor of shape (seq_len, batch size, d_in)

    Performs partial forwad pass of encoder, to incrementally pass to encoder for attention visualization
    '''

    src = model.embedding(x)*math.sqrt(model.d_model)
    src = model.cls(src)
    src = model.pos_encoder(src)
    return src

def visualize_attn(w0, b0, src, head):
    '''
    Arguments:
        w0: input projection, as defined by torch
        b0: input bias, as defined by torch
        src: tensor of shape (seq_len, batch size, d_in)
        head: head to calculate attention for
    
    Returns:
        alpha: attention scores at the current layer and head.
    '''
    s = nn.Softmax(dim=1)
    w0_t = torch.transpose(w0, 0,1)
    qkv = torch.matmul(src, w0_t)
    qkv = qkv+b0

    head_start = head*7
    head_end = head_start + 7
    q = qkv[:, :, :140].squeeze()
    k = qkv[:, :, 140:280].squeeze()
    v = qkv[:, : ,280:].squeeze()
    
    q_i = q[:, head_start:head_end]
    k_i = k[:, head_start:head_end]
    v_i = v[:, head_start:head_end]

    qk_t = torch.matmul(q_i, torch.transpose(k_i, 0, 1))
    qk_t = qk_t.detach()

    alpha = s(qk_t)

    return alpha.detach()