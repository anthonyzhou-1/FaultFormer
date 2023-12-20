import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import re 

class SignalDataset(Dataset):
    def __init__(self, data_path, label_path=None, std=True):
        '''
        Arguments:
            data_path: Path to data
            label_path: Path to labels
        '''

        if(data_path == None):
            raise Exception("Must specify path to data")

        self.data = torch.load(data_path)

        if(label_path != None):
            self.labels = torch.load(label_path)
        else:
            self.labels = None

        self.std = std

    def __len__(self):
        return len(self.data)
    
    def standardize(self, x):
        means = x.mean(dim=0, keepdim=True)
        stds = x.std(dim=0, keepdim=True)
        normalized_data = (x - means) / stds
        return normalized_data

    def __getitem__(self, idx):

        data = self.data[idx].type(torch.FloatTensor)
        if self.std:
            data = self.standardize(data)

        if self.labels is None:
            return data

        else: 
            return data, self.labels[idx].type(torch.LongTensor)
    

def get_fourier_features(data, k=40):
    '''
    Arguments:
        data: Tensor of shape (seq_len)
        k: Top k fourier modes to truncate input to
    
    Returns:
        features: Tensor of shape (k, 3)
    '''

    data_spectral = torch.fft.rfft(data)
    freqs = torch.fft.rfftfreq(data.shape[-1]).to(data.device)

    data_r = data_spectral.real
    data_i = data_spectral.imag
    mag = torch.sqrt(data_r*data_r + data_i*data_i)

    max_amp, max_idx = torch.topk(mag, k)

    top_r, top_i, top_f = torch.Tensor().to(data.device), torch.Tensor().to(data.device), torch.Tensor().to(data.device)
    for i in range(len(data_r)):
        top_r = torch.cat([top_r, data_r[i][max_idx[i]].unsqueeze(0)], dim = 0)
        top_i = torch.cat([top_i, data_i[i][max_idx[i]].unsqueeze(0)], dim = 0)
        top_f = torch.cat([top_f, freqs[max_idx[i]].unsqueeze(0)], dim = 0)

    features = torch.stack((top_r, top_i, top_f), dim=2)
    
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
    m = torch.nn.Upsample(scale_factor=2, mode = 'linear')
    aug = m(cropped)
    aug = torch.flatten(aug)
    return aug.to(x.device)

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

    return aug.to(x.device)

def gaussian_noise(x, std=None):
    '''
    Arguments:
        x: Tensor of shape (seq_len)
        std: standard deviation of Gaussian Noise to add
    
    Returns: 
        aug: Tensor of shape (seq_len)

    Adds gaussian noise to signal. std values between 0 and .1
    '''
    device = x.device
    if std== None:
        std = .05*torch.rand((1,)).to(device)

    noise = std*(torch.randn(x.shape).to(device))
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

    return torch.roll(x, k, dims).to(x.device)

def random_augmentation(x):
    '''
    Arguments:
        x: Tensor of shape (seq_len)
    
    Returns: 
        aug: Tensor of shape (seq_len)

    Randomly augments signal
    '''
    aug_idx = torch.randint(0,8, (1,)).item()

    if(aug_idx == 0):
        x = gaussian_noise(x)
    elif(aug_idx == 1):
        x = cutout(x)
    elif(aug_idx == 2):
        x = shift(x)
    elif(aug_idx == 3):
        x = crop(x)
    elif(aug_idx == 4):
        x = cutout(shift(x))
    elif(aug_idx == 5):
        x = cutout(gaussian_noise(x))
    elif(aug_idx == 6):
        x = crop(shift(x))
    elif(aug_idx == 7):
        x = crop(gaussian_noise(x))

    return x

def process_dict(state_dict: OrderedDict) -> OrderedDict:
    '''
    Processes state dict to remove the 'module.' prefix'''

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    
    return model_dict