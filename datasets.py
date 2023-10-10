import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
from utils import cutout, gaussian_noise, shift, crop, get_fourier_features

class AugmentedDataset(Dataset):
    def __init__(self, data_path = None, label_path = None, p_no_aug=1, p_two_aug=0, fourier=True, k = 4, test = False, mask=False):
        '''
        Arguments:
            data_path: Path to data
            label_path: Path to labels
            p_no_aug: Probability of no data augmentation
            p_two_aug: Probability of applying two data augmentations
            fourier: Flag to implement fourier downsampling
            k: factor to decimate signal down to, if not using fourier downsampling
            test: Flag for testing mode (no augmentations applied during testing)
            mask: Flag for masking mode (no labels are passed when predictive pretraining)
        '''

        if(data_path == None):
            raise Exception("Must specify path to data")

        self.data = torch.load(data_path)

        if(label_path != None):
            self.labels = torch.load(label_path)
        
        self.aug = ["noise", "cutout", "shift", "crop"]
        self.p_no_aug = p_no_aug
        self.p_two_aug = p_two_aug
        self.fourier = fourier
        self.test = test
        self.mask = mask
        self.k = k

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Processes a single input sequence by augmenting and downsampling
        '''

        x = self.data[idx]

        if(self.test == False):
            sample = torch.rand(1).item()

            if(sample>self.p_no_aug):
                sample_aug = torch.rand(1).item()

                if(sample_aug>self.p_two_aug):
                    aug_idx = torch.randint(0,4, (1,)).item()
                    aug = self.aug[aug_idx]

                    if(aug == "noise"):
                        x = gaussian_noise(x)
                    elif(aug == "cutout"):
                        x = cutout(x)
                    elif(aug == "shift"):
                        x = shift(x)
                    else:
                        x = crop(x)
                else:
                    aug_idx = torch.randint(0,4, (1,)).item()
                    if(aug_idx == 0):
                        x = cutout(shift(x))
                    elif(aug_idx == 1):
                        x = cutout(gaussian_noise(x))
                    elif(aug_idx == 2):
                        x = crop(shift(x))
                    elif(aug_idx == 3):
                        x = crop(gaussian_noise(x))

        if(self.fourier):
            x = get_fourier_features(x, k=40)
        else:
            x = signal.decimate(x, self.k)
            x = torch.from_numpy(x.copy())
            x = x.unsqueeze(1)

        if(self.mask):
            return x

        return x, self.labels[idx]