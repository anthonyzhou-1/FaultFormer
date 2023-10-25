# FaultFormer 

## Data
- Data is available at: http://manufacturingnet.io/html/datasets.html
- The codebase expects data in raw, unfeaturized form. In most cases, this means tensors of shape (batch_size, seq_len)
- Data in the codebase was organized into the following folder structure, although any structure can be used. (Will have to modify data paths, etc.)
- CWRU Data is organized in the /CWRU folder. 
    - /kfold
        - Holds main training and testing data. 
        - data is a torch tensor of shape (n_samples, n_data_points), which is (2240, 1600) for training, and (560, 1600) for test
        - labels is a torch tensor of shape (n_samples), which is (2240) for training, and (560) for test
        - Each fold has a unique train-test split of the raw data/labels. The best model was trained on fold 1
    - signal_data.npy 
        - Raw data of shape (2800, 1600)
    - signal_data_labels.npy
        - Raw labels of shape (2800)
- Paderborn Data is organized in the same way.

## Training
- Hyperparameters on the CWRU Dataset: 
    - params = {
        "batch_size": 1024,
        "epochs": 11000,
        "d_in": 3,
        "d_model": 140,
        "nhead": 20,
        "d_hid": 300,
        "nlayers": 6,
        "dropout": 0.3,
        "warmup": 4000,
        "seq_len": 40,
        "n_classes": 10,
        "model": "Transformer_cls",
        "fourier": True,
        "p_no_aug": .1,
        "p_two_aug": .5,
    }

- Hyperparameters on the Paderborn Dataset: 
    - params = {
        "batch_size": 1024,
        "epochs": 1000,
        "d_in": 3,
        "d_model": 256,
        "nhead": 16,
        "d_hid": 512,
        "nlayers": 6,
        "dropout": 0.3,
        "warmup": 4000,
        "seq_len": 40,
        "n_classes": 3,
        "model": "Transformer_cls",
        "fourier": True,
        "p_no_aug": .1,
        "p_two_aug": .5,
    }

- examples.ipynb contains examples explaining how to use codebase for training, pretraining, and visualizing results


