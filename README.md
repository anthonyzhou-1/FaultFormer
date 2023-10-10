# FaultFormer 

## Data
- Data is organized in the /data folder. 
    - /5folddata
        - Holds main training and testing data. 
        - data is a torch tensor of shape (n_samples, n_data_points), which is (2240, 1600) for training, and (560, 1600) for test
        - labels is a torch tensor of shape (n_samples), which is (2240) for training, and (560) for test
        - Each fold has a unique train-test split of the raw data/labels. The best model was trained on fold 1
    - signal_data.npy 
        - Raw data of shape (2800, 1600)
    - signal_data_labels.npy
        - Raw labels of shape (2800)

## Training
- Hyperparameters for base model: 
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
        "d_lin": 512,
        "d_out": 64,
        "n_classes": 10,
        "model": "Transformer_cls",
        "fourier": True,
        "p_no_aug": .1,
        "p_two_aug": .5,
    }

- examples.ipynb contains examples explaining how to use codebase for training, pretraining, and visualizing results


