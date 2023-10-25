import torch
from train import train_model
from datasets import AugmentedDataset
from torch.utils.data import DataLoader
from utils import collate_fn_transpose
from models import TransformerModel_cls
from utils import NoamOpt

params = {
        "batch_size": 200,
        "epochs": 1000,
        "d_in": 1,
        "d_model": 128,
        "nhead": 16,
        "d_hid": 256,
        "nlayers": 4,
        "dropout": 0.3,
        "warmup": 4000,
        "seq_len": 250,
        "n_classes": 3,
        "model": "Transformer_cls",
        "p_no_aug": 1,
        "p_two_aug": 0,
        "fourier": False,
    }

model = TransformerModel_cls(d_in = params["d_in"], 
                            d_model = params["d_model"], 
                            nhead = params["nhead"], 
                            d_hid = params["d_hid"],
                            nlayers = params["nlayers"], 
                            seq_len = params["seq_len"], 
                            dropout = params["dropout"],
                            n_classes = params["n_classes"])

fold = 0
train_dir = "CwRU/kfold/training_data_" + str(fold)
test_dir = "CWRU/kfold/test_data_" + str(fold)
train_dir_l = "CWRU/kfold/training_labels_" + str(fold)
test_dir_l = "CWRU/kfold/test_labels_" + str(fold)

batch_size = params["batch_size"]
p_no_aug = params["p_no_aug"]
p_two_aug = params["p_two_aug"]

training_data = AugmentedDataset(data_path=train_dir, label_path=train_dir_l, p_no_aug=p_no_aug, p_two_aug=p_two_aug, fourier=False, test=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn = collate_fn_transpose)

test_data = AugmentedDataset(data_path=test_dir, label_path=test_dir_l, fourier=False, test=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn = collate_fn_transpose)

optimizer = NoamOpt(params["d_model"], params["warmup"],
                torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))

train_model(model, params, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer = optimizer)