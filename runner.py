import torch
from train import train_model
from datasets import AugmentedDataset
from torch.utils.data import DataLoader
from utils import collate_fn_transpose
from models import TransformerModel_cls

params = {
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

model = TransformerModel_cls(d_in = params["d_in"], 
                            d_model = params["d_model"], 
                            nhead = params["nhead"], 
                            d_hid = params["d_hid"],
                            nlayers = params["nlayers"], 
                            seq_len = params["seq_len"], 
                            dropout = params["dropout"])

fold = 1
train_dir = "data/5folddata/training_data_" + str(fold)
test_dir = "data/5folddata/test_data_" + str(fold)
train_dir_l = "data/5folddata/training_labels_" + str(fold)
test_dir_l = "data/5folddata/test_labels_" + str(fold)

batch_size = params["batch_size"]
p_no_aug = params["p_no_aug"]
p_two_aug = params["p_two_aug"]

training_data = AugmentedDataset(data_path=train_dir, label_path=train_dir_l, p_no_aug=p_no_aug, p_two_aug=p_two_aug, fourier=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn = collate_fn_transpose)

test_data = AugmentedDataset(data_path=test_dir, label_path=test_dir_l, fourier=True, test=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn = collate_fn_transpose)

train_model(model, params, train_dataloader=train_dataloader, test_dataloader=test_dataloader)