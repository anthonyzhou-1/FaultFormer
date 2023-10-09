import torch
from torch import nn
from end_to_end import train_model
from fault_dataset import AugmentedDataset, AugmentedDatasetTest, FaultDataset
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn_transpose, NoamOpt

from models import TransformerModel_cls

params = {
    "batch_size": 64,
    "epochs": 11000,
    "d_in": 1,
    "d_model": 100,
    "nhead": 20,
    "d_hid": 200,
    "nlayers": 3,
    "dropout": 0.3,
    "warmup": 4000,
    "seq_len": 400,
    "d_lin": 512,
    "d_out": 64,
    "freeze": False,
    "n_classes": 10,
    "lr_head": None,
    "model": "Transformer_cls",
    "path": None,
    "fourier": False,
    "margin": .05,
    "p_no_aug": .1,
    "p_two_aug": .5,
    "fold": 1,
}

model = TransformerModel_cls(  d_in = params["d_in"], 
                            d_model = params["d_model"], 
                            nhead = params["nhead"], 
                            d_hid = params["d_hid"],
                            nlayers = params["nlayers"], 
                            seq_len = params["seq_len"], 
                            dropout = params["dropout"])

fold = params["fold"]
train_dir = "data/5folddata/training_data_" + str(fold)
test_dir = "data/5folddata/test_data_" + str(fold)
train_dir_l = "data/5folddata/training_labels_" + str(fold)
test_dir_l = "data/5folddata/test_labels_" + str(fold)

batch_size = params["batch_size"]
p_no_aug = params["p_no_aug"]
p_two_aug = params["p_two_aug"]

training_data = AugmentedDataset(p_no_aug, p_two_aug, fourier=False, data_path=train_dir, label_path=train_dir_l)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn = collate_fn_transpose)

test_data = AugmentedDatasetTest(fourier=False, data_path=test_dir, label_path=test_dir_l)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn = collate_fn_transpose)

train_model(model, params, train_dataloader=train_dataloader, test_dataloader=test_dataloader)





