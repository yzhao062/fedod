# this is to calculate baseline for kNN

import os
import torch
from torch import nn

from torch import optim

import numpy as np
from sklearn.utils import check_array
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd

# from torchmetrics.functional import spearman_corrcoef
from scipy.io import loadmat

from pyod.utils.utility import standardizer

from time import time

def bottomk(A, k, dim=1):
    if len(A.shape) == 1:
        dim = 0
    # tk = torch.topk(A * -1, k, dim=dim)
    # see parameter https://pytorch.org/docs/stable/generated/torch.topk.html
    tk = torch.topk(A, k, dim=dim, largest=False)
    return tk[0].cpu(), tk[1].cpu()

from pyod.utils.utility import precision_n_scores
class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None):
        super(PyODDataset, self).__init__()
        self.X = X
        # self.mean = mean
        # self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.X[idx, :]
        sample_torch = torch.from_numpy(sample)
        # print(sample_torch.shape)

        # calculate the local knn
        # dist = torch.cdist(sample_torch, sample_torch)

        return sample_torch, idx
    

def collate_batch2(batch):
    samples = []
    idxs = []
    
    for (sample, idx) in batch:
        samples.append(sample)
        idxs.append(idx)
    
    # print(len(samples))
    # samples = torch.Tensor(samples)
    samples = torch.stack(samples)
    # samples = sample.float()
    # print(samples.shape)
    idxs  = torch.tensor(idxs)
    # print(samples)
    dists = torch.cdist(samples, samples)
    
    return samples.float(), dists.float(), idxs


import torchsort

def corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()*-1


def spearman(
    target,
    pred,
    regularization="l2",
    regularization_strength=1.0,
):
    pred = torchsort.soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    return corrcoef(target, pred / pred.shape[-1])


class NeuralNetwork(nn.Module):
    def __init__(self, n_features, n_samples, hidden_neuron=16, n_layers=2):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        layer_list = []
        
        layer_list.append(nn.Linear(n_features, hidden_neuron)),
        layer_list.append(nn.ELU())
        for i in range(n_layers):

            layer_list.append(nn.Linear(hidden_neuron, hidden_neuron))
            layer_list.append(nn.ELU())
        
        layer_list.append(nn.Linear(hidden_neuron, n_samples))
        
        self.simple_nn = nn.Sequential(*layer_list)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.simple_nn(x)
        return logits
    
# if __name__ == "__main__": 
def train(mat_file):

    # mat_file = '19_annthyroid'
    prediction_time = 0
    calc_time = 0 
    
    
    X = pd.read_csv(os.path.join('data', mat_file+'_X.csv'), header=None).to_numpy()
    y = pd.read_csv(os.path.join('data', mat_file+'_y.csv'), header=None).to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, shuffle=True, random_state=42)
 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, shuffle=True, random_state=42)
    
    
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    
    X_train, scalar = standardizer(X_train, keep_scalar=True)
    X_test = scalar.transform(X_test)
    X_valid = scalar.transform(X_valid)
    
    batch_size = 200
    
    train_set = PyODDataset(X=X_train)    
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch2,
                                               shuffle=False)
    
    test_set = PyODDataset(X=X_test)    
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch2,
                                               shuffle=False)
    
    
    valid_set = PyODDataset(X=X_valid)    
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch2,
                                               shuffle=False)
    
    
    k=10

    batch_dist = []
    for batch_idx, batch in enumerate(test_loader):
        # print(len(batch[1][:, k]))
        batch_dist.extend(batch[1][:, k].numpy().tolist())
    
    print(mat_file, 'test base', roc_auc_score(y_test, batch_dist))
    return roc_auc_score(y_test, batch_dist)

# Initialize a pandas DataFrame
df = pd.DataFrame(columns=['file', 'roc'])

files = pd.read_csv('file_list.csv', header=None).values.tolist()

for i, mat_file in enumerate(files[25:]):   
    real_roc = train(mat_file=mat_file[0])
    print(mat_file)
    # Append the list to the DataFrame
    df = df.append(pd.Series([mat_file, real_roc], index=df.columns), ignore_index=True)