# -*- coding: utf-8 -*-
"""
Created on Fri May  6 22:45:31 2022

@author: yzhao
"""

import os
from functools import partial
import itertools

import torch
from torch import nn
from sklearn.cluster import KMeans
from combo.models.cluster_comb import ClustererEnsemble
from combo.models.cluster_eac import EAC

from torch import optim

import numpy as np
from sklearn.utils import check_array
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from scipy.io import loadmat

from pyod.utils.utility import standardizer
from functools import reduce
import pandas as pd

import time
from scipy.stats import rankdata


# class nn_svd(torch.nn.Module):
#     def __init__(self, n, m, k):
#         super(nn_svd, self).__init__()
#         self.linear = torch.nn.Linear(inputSize, outputSize)

#     def forward(self, x):
#         out = self.linear(x)
#         return out


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.y = y
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
        
        # sample_y = torch.from_numpy(self.y[idx])
        
        # print(sample_y.shape)
        # print(sample_torch.shape)

        # calculate the local knn
        # dist = torch.cdist(sample_torch, sample_torch)
        # assert(sample_torch.shape[0] == len(sample_y))
        
        return sample_torch, idx
    

if __name__ == "__main__": 
# def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prediction_time = 0
    calc_time = 0

    epochs = 1000
    batch_size = 200
    n_estimators = 10

    # mat_file = '9_PenDigits'
    # mat_file = '13_Stamps'
    # mat_file = '21_breastw'
    # mat_file = '22_glass'
    # mat_file = '26_mammography'
    # mat_file = '24_letter'
    # mat_file = '1_ALOI'
    mat_file = '36_vertebral'
    # mat_file = '39_wine'
    
    X = pd.read_csv(os.path.join('data', mat_file+'_X.csv'), header=None).to_numpy()
    y = pd.read_csv(os.path.join('data', mat_file+'_y.csv'), header=None).to_numpy()
    
    X, scalar = standardizer(X, keep_scalar=True)

    train_set = PyODDataset(X=X)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    k = min(n_samples, n_features)
    
    
    assert(k <= n_samples)
    assert(k <= n_features)
    
    U = torch.rand([n_samples, k], requires_grad=True, device=device)
    # S = torch.zeros([k, k], requires_grad=True, device=device)
    # sl = torch.rand(k)
    V = torch.rand([k, n_features], requires_grad=True, device=device)
    
    # print(sl)
    
    loss_function = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam([U] + [sl] + [V], lr=0.01)
    optimizer = torch.optim.Adam([U] + [V], lr=0.01)
    
    train_losses = []
    reconstruction = []
    
    for epoch in range(epochs):
        
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # print(epoch, batch_idx)

            inputs = batch[0].to(device).float()
            local_idx = batch[1]
            
            optimizer.zero_grad()
            
            # S = torch.diag_embed(sl).to(device)
            # this is available
            # pred = torch.matmul(torch.matmul(U, S), V)
            pred = torch.matmul(U, V)
            
            # local pred
            local_pred = pred[local_idx, :]
            
            loss = loss_function(inputs, local_pred)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print('epoch {}, loss {}'.format(epoch, total_loss))
        train_losses.append(total_loss)
        
        with torch.no_grad():
            # pred = torch.matmul(torch.matmul(U, S), V)
            pred = torch.matmul(U, V)
            mse = loss_function(torch.from_numpy(X).to(device), pred)
            print('epoch {}, mse {}'.format(epoch, mse))
            reconstruction.append(mse.item())
            print()
    np.save(os.path.join('data', mat_file+'_U.npy'), U.cpu().detach().numpy())
#%%
def bottomk(A, k, dim=1):
    if len(A.shape) == 1:
        dim = 0
    # tk = torch.topk(A * -1, k, dim=dim)
    # see parameter https://pytorch.org/docs/stable/generated/torch.topk.html
    tk = torch.topk(A, k, dim=dim, largest=False)
    return tk[0].cpu(), tk[1].cpu()

with torch.no_grad():
    Ut, St, Vt = torch.linalg.svd(torch.from_numpy(X).to(device), full_matrices=False)
    pred = torch.matmul(torch.matmul(Ut, torch.diag_embed(St)), Vt)
    mse = loss_function(torch.from_numpy(X).to(device), pred)
    
    print('torch mse', mse.item())
    
    real_dist = torch.cdist(torch.tensor(Ut).float(), torch.tensor(Ut).float())
    dist_real, idx_real = bottomk(real_dist, k=10)
    
    print('roc is', roc_auc_score(y, dist_real[:, -1]))

#%%

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)

plt.plot(x_range, np.log(train_losses))
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("local training loss (MSE)")
plt.title(mat_file)
plt.show()

#%%
plt.style.use('seaborn-whitegrid')
import numpy as np

plt.figure(figsize=(6,5))

x_range = np.arange(epochs)

# plt.plot(x_range, train_rocs, label='train')
# plt.plot(x_range, valid_rocs, label='valid')
plt.plot(x_range, reconstruction, label='reconstruction by nn', color='black')
# plt.hlines(lr_test_roc, xmin=0, xmax=epochs, label='classical lr roc', linestyle='--', color='black')

# plt.plot(x_range, test_aps, label='test ap', color='red')
plt.hlines(mse.item(), xmin=0, xmax=epochs, label='reconstruction by decomposation', linestyle='--', color='red')


# plt.plot(x_range, valid_aps, label='valid ap')
# plt.plot(x_range, test_aps, label='test ap')
# plt.hlines(test_ap_iforest, xmin=0, xmax=epochs, label='test ap', linestyle='--')
# plt.xticks([0,10, 20, 30, 50])
# plt.grid(False)
plt.xlabel("number of epochs", size=12)
plt.ylabel("reconstruction error", size=15)
plt.title(mat_file)

plt.legend()
plt.show()