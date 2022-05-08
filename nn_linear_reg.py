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

import numpy as np
from sklearn.linear_model import LinearRegression
# create dummy data for training
# x_values = [i for i in range(11)]
# x_train = np.array(x_values, dtype=np.float32)
# x_train = x_train.reshape(-1, 1)

# y_values = [2*i + 1 for i in x_values]
# y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
    
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
        
        sample_y = torch.from_numpy(self.y[idx])
        
        # print(sample_y.shape)
        # print(sample_torch.shape)

        # calculate the local knn
        # dist = torch.cdist(sample_torch, sample_torch)
        # assert(sample_torch.shape[0] == len(sample_y))
        
        return sample_torch,sample_y, idx

    


#%%

if __name__ == "__main__": 
# def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prediction_time = 0
    calc_time = 0

    epochs = 100
    batch_size = 200
    n_estimators = 10

    mat_file = '9_PenDigits'
    # mat_file = '13_Stamps'
    # mat_file = '21_breastw'
    # mat_file = '22_glass'
    # mat_file = '26_mammography'
    # mat_file = '36_vertebral'
    # mat_file = '39_wine'
    # mat_file = 'pendigits.mat'
    # mat_file = 'letter.mat'
    # mat_file = 'mnist.mat'
    # mat_file = 'annthyroid.mat'
    
    # mat_file = 'arrhythmia.mat'
    
    # mat = loadmat(os.path.join('data', mat_file))
    # X = mat['X']
    # y = mat['y'].ravel()

    X = pd.read_csv(os.path.join('data', mat_file+'_X.csv'), header=None).to_numpy()
    y = pd.read_csv(os.path.join('data', mat_file+'_y.csv'), header=None).to_numpy()

    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, shuffle=True,
                                                              random_state=42)
    
    
    X_train, scalar = standardizer(X_train, keep_scalar=True)
    X_test = scalar.transform(X_test)
    X_valid = scalar.transform(X_valid)
    
    inputDim = X_train.shape[1]        # takes variable 'x' 
    outputDim = 1       # takes variable 'y'
    learningRate = 0.03

    
    model = linearRegression(inputDim, outputDim)
    mode = model.to(device)
        
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    
    train_set = PyODDataset(X=X_train, y=y_train)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    train_losses = []
    
    valid_rocs = []
    valid_aps = []
    
    test_rocs = []
    test_aps = []
    
    best_valid = 0
    best_test = 0
    
    for epoch in range(epochs):
        
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # print(epoch, batch_idx)

            inputs = batch[0].to(device).float()
            labels = batch[1].to(device).float()
    
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()
        
            # get output from the model, given the inputs
            outputs = model(inputs)
        
            # get loss for the predicted output
            loss = criterion(outputs, labels)
            # get gradients w.r.t to parameters
            loss.backward()
        
            # update parameters
            optimizer.step()
            total_loss += loss.item()
        
        train_losses.append(total_loss)
        print('epoch {}, loss {}'.format(epoch, total_loss))
        
        with torch.no_grad():

            # doing evaluation
            y_pred = model(torch.from_numpy(X_valid).to(device).float()).cpu().numpy()
            valid_rocs.append(roc_auc_score(y_valid, y_pred))
            valid_aps.append(average_precision_score(y_valid, y_pred))
            
            y_pred_test = model(torch.from_numpy(X_test).to(device).float()).cpu().numpy()
            test_rocs.append(roc_auc_score(y_test, y_pred_test))
            test_aps.append(average_precision_score(y_test, y_pred_test))
            
            if best_valid <= roc_auc_score(y_valid, y_pred):
                best_valid = roc_auc_score(y_valid, y_pred)
                best_valid_ap = average_precision_score(y_valid, y_pred)
                
                best_test = roc_auc_score(y_test, y_pred_test)
                best_test_ap = average_precision_score(y_test, y_pred_test)
                
        print(epoch, 'pred valid roc {}, ap {}'.format(best_valid, best_valid_ap))
        print(epoch, 'pred test roc {}, ap {}'.format(best_test, best_test_ap))
        print()

coef_nn = model.state_dict()['linear.weight'][0].cpu().numpy().tolist()
bias_nn = model.state_dict()['linear.bias'][0].cpu().numpy().tolist()

print('best', best_test, best_test_ap)

#%%

clf = LinearRegression()
clf.fit(X_train, y_train)

coef_sk = clf.coef_[0].tolist() 
bias_sk = clf.intercept_[0].tolist()

pred_test = clf.predict(X_test)
lr_test_roc = roc_auc_score(y_test, pred_test)
lr_test_ap = average_precision_score(y_test, pred_test)
print('lr', lr_test_roc, lr_test_ap)

#%%

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)

plt.plot(x_range, np.log(train_losses))
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("log of training loss (MSE)")
plt.title(mat_file)
plt.show()


#%%
plt.style.use('seaborn-whitegrid')
import numpy as np

plt.figure(figsize=(6,5))

x_range = np.arange(epochs)

# plt.plot(x_range, train_rocs, label='train')
# plt.plot(x_range, valid_rocs, label='valid')
plt.plot(x_range, test_rocs, label='test roc', color='black')
plt.hlines(lr_test_roc, xmin=0, xmax=epochs, label='classical lr roc', linestyle='--', color='black')

plt.plot(x_range, test_aps, label='test ap', color='red')
plt.hlines(lr_test_ap, xmin=0, xmax=epochs, label='classical lr ap', linestyle='--', color='red')


# plt.plot(x_range, valid_aps, label='valid ap')
# plt.plot(x_range, test_aps, label='test ap')
# plt.hlines(test_ap_iforest, xmin=0, xmax=epochs, label='test ap', linestyle='--')
# plt.xticks([0,10, 20, 30, 50])
# plt.grid(False)
plt.xlabel("number of epochs", size=12)
plt.ylabel("performance (roc and ap)", size=15)
plt.title(mat_file)

plt.legend()
plt.show()
