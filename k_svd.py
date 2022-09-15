#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:33:15 2022

@author: yuezhao
"""

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

from scipy.io import loadmat

from pyod.utils.utility import standardizer

from time import time
from utility import validate_device

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
    
if __name__ == "__main__": 
# def train():
    # contamination = 0.1  # percentage of outliers
    # n_train = 10000  # number of training points
    # n_test = 2000  # number of testing points
    # n_features = 100  # number of features

    # # Generate sample data
    # X_train, y, X_test, y_test = \
    #     generate_data(n_train=n_train,
    #                   n_test=n_test,
    #                   n_features=n_features,
    #                   contamination=contamination,
    #                   random_state=42)
        
    prediction_time = 0
    calc_time = 0 
    
    # mat_file = 'pendigits.mat'
    # mat_file = 'letter.mat'
    # mat_file = 'mnist.mat'
    # mat_file = 'annthyroid.mat'

    # mat_file = '19_annthyroid'
    # mat_file = '27_mnist'    
    # mat_file = '24_letter'
    # mat_file = '13_Stamps'
    # mat_file = '21_breastw'
    # mat_file = '22_glass'
    # mat_file = '26_mammography'
    # mat_file = '24_letter'
    # mat_file = '1_ALOI'
    mat_file = '36_vertebral'
    # mat_file = '36_vertebral'
    # mat_file = '39_wine'
    # mat_file = '1_ALOI'
      # mat_file = 'letter.mat'
      # mat_file = 'mnist.mat'
      # mat_file = 'annthyroid.mat'
      
      # mat_file = 'arrhythmia.mat'
      
      # mat = loadmat(os.path.join('data', mat_file))
      # X = mat['X']
      # y = mat['y'].ravel()
    
    X = np.load(os.path.join('data', mat_file+'_U.npy'))
    y = pd.read_csv(os.path.join('data', mat_file+'_y.csv'), header=None).to_numpy()

    # mat = loadmat(os.path.join('data', mat_file))
    # X = mat['X']
    # y = mat['y'].ravel()
    
    
    X_train, scalar = standardizer(X, keep_scalar=True)
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    
    batch_size = 200
    
    train_set = PyODDataset(X=X_train)    
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch2,
                                               shuffle=False)
    
    
    device = validate_device(0)
    
    model = NeuralNetwork(n_features=n_features, n_samples=n_samples, hidden_neuron=64, n_layers=2)
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=.01)
    # criterion = nn.NLLLoss()
    epochs = 100
    
    
    train_losses = []
    inter_track = []
    kendall_tracker = []
    
    best_train = 0
    
    k=20
    # mse_tracker = []
    for e in range(epochs): 
        
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            # print(batch_idx, batch[0].shape, batch[1].shape)
            
            dist_label = batch[1].to(device)
            idx = batch[2].to(device)
            # print(batch_idx, batch[0].shape, batch[1].shape, batch[2].shape)
            
            pred_dist = model(batch[0].to(device))
            select_dist = pred_dist[:, idx].to(device)
            criterion = nn.MSELoss()
            # criterion = nn.HuberLoss()
            loss = criterion(select_dist, dist_label)
            # loss = spearman(dist_label, select_dist)
            # loss = spearman(select_dist, dist_label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            
        
        print('epoch', e, 'MSE', total_loss, )
        train_losses.append(total_loss)
        # kendall_tracker.append(total_kendall)
        
        total_inter = 0
        with torch.no_grad():
            #on the validation
            model.eval()
            
    
        
        # this is for validation but not available
        pred_dist = model(torch.tensor(X_train).float().to(device))
        dist, idx = bottomk(pred_dist, k=k)

        real_dist = torch.cdist(torch.tensor(X_train).float(), torch.tensor(X_train).float())
        dist_real, idx_real = bottomk(real_dist, k=k)


        
        true_k_values = []
        # use the index to calculate true distance
        for l in range(X_train.shape[0]):
            distance_to_train = X_train[idx[l],:]
            real_dist_now = torch.cdist(torch.tensor(X_train[l, :].reshape(1, n_features)).float(), torch.tensor(distance_to_train).float())
            true_k_values.append(real_dist_now.max().item())
        
        

        real_dist = torch.cdist(torch.tensor(X_train).float(), torch.tensor(X_train).float())
        dist_real, idx_real = bottomk(real_dist, k=k)

        print('test real', roc_auc_score(y, dist_real[:, -1].numpy()))
        # print('test approx', roc_auc_score(y_test, dist[:, -1].detach().numpy()))
        print('test approx2', roc_auc_score(y, true_k_values))
        

        # print('test real ap', average_precision_score(y_test, dist_real[:, -1].detach().numpy()))
        # print('test approx ap', average_precision_score(y_test, dist[:, -1].detach().numpy()))
        # print('test approx2', average_precision_score(y_test, true_k_values))
        
        # print('test real prn', precision_n_scores(y_test, dist_real[:, -1].detach().numpy()))
        # print('test approx prn', precision_n_scores(y_test, dist[:, -1].detach().numpy()))
        # print('test approx2', precision_n_scores(y_test, true_k_values))
        

        # print()
        
        if roc_auc_score(y, true_k_values) >= best_train:
            best_train = roc_auc_score(y, true_k_values)
            
        
        
        # print('best test', best_test_roc, "|", best_test_ap, "|",best_test_prn,"|", best_valid/X_valid.shape[0]/k)
        # print('best test2', best_test_roc2, "|", best_test_ap2, "|",best_test_prn2,"|", best_valid/X_valid.shape[0]/k)
    
    print(k, 'real', roc_auc_score(y, dist_real[:, -1].numpy()))
    print(k, 'approximate', roc_auc_score(y, true_k_values), best_train, (best_train+roc_auc_score(y, true_k_values))/2)

