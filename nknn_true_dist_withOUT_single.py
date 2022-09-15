#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:33:15 2022

@author: yuezhao
"""

import os
import torch
from torch import nn
import pandas as pd
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


# import torchsort

# def corrcoef(target, pred):
#     pred_n = pred - pred.mean()
#     target_n = target - target.mean()
#     pred_n = pred_n / pred_n.norm()
#     target_n = target_n / target_n.norm()
#     return (pred_n * target_n).sum()*-1


# def spearman(
#     target,
#     pred,
#     regularization="l2",
#     regularization_strength=1.0,
# ):
#     pred = torchsort.soft_rank(
#         pred,
#         regularization=regularization,
#         regularization_strength=regularization_strength,
#     )
#     return corrcoef(target, pred / pred.shape[-1])


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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prediction_time = 0
    calc_time = 0

    epochs = 100
    batch_size = 200
    
    n_estimators = 10
    k=10
    
    # # mat_file = 'pendigits.mat'
    # mat_file = 'letter.mat'
    # # mat_file = 'mnist.mat'
    # # mat_file = 'annthyroid.mat'

    # mat_file = '9_PenDigits'
    # mat_file = '30_penDigits'
    mat_file = '19_annthyroid'
    # mat_file = '13_Stamps'
    # mat_file = '21_breastw'
    # mat_file = '22_glass'
    # mat_file = '26_mammography'
    # mat_file = '24_letter'
    # mat_file = '1_ALOI'
    # mat_file = '36_vertebral'
    # mat_file = '39_wine'
    
    X = pd.read_csv(os.path.join('data', mat_file+'_X.csv'), header=None).to_numpy()
    y = pd.read_csv(os.path.join('data', mat_file+'_y.csv'), header=None).to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.2, shuffle=True, random_state=42)
    
    
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    
    X_train, scalar = standardizer(X_train, keep_scalar=True)
    X_test = scalar.transform(X_test)
    X_valid = scalar.transform(X_valid)
    
    
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
    
    roc_list = []
    roc_list2 = []
    
    ap_list = []
    ap_list2 = []
    
    for n in range(n_estimators):
    
        model = NeuralNetwork(n_features=n_features, n_samples=n_samples, hidden_neuron=64, n_layers=2)
        
        optimizer = optim.Adam(model.parameters(), lr=.01)
    
    
        train_losses = []
        inter_track = []
        kendall_tracker = []
        
        best_valid = 0
        best_test_roc = 0
        best_test_prn = 0
        best_test_ap = 0
        best_inter = 0
        
        # mse_tracker = []
        for e in range(epochs): 
            
            total_loss = 0
            total_kendall = 0
            total_ndcg = 0
            
            for batch_idx, batch in enumerate(train_loader):
                
                if batch_idx == n:
                    optimizer.zero_grad()
                    # print(batch_idx, batch[0].shape, batch[1].shape)
                    
                    dist_label = batch[1]
                    idx = batch[2]
                    # print(batch_idx, batch[0].shape, batch[1].shape, batch[2].shape)
                    
                    pred_dist = model(batch[0])
                    select_dist = pred_dist[:, idx]
                    criterion = nn.MSELoss()
                    # criterion = nn.HuberLoss()
                    loss = criterion(select_dist, dist_label)
                    # loss = spearman(dist_label, select_dist)
                    # loss = spearman(select_dist, dist_label)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
            
            print('epoch', e, 'MSE', total_loss, 'kendall', total_kendall, 'ndcg', total_ndcg)
            train_losses.append(total_loss)
            # kendall_tracker.append(total_kendall)
            
            total_inter = 0
            with torch.no_grad():
                #on the validation
                model.eval()
                pred_labels = model(torch.tensor(X_valid).float())
                valid_labels = torch.cdist(torch.tensor(X_valid).float(), torch.tensor(X_train).float())
                
                
                # ndcg_test = 0
                # for k in range(test_labels.shape[0]):
                    # ndcg_test += kendalltau([test_labels[k, :].numpy()],[pred_labels[k, :].numpy()])
                topks, indx = bottomk(valid_labels, k=k)
                topks_pred, indx_pred = bottomk(pred_labels, k=k)
                
                for h in range(valid_labels.shape[0]):
                    total_inter += len(np.intersect1d(indx[h, :], indx_pred[h, :]))
                    # print(len(np.intersect1d(indx[h, :], indx_pred[h, :])))
            
            
            print('total intersec', total_inter)
            print()
            
            inter_track.append(total_inter/X_valid.shape[0]/k)
            
            # this is for validation but not available
            pred_dist = model(torch.tensor(X_valid).float())
            dist, idx = bottomk(pred_dist, k=k)
    
            real_dist = torch.cdist(torch.tensor(X_valid).float(), torch.tensor(X_train).float())
            dist_real, idx_real = bottomk(real_dist, k=k)
    
    
    
            print('valid real', roc_auc_score(y_valid, dist_real[:, -1].numpy()))
            print('valid approx', roc_auc_score(y_valid, dist[:, -1].detach().numpy()))
            
            print('valid real prn', precision_n_scores(y_valid, dist_real[:, -1].detach().numpy()))
            print('valid approx prn', precision_n_scores(y_valid, dist[:, -1].detach().numpy()))
            
            print('valid real ap', average_precision_score(y_valid, dist_real[:, -1].detach().numpy()))
            print('valid approx ap', average_precision_score(y_valid, dist[:, -1].detach().numpy()))
            
            print()
    
            
            start = time()
            print()
            # this is for test 
            pred_dist = model(torch.tensor(X_test).float())
            dist, idx = bottomk(pred_dist, k=k)
            print('1', time()- start)
            prediction_time += time()- start
            
            # raise ValueError()
            
            true_k_values = []
            # use the index to calculate true distance
            for l in range(X_test.shape[0]):
                distance_to_train = X_train[idx[l],:]
                real_dist_now = torch.cdist(torch.tensor(X_test[l, :].reshape(1, n_features)).float(), torch.tensor(distance_to_train).float())
                true_k_values.append(real_dist_now.max().item())
            
            
            start = time()
            real_dist = torch.cdist(torch.tensor(X_test).float(), torch.tensor(X_train).float())
            dist_real, idx_real = bottomk(real_dist, k=k)
            print('2', time()- start)
            calc_time += time()- start
    
            print('test real', roc_auc_score(y_test, dist_real[:, -1].numpy()))
            print('test approx', roc_auc_score(y_test, dist[:, -1].detach().numpy()))
            print('test approx2', roc_auc_score(y_test, true_k_values))
            
    
            # print('test real ap', average_precision_score(y_test, dist_real[:, -1].detach().numpy()))
            # print('test approx ap', average_precision_score(y_test, dist[:, -1].detach().numpy()))
            # print('test approx2', average_precision_score(y_test, true_k_values))
            
            # print('test real prn', precision_n_scores(y_test, dist_real[:, -1].detach().numpy()))
            # print('test approx prn', precision_n_scores(y_test, dist[:, -1].detach().numpy()))
            # print('test approx2', precision_n_scores(y_test, true_k_values))
            
    
            print()
            
            if total_inter >= best_valid:
                best_valid = total_inter
                best_test_roc = roc_auc_score(y_test, dist[:, -1].detach().numpy())
                best_test_ap = average_precision_score(y_test, dist[:, -1].detach().numpy())
                best_test_prn = precision_n_scores(y_test, dist[:, -1].detach().numpy())
                
                best_test_roc2 = roc_auc_score(y_test, true_k_values)
                best_test_ap2 = average_precision_score(y_test, true_k_values)
                best_test_prn2 = precision_n_scores(y_test, true_k_values)
                
            
            
            print('best test', best_test_roc, "|", best_test_ap, "|",best_test_prn,"|", best_valid/X_valid.shape[0]/k)
            print('best test2', best_test_roc2, "|", best_test_ap2, "|",best_test_prn2,"|", best_valid/X_valid.shape[0]/k)
            print("****************************")
            print('batch', n, 'epoch', e)
        
        roc_list.append(best_test_roc)
        roc_list2.append(best_test_roc2)
        
        ap_list.append(best_test_ap)
        ap_list2.append(best_test_ap2)
        
        print('real', roc_auc_score(y_test, dist_real[:, -1].numpy()), "|", average_precision_score(y_test, dist_real[:, -1].detach().numpy()), "|",precision_n_scores(y_test, dist_real[:, -1].detach().numpy()),"|", 1)
        print('prediction time', prediction_time)
        print('cal time', calc_time)
        
        print('avg perf', np.mean(roc_list))
        print('avg perf2', np.mean(roc_list2))
        
        print('avg ap', np.mean(ap_list))
        print('avg ap2', np.mean(ap_list2))
        
    #     # we could calculate 
    
    
#%%

for i in range(11):
    print(i, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    train()
    print(i, "********************************************")

#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)
    
plt.plot(x_range, train_losses)
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("training loss (MSE)")


#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)
    
plt.plot(x_range, inter_track)
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("the topk overlapping rate on the test")

    
    
    
    
    