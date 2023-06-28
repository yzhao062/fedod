# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:02:45 2022

@author: yuezhao
"""

import os
from functools import partial
import itertools

import torch
from torch import nn
from sklearn.cluster import KMeans
from combo.models.cluster_comb import ClustererEnsemble
from combo.models.cluster_eac import EAC

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
from functools import reduce
import pandas as pd

import time
from scipy.stats import rankdata

from sklearn.metrics import homogeneity_score, adjusted_mutual_info_score, \
    completeness_score, homogeneity_completeness_v_measure, adjusted_rand_score

from pyod.utils.utility import precision_n_scores
from pyod.models.iforest import IForest

def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', [a, b])
    # res = contract('ij,ik->ijk', [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])

    return res

def torch_bin(x, cut_points, device, temperature=0.1):
    # x is a N-by-1 matrix (column vector)
    # cut_points is a D-dim vector (D is the number of cut-points)
    # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    D = cut_points.shape[0]
    W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1, device=device), [1, -1])
    cut_points, _ = torch.sort(cut_points)  # make sure cut_points is monotonically increasing
    # print(cut_points)
    b = torch.cumsum(torch.cat([torch.zeros([1], device=device), -cut_points], 0),0)

    h = torch.matmul(x, W) + b
    res = torch.exp(h-torch.max(h))
    res = res/torch.sum(res, dim=-1, keepdim=True)
    return h

def nn_decision_tree(x, cut_points_list, leaf_score, device, temperature=0.1):
    # cut_points_list contains the cut_points for each dimension of feature
    leaf = reduce(torch_kron_prod,
                  map(lambda z: torch_bin(x[:, z[0]:z[0] + 1], z[1], device, temperature), enumerate(cut_points_list)))
    return torch.matmul(leaf, leaf_score)


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


def collate_batch2(batch, n_estimators, n_trials=10):
    samples = []
    idxs = []

    # print(k)

    for (sample, idx) in batch:
        samples.append(sample)
        idxs.append(idx)

    samples = torch.stack(samples)
    # print(samples.shape)
    idxs = torch.tensor(idxs)
    # print(samples)
    
    outputs = np.zeros([n_trials, samples.shape[0]])
    for i in range(n_trials):    
        clf = IForest(n_estimators=n_estimators)
        clf.fit(samples.numpy())
        outputs[i, :] = clf.decision_scores_
    
    ground_truth = np.mean(outputs, axis=0)
    ground_truth = rankdata(ground_truth)/len(ground_truth)
    # return local sample, sample index, and also local labels
    return samples.float(), idxs, torch.from_numpy(ground_truth).reshape(len(clf.decision_scores_), 1)



if __name__ == "__main__": 
# def train():
    
    files = pd.read_csv('file_list.csv', header=None).values.tolist()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    warmup_epochs = 30
    batch_size = 200
    n_estimators = 100
    n_trials = 5
    
    start = 30
    end = 39
    
    # for i, mat_file in enumerate(files[0:19]): 
    for i, mat_file in enumerate(files[start:end]):         
        mat_file = mat_file[0]
    # mat_file = '1_ALOI'
    # mat_file = '2_Annthyroid'
    # mat_file = '4_Cardiotocography'
    # mat_file = '5_Glass' # maybe
    # mat_file = '7_InternetAds'
    # mat_file = '10_Pima' # maybe
    # mat_file = '12_SpamBase'

    # mat_file = '14_Waveform'
    # mat_file = ''
    # mat_file = ''
    
    
    
    # mat_file = '8_PageBlocks'
    # mat_file = '9_PenDigits'
    # mat_file = '11_Shuttle'
    # mat_file = '3_Arrhythmia'
    # mat_file = '6_HeartDisease' # much better than the original
    # mat_file = '13_Stamps'
    
    # mat_file = '13_Stamps'
    # mat_file = '21_breastw'
    # mat_file = '22_glass'
    # mat_file = '26_mammography'
    # mat_file = '36_vertebral'
    # mat_file = '39_wine'


        X = pd.read_csv(os.path.join('data', mat_file+'_X.csv'), header=None).to_numpy()
        y = pd.read_csv(os.path.join('data', mat_file+'_y.csv'), header=None).to_numpy()
    
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, shuffle=True,
                                                                  random_state=42)
        
        roc_scores_train = []
        ap_scores_train = []
        
        roc_scores_test = []
        ap_scores_test = []
        
    
        outputs = np.zeros([n_trials, X_train.shape[0]]) 
        test_tree = np.zeros([n_trials, X_test.shape[0]]) 
        for i in range(n_trials):    
            clf = IForest(n_estimators=n_estimators, max_samples=batch_size)
            clf.fit(X_train)
            outputs[i, :] = clf.decision_scores_
            
            # roc_scores_train.append(roc_auc_score(y_train, clf.decision_scores_))
            # ap_scores_train.append(average_precision_score(y_train, clf.decision_scores_))
            
            test_tree[i, :] = clf.predict(X_test)
            # roc_scores_test.append(roc_auc_score(y_test, test_tree))
            # ap_scores_test.append(average_precision_score(y_test, test_tree))
        
        ground_truth = np.mean(outputs, axis=0)
        ground_truth = rankdata(ground_truth)/len(ground_truth)
        # print('train roc {}, ap {}'.format(np.mean(roc_scores_train), np.mean(ap_scores_train)))
        # print('test roc {}, ap {}'.format(np.mean(roc_scores_test), np.mean(ap_scores_test)))
        print('train roc {}, ap {}'.format(roc_auc_score(y_train, ground_truth), average_precision_score(y_train, ground_truth)))
        train_ground_roc = roc_auc_score(y_train, ground_truth)
        
        ground_truth = np.mean(test_tree, axis=0)
        ground_truth = rankdata(ground_truth)/len(ground_truth)
        print('test roc {}, ap {}'.format(roc_auc_score(y_test, ground_truth), average_precision_score(y_test, ground_truth)))
        print()
        
        test_roc_iforest = roc_auc_score(y_test, ground_truth)
        test_ap_iforest = average_precision_score(y_test, ground_truth)
    
        
        train_set = PyODDataset(X=X_train)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   collate_fn=partial(collate_batch2, n_estimators=n_estimators, n_trials=n_trials),
                                                   shuffle=True)
    
        test_set = PyODDataset(X=X_test)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  collate_fn=partial(collate_batch2, n_estimators=n_estimators, n_trials=n_trials),
                                                  shuffle=True)
    
        valid_set = PyODDataset(X=X_valid)
        valid_loader = torch.utils.data.DataLoader(valid_set,
                                                   batch_size=batch_size,
                                                   collate_fn=partial(collate_batch2, n_estimators=n_estimators, n_trials=n_trials),
                                                   shuffle=True)
        
    
        
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        
        X_test = torch.from_numpy(X_test).float()
        # y_test = torch.from_numpy(y_test).long()
        X_valid = torch.from_numpy(X_valid).float()
        X_test = X_test.to(device)
        X_valid = X_valid.to(device)
        
        ####################### this is relatively hard to assign*******************
        ### both k and n_c can be varying
        k = 2
        # n_c = int(n_features*0.1)
        # if n_c <= 5:
        #     n_c = 5
        n_c = 5
        # num_cut = [k]* int(n_features)  # "Petal length" and "Petal width"
        num_cut = [k]* n_c  # "Petal length" and "Petal width"
        # num_cut = [k]*n_c  # "Petal length" and "Petal width"
        #############################################################################
        
        temperature = 0.1
        num_leaf = np.prod(np.array(num_cut) + 1)
        num_class = 1
        # num_class = 3
        
        cut_points_list = [torch.rand([i], requires_grad=True, device=device) for i in num_cut]
        leaf_score = torch.rand([num_leaf, num_class], requires_grad=True, device=device)
        
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(cut_points_list + [leaf_score], lr=0.05)
        
        best_valid = 0
        best_test = 0
        best_train = 0
    
        train_losses = []
        train_rocs = []
        train_aps = []
        
        
        valid_rocs = []
        valid_aps = []
        
        test_rocs = []
        test_aps = []
        
        
        for e in range(epochs):
        
            train_preds = []
            for batch_idx, batch in enumerate(train_loader):
                
                batch_y = batch[2].to(device).float()
                batch = batch[0].to(device)
                
    
                optimizer.zero_grad()
                y_pred = nn_decision_tree(batch, cut_points_list, leaf_score, device=device, temperature=temperature)
                
                loss = loss_function(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                
                
                train_preds.extend(y_pred.cpu().detach().numpy().tolist())
                
            train_losses.append(loss.item())
            
            train_roc = roc_auc_score(y_train, train_preds)
            train_ap = average_precision_score(y_train, train_preds)
            print(e, 'pred train roc {}, ap {}'.format(train_roc, train_ap))
            
            train_rocs.append(train_roc)
            train_aps.append(train_ap)
            
            with torch.no_grad():
                ## make evaluation
                y_pred = nn_decision_tree(X_valid, cut_points_list, leaf_score, device=device, temperature=temperature)
                y_numpy = y_pred.detach().cpu().numpy()
                y_numpy = np.nan_to_num(y_numpy)
                # print('valid roc %.4f' % (roc_auc_score(y_valid, y_numpy)))
                valid_rocs.append(roc_auc_score(y_valid, y_numpy))
                valid_aps.append(average_precision_score(y_valid, y_numpy))
                
                if best_valid <= roc_auc_score(y_valid, y_numpy):
                    best_valid = roc_auc_score(y_valid, y_numpy)
                    best_valid_ap = average_precision_score(y_valid, y_numpy)
                    
                    ## make prediction
                    y_pred = nn_decision_tree(X_test, cut_points_list, leaf_score, device=device, temperature=temperature)
                    y_numpy = y_pred.detach().cpu().numpy()
                    y_numpy = np.nan_to_num(y_numpy)
                    # print('test roc %.4f' % (roc_auc_score(y_test, y_numpy)))
                    
                    best_test = roc_auc_score(y_test, y_numpy)
                    best_test_ap = average_precision_score(y_test, y_numpy)
                    best_train = train_roc
                
                test_rocs.append(best_test)
                test_aps.append(best_test_ap)
                    
            print(e, 'pred valid roc {}, ap {}'.format(best_valid, best_valid_ap))
            print(e, 'pred test roc {}, ap {}'.format(best_test, best_test_ap))
            print('ground roc', test_roc_iforest)
            print(start, end)
            print()
        
        print('ground roc', test_roc_iforest)
        print('avg perf', np.mean(test_rocs[warmup_epochs:]))
         
        print('ground ap', test_ap_iforest) 
        print('avg ap', np.mean(test_aps[warmup_epochs:]))
        
        print(mat_file, k, n_c)
        
        plt.style.use('seaborn-whitegrid')
        import numpy as np
        
        plt.figure(figsize=(6,5))
        
        x_range = np.arange(epochs)
        
        # plt.plot(x_range, train_rocs, label='train')
        # plt.plot(x_range, valid_rocs, label='valid')
        plt.plot(x_range, test_rocs, label='test', c='red')
        plt.hlines(test_roc_iforest, xmin=0, xmax=epochs, label='test by tree', linestyle='--', color='red')
        
        # plt.plot(x_range, train_rocs, label='train', c='blue')
        # plt.hlines(train_ground_roc, xmin=0, xmax=epochs, label='train by tree', linestyle='--', color='blue')
        
        plt.hlines(best_test, xmin=0, xmax=epochs, label='best test', linestyle='-', color='green')
        # plt.hlines(best_train, xmin=0, xmax=epochs, label='best train', linestyle='-', color='purple')
        # plt.plot(x_range, valid_aps, label='valid ap')
        # plt.plot(x_range, test_aps, label='test ap')
        # plt.hlines(test_ap_iforest, xmin=0, xmax=epochs, label='test ap', linestyle='--')
        # plt.xticks([0,10, 20, 30, 50])
        # plt.grid(False)
        plt.xlabel("number of epochs", size=12)
        plt.ylabel("roc", size=15)
        plt.title(mat_file)
        
        plt.legend()
        plt.savefig(mat_file+'.png')
        plt.show()
    
    
    # plt.savefig(mat_file+'.png')


