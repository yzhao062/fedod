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
import pandas as pd

import time

from sklearn.metrics import homogeneity_score, adjusted_mutual_info_score, \
    completeness_score, homogeneity_completeness_v_measure, adjusted_rand_score

from pyod.utils.utility import precision_n_scores
import warnings
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from scipy.spatial.distance import cdist


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


def collate_batch2(batch, k):
    samples = []
    idxs = []

    # print(k)

    for (sample, idx) in batch:
        samples.append(sample)
        idxs.append(idx)

    samples = torch.stack(samples)
    idxs = torch.tensor(idxs)


    estimators = [KMeans(n_clusters=k, random_state=0),
                  KMeans(n_clusters=k, random_state=1),
                  KMeans(n_clusters=k, random_state=2),]

    # clf = KMeans(n_clusters=k)
    clf = ClustererEnsemble(estimators, n_clusters=k)
    clf.fit(samples.numpy())
    # print(clf.labels_)

    # return local sample, sample index, and also local labels
    return samples.float(), idxs, torch.from_numpy(clf.labels_)


def get_scores(cluster_labels_, X_test, y_test, k, n_samples, n_features):
    # cluster_labels_ = w
    # cluster_labels_ = test_labels
    cluster_sizes_ = np.bincount(cluster_labels_)
    n_clusters_ = cluster_sizes_.shape[0]
    
    if n_clusters_ != k:
        warnings.warn("The chosen clustering for CBLOF forms {0} clusters"
                      "which is inconsistent with n_clusters ({1}).".
                      format(n_clusters_, k))
    
    cluster_centers_ = np.zeros([k, n_features])
    for i in range(k):
        cluster_centers_[i, :] = np.mean(
            X_test[np.where(cluster_labels_ == i)], axis=0)
    
    cluster_centers_ = np.nan_to_num(cluster_centers_)
    # Get the actual number of clusters
    # self.n_clusters_ = self.cluster_sizes_.shape[0]
    
    
    # Sort the index of clusters by the number of samples belonging to it
    size_clusters = np.bincount(cluster_labels_)
    
    # Sort the order from the largest to the smallest
    sorted_cluster_indices = np.argsort(size_clusters * -1)
    
    # Initialize the lists of index that fulfill the requirements by
    # either alpha or beta
    alpha_list = []
    beta_list = []
    
    for i in range(1, n_clusters_):
        temp_sum = np.sum(size_clusters[sorted_cluster_indices[:i]])
        if temp_sum >= n_samples * 0.9:
            alpha_list.append(i)
    
        if size_clusters[sorted_cluster_indices[i - 1]] / size_clusters[
            sorted_cluster_indices[i]] >= 5:
            beta_list.append(i)
    
        # Find the separation index fulfills both alpha and beta
    intersection = np.intersect1d(alpha_list, beta_list)
    
    if len(intersection) > 0:
        _clustering_threshold = intersection[0]
    elif len(alpha_list) > 0:
        _clustering_threshold = alpha_list[0]
    elif len(beta_list) > 0:
        _clustering_threshold = beta_list[0]
    else:
        raise ValueError("Could not form valid cluster separation. Please "
                         "change n_clusters or change clustering method")
    
    small_cluster_labels_ = sorted_cluster_indices[_clustering_threshold:]
    large_cluster_labels_ = sorted_cluster_indices[0:_clustering_threshold]
    
    # No need to calculate small cluster center
    # self.small_cluster_centers_ = self.cluster_centers_[
    #     self.small_cluster_labels_]
    
    _large_cluster_centers = cluster_centers_[large_cluster_labels_]
    
    # Initialize the score array
    scores = np.zeros([X_test.shape[0], ])
    
    small_indices = np.where(
        np.isin(cluster_labels_, small_cluster_labels_))[0]
    large_indices = np.where(
        np.isin(cluster_labels_, large_cluster_labels_))[0]
    
    if small_indices.shape[0] != 0:
        # Calculate the outlier factor for the samples in small clusters
        dist_to_large_center = cdist(X_test[small_indices, :],
                                     _large_cluster_centers)
    
        scores[small_indices] = np.min(dist_to_large_center, axis=1)
    
    if large_indices.shape[0] != 0:
        # Calculate the outlier factor for the samples in large clusters
        large_centers = cluster_centers_[cluster_labels_[large_indices]]
    
        scores[large_indices] = pairwise_distances_no_broadcast(
            X_test[large_indices, :], large_centers)
    
    # print(roc_auc_score(y_test, scores))
    return roc_auc_score(y_test, scores)


class NeuralNetwork(nn.Module):
    def __init__(self, n_features, n_samples, hidden_neuron=16, n_layers=2, k=5):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        layer_list = []

        layer_list.append(nn.Linear(n_features, hidden_neuron)),
        layer_list.append(nn.ELU())
        for i in range(n_layers):
            layer_list.append(nn.Linear(hidden_neuron, hidden_neuron))
            layer_list.append(nn.ELU())

        layer_list.append(nn.Linear(hidden_neuron, k))

        self.simple_nn = nn.Sequential(*layer_list)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.simple_nn(x)
        return logits


# if __name__ == "__main__": 
def train(mat_file):

        
    prediction_time = 0
     
    
    calc_time = 0

    epochs = 200
    k = 10
    
    X = pd.read_csv(os.path.join('data', mat_file+'_X.csv'), header=None).to_numpy()
    y = pd.read_csv(os.path.join('data', mat_file+'_y.csv'), header=None).to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, shuffle=True, random_state=42)
 

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    X_train, scalar = standardizer(X_train, keep_scalar=True)
    X_test = scalar.transform(X_test)
    X_valid = scalar.transform(X_valid)
    
    gpu_time = []
    cpu_time = []
    # clf = KMeans(n_clusters=k)
    # clf.fit(X_train)


    estimators = [KMeans(n_clusters=k, random_state=0),
                  KMeans(n_clusters=k, random_state=1),
                  KMeans(n_clusters=k, random_state=2),]

    clf = ClustererEnsemble(estimators, n_clusters=k)

    clf.fit(X_train)
    

    batch_size = 200

    train_set = PyODDataset(X=X_train)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               collate_fn=partial(collate_batch2, k=k),
                                               shuffle=True)

    test_set = PyODDataset(X=X_test)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              collate_fn=collate_batch2,
                                              shuffle=True)

    valid_set = PyODDataset(X=X_valid)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch2,
                                               shuffle=True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    model = NeuralNetwork(n_features=n_features, n_samples=n_samples, hidden_neuron=64, n_layers=3,
                          k=k)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=.001)
    # criterion = nn.NLLLoss()

    all_pairs = list(itertools.combinations(range(k), 2))
    # mse_tracker = []

    best_valid = 0
    best_test = 0
    train_losses = []
    train_accu = []
    valid_accu = []
    test_accu = []
    
    # print(X.shape)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_valid.shape)

    for e in range(epochs):

        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):


            loss = 0
            optimizer.zero_grad()


            output = model(batch[0].to(device))
            # output = torch.argmax(output, axis=1)

            # intra distance minimize
            for ih in range(k):

                k_ind = torch.where(batch[2] == ih)[0]
                k_pred = output[k_ind, :]


                loss += torch.cdist(k_pred, k_pred).sum()

            # inter distance maximize
            for pair in all_pairs:
                pair_ind_1 = torch.where(batch[2] == pair[0])[0]
                pair_ind_2 = torch.where(batch[2] == pair[1])[0]

                cdist12 = torch.cdist(output[pair_ind_1, :], output[pair_ind_2, :]).sum()
                loss += cdist12 * -1
                # print(cdist12.shape)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()


        train_losses.append(total_loss)

        # validation loss
        with torch.no_grad():
            model.eval()

            # on the train
            pred_prob = model(torch.tensor(X_train).float().to(device))
            pred_labels = torch.argmax(pred_prob, axis=1)
            w = pred_labels.cpu().numpy()
            train_labels = clf.labels_
            train_completeness = completeness_score(train_labels, w)
            train_accu.append(train_completeness)

            # on the validation

            pred_prob = model(torch.tensor(X_valid).float().to(device))
            pred_labels = torch.argmax(pred_prob, axis=1)
            # valid_labels = torch.cdist(torch.tensor(X_valid).float(), torch.tensor(X_train).float())

            w = pred_labels.cpu().numpy()
            valid_labels = clf.predict(X_valid)

            valid_completeness = completeness_score(valid_labels, w)
            valid_accu.append(valid_completeness)

            # print(adjusted_mutual_info_score(valid_labels, w))
            # print(adjusted_rand_score(valid_labels, w))
            # print(homogeneity_completeness_v_measure(valid_labels, w))
            # print(completeness_score(valid_labels, w))

            if best_valid <= valid_completeness:
                best_valid = valid_completeness

                # do the evaluation only for the test when it is promising
                start = time.time()
                pred_prob = model(torch.tensor(X_test).float().to(device))
                pred_labels = torch.argmax(pred_prob, axis=1)
                ours_test = pred_labels.cpu().numpy()
                gpu_time.append(time.time()-start)
                
                start = time.time()
                test_labels = clf.predict(X_test)
                cpu_time.append(time.time()-start)
                
                best_test = completeness_score(test_labels, ours_test)
                

                pred_train = model(torch.tensor(X_train).float().to(device))
                pred_labels = torch.argmax(pred_train, axis=1)
                ours_train = pred_labels.cpu().numpy()
                

                # train_labels = clf.predict(X_train)

                
            print('epoch', e)
    print('test completeness', best_test, mat_file)
    # print(np.sum(cpu_time), np.sum(gpu_time))
    
    # return test_labels, ours_test, X_test, y_test, k, n_samples, n_features
    return clf.labels_, ours_train, X_train, y_train, k, n_samples, n_features

#%%
# # Initialize a pandas DataFrame
df = pd.DataFrame(columns=['file', 'real', 'ours',])

files = pd.read_csv('file_list.csv', header=None).values.tolist()
for i, mat_file in enumerate(files[5:]): 
# for i, mat_file in enumerate(files[4:]):  
# for i, mat_file in enumerate(files[23:]):  
# for i, mat_file in enumerate(files[14:]):  
    test_labels, ours_test, X_test, y_test, k, n_samples, n_features = train(mat_file=mat_file[0])

    ours = get_scores(ours_test, X_test, y_test, k, n_samples, n_features)
    ground = get_scores(test_labels, X_test, y_test, k, n_samples, n_features)
    
    df = df.append(pd.Series([mat_file, ground, ours], index=df.columns), ignore_index=True)
# # %%
# test_tracker = []
# for i in range(20):
#     print(i, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#     train()
#     print(i, "********************************************")

# # %%
# import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')
# import numpy as np

# x_range = np.arange(epochs)

# plt.plot(x_range, train_losses)
# # plt.grid(False)
# plt.xlabel("number of epochs")
# plt.ylabel("training loss (intra cluster dist - inter cluster dist)")
# plt.show()

# # %%
# import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')
# import numpy as np

# x_range = np.arange(epochs)

# plt.plot(x_range, train_accu, label='train')
# plt.plot(x_range, valid_accu, label='valid')
# # plt.grid(False)
# plt.xlabel("number of epochs")
# plt.ylabel("accuracy (compleness score)")
# plt.title(mat_file)

# plt.legend()
# plt.show()
