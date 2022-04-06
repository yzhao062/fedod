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

from time import time

from sklearn.metrics import homogeneity_score, adjusted_mutual_info_score, \
    completeness_score, homogeneity_completeness_v_measure, adjusted_rand_score

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


def collate_batch2(batch, k):
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

    estimators = [KMeans(n_clusters=k, random_state=0),
                  KMeans(n_clusters=k, random_state=1),
                  KMeans(n_clusters=k, random_state=2),]

    # clf = KMeans(n_clusters=k)
    clf = ClustererEnsemble(estimators, n_clusters=k)
    clf.fit(samples.numpy())
    # print(clf.labels_)

    # return local sample, sample index, and also local labels
    return samples.float(), idxs, torch.from_numpy(clf.labels_)


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
def train():
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

    epochs = 200
    k = 5

    # mat_file = 'pendigits.mat'
    # mat_file = 'letter.mat'
    # mat_file = 'mnist.mat'
    mat_file = 'annthyroid.mat'

    mat = loadmat(os.path.join('data', mat_file))
    X = mat['X']
    y = mat['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.2, shuffle=True,
                                                          random_state=42)

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    X_train, scalar = standardizer(X_train, keep_scalar=True)
    X_test = scalar.transform(X_test)
    X_valid = scalar.transform(X_valid)

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

    for e in range(epochs):

        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):

            # print('batch', batch_idx)
            loss = 0
            optimizer.zero_grad()
            # print(batch_idx, batch[0].shape, batch[1].shape, batch[2].shape)

            # batch 2 is the local ground truth
            # print(batch[2])

            output = model(batch[0].to(device))
            # output = torch.argmax(output, axis=1)

            # intra distance minimize
            for ih in range(k):
                # print(torch.where(batch[2] == ih)[0].shape)
                k_ind = torch.where(batch[2] == ih)[0]
                k_pred = output[k_ind, :]
                # print(k_pred.shape)

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

        print('epoch', e, 'loss', total_loss)
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
            print(completeness_score(valid_labels, w))

            if best_valid <= valid_completeness:
                best_valid = valid_completeness

                # do the evaluation only for the test when it is promising
                pred_prob = model(torch.tensor(X_test).float().to(device))
                pred_labels = torch.argmax(pred_prob, axis=1)

                w = pred_labels.cpu().numpy()
                test_labels = clf.predict(X_test)

                best_test = completeness_score(test_labels, w)
            # print()
    print('test completeness', best_test)


# %%
test_tracker = []
for i in range(15):
    print(i, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    train()
    print(i, "********************************************")

# %%
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)

plt.plot(x_range, train_losses)
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("training loss (intra cluster dist - inter cluster dist)")
plt.show()

# %%
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)

plt.plot(x_range, train_accu, label='train')
plt.plot(x_range, valid_accu, label='valid')
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("accuracy (compleness score)")
plt.title(mat_file)

plt.legend()
plt.show()
