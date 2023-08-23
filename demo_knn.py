#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:33:15 2022

@author: yuezhao
"""

import os
from time import time

import numpy as np
import pandas as pd
import torch
from pyod.utils.utility import standardizer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim

from utility import bottomk, PyODDataset
from pyod.utils.utility import precision_n_scores


def custom_batch(batch):
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
	idxs = torch.tensor(idxs)
	# print(samples)
	dists = torch.cdist(samples, samples)

	return samples.float(), dists.float(), idxs


# import torchsort


def corrcoef(target, pred):
	pred_n = pred - pred.mean()
	target_n = target - target.mean()
	pred_n = pred_n / pred_n.norm()
	target_n = target_n / target_n.norm()
	return (pred_n * target_n).sum() * -1


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


def train(mat_file):
	prediction_time = 0
	calc_time = 0

	X = pd.read_csv(os.path.join('data', mat_file + '_X.csv'),
					header=None).to_numpy()
	y = pd.read_csv(os.path.join('data', mat_file + '_y.csv'),
					header=None).to_numpy()

	X_train, X_test, y_train, y_test = train_test_split(X, y,
														train_size=0.8,
														shuffle=True,
														random_state=42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train,
														  y_train,
														  train_size=0.8,
														  shuffle=True,
														  random_state=42)

	n_samples = X_train.shape[0]
	n_features = X_train.shape[1]

	X_train, scalar = standardizer(X_train, keep_scalar=True)
	X_test = scalar.transform(X_test)
	X_valid = scalar.transform(X_valid)

	batch_size = 200

	train_set = PyODDataset(X=X_train)
	train_loader = torch.utils.data.DataLoader(train_set,
											   batch_size=batch_size,
											   collate_fn=custom_batch,
											   shuffle=False)

	test_set = PyODDataset(X=X_test)
	test_loader = torch.utils.data.DataLoader(test_set,
											  batch_size=batch_size,
											  collate_fn=custom_batch,
											  shuffle=False)

	valid_set = PyODDataset(X=X_valid)
	valid_loader = torch.utils.data.DataLoader(valid_set,
											   batch_size=batch_size,
											   collate_fn=custom_batch,
											   shuffle=False)

	hidden_neuron = 128
	n_layers = 4
	model = NeuralNetwork(n_features=n_features, n_samples=n_samples,
						  hidden_neuron=hidden_neuron, n_layers=n_layers)

	optimizer = optim.Adam(model.parameters(), lr=.01)
	epochs = 50
	k = 10

	train_losses = []
	inter_track = []
	kendall_tracker = []

	best_valid = 0
	best_test_roc = 0
	best_test_prn = 0
	best_test_ap = 0
	best_inter = 0

	roc1s = []
	roc2s = []

	# mse_tracker = []
	for e in range(epochs):

		total_loss = 0
		total_kendall = 0
		total_ndcg = 0

		for batch_idx, batch in enumerate(train_loader):
			optimizer.zero_grad()

			dist_label = batch[1]
			idx = batch[2]

			pred_dist = model(batch[0])
			select_dist = pred_dist[:, idx]
			criterion = nn.MSELoss()
			# criterion = nn.HuberLoss()
			loss = criterion(select_dist, dist_label)
			# loss += spearman(dist_label, select_dist)

			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		print('epoch', e, 'MSE', total_loss, 'kendall', total_kendall, 'ndcg',
			  total_ndcg)
		train_losses.append(total_loss)

		total_inter = 0
		with torch.no_grad():
			# on the validation
			model.eval()
			pred_labels = model(torch.tensor(X_valid).float())
			valid_labels = torch.cdist(torch.tensor(X_valid).float(),
									   torch.tensor(X_train).float())

			topks, indx = bottomk(valid_labels, k=k)
			topks_pred, indx_pred = bottomk(pred_labels, k=k)

			for h in range(valid_labels.shape[0]):
				total_inter += len(np.intersect1d(indx[h, :], indx_pred[h, :]))


		inter_track.append(total_inter / X_valid.shape[0] / k)

		# this is for validation but not available
		pred_dist = model(torch.tensor(X_valid).float())
		dist, idx = bottomk(pred_dist, k=k)

		real_dist = torch.cdist(torch.tensor(X_valid).float(),
								torch.tensor(X_train).float())
		dist_real, idx_real = bottomk(real_dist, k=k)


		start = time()
		print()
		# this is for test
		pred_dist = model(torch.tensor(X_test).float())
		dist, idx = bottomk(pred_dist, k=k)
		prediction_time += time() - start

		true_k_values = []
		# use the index to calculate true distance
		for l in range(X_test.shape[0]):
			distance_to_train = X_train[idx[l], :]
			real_dist_now = torch.cdist(
				torch.tensor(X_test[l, :].reshape(1, n_features)).float(),
				torch.tensor(distance_to_train).float())
			true_k_values.append(real_dist_now.max().item())

		start = time()
		real_dist = torch.cdist(torch.tensor(X_test).float(),
								torch.tensor(X_train).float())
		dist_real, idx_real = bottomk(real_dist, k=k)
		calc_time += time() - start

		roc1s.append(roc_auc_score(y_test, dist[:, -1].detach().numpy()))
		roc2s.append(roc_auc_score(y_test, true_k_values))

		if total_inter >= best_valid:
			best_valid = total_inter
			best_test_roc = roc_auc_score(y_test, dist[:, -1].detach().numpy())
			best_test_ap = average_precision_score(y_test,
												   dist[:, -1].detach().numpy())
			best_test_prn = precision_n_scores(y_test,
											   dist[:, -1].detach().numpy())

			best_test_roc2 = roc_auc_score(y_test, true_k_values)
			best_test_ap2 = average_precision_score(y_test, true_k_values)
			best_test_prn2 = precision_n_scores(y_test, true_k_values)

	return roc_auc_score(y_test, dist_real[:,
								 -1].numpy()), best_test_roc, best_test_roc2, roc1s, roc2s


# Initialize a pandas DataFrame
df = pd.DataFrame(columns=['file', 'ground truth', 'ours 1', 'ours 2'])

files = pd.read_csv('file_list.csv', header=None).values.tolist()
for i, mat_file in enumerate(files[:10]):
	real_roc, roc1, roc2, roc1s, roc2s = train(mat_file=mat_file[0])
	print(mat_file)
    
	# Append the list to the DataFrame
	df = df.append(
		pd.Series([mat_file, real_roc, roc1, roc2, ], index=df.columns),
		ignore_index=True)
    
# as a demo, just show the performance on the first 10 datasets

print(df)
