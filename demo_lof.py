#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:33:15 2022

@author: yuezhao
"""

import os

import numpy as np
import pandas as pd
import scipy as sp
import torch
from pyod.utils.utility import standardizer
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim

from utility import bottomk, PyODDataset


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


# if __name__ == "__main__": 
def train(mat_file):
	prediction_time = 0

	X = pd.read_csv(os.path.join('data', mat_file + '_X.csv'),
					header=None).to_numpy()
	y = pd.read_csv(os.path.join('data', mat_file + '_y.csv'),
					header=None).to_numpy()

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
														shuffle=True,
														random_state=42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
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

	model = NeuralNetwork(n_features=n_features, n_samples=n_samples,
						  hidden_neuron=64, n_layers=2)

	optimizer = optim.Adam(model.parameters(), lr=.01)
	# criterion = nn.NLLLoss()
	epochs = 100
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
			# print(batch_idx, batch[0].shape, batch[1].shape)

			dist_label = batch[1]
			idx = batch[2]
			# print(batch_idx, batch[0].shape, batch[1].shape, batch[2].shape)

			pred_dist = model(batch[0])
			select_dist = pred_dist[:, idx]
			criterion = nn.MSELoss()
			# criterion = nn.HuberLoss()
			loss = criterion(select_dist, dist_label)
# 			loss += spearman(dist_label, select_dist)
			# loss = spearman(dist_label, select_dist)

			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		print('epoch', e, 'MSE', total_loss, 'kendall', total_kendall, 'ndcg',
			  total_ndcg)
		train_losses.append(total_loss)
		# kendall_tracker.append(total_kendall)

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

	
		print()

		if total_inter >= best_valid:
			best_valid = total_inter

			with torch.no_grad():
				# on the validation
				model.eval()
				final_dist = model(torch.tensor(X_train).float())

	# make sure all the distance prediction is positive
	if (final_dist < 0).sum() > 0:
		final_dist = final_dist - final_dist.min()

	# find the k nearst neighbors of all samples
	knn_dist, knn_inds = torch.topk(final_dist, k=k + 1, largest=False,
									sorted=True)
	# knn_dist, knn_inds = bottomk(sorted1, k=k)
	# knn_dist, knn_inds = bottomk(dist, k=k)
	knn_dist, knn_inds = knn_dist[:, 1:], knn_inds[:, 1:]

	# this is the index of kNN's index
	knn_inds_flat = torch.flatten(knn_inds).long()
	knn_dist_flat = torch.flatten(knn_dist)

	# for each sample, find their kNN's *kth* neighbor's distance
	# -1 is for selecting the kth distance
	knn_kth_dist = torch.index_select(knn_dist, 0, knn_inds_flat)[:, -1]
	knn_kth_inds = torch.index_select(knn_inds, 0, knn_inds_flat)[:, -1]

	# to calculate the reachable distance, we need to compare these two distances
	raw_smaller = torch.where(knn_dist_flat < knn_kth_dist)[0]

	# let's override the place where it is not the case
	# this can save one variable
	knn_dist_flat[raw_smaller] = knn_kth_dist[raw_smaller]
	# print(knn_dist_flat[:10])

	# then we need to calculate the average reachability distance

	# this result in [n, k] shape
	ar = torch.mean(knn_dist_flat.view(-1, k), dim=1)

	# harmonic mean give the exact result!
	# todo: harmonic mean can be written in PyTorch as well
	ar_nn = sp.stats.hmean(
		torch.index_select(ar, 0, knn_inds_flat).view(-1, k).numpy(),
		axis=1)
	assert (len(ar_nn) == len(ar))

	scores = (ar / ar_nn).cpu().numpy()

	return X_train, y_train, scores


# Initialize a pandas DataFrame
df = pd.DataFrame(columns=['file', 'ground truth', 'ours'])

files = pd.read_csv('file_list.csv', header=None).values.tolist()
# for i, mat_file in enumerate(files[1:]): 
for i, mat_file in enumerate(files[1:10]):
	X_train, y_train, scores = train(mat_file=mat_file[0])
	print(mat_file)
	print('ours', roc_auc_score(y_train, scores))
    

	from pyod.models.lof import LOF

	clf_name = 'LOF-PyOD'
	clf = LOF(n_neighbors=10)

	clf.fit(X_train)
	y_train_scores = clf.decision_scores_
	print('pyod', roc_auc_score(y_train, y_train_scores))
	# Append the list to the DataFrame
	df = df.append(pd.Series([mat_file, roc_auc_score(y_train, y_train_scores),
							  roc_auc_score(y_train, scores)],
							 index=df.columns), ignore_index=True)

print(df)