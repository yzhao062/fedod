#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:02:45 2022

@author: yuezhao
"""

############################################################################
############################################################################

"""Contains Tabular Transformer Model definition."""
from itertools import cycle
import torch.nn.init as init

import torch
import torch.nn as nn
import sys
from npt_modules import MHSA


## NPT args
import yaml
from pathlib import Path


'''
    1. Make sure the arugements are coming in properly
    2. Make sure the data set details are coming in properly
    3. Run the and fix the issues  
'''


class NPTModel(nn.Module):
    """Non-Parametric Transformers.

    Applies Multi-Head Self-Attention blocks between datapoints,
    and within each datapoint.

    For all model variants, we expect a list of input data, `X_ragged`:
    ```
        len(X_ragged) == N
        X_ragged[i].shape == (D, H_i)
    ```
    In other words, we have `N` input samples. All samples share the same
    number of `D` features, where each feature i is encoded in `H_i`
    dimensions. "Encoding" here refers to the data preprocessing, i.e. the
    one-hot-encoding for categorical features, as well as well as adding
    the mask tokens. (Note that this is done by the code and the user is
    expected to provide datasets as given in `npt.data_loaders`.)

    High-level model overview:

    Initially in NPTModel, `self.in_embedding()` linearly embeds each of the
    `D` feature columns to a shared embedding dimension `E`.
    We learn separate embedding weights for each column.
    This allows us to obtain the embedded data matrix `X_emb` as a
    three-dimensional tensor of shape `(N, D, E)`.
    `E` is referred to as `dim_feat` in the code below.

    After embedding the data, we apply NPT.
    See `build_npt()` for further information.
    NPT applies a series of attention blocks on the input.

    We eventually obtain output of shape `(N, D, E)`,
    which is projected back to the dimensions of the input `X_ragged` using
    `self.out_embedding()`, which applies a learned linear embedding to
    each column `D` separately.
    """
    def __init__(self, device=None,n_samples=None):
        """Initialise NPTModel.

        Args:
            c: wandb config
            metadata: Dict, from which we retrieve:
                input_feature_dims: List[int], used to specify the number of
                    one-hot encoded dimensions for each feature in the table
                    (used when reloading models from checkpoints).
                cat_features: List[int], indices of categorical features, used
                    in model initialization if using feature type embeddings.
                num_features: List[int], indices of numerical features, used
                    in model initialization if using feature type embeddings.
                cat_target_cols: List[int], indices of categorical target
                    columns, used if there is a special embedding dimension
                    for target cols.
                num_target_cols: List[int], indices of numerical target
                    columns, used if there is a special embedding dimension
                    for target cols.
            device: Optional[int].
        """
        self.n_samples = n_samples
        c = yaml.safe_load(Path('npt_args.yml').read_text())
        metadata = yaml.safe_load(Path('metadata.yml').read_text())
        super().__init__()
        self.mp_distributed = False
        # *** Extract Configs ***
        # cannot deepcopy wandb config.
        #if c.mp_distributed:
        #    self.c = Args(c.__dict__)
        #else:
        self.c = c
        # * Main model configuration *
        self.device = device
        # * Dataset Metadata *
        input_feature_dims = [1] * 100 #metadata['input_feature_dims']
        cat_features = metadata['cat_features']
        num_features = metadata['num_features']
        # * Dimensionality Configs *
        # how many attention blocks are stacked after each other
        self.stacking_depth = c['model_stacking_depth']
        # the shared embedding dimension of each attribute is given by
        self.dim_hidden = c['model_dim_hidden']
        # we use num_heads attention heads
        self.num_heads = c['model_num_heads']
        # how many feature columns are in the input data
        self.num_input_features = 100#len(input_feature_dims)

        # *** Build Model ***
        # We immediately embed each element
        # (i.e., a table with N rows and D columns has N x D elements)
        # to the hidden_dim. Similarly, in the output, we will "de-embed"
        # from this hidden_dim.

        # Build encoder
        self.enc = self.get_npt()
        # *** Input In/Out Embeddings ***

        # In-Embedding
        # Linearly embeds each of the `D` [len(input_feature_dims)] feature
        # columns to a shared embedding dimension E [dim_hidden].
        # Before the embedding, each column has its own dimensionionality
        # H_j [dim_feature_encoding], given by the encoding dimension of the
        # feature (e.g. This is given by the one-hot-encoding size for
        # categorical variables + one dimension for the mask token and two-
        # dimensional for continuous variables (scalar + mask_token)).
        # See docstring of NPTModel for further context.

        # Since bert masking is turned off and all the features in our dataset are numerical commenting self.inembeddin logic
        
        self.in_embedding = nn.ModuleList([
            nn.Linear(dim_feature_encoding, self.dim_hidden)
            for dim_feature_encoding in input_feature_dims])

        # Out embedding.
        # The outputs of the AttentionBlocks have shape (N, D, E)
        # [N, len(input_feature_dim), dim_hidden].
        # For each of the column j, we then project back to the dimensionality
        # of that column in the input (N, H_j-1), subtracting 1, because we do
        # not predict the mask tokens, which were present in the input.

        # Need to remove the mask column if we are using BERT augmentation,
        # otherwise we just project to the same size as the input.
        self.model_bert_augmentation  = False
        if self.model_bert_augmentation:
            get_dim_feature_out = lambda x: x - 1
        else:
            get_dim_feature_out = lambda x: x

        self.out_embedding = nn.ModuleList([
            nn.Linear(
                self.dim_hidden,
                get_dim_feature_out(dim_feature_encoding))
            for dim_feature_encoding in input_feature_dims])
        #nn.Linear(1,32) #
        self.classifier =  nn.Linear(100,self.n_samples) #nn.Sequential( #nn.Linear(32,self.n_samples)
                            #nn.Linear(32,self.n_samples),
                            #nn.LeakyReLU()
                           # nn.Linear(200,200)
                        #)

        # *** Gradient Clipping ***
        self.exp_gradient_clipping = 1
        if self.exp_gradient_clipping:
            clip_value = self.exp_gradient_clipping
            #print(f'Clipping gradients to value {clip_value}.')
            for p in self.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def get_npt(self):
        """
        A model performing "flattened" attention over the rows and
        "nested" attention over the columns.

        This is reasonable if we don't aim to maintain column equivariance
        (which we essentially never do, because of the column-specific
        feature embeddings at the input and output of the NPT encoder).

        This is done by concatenating the feature outputs of column
        attention and inputting them to row attention. Therefore, it requires
        reshaping between each block, splitting, and concatenation.
        """
        if self.stacking_depth < 2:
            raise ValueError(
                f'Stacking depth {self.stacking_depth} invalid.'
                f'Minimum stacking depth is 2.')
        if self.stacking_depth % 2 != 0:
            raise ValueError('Please provide an even stacking depth.')


        #print('Building NPT.')

        # *** Construct arguments for row and column attention. ***

        row_att_args = {'c': self.c} # Why are the arguments the same in the row and column attention?
        col_att_args = {'c': self.c}

        # Perform attention over rows first
        att_args = cycle([row_att_args, col_att_args])
        AttentionBlocks = cycle([MHSA])

        D = self.num_input_features
        
        enc = []

        #if self.c.model_hybrid_debug:
        #    enc.append(Print())

        # Reshape to flattened representation (1, N, D*dim_input)
        enc.append(ReshapeToFlat())

        enc = self.build_hybrid_enc(
            enc, AttentionBlocks, att_args, D)
        #print("yo! after build_enc")
        enc = nn.Sequential(*enc)
        return enc

    def build_hybrid_enc(self, enc, AttentionBlocks, att_args, D):
        final_shape = 'flat'

        #if self.c.model_hybrid_debug:
        #    stack = [Print()]
        #else:
        stack = []

        layer_index = 0

        while layer_index < self.stacking_depth:
            #print("count ", layer_index)
            '''
            if layer_index % 2 == 1:
            # Input is already in nested shape (N, D, E)
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden, self.dim_hidden, self.dim_hidden,
                    **next(att_args)))
                # Reshape to flattened representation
                stack.append(ReshapeToFlat())
                final_shape = 'flat'
                #if self.c.model_hybrid_debug:
                #    stack.append(Print())
            else:
            '''
            # Input is already in flattened shape (1, N, D*E)

            # Attend between instances N
            # whenever we attend over the instances,
            # we consider dim_hidden = self.c.dim_hidden * D
            stack.append(next(AttentionBlocks)(
                self.dim_hidden * D, self.dim_hidden * D,
                self.dim_hidden * D,
                **next(att_args)))
            
            #stack.append(ReshapeToFlat())
            #final_shape = 'flat'
            # Reshape to nested representation
            #stack.append(ReshapeToNested(D=D))
            #final_shape = 'nested'

            #if self.c.model_hybrid_debug:
            #    stack.append(Print())
            
            # Conglomerate the stack into the encoder thus far
            enc += stack
            stack = []

            layer_index += 1

        # Reshape to nested representation, for correct treatment
        # after enc
        if final_shape == 'flat':
            enc.append(ReshapeToNested(D=D))

        return enc

    def forward(self, X_ragged, X_labels=None, eval_model=None):
        #in_dims = [X_ragged[0].shape[0], len(X_ragged), -1]

        # encode ragged input array D x {(NxH_j)}_j to NxDxE)
        X = [embed(X_ragged[i]) for i, embed in enumerate(self.in_embedding)]
        #print("DEBUG: The shape of X after in_embedding: ",len(X))
        # The output X here should be: 32x200x128
        X = torch.stack(X, 1) # what is the meaning of this? 
        #print("DEBUG: The shape of X after in_embedding and stacking: ",X.shape)
        # apply NPT
        X = self.enc(X)
        #print("Shape of the encoder output: ", X.shape)
        #print("yo")
        X_ragged = torch.randn(100,X.shape[0],1)
        for i, de_embed in enumerate(self.out_embedding):
            X_ragged[i,:,:] = de_embed(X[:,i,:])

        X_ragged = X_ragged.permute([1, 0, 2]).to('cuda:0')

        output_new = torch.randn(X.shape[0],self.n_samples)
        for i in range(len(X_ragged)):
            output_new[i] = self.classifier(X_ragged[i].t())

        return output_new


class Permute(nn.Module):
    """Permutation as nn.Module to include in nn.Sequential."""
    def __init__(self, idxs):
        super(Permute, self).__init__()
        self.idxs = idxs

    def forward(self, X):
        return X.permute(self.idxs)


class ReshapeToFlat(nn.Module):
    """Reshapes a tensor of shape (N, D, E) to (1, N, D*E)."""
    def __init__(self):
        super(ReshapeToFlat, self).__init__()

    @staticmethod
    def forward(X):
        return X.reshape(1, X.size(0), -1)


class ReshapeToNested(nn.Module):
    """Reshapes a tensor of shape (1, N, D*E) to (N, D, E)."""
    def __init__(self, D):
        super(ReshapeToNested, self).__init__()
        self.D = D

    def forward(self, X):
        return X.reshape(X.size(1), self.D, -1)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('Debug', x.shape)
        return x

####################################################################################
####################################################################################

import os
from functools import partial
import itertools

import torch
from torch import nn
from sklearn.cluster import KMeans
from combo.combo.models.cluster_comb import ClustererEnsemble
from combo.combo.models.cluster_eac import EAC

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

import time

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
    kmsamples = []
    # print(k)

    for (sample, idx) in batch:
        #print(sample.shape)
        samples.append(sample)
        idxs.append(idx)
        sample = sample[:,0]
        kmsamples.append(sample)

    samples = torch.stack(samples)
    kmean_samples = torch.stack(kmsamples)
    # print(samples.shape)
    idxs = torch.tensor(idxs)
    # print(samples)

    estimators = [KMeans(n_clusters=k, random_state=0),
                  KMeans(n_clusters=k, random_state=1),
                  KMeans(n_clusters=k, random_state=2),]

    # clf = KMeans(n_clusters=k)
    clf = ClustererEnsemble(estimators, n_clusters=k)
    clf.fit(kmean_samples.numpy())
    # print(clf.labels_)

    # return local sample, sample index, and also local labels
    return samples.float(), idxs, torch.from_numpy(clf.labels_)


class NeuralNetwork(nn.Module):
    def __init__(self, n_features, n_samples, hidden_neuron=16, n_layers=2, k=5):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        layer_list = []

        layer = nn.Linear(n_features, hidden_neuron)
        init.xavier_normal_(layer.weight)

        layer_list.append(layer),
        layer_list.append(nn.ELU())
        for i in range(n_layers):
            layer = nn.Linear(hidden_neuron, hidden_neuron)
            init.xavier_normal_(layer.weight)
            layer_list.append(layer)
            layer_list.append(nn.ELU())
        
        layer = nn.Linear(hidden_neuron, k)
        init.xavier_normal_(layer.weight)
        layer_list.append(layer)

        self.simple_nn = nn.Sequential(*layer_list)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.simple_nn(x)
        return logits


#if __name__ == "__main__": 
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
    global epochs
    epochs = 500
    k = 5
    global valid_roc
    global test_roc
    global test_roc_approx_2
    global valid_ap
    global test_ap
    global test_ap_approx_2
    global valid_prn
    global test_prn
    global test_prn_approx_2


    valid_roc = []
    test_roc = []
    valid_ap = []
    test_ap = []
    valid_prn = []
    test_prn = []

    test_roc_approx_2   = []
    test_ap_approx_2    = []
    test_prn_approx_2   = []
    global mat_file
    #mat_file = 'pendigits.mat'
    #mat_file = 'letter.mat'
    mat_file = 'mnist.mat'
    # mat_file = 'annthyroid.mat'

    mat = loadmat(os.path.join('data', mat_file))
    X = mat['X']
    y = mat['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.2, shuffle=True,
                                                          random_state=42)

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_samples_test = X_test.shape[0]
    n_samples_valid = X_valid.shape[0]

    X_train, scalar = standardizer(X_train, keep_scalar=True)
    X_test = scalar.transform(X_test)
    X_valid = scalar.transform(X_valid)
    

    X_train_e = X_train.reshape((n_samples,n_features,1))
    X_test_e  = X_test.reshape((n_samples_test,n_features,1))
    X_valid_e = X_valid.reshape((n_samples_valid,n_features,1))

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

    train_set = PyODDataset(X=X_train_e)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               collate_fn=partial(collate_batch2, k=k),
                                               shuffle=False)

    test_set = PyODDataset(X=X_test_e)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              collate_fn=collate_batch2,
                                              shuffle=False)

    valid_set = PyODDataset(X=X_valid_e)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch2,
                                               shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    exp_device = 'cuda:0'

    #model = NeuralNetwork(n_features=n_features, n_samples=n_samples, hidden_neuron=64, n_layers=3,
    #                      k=k).to('cuda:0')
    model = NPTModel(device=exp_device, n_samples=k).to('cuda:0')

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    #criterion = nn.NLLLoss()
    all_pairs = list(itertools.combinations(range(k), 2))
    # mse_tracker = []

    best_valid = 0
    best_test = 0
    global train_losses
    train_losses = []
    global train_accu
    train_accu = []
    global valid_accu
    valid_accu = []
    test_accu = []

    for e in range(epochs):
        #print('589')
        total_loss = 0
        #loss =0
        for batch_idx, batch in enumerate(train_loader):

            # print('batch', batch_idx)
            loss = 0
            optimizer.zero_grad()
            # print(batch_idx, batch[0].shape, batch[1].shape, batch[2].shape)

            # batch 2 is the local ground truth
            # print(batch[2])

            output = model(batch[0].permute(1,0,2).to(exp_device))
            print(output.shape)
            # output = torch.argmax(output, axis=1)
            #print('605')

            # intra distance minimize
            '''
            for ih in range(k):
                # print(torch.where(batch[2] == ih)[0].shape)
                k_ind = torch.where(batch[2] == ih)[0]
                k_pred = output[k_ind, :]
                # find the centroid 
                mu = k_pred.mean()
                # Calculate the wcss
                result = abs(k_pred - mu)
                loss += result.sum()
            #print("Intra Distance:",loss)
            '''
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


                #print(loss)
            #print("finish")
            print(" ")
            total_loss += loss.item()
            #print("after loss")
            loss.backward()
            #print("after back")
            optimizer.step()

        print('epoch', e, 'loss', total_loss)
        train_losses.append(total_loss)

        # validation loss
        with torch.no_grad():
            model.eval()
            #print("636")

            # on the train
            pred_prob = model(torch.tensor(X_train_e).permute(1,0,2).float().to(exp_device))
            pred_labels = torch.argmax(pred_prob, axis=1)
            w = pred_labels.cpu().numpy()
            train_labels = clf.labels_
            train_completeness = completeness_score(train_labels, w)
            print("Train Completeness: ", train_completeness)
            train_accu.append(train_completeness)

            # on the validation

            pred_prob = model(torch.tensor(X_valid_e).permute(1,0,2).float().to(exp_device))
            pred_labels = torch.argmax(pred_prob, axis=1)
            # valid_labels = torch.cdist(torch.tensor(X_valid).float(), torch.tensor(X_train).float())
            #print("650")

            w = pred_labels.cpu().numpy()
            valid_labels = clf.predict(X_valid)

            valid_completeness = completeness_score(valid_labels, w)
            valid_accu.append(valid_completeness)

            # print(adjusted_mutual_info_score(valid_labels, w))
            # print(adjusted_rand_score(valid_labels, w))
            # print(homogeneity_completeness_v_measure(valid_labels, w))
            print("Valid Completeness: " ,valid_completeness)

            #print('666')
            if best_valid <= valid_completeness:
                #print("ok")
                best_valid = valid_completeness
                # do the evaluation only for the test when it is promising
                start = time.time()
                pred_prob = model(torch.tensor(X_test_e).permute(1,0,2).float().to(exp_device))
                pred_labels = torch.argmax(pred_prob, axis=1)
                w = pred_labels.cpu().numpy()
                gpu_time.append(time.time()-start)
                

                start = time.time()
                test_labels = clf.predict(X_test)
                cpu_time.append(time.time()-start)

                best_test = completeness_score(test_labels, w)
            #print("680")

            # print()
    print('test completeness', best_test)
    print(np.sum(cpu_time), np.sum(gpu_time))

# %%
test_tracker = []
for i in range(1):
    print(i, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    train()
    print(i, "********************************************")

# %%
import matplotlib.pyplot as plt
import numpy as np

x_range = np.arange(epochs)

plt.plot(x_range, train_losses)
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("training loss (intra cluster dist - inter cluster dist)")
plt.show()
plt.savefig("cluster_loss.png")
plt.clf()

# %%
plt.plot(x_range, train_accu, label='train')
plt.plot(x_range, valid_accu, label='valid')
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("accuracy (compleness score)")
plt.title(mat_file)
plt.legend()
plt.savefig("cluster_accu.png")
plt.show()
