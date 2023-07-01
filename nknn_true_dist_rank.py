
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
#from npt.NPTModel import NPTModel
import numpy as np
from sklearn.utils import check_array
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchsummary import summary



    
############################################################################
############################################################################

"""Contains Tabular Transformer Model definition."""
from itertools import cycle

import torch
import torch.nn as nn
import sys
from npt_modules import MHSA
#from npt.utils.config_utils import Args
#from npt.utils.encode_utils import torch_cast_to_dtype


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
        self.metadata = yaml.safe_load(Path('metadata.yml').read_text())
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
        input_feature_dims = [1] * self.metadata['input_feature_dims']
        cat_features = self.metadata['cat_features']
        num_features = self.metadata['num_features']
        # * Dimensionality Configs *
        # how many attention blocks are stacked after each other
        self.stacking_depth = c['model_stacking_depth']
        # the shared embedding dimension of each attribute is given by
        self.dim_hidden = c['model_dim_hidden']
        # we use num_heads attention heads
        self.num_heads = c['model_num_heads']
        # how many feature columns are in the input data
        self.num_input_features = len(input_feature_dims)

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
        self.classifier =  nn.Linear(self.metadata['input_feature_dims'],self.n_samples)#nn.Sequential( 
                           # nn.Linear(32,200),
                           # nn.ReLU(),
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
        #print("[Check] After build_enc")
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
        X_ragged = torch.randn(self.metadata['input_feature_dims'],X.shape[0],1)
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



# from torchmetrics.functional import spearman_corrcoef
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
    """
    PyOD Dataset class for PyTorch Dataloader
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
    samples_dist = []
    idxs = []
    
    for (sample, idx) in batch:
        samples.append(sample)
        idxs.append(idx)
        sample = sample[:,0]
        #print("****************** Shape after squeeze: ",sample.shape)
        samples_dist.append(sample)
    #print("[Collate] Length of samples :",len(samples))
    # samples = torch.Tensor(samples)
    samples = torch.stack(samples)
    samples_dist = torch.stack(samples_dist)
    # samples = sample.float()
    #print(samples.shape)
    idxs  = torch.tensor(idxs)
    #print(idxs)
    dists = torch.cdist(samples_dist, samples_dist)
    #print("******************: [Collate] dist shape: ",dists.shape)
    return samples.float(), dists.float(), idxs


import torchsort

def corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()*-1


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
    
#if __name__ == "__main__": 
def train():
    #metadata = yaml.safe_load(Path('metadata.yml').read_text())
    #contamination = 0.1  # percentage of outliers
    #n_train = 10000  # number of training points
    #n_test = 2000  # number of testing points
    #n_features = metadata['input_feature_dims']  # number of features

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

     # Generate sample data
    #X_train, y, X_test, y_test = \
    #     generate_data(n_train=n_train,
    #                   n_test=n_test,
    #                   n_features=n_features,
    #                   contamination=contamination,
    #                   random_state=42)
    prediction_time = 0
    calc_time = 0 
    
    #mat_file = 'pendigits.mat'
    #mat_file = 'letter.mat'
    #mat_file = 'mnist.mat'
    mat_file = 'annthyroid.mat'

    mat = loadmat(os.path.join('data', mat_file))
    X = mat['X']
    y = mat['y'].ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, shuffle=True, random_state=42)
    
    
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_samples_test      = X_test.shape[0]
    n_samples_valid     = X_valid.shape[0]

    X_train, scalar = standardizer(X_train, keep_scalar=True)
    X_test = scalar.transform(X_test)
    X_valid = scalar.transform(X_valid)

    X_train_e = X_train.reshape((n_samples,n_features,1))
    X_test_e  = X_test.reshape((n_samples_test,n_features,1))
    X_valid_e = X_valid.reshape((n_samples_valid,n_features,1))


    '''
    # Add another dimension over here for bert masking
    n_samples_train     = n_samples

    zeros_vector_train  = np.zeros((n_samples_train, n_features, 1))
    zeros_vector_test   = np.zeros((n_samples_test, n_features, 1))
    zeros_vector_valid  = np.zeros((n_samples_valid, n_features, 1))

    # Add new 3D axis
    X_train_n =   X_train[...,np.newaxis]
    X_test_n  =   X_test[...,np.newaxis]
    X_valid_n =   X_valid[...,np.newaxis]

    encoded_X_train = np.concatenate((X_train_n, zeros_vector_train), axis=-1) 
    encoded_X_test  = np.concatenate((X_test_n, zeros_vector_test), axis=-1) 
    encoded_X_valid = np.concatenate((X_valid_n, zeros_vector_valid), axis=-1)

    for i in range(n_samples):
        encoded_X_train[i] = encoded_X_train[i].reshape(-1,2)
    for i in range(n_samples_test):
        encoded_X_test[i] = encoded_X_test[i].reshape(-1,2)
    for i in range(n_samples_valid):
        encoded_X_valid[i] = encoded_X_valid[i].reshape(-1,2)
    #encoded_X_train = encoded_X_train.reshape(n_samples,-1,2)
    #print("****************** Total train samples:",n_samples)
    #print("[TRAIN] The shape of training data: ", encoded_X_train.shape)
    #print("[TRAIN] The first few rows of training data: ", encoded_X_train[:4,:,:])
    '''
    batch_size = 200
    
    train_set = PyODDataset(X=X_train_e)    
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch2,
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
    
    
    #model = NeuralNetwork(n_features=n_features, n_samples=n_samples, hidden_neuron=64, n_layers=2).to('cuda:0')
    exp_device = 'cuda:0'
    model = NPTModel(device=exp_device, n_samples=n_samples).to('cuda:0')
#    summary(model, input_size=(32, 1, 2), device='cpu')
    optimizer = optim.Adam(model.parameters(), lr=5e-3) # 3 really good but slow
    # criterion = nn.NLLLoss()
    global epochs
    epochs = 200
    k=10
    
    global train_losses
    train_losses = []
    global inter_track
    inter_track = []
    kendall_tracker = []
    
    best_valid = 0
    best_test_roc = 0
    best_test_prn = 0
    best_test_ap = 0
    best_inter = 0
    
    debug = 0
    # mse_tracker = []
    for e in range(epochs): 
        
        total_loss = 0
        total_kendall = 0
        total_ndcg = 0
        
        for batch_idx, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            #print(batch_idx, batch[0].shape, batch[1].shape)
            dist_label = batch[1]#.to('cuda:0')
            idx = batch[2]
            #print(idx.shape)
            #print(batch_idx, batch[0].shape, batch[1].shape, batch[2].shape)
            pred_dist = model(batch[0].permute(1,0,2).to('cuda:0'))
            #pred_dist = model(batch[0].to('cuda:0'))

            if debug:
                print("[TRAIN] Forward Pass output:", pred_dist[:,idx].shape)
                print("[TRAIN] Forward Pass output:", dist_label.shape)
            #print(idx.tolist())
            
            select_dist = pred_dist[:,idx]
            criterion = nn.MSELoss()
            #criterion = nn.HuberLoss()
            loss = criterion(select_dist, dist_label)
            
            loss += spearman(dist_label, select_dist)
            #loss = spearman(dist_label, select_dist)
            #print(loss)
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
            pred_labels = model(torch.tensor(X_valid_e).permute(1,0,2).float().to('cuda:0'))
            #pred_labels = model(torch.tensor(X_valid).float().to('cuda:0'))
            valid_labels = torch.cdist(torch.tensor(X_valid).float(), torch.tensor(X_train).float())
            # ndcg_test = 0
            # for k in range(test_labels.shape[0]):
                # ndcg_test += kendalltau([test_labels[k, :].numpy()],[pred_labels[k, :].numpy()])
            topks, indx = bottomk(valid_labels, k=k)
            topks_pred, indx_pred = bottomk(pred_labels, k=k)
            '''
            print("The shape of the valid labels: ", valid_labels.shape)
            print("The shape of the pred labels: ", pred_labels.shape)
            print("The shape of the indx labels: ", indx.shape)
            print("The shape of the indx_pred labels: ", indx_pred.shape)
            '''
            for h in range(valid_labels.shape[0]):
                total_inter += len(np.intersect1d(indx[h, :], indx_pred[h, :]))
                # print(len(np.intersect1d(indx[h, :], indx_pred[h, :])))
        
        
        print('total intersec', total_inter)
        print()
        
        inter_track.append(total_inter/X_valid.shape[0]/k)
        
        # this is for validation but not available
        pred_dist = model(torch.tensor(X_valid_e).permute(1,0,2).float().to('cuda:0'))
        #pred_dist = model(torch.tensor(X_valid).float().to('cuda:0'))
        dist, idx = bottomk(pred_dist, k=k)

        real_dist = torch.cdist(torch.tensor(X_valid).float(), torch.tensor(X_train).float())
        dist_real, idx_real = bottomk(real_dist, k=k)

        global valid_roc_real
        valid_roc_real  = roc_auc_score(y_valid, dist_real[:, -1].numpy())
        valid_roc_approx = roc_auc_score(y_valid, dist[:, -1].detach().numpy())
        print('valid real', valid_roc_real)
        print('valid approx', valid_roc_approx)
        valid_roc.append(valid_roc_approx)
        
        global valid_prn_real
        valid_prn_real = precision_n_scores(y_valid, dist_real[:, -1].detach().numpy())
        valid_prn_approx = precision_n_scores(y_valid, dist[:, -1].detach().numpy())
        print('valid real prn', valid_prn_real)
        print('valid approx prn', valid_prn_approx)
        valid_prn.append(valid_prn_approx)

        global valid_ap_real
        valid_ap_real = average_precision_score(y_valid, dist_real[:, -1].detach().numpy())
        valid_ap_approx = average_precision_score(y_valid, dist[:, -1].detach().numpy())
        print('valid real ap', valid_ap_real)
        print('valid approx ap', valid_ap_approx)
        valid_ap.append(valid_ap_approx)
        print()

        
        start = time()
        print()
        # this is for test 
        pred_dist = model(torch.tensor(X_test_e).permute(1,0,2).float().to('cuda:0'))
        #pred_dist = model(torch.tensor(X_test).float().to('cuda:0'))

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

        global test_roc_real
        test_roc_real = roc_auc_score(y_test, dist_real[:, -1].numpy())
        test_roc_approx = roc_auc_score(y_test, dist[:, -1].detach().numpy())
        test_roc_approx2 = roc_auc_score(y_test, true_k_values)
        print('test real', test_roc_real)
        print('test approx', test_roc_approx)
        print('test approx2', test_roc_approx2)
        test_roc.append(test_roc_approx)
        test_roc_approx_2.append(test_roc_approx2)

        global test_ap_real
        test_ap_real = average_precision_score(y_test, dist_real[:, -1].detach().numpy())
        test_ap_approx = average_precision_score(y_test, dist[:, -1].detach().numpy())
        test_ap_approx2 = average_precision_score(y_test, true_k_values)
        print('test real ap', test_ap_real)
        print('test approx ap', test_ap_approx)
        print('test approx2', test_ap_approx2)
        test_ap.append(test_ap_approx)
        test_ap_approx_2.append(test_ap_approx2)

        global test_prn_real
        test_prn_real = precision_n_scores(y_test, dist_real[:, -1].detach().numpy())
        test_prn_approx = precision_n_scores(y_test, dist[:, -1].detach().numpy())
        test_prn_approx2 = precision_n_scores(y_test, true_k_values)
        print('test real prn', test_prn_real)
        print('test approx prn', test_prn_approx)
        print('test approx2', test_prn_approx2)
        test_prn.append(test_prn_approx)
        test_prn_approx_2.append(test_prn_approx2)


        print()
        
        if total_inter >= best_valid:
            best_valid = total_inter
            best_test_roc = roc_auc_score(y_test, dist[:, -1].detach().numpy())
            best_test_ap = average_precision_score(y_test, dist[:, -1].detach().numpy())
            best_test_prn = precision_n_scores(y_test, dist[:, -1].detach().numpy())
            
            best_test_roc2 = roc_auc_score(y_test, true_k_values)
            best_test_ap2 = average_precision_score(y_test, true_k_values)
            best_test_prn2 = precision_n_scores(y_test, true_k_values)
            
            # best_inter
        
        
        print('best test', best_test_roc, "|", best_test_ap, "|",best_test_prn,"|", best_valid/X_valid.shape[0]/k)
        print('best test2', best_test_roc2, "|", best_test_ap2, "|",best_test_prn2,"|", best_valid/X_valid.shape[0]/k)
    
    print('real', roc_auc_score(y_test, dist_real[:, -1].numpy()), "|", average_precision_score(y_test, dist_real[:, -1].detach().numpy()), "|",precision_n_scores(y_test, dist_real[:, -1].detach().numpy()),"|", 1)
    print('prediction time', prediction_time)
    print('cal time', calc_time)
    #     # we could calculate 
    
    
#%%

for i in range(1):
    print(i, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    train()
    print(i, "********************************************")

#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)
    
#plt.plot(x_range, train_losses)
#plt.grid(False)
#plt.xlabel("number of epochs")
#plt.ylabel("training loss (MSE)")

fig, ax = plt.subplots(2, 3, figsize=(16, 8))
'''
ax[0,0].plot(x_range, train_losses)
ax[0,0].set_xlabel('Epoch')
ax[0,0].set_ylabel('Train Loss')
'''
ax[0,0].plot(x_range, valid_roc)
ax[0,0].plot(x_range, [valid_roc_real]*len(x_range))
ax[0,0].set_xlabel('Epoch')
ax[0,0].set_ylabel('Validation ROC (↑)')

ax[0,1].plot(x_range, valid_prn)
ax[0,1].plot(x_range, [valid_prn_real]*len(x_range))
ax[0,1].set_xlabel('Epoch')
ax[0,1].set_ylabel('Validation Precision N (↑)')

ax[0,2].plot(x_range, valid_ap)
ax[0,2].plot(x_range, [valid_ap_real]*len(x_range))
ax[0,2].set_xlabel('Epoch')
ax[0,2].set_ylabel('Validation Average Precision (↑)')

ax[1,0].plot(x_range, test_roc,label='Approx ROC 1')
ax[1,0].plot(x_range, test_roc_approx_2,label='Approx ROC 2')
ax[1,0].plot(x_range, [test_roc_real]*len(x_range),label='Real ROC')
ax[1,0].legend()
ax[1,0].set_xlabel('Epoch')
ax[1,0].set_ylabel('Test ROC (↑)')

ax[1,1].plot(x_range, test_prn,label='Approx PRN 1')
ax[1,1].plot(x_range, test_prn_approx_2,label='Approx PRN 2')
ax[1,1].plot(x_range, [test_prn_real]*len(x_range),label='Real PRN')
ax[1,1].legend()
ax[1,1].set_xlabel('Epoch')
ax[1,1].set_ylabel('Test Precision N(↑)')

ax[1,2].plot(x_range, test_ap,label='Approx AP 1')
ax[1,2].plot(x_range, test_ap_approx_2,label='Approx AP 2')
ax[1,2].plot(x_range, [test_ap_real]*len(x_range),label='Real AP')
ax[1,2].legend()
ax[1,2].set_xlabel('Epoch')
ax[1,2].set_ylabel('Test Average Precision (↑)')

plt.tight_layout()
plt.savefig('train_valid.png')
print("Plot Saved")
plt.clf()

#%%
#import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
#import numpy as np

#x_range = np.arange(epochs)
    
#plt.plot(x_range, inter_track)
#plt.xlabel("number of epochs")
#plt.ylabel("the topk overlapping rate on the test")

'''
        total_inter = 0
        with torch.no_grad():
            #on the validation
            model.eval()
            pred_labels = model(torch.tensor(X_valid).permute(1,0,2).float().to('cuda:0') )
            print(pred_labels.shape)
            
            valid_labels = torch.cdist(torch.tensor(X_valid).float(), torch.tensor(X_train).float())
            print(valid_labels.shape)
            
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
        #print("***************** The input passed: ", torch.tensor(encoded_X_valid).permute(1,0,2).float().shape)
        pred_dist = model(torch.tensor(X_valid).permute(1,0,2).float().to('cuda:0'))
        dist, idx = bottomk(pred_dist, k=k)

        real_dist = torch.cdist(torch.tensor(X_valid).float(), torch.tensor(X_train).float())
        dist_real, idx_real = bottomk(real_dist, k=k)


        #print("****************** Real Dist Shape: ",real_dist.shape)
        #print("****************** Pred Dist Shape: ",dist[:, -1].shape)
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
        pred_dist = model(torch.tensor(X_test).permute(1,0,2).float().to('cuda:0'))
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
        

        print('test real ap', average_precision_score(y_test, dist_real[:, -1].detach().numpy()))
        print('test approx ap', average_precision_score(y_test, dist[:, -1].detach().numpy()))
        print('test approx2', average_precision_score(y_test, true_k_values))
        
        print('test real prn', precision_n_scores(y_test, dist_real[:, -1].detach().numpy()))
        print('test approx prn', precision_n_scores(y_test, dist[:, -1].detach().numpy()))
        print('test approx2', precision_n_scores(y_test, true_k_values))
        

        print()
        
        if total_inter >= best_valid:
            best_valid = total_inter
            best_test_roc = roc_auc_score(y_test, dist[:, -1].detach().numpy())
            best_test_ap = average_precision_score(y_test, dist[:, -1].detach().numpy())
            best_test_prn = precision_n_scores(y_test, dist[:, -1].detach().numpy())
            
            best_test_roc2 = roc_auc_score(y_test, true_k_values)
            best_test_ap2 = average_precision_score(y_test, true_k_values)
            best_test_prn2 = precision_n_scores(y_test, true_k_values)
            
            # best_inter
        
        print('best test', best_test_roc, "|", best_test_ap, "|",best_test_prn,"|", best_valid/X_valid.shape[0]/k)
        print('best test2', best_test_roc2, "|", best_test_ap2, "|",best_test_prn2,"|", best_valid/X_valid.shape[0]/k)
        
    #print('real', roc_auc_score(y_test, dist_real[:, -1].numpy()), "|", average_precision_score(y_test, dist_real[:, -1].detach().numpy()), "|",precision_n_scores(y_test, dist_real[:, -1].detach().numpy()),"|", 1)
    #print('prediction time', prediction_time)
    #print('cal time', calc_time)
    print(train_losses)
    #     # we could calculate 

    
    
#%%

for i in range(1):
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
plt.savefig("/home/ubuntu/fedod/fedod/fig1.png")


#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)
    
plt.plot(x_range, inter_track)
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("the topk overlapping rate on the test")
plt.savefig("/home/ubuntu/fedod/fedod/fig2.png")
    '''
    
