# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:25:22 2022

@author: yzhao
"""

import warnings
import torch
import sklearn
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

import numpy as np
from numpy import percentile
import numbers

from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length

from pyod.utils.utility import check_parameter


def validate_device(gpu_id):
    """Validate the input device id (GPU id) is valid on the given
    machine. If no GPU is presented, return 'cpu'.
    Parameters
    ----------
    gpu_id : int
        GPU id to be used. The function will validate the usability
        of the GPU. If failed, return device as 'cpu'.
    Returns
    -------
    device_id : str
        Valid device id, e.g., 'cuda:0' or 'cpu'
    """
    # if it is cpu
    if gpu_id == -1:
        return 'cpu'

    # cast to int for checking
    gpu_id = int(gpu_id)

    # if gpu is available
    if torch.cuda.is_available():
        # check if gpu id is between 0 and the total number of GPUs
        check_parameter(gpu_id, 0, torch.cuda.device_count(), param_name='gpu id', include_left=True,
                        include_right=False)
        device_id = 'cuda:{}'.format(gpu_id)
    else:
        if gpu_id != 'cpu':
            warnings.warn('The cuda is not available. Set to cpu.')
        device_id = 'cpu'

    return device_id


def bottomk(A, k, dim=1):
    if len(A.shape) == 1:
        dim = 0
    # tk = torch.topk(A * -1, k, dim=dim)
    # see parameter https://pytorch.org/docs/stable/generated/torch.topk.html
    tk = torch.topk(A, k, dim=dim, largest=False)
    return tk[0].cpu(), tk[1].cpu()


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

        return sample_torch, idx
