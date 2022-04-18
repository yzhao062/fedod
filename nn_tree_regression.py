# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from opt_einsum import contract
import pandas as pd

np.random.seed(42)
torch.manual_seed(42)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

mat_file = '19_annthyroid'
# mat_file = 'pendigits.mat'
# mat_file = 'letter.mat'
# mat_file = 'mnist.mat'
# mat_file = 'annthyroid.mat'

# mat_file = 'arrhythmia.mat'

# mat = loadmat(os.path.join('data', mat_file))
# X = mat['X']
# y = mat['y'].ravel()

X = pd.read_csv(os.path.join('data', mat_file+'_X.csv'), header=None).to_numpy()
y = pd.read_csv(os.path.join('data', mat_file+'_y.csv'), header=None).to_numpy()

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.2, shuffle=True, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.2, shuffle=True,
                                                          random_state=42)

# for classicial one
# clf = DecisionTreeClassifier()
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)
train_tree = clf.predict(X_train)
print(roc_auc_score(y_train, train_tree))

test_tree = clf.predict(X_test)
print(roc_auc_score(y_test, test_tree))

#### for neural models
X_train = torch.from_numpy(X_train).float()
# y_train = torch.from_numpy(y_train).long()
y_train = torch.from_numpy(y_train).float().reshape(len(y_train),1)
X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).long()
X_valid = torch.from_numpy(X_valid).float()

n_samples = X_train.shape[0]
n_features = X_train.shape[1]

####################### this is relatively hard to assign*******************
### both k and n_c can be varying
k = 1
# n_c = int(n_features*0.1)
num_cut = [k]*n_features  # "Petal length" and "Petal width"
# num_cut = [k]*n_c  # "Petal length" and "Petal width"
#############################################################################

temperature = 0.1
num_leaf = np.prod(np.array(num_cut) + 1)
num_class = 1
# num_class = 3

cut_points_list = [torch.rand([i], requires_grad=True, device=device) for i in num_cut]
leaf_score = torch.rand([num_leaf, num_class], requires_grad=True, device=device)


X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
X_valid = X_valid.to(device)

# loss_function = torch.nn.CrossEntropyLoss()
loss_function = torch.nn.MSELoss()
# loss_function = torch.nn.L1Loss()
# loss_function = torch.nn.HuberLoss()
optimizer = torch.optim.Adam(cut_points_list + [leaf_score], lr=0.1)

best_valid = 0
best_test = 0

train_losses = []
train_roc = []
valid_roc = []
# test_roc = []

epochs = 1000

for i in range(epochs):
    optimizer.zero_grad()
    y_pred = nn_decision_tree(X_train, cut_points_list, leaf_score, device=device, temperature=temperature)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    if i % 1 == 0:
    # y_numpy = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        y_numpy = y_pred.detach().cpu().numpy()
        y_numpy = np.nan_to_num(y_numpy)
        print(i, 'loss is {loss}; roc is {roc}'.format(loss=loss.detach().cpu().numpy(), roc=roc_auc_score(y_train.cpu().numpy(), y_numpy)))
        train_roc.append(roc_auc_score(y_train.cpu().numpy(), y_numpy))
        
        with torch.no_grad():
            
            ## make evaluation
            y_pred = nn_decision_tree(X_valid, cut_points_list, leaf_score, device=device, temperature=temperature)
            y_numpy = y_pred.detach().cpu().numpy()
            y_numpy = np.nan_to_num(y_numpy)
            print('valid roc %.4f' % (roc_auc_score(y_valid, y_numpy)))
            valid_roc.append(roc_auc_score(y_valid, y_numpy))
            
            if best_valid <= roc_auc_score(y_valid, y_numpy):
                best_valid = roc_auc_score(y_valid, y_numpy)
                
                ## make prediction
                y_pred = nn_decision_tree(X_test, cut_points_list, leaf_score, device=device, temperature=temperature)
                y_numpy = y_pred.detach().cpu().numpy()
                y_numpy = np.nan_to_num(y_numpy)
                print('test roc %.4f' % (roc_auc_score(y_test, y_numpy)))
                
                best_test = roc_auc_score(y_test, y_numpy)

print('final test', best_test)    

#%%

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)

plt.plot(x_range, np.log(train_losses))
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("log of training loss (MSE)")
plt.show()

import matplotlib.pyplot as plt

#%%
plt.style.use('seaborn-whitegrid')
import numpy as np

x_range = np.arange(epochs)

plt.plot(x_range, train_roc, label='train')
plt.plot(x_range, valid_roc, label='valid')
# plt.xticks([0,10, 20, 30, 50])
# plt.grid(False)
plt.xlabel("number of epochs")
plt.ylabel("accuracy (compleness score)")
plt.title(mat_file)

plt.legend()
plt.show()