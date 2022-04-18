# -*- coding: utf-8 -*-

import numpy as np
import torch
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn import datasets

np.random.seed(1943)
torch.manual_seed(1943)

def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', [a, b])
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

iris = datasets.load_iris()

# X = feature[:, 2:4]  # use "Petal length" and "Petal width" only
X = iris['data'] # use "Petal length" and "Petal width" only
y = iris['target']
d = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()

X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).long()

# y = torch.from_numpy(y).float()

k = 1
num_cut = [k]*d  # "Petal length" and "Petal width"
num_leaf = np.prod(np.array(num_cut) + 1)
# num_class = 1
num_class = 3

cut_points_list = [torch.rand([i], requires_grad=True, device=device) for i in num_cut]
leaf_score = torch.rand([num_leaf, num_class], requires_grad=True, device=device)


X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)


loss_function = torch.nn.CrossEntropyLoss()
# loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(cut_points_list + [leaf_score], lr=0.01)

for i in range(1000):
    optimizer.zero_grad()
    y_pred = nn_decision_tree(X_train, cut_points_list, leaf_score, device=device, temperature=0.1)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(loss.detach().cpu().numpy())
    y_numpy = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
print('train error rate %.2f' % (1-np.mean(y_numpy==y_train.cpu().numpy())))

#%% make prediction
y_pred = nn_decision_tree(X_test, cut_points_list, leaf_score, device=device, temperature=0.1)
y_numpy = np.argmax(y_pred.detach().numpy(), axis=1)
print('test error rate %.2f' % (1-np.mean(y_numpy==y_test)))