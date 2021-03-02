import torch
from matplotlib import pyplot as plt

from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def new(dim):
    return None


def update_batch(index, data):
    data = tensor(data)
    X_j = LazyTensor(data[None, :, :])
    return X_j


def get_nn_batch(index, queries, n_neighbors=16):
    X_j = index
    X_i = LazyTensor(tensor(queries)[:, None, :])
    D_ij = ((X_i - X_j) ** 2).sum(-1)
    return D_ij.argKmin(n_neighbors, dim=1)


def convert_array(x):
    return tensor(x)
