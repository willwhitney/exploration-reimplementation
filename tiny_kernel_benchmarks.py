import numpy as np
import math
import time
import torch
import jax
from jax import numpy as jnp

from pykeops.torch import LazyTensor

from densities.kernel_count import _scaled_normal_diag_pdf_batchedmean


use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


N = 10000
Q = 1
dim = 32
sigma = 1e-1
iterations = 10
data = np.random.uniform(low=-1, high=1, size=(N, dim)).astype(np.float32)
queries = np.random.uniform(low=-1, high=1, size=(iterations, Q, dim)).astype(np.float32)
# query = data

x_i = LazyTensor( tensor(data[:,None,:]) )
# x_q = LazyTensor( tensor(query[None,:,:]) )


data = jnp.array(data)


def keops_count(query):
    x_q = LazyTensor( tensor(query[None,:,:]) )
    print(x_i.shape)
    print(x_q.shape)
    D_ij = (((x_i - x_q) / sigma)**2).sum(dim=2)
    K_ij = (- 0.5 * D_ij).exp()
    import ipdb; ipdb.set_trace()
    return K_ij.sum(dim=2).sum(dim=0)


cov = jnp.eye(dim) * sigma ** 2
normalizer = (jnp.linalg.det(cov) ** (- 1/2))


@jax.jit
def jax_count_single(single_query):
    probs_per = _scaled_normal_diag_pdf_batchedmean(data, cov, single_query)
    return probs_per.sum() / normalizer
jax_count = jax.vmap(jax_count_single)


k_start = time.time()
for i in range(iterations):
    _ = keops_count(queries[i])
k_end = time.time()


j_start = time.time()
for i in range(iterations):
    _ = jax_count(queries[i])
j_end = time.time()

print(f"Elapsed keops: {k_end - k_start :.4f}")
print(f"Elapsed jax: {j_end - j_start :.4f}")
