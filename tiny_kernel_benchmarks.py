import numpy as np
import math
import time
import torch
import jax
from jax import numpy as jnp

from pykeops.torch import LazyTensor

from densities.kernel_count import _scaled_normal_diag_pdf_batchedmean
from densities import faiss_base


use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


Q = 4096
sigma = 1e-1
iterations = 10

for dim in [4, 16, 64]:
    for N in [4096, 16384, 65536]:
        print(f"Dimension: {dim}, Dataset size: {N}")
        data = np.random.uniform(low=-1, high=1, size=(N, dim)).astype(np.float32)
        queries = np.random.uniform(low=-1, high=1, size=(iterations, Q, dim)).astype(np.float32)
        x_i = LazyTensor( tensor(data[:,None,:]) )
        j_data = jnp.array(data)


        def keops_count(query):
            x_q = LazyTensor( tensor(query[None,:,:]) )
            D_ij = (((x_i - x_q) / sigma)**2).sum(dim=2)
            K_ij = (- 0.5 * D_ij).exp()
            return K_ij.sum(dim=2).sum(dim=0)


        cov = jnp.eye(dim) * sigma ** 2
        normalizer = (jnp.linalg.det(cov) ** (- 1/2))


        @jax.jit
        def jax_count_single(single_query):
            probs_per = _scaled_normal_diag_pdf_batchedmean(j_data, cov, single_query)
            return probs_per.sum() / normalizer
        jax_count = jax.vmap(jax_count_single)


        faiss_index = faiss_base.new(dim)
        faiss_index = faiss_base.update_batch(faiss_index, data)

        def faiss_count(query):
            return faiss_base.get_nn_batch(faiss_index, query, n_neighbors=16)

        keops_count(queries[0])
        k_start = time.time()
        for i in range(iterations):
            _ = keops_count(queries[i])
        k_end = time.time()

        j_start = time.time()
        for i in range(iterations):
            _ = jax_count(queries[i])
        j_end = time.time()

        f_start = time.time()
        for i in range(iterations):
            _ = faiss_count(queries[i])
        f_end = time.time()


        print(f"Elapsed jax / kquery: {(j_end - j_start) / (Q * iterations) * 1000 :.4e}")
        print(f"Elapsed faiss / kquery: {(f_end - f_start) / (Q * iterations) * 1000 :.4e}")
        print(f"Elapsed keops / kquery: {(k_end - k_start) / (Q * iterations) * 1000 :.4e}")
        print()
