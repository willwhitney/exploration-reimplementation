import time
import numpy as np

from densities import annoy_base
from densities import faiss_base
from densities import nms_base
from densities import keops_base


N = 10000
d = 32
qsize = 8192
iterations = 100

data = np.random.uniform(low=-1, high=1, size=(N, d)).astype(np.float32)
queries = np.random.uniform(low=-1, high=1, size=(qsize, d)).astype(np.float32)

for knn in [faiss_base, keops_base]:
    print((f"{knn.__name__}, N={N}, d={d}, qsize={qsize}, iter={iterations}"))
    index = knn.new(d)
    index = knn.update_batch(index, data)

    knn_queries = knn.convert_array(queries)
    start = time.time()
    for _ in range(iterations):
        neighbors = knn.get_nn_batch(index, knn_queries, n_neighbors=10)
    end = time.time()
    elapsed = end - start
    print(f"Read time: {elapsed}")
    print()
