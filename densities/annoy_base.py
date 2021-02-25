import numpy as np
import annoy


def new(dim):
    return annoy.AnnoyIndex(dim, 'euclidean')


def update_batch(index, data):
    for d in data:
        index.add_item(index.get_n_items(), d)
    index.build(10, n_jobs=16)
    return index


def get_nn_batch(index, queries, n_neighbors=16):
    neighbors = np.zeros((queries.shape[0], n_neighbors)).astype(int)
    for i in range(queries.shape[0]):
        ns = index.get_nns_by_vector(queries[i], n_neighbors)
        neighbors[i] = ns
    return neighbors
