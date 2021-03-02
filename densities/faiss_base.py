import numpy as np
import faiss


def new(dim):
    faiss.omp_set_num_threads(16)
    index = faiss.IndexHNSWFlat(dim, 16)

    # quantizer = faiss.IndexHNSWFlat(dim, 16)
    # quantizer.hnsw.efConstruction = 40
    # quantizer.hnsw.efSearch = 20
    # index = faiss.IndexIVFFlat(quantizer, dim, 4096)
    # index.cp.min_points_per_centroid = 5   # quiet warning
    # index.quantizer_trains_alone = 2
    # index.nprobe = 16

    # index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, 16)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 20
    # index.hnsw.search_bounded_queue = True
    return index


def update_batch(index, data):
    index.train(data)
    index.add(data)
    return index


def get_nn_batch(index, queries, n_neighbors=16):
    return index.search_and_reconstruct(queries, n_neighbors)


def convert_array(x):
    return x
