import numpy as np
import nmslib


THREADS = 16


def new(dim):
    index = nmslib.init(method='hnsw', space='l2',
                        data_type=nmslib.DataType.DENSE_VECTOR)
    return index


def update_batch(index, data):
    index.addDataPointBatch(data)
    M = 16
    efC = 40
    index_time_params = {'M': M,
                         'indexThreadQty': THREADS,
                         'efConstruction': efC}
    index.createIndex(index_time_params)

    efS = 20
    query_time_params = {'efSearch': efS}
    index.setQueryTimeParams(query_time_params)
    return index


def get_nn_batch(index, queries, n_neighbors=16):
    return index.knnQueryBatch(queries, n_neighbors, num_threads=THREADS)
