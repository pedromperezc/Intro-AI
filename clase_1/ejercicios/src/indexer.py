import numpy as np


class Indexer(object):
    def __init__(self, users_id):
        self.users_id = users_id
        indexes = np.unique(self.users_id, return_index=True)
        id2idx = np.ones(indexes[0].max() + 1, dtype=np.int64) * -1
        id2idx[indexes[0]] = indexes[1]
        self.id2idx = id2idx

    def get_users_id(self, ids):
        idx2id = self.users_id[ids]
        return idx2id


    def get_users_idx(self, ids):
        id2idx = self.id2idx[ids]
        return id2idx
