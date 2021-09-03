import numpy as np
from clase_1.ejercicios.src.indexer import Indexer
a = np.array([15, 12, 14, 10, 1, 2, 1], dtype=np.int64)
#
indexer = Indexer(a)
#
id2idx = indexer.get_users_idx([6])
print(id2idx)
#
idx2id = indexer.get_users_id([0])
print(idx2id)
#
#


# from clase_1.ejercicios.src.indexer import Indexer
# a = np.array([15, 12, 14, 10, 1, 2, 1], dtype=np.int64)
# indexer = Indexer(a)
#
# id2idx = indexer.get_idxs([15])
# print (id2idx)
#
# idx2id = indexer.get_ids([5])
# print(idx2id)