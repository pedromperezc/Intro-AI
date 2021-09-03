from unittest import TestCase
import numpy as np
from clase_1.ejercicios.src.indexer_v2 import Indexer


class IndexerTestCase(TestCase):

    def test_indexer(self):
        a = np.array([15, 12, 14, 10, 1, 2, 1], dtype=np.int64)
        indexer = Indexer(a)

        idxs = indexer.get_users_idx([15, 14, 1])
        np.testing.assert_equal(np.array([0, 2, 4]), idxs)

        ids = indexer.get_users_id([0, 2, 4])
        np.testing.assert_equal(np.array([15, 14, 1]), ids)
