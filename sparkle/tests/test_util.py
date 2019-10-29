"""
Shared testing utilities and unit tests of sparkle.util methods.
"""
import unittest
import numpy as np

from pyspark.context import SparkContext
from pyspark.sql import SparkSession

from pyspark.mllib.linalg import Matrices as OldMatrices
from pyspark.mllib.linalg.distributed import BlockMatrix

import sparkle


class SparkTestCase(unittest.TestCase):

    def setUp(self):
        class_name = self.__class__.__name__
        self.sc = SparkContext('local[4]', class_name)
        self.spark = SparkSession(self.sc)

    def tearDown(self):
        self.spark.stop()


class UtilTests(SparkTestCase):

    def test_computeRowSums(self):
        dm1 = OldMatrices.dense(3, 2, [1, 2, 3, 4, 5, 6])
        dm2 = OldMatrices.dense(3, 2, [7, 8, 9, 10, 11, 12])
        dm3 = OldMatrices.dense(3, 2, [13, 14, 15, 16, 17, 18])
        dm4 = OldMatrices.dense(3, 2, [19, 20, 21, 22, 23, 24])
        blocks = self.sc.parallelize([
            ((0, 0), dm1), ((0, 1), dm2), ((1, 0), dm3), ((1, 1), dm4)
        ])
        mat = BlockMatrix(blocks, 3, 2)
        rowSums = sparkle.util._computeRowSums(mat)
        self.assertTrue(np.all(rowSums == [48, 66, 84, 102]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
