"""
Unit tests for sparkle.distributed
"""
import numpy as np
import unittest

from sparkle.distributed import *
from sparkle.tests.test import SparkTestCase

class ColBlockMatrixTests(SparkTestCase):

    def test_leftMultiply(self):
        local = [[1,2,3,4,5,6,7,8,9],
                 [1,2,3,4,5,6,7,8,9],
                 [1,2,3,4,5,6,7,8,9]]
        colBlockMat = makeColBlockMatrix(local, 3)
        self.assertTrue(
            np.all(colBlockMat.toLocalMatrix() == local)
        )
        matrix = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        newColBlockMat = colBlockMat.leftMultiply(matrix)
        newLocal = [[  6,  12,  18,  24,  30,  36,  42,  48,  54],
                    [ 15,  30,  45,  60,  75,  90, 105, 120, 135],
                    [ 24,  48,  72,  96, 120, 144, 168, 192, 216],
                    [ 33,  66,  99, 132, 165, 198, 231, 264, 297]]
        self.assertTrue(
            np.all(newColBlockMat.toLocalMatrix() == newLocal)
        )

    def test_maxL2NormCol(self):
        local = [[1,2,3,4,5,6,7,8,10,9,10],
                 [1,2,3,4,5,6,7,8, 9,9,10],
                 [1,2,3,4,5,6,7,8,11,9,10]]
        colBlockMat = makeColBlockMatrix(local, 3)
        col1 = colBlockMat.maxL2NormCol()
        self.assertTrue(col1[0] == np.sqrt(302))
        self.assertTrue(np.all(col1[1] == [10,9,11]))
        col2 = colBlockMat.maxL2NormCol([0,1])
        self.assertTrue(col2[0] == np.sqrt(200))
        self.assertTrue(np.all(col2[1] == [10,10]))

    def test_subtractOuter(self):
        local = [[1,2,3,4,5,6,7,8,9],
                 [1,2,3,4,5,6,7,8,9],
                 [1,2,3,4,5,6,7,8,9]]
        colBlockMat = makeColBlockMatrix(local, 3)
        left = [1,1,1]
        right = [1,2,3,4,5,6,7,8,9]
        colBlockMat.subtractOuter(left, right)
        self.assertTrue(
            np.all(colBlockMat.toLocalMatrix() == np.zeros((3,9)))
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
