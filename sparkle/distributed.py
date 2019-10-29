import numpy as np

from pyspark import RDD
from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import DistributedMatrix

__all__ = ['ColBlockMatrix', 'makeColBlockMatrix']


def _max_l2_norm_col(block, p):
    """
    Find the column with the largest L2 norm in this `block` 
    and return it as a (norm, column) tuple.

    :param block: A tuple of the form (index, np.array) as
                  in the RDD elements of a ColBlockMatrix.
    :param p: An optional sparsity pattern which determines which
              rows of this block to consider in computing the column 
              L2 norms. Format must be valid for numpy advanced array 
              indexing. If not provided, all rows used.
    """
    matrix = np.array(block[1])  # TODO: why is this not already an np.array?
    if p is not None:
        matrix = matrix[p,:]
    norms = np.linalg.norm(matrix, axis=0)
    max_index = np.argmax(norms)
    return (norms[max_index], matrix[:,max_index])


def _compare_l2_norms(col1, col2):
    """
    Compares two tuples as returned by _max_l2_norm_col
    returning the one with the larger L2 norm.
    """
    return max(col1, col2, key = lambda x: x[0])


def _subtract_outer_from_block(block, colsPerBlock, left, right):
    """
    Subtract the appropriate block of the outer product of
    `left` and `right` from this `block`

    :param block: A tuple of the form (index, np.array) as
                  in the RDD elements of a ColBlockMatrix.
    :param colsPerBlock: The number of columns that we expect each
                         block to have (may not be true of last block).
    :param left: A column vector with dimension equal to the
                 number of rows of this block.
    :param right: A column vector with dimension greater than or equal
                  to the number of columns of this block.
    """
    matrix = np.array(block[1])
    first_col = block[0] * colsPerBlock
    stop_col = first_col + matrix.shape[1]
    sub_right = right[first_col:stop_col]
    new_matrix = matrix - np.outer(left, sub_right)
    return (block[0], new_matrix)


def _broadcast_np_array(ary):
    """
    Take a numpy.array and broadcast it to all workers. If the array's size in
    memory is greater than or equal to 2 GB, first split the array into
    manageable chunks and return a list of broadcast variables.

    :param ary: The numpy.array to broadcast.
    """
    gbs = ary.nbytes / 2**30
    sc = SparkContext.getOrCreate()
    if gbs >= 2: ## Spark can only serialize objects of size less than 2048 MB
        nBlocks = np.floor(gbs / 2) + 1 ## add 1 block to ensure below limit
        colsPerBlock = int(np.floor(ary.shape[1] / nBlocks))
        splits = range(colsPerBlock, ary.shape[1], colsPerBlock)
        blocks = np.split(ary, splits, axis=1)
        blocks = list(zip(range(len(blocks)), blocks))
        return [sc.broadcast(el) for el in blocks]
    else:
        return sc.broadcast(ary)


def _reassemble_broadcasted_np_array(broadcast):
    """
    Extract (and put back together) an np.array that was broadcasted (and split
    up) via _broadcast_np_array.

    :param broadcast: A single pyspark.Broadcast or list of them containing the
                      numpy.array.
    """
    if isinstance(broadcast, list):
        blocks = [b.value for b in broadcast]
        blocks = sorted(blocks, key = lambda x: x[0])
        blocks = [x[1] for x in blocks]
        return np.hstack(blocks)
    else:
        return broadcast.value


def _unpersist_broadcasted_np_array(broadcast):
    """
    Unpersist a single pyspark.Broadcast variable or a list of them.

    :param broadcast: A single pyspark.Broadcast or list of them.
    """
    if isinstance(broadcast, list):
        [b.unpersist() for b in broadcast]
    else:
        broadcast.unpersist()
    return None


def makeColBlockMatrix(matrix, colsPerBlock, cache=True):
    """
    Take an in-memory matrix and make a distributed ColBlockMatrix.
    
    :param matrix: The matrix to distribute.
    :param colsPerBlock: The number of columns that make up each column block.
    """
    sc = SparkContext.getOrCreate()
    matrix = np.array(matrix)
    splits = range(colsPerBlock, matrix.shape[1], colsPerBlock)
    blocks = np.split(matrix, splits, axis=1)
    blocks = sc.parallelize(zip(range(len(blocks)), blocks), len(blocks))
    if cache:
        return ColBlockMatrix(blocks, colsPerBlock).cache()
    else:
        return ColBlockMatrix(blocks, colsPerBlock)


class ColBlockMatrix(DistributedMatrix):
    """
    Represents a distributed block matrix where each block contains
    all of the rows and a small number of the columns. Heavily
    influenced by pyspark.mllib.linalg.distributed.BlockMatrix.
    However, note that there is much less error checking and
    validation here.

    :param colBlocks: An RDD of column blocks (blockColIndex, sub-matrix)
                      that form this distributed matrix. The sub-matrices
                      are numpy.ndarrays and should all have the same number 
                      of rows colsPerBlock columns.
    :param colsPerBlock: Number of columns that make up each block.
                         The blocks forming the final columns are not
                         required to have the given number of columns.
    """
    def __init__(self, colBlocks, colsPerBlock):
        if not isinstance(colBlocks, RDD):
            raise TypeError("blocks should be an RDD of sub-matrix column "
                            "blocks as (int, matrix) tuples, got %s" %
                            type(colBlocks))
        self.__colBlocks = colBlocks
        self.__colsPerBlock = colsPerBlock

    @property
    def colBlocks(self):
        """
        The RDD of sub-matrix column blocks.
        """
        return self.__colBlocks

    @property
    def colsPerBlock(self):
        """
        Number of columns that make up each column block.
        """
        return self.__colsPerBlock

    def leftMultiply(self, matrix, broadcast=True):
        """
        Multiplies each block of this ColumnBlockMatrix by a numpy.array
        on the left. The number of columns of `matrix` must be the same as
        the number of rows of this ColumnBlockMatrix.

        :param matrix: The numpy.array
        :param broadcast: Whether or not to broadcast the numpy.array.
        """
        colBlocks = self.colBlocks
        if broadcast:
            b = _broadcast_np_array(matrix)
            newColBlocks = colBlocks.map(lambda x: (x[0], np.matmul(
                _reassemble_broadcasted_np_array(b), x[1]))
            )
            _unpersist_broadcasted_np_array(b)
        else:
            newColBlocks = colBlocks.map(lambda x: (x[0], np.matmul(matrix, x[1])))
        return ColBlockMatrix(newColBlocks, self.colsPerBlock)

    def toLocalMatrix(self):
        """
        Collect the distributed matrix on the driver as a numpy.array.
        """
        colBlocks = sorted(self.colBlocks.collect(), key = lambda x: x[0])
        colBlocks = [x[1] for x in colBlocks]
        return np.hstack(colBlocks)

    def maxL2NormCol(self, p=None):
        """
        Find and return the column of this ColBlockMatrix that has 
        the largest L2 norm, optionally only considering the rows 
        specified by the sparsity pattern `p`.

        :param p: An optional sparsity pattern which determines which
                  rows of this ColBlockMatrix to consider in computing
                  the column L2 norms. Format must be valid for numpy
                  advanced array indexing. If not provided, all rows used.
        """
        if p is not None:
            assert np.size(p) > 0,\
                "In maxL2NormCol: pattern must be None or have size > 0"
        colBlocks = self.colBlocks
        return colBlocks.map(lambda x: _max_l2_norm_col(x, p))\
                        .reduce(_compare_l2_norms)

    def subtractOuter(self, left, right, mutate_self = True):
        """
        Subtract the outer product of `left` and `right` from this ColBlockMatrix.

        :param left: A list or numpy.array with dimension equal to the
                     number of rows of this ColBlockMatrix.
        :param right: A list or numpy.array with dimension equal to the
                      number of columns of this ColBlockMatrix.
        """
        colBlocks = self.colBlocks
        colsPerBlock = self.colsPerBlock
        newColBlocks = colBlocks.map(
            lambda x: _subtract_outer_from_block(
                x, colsPerBlock, left, right
            )
        )
        if mutate_self:
            self.__colBlocks = newColBlocks
        else:
            return ColBlockMatrix(newColBlocks, colsPerBlock)

    def cache(self):
        self.__colBlocks.cache()
        return self

    def unpersist(self):
        self.__colBlocks.unpersist()
