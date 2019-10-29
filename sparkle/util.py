import numpy as np
import pickle

from pyspark import SparkContext
from pyspark.mllib.linalg import Matrices as OldMatrices, Vectors as OldVectors
from pyspark.mllib.linalg.distributed import BlockMatrix, IndexedRowMatrix


__all__ = ["_checkState", "_computeRowSums", "_standardize", "_normalize",
           "_getColumns", "_colVectorToBlockMatrix", "_thresholdColVector",
           "_thresholding", "_save", "_load"]


def _checkState(squareLoss, nIter, maxIter, tol, verbose=False):
    """
    Compares the ( squareLoss[ i ] - squareLoss[ i - 1 ] ) / squareLoss[ i - 1 ]
    to epsilon and outputs 0 if it's greater than epsilon and 1 otherwise.
    It will output -1 if the sparsity parameter is too high i.e.
    squareLoss[ iter - 1 ] == 0 and 2 if it has reached maximum iterations.
    """
    state = 0

    if nIter > 2:
        if squareLoss[nIter - 2] == 0:
            if verbose:
                print("Sparsity parameter may be too high!")
            state = -1
        elif np.abs((squareLoss[nIter - 2] - squareLoss[nIter - 3]) /
                    squareLoss[nIter - 3]) <= tol:
            if verbose:
                print("Convergence criteria met!")
            state = 1
        elif nIter == maxIter:
            if verbose:
                print("Reached maximum number of iterations!")
            state = 2

    return state


def _computeRowSums(blockMat, power=1):
    """
    Sum the rows of blockMat.

    :param blockMat: A pyspark.mllib.linalg.distributed.BlockMatrix.
    :param    power: The power by which to raise each entry of blockMat.
    """
    sc = SparkContext.getOrCreate()
    numCols = blockMat.numCols()
    colsPerBlock = blockMat.colsPerBlock

    def seqOp(zeroVal, block):
        _, j = block[0]
        colSums = np.apply_along_axis(lambda x: (x**power).sum(), 0,
                                      block[1].toArray())
        col0 = j * colsPerBlock
        for i in range(col0, col0+len(colSums)):
            zeroVal[i] += colSums[i - col0]
        return zeroVal

    rowSums = blockMat.blocks.aggregate(
        np.zeros(numCols), seqOp, lambda x, y: x + y
    )
    return rowSums


def _standardize(blockMat, center=0, scale=1):
    """
    Standardize blockMat columns by subtracting center and dividing
    by scale.

    :param blockMat: A pyspark.mllib.linalg.distributed.BlockMatrix.
    :param   center: Either a scalar value which will be subtracted from
                     all entries of blockMat, or a 1D array of length
                     blockMat.numCols(), in which case center[j] will be
                     subtracted from the entries in blockMat column j.
    :param    scale: Either a scalar value which will divide all entries in
                     blockMat, or a 1D array of length blockMat.numCols(),
                     in which case scale[j] will divide the entries in
                     blockMat column j.
    """
    sc = SparkContext.getOrCreate()
    colsPerBlock = sc.broadcast(blockMat.colsPerBlock)
    cb = sc.broadcast(center)
    sb = sc.broadcast(scale)
    def g(block):
        i, j = block[0]
        mat = block[1].toArray()
        n, m = mat.shape
        col0 = colsPerBlock.value * j
        blockCenter = cb.value if np.isscalar(cb.value) else cb.value[col0:(col0+m)]
        blockScale = sb.value if np.isscalar(sb.value) else sb.value[col0:(col0+m)]
        newmat = (mat - blockCenter) / blockScale
        newmat = OldMatrices.dense(n, m, newmat.ravel(order='F'))
        return ((i, j), newmat)
    newBlocks = blockMat.blocks.map(g)
    colsPerBlock.unpersist()
    cb.unpersist()
    sb.unpersist()
    return BlockMatrix(newBlocks, rowsPerBlock=blockMat.rowsPerBlock,
                       colsPerBlock=blockMat.colsPerBlock)


def _normalize(blockMat, norm):
    """
    Normalize blockMat by dividing all entries by norm.
    """
    def g(block):
        newmat = OldMatrices.dense(block[1].numRows, block[1].numCols,
                                   block[1].toArray() / norm)
        return (block[0], newmat)
    newBlocks = blockMat.blocks.map(g)
    return BlockMatrix(newBlocks, rowsPerBlock=blockMat.rowsPerBlock,
                       colsPerBlock=blockMat.colsPerBlock)


def _thresholdColVector(blockMat, rho):
    """
    Apply soft-thresholding to a column vector BlockMatrix.
    """
    def g(block):
        blockArr = block[1].toArray().ravel()
        newmat = OldMatrices.dense(
            block[1].numRows, block[1].numCols,
            np.sign(blockArr)*np.maximum(0, np.abs(blockArr) - rho)
        )
        return (block[0], newmat)
    newBlocks = blockMat.blocks.map(g)
    return BlockMatrix(newBlocks, rowsPerBlock=blockMat.rowsPerBlock,
                       colsPerBlock=blockMat.colsPerBlock)


def _thresholding(z, rho):
    """
    Apply soft thresholding to the vector `z` using the
    sparsity parameter `rho`.

    :param z: A vector to be thresholded.
    :param rho: A sparsity parameter that determines
                how strong the threshold will be.
    """
    return np.sign(z) * np.maximum(0, np.abs(z) - rho)


def _getColumns(blockMat, j, norm=1):
    """
    Returns column(s) j of the input BlockMatrix as a BlockMatrix with
    the same number of rowsPerBlock.
    """
    sc = SparkContext.getOrCreate()
    if np.isscalar(j):
        colsPerBlock = blockMat.colsPerBlock
        jBlockCol = j // colsPerBlock
        jInBlock = j % colsPerBlock
        jBlocks = blockMat.blocks.filter(lambda x: x[0][1] == jBlockCol)
        def g(block):
            colJ = block[1].toArray()[:,jInBlock] / norm
            return ((block[0][0], 0), OldMatrices.dense(len(colJ), 1, colJ))
        colJBlocks = jBlocks.map(g)
        return BlockMatrix(colJBlocks, rowsPerBlock=blockMat.rowsPerBlock,
                           colsPerBlock=1, numCols=1)
    else:
        j_b = sc.broadcast(j)
        blockMat_red = blockMat.toIndexedRowMatrix()
        rows_red = blockMat_red.rows.map(
            lambda row: (row.index,
                         OldVectors.dense(row.vector.toArray()[j_b.value] / norm))
        )
        j_b.unpersist()
        return IndexedRowMatrix(rows_red).toBlockMatrix(
            rowsPerBlock=blockMat.rowsPerBlock,
            colsPerBlock=min(len(j), blockMat.colsPerBlock)
        )


def _colVectorToBlockMatrix(vec, rowsPerBlock, numSlices=None):
    sc = SparkContext.getOrCreate()
    remainder = len(vec) % rowsPerBlock
    if rowsPerBlock >= len(vec):
        splits = [vec]
    elif remainder == 0:
        splits = np.split(vec, len(vec) // rowsPerBlock)
    else:
        head = vec[:-remainder]
        splits = np.split(head, len(head) // rowsPerBlock)
        splits.append(vec[-remainder:])
    blocks = sc.parallelize(
        [((i, 0), OldMatrices.dense(len(split), 1, split))
         for i, split in zip(range(len(splits)), splits)],
        numSlices = numSlices
    )
    return BlockMatrix(blocks, rowsPerBlock, 1, len(vec), 1)


def _genLowRankData(n, p1, p2 = None, rowsPerBlock=100, colsPerBlockX=100,
                    colsPerBlockY=100, numSlices=None, rho1=0.1, rho2=0.1,
                    eps=0.1, seed = 1):
    sc = SparkContext.getOrCreate()
    np.random.seed(seed)
    q1 = round(rho1*p1/2)
    w1 = _colVectorToBlockMatrix(
        np.repeat([1, -1, 0], [q1, q1, p1 - 2*q1]) +
        np.random.normal(scale=eps, size=p1),
        rowsPerBlock = colsPerBlockX,
        numSlices = numSlices
    )
    u = _colVectorToBlockMatrix(
        np.random.normal(size=n), rowsPerBlock, numSlices
    )
    X = u.multiply(w1.transpose())
    if p2 is not None:
        q2 = round(rho2*p2/2)
        w2 = _colVectorToBlockMatrix(
            np.repeat([0, 1, -1], [q2, q2, p2 - 2*q2]) +
                np.random.normal(scale=eps, size=p2),
            rowsPerBlock = colsPerBlockY,
            numSlices = numSlices
        )
        Y = u.multiply(w2.transpose())
        return (X, Y)
    return X


def _save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
