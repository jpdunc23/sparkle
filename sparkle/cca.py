import numpy as np
import gc

from multiprocessing.pool import ThreadPool

from scipy.linalg import pinv
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

from .util import _checkState, _thresholding, _save, _load
from .distributed import *

from pyspark import keyword_only, SparkContext
from pyspark.ml import Estimator, Model
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param import *
from pyspark.ml.param.shared import HasMaxIter, HasStandardization, HasTol
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel


class CCA(Estimator, HasMaxIter, HasStandardization, HasTol):
    verbose = Param(Params._dummy(), "verbose", "if True, print progress to STDOUT")
    k = Param(Params._dummy(), "k", "the number of canonical vectors",
              typeConverter=TypeConverters.toInt)

    rhos = Param(Params._dummy(), "rhos", "the sparsity parameters in [0,1]",
                 typeConverter=TypeConverters.toList)

    colsPerBlock = Param(Params._dummy(), "colsPerBlock",
                         "Number of columns in the blocks of the two input "
                         "matrices when they are distributed",
                         typeConverter=TypeConverters.toList)

    broadcast = Param(Params._dummy(), "broadcast",
                      "whether or not to broadcast during distributed matrix "
                      "multiplication",
                      typeConverter=TypeConverters.toBoolean)

    caching = Param(Params._dummy(), "caching",
                      "whether or not to cache underlying RDDs of ColBlockMatrices",
                      typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self, k=1, maxIter=1000, rhos=[0.1], colsPerBlock=[1024],
                 broadcast=True, caching=True, standardization=True, tol=1e-4,
                 verbose=False):
        super(CCA, self).__init__()
        self._setDefault(
            k=1, maxIter=1000, rhos=[0.1], colsPerBlock=[1024], broadcast=True,
            caching=True, standardization=True, tol=1e-4, verbose=False
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setVerbose(self, verbose):
        """
        Sets the value of :py:attr:`verbose`.
        """
        return self._set(verbose=verbose)

    def getVerbose(self):
        """
        Gets the value of verbose or its default value.
        """
        return self.getOrDefault(self.verbose)

    @keyword_only
    def setParams(self, k=1, maxIter=1000, rhos=[0.1],
                  colsPerBlock=[1024], broadcast=True, caching=True,
                  standardization=True, tol=1e-4, verbose=False):
        self.setK(k)
        self.setMaxIter(maxIter)
        self.setRhos(rhos)
        self.setColsPerBlock(colsPerBlock)
        self.setBroadcast(broadcast)
        self.setCaching(caching)
        self.setStandardization(standardization)
        self.setTol(tol)
        self.setVerbose(verbose)

    def setRhos(self, value):
        """
        Sets the value of :py:attr:`rhos`.
        """
        assert min(value) >= 0 and max(value) <= 1, \
            "all rhos must be in [0, 1]"
        return self._set(rhos=value)

    def getRhos(self):
        """
        Gets the value of :py:attr:`rhos` or its default value.
        """
        return self.getOrDefault(self.rhos)

    def setColsPerBlock(self, value):
        """
        Sets the value of :py:attr:`colsPerBlock`.
        """
        # assert len(value) == 2, "colsPerBlock must be a two-tuple"
        assert min(value) > 0, "colsPerBlock must be positive"
        return self._set(colsPerBlock=value)

    def getColsPerBlock(self):
        """
        Gets the value of :py:attr:`colsPerBlock`. or its default value.
        """
        return self.getOrDefault(self.colsPerBlock)

    def setBroadcast(self, value):
        """
        Sets the value of :py:attr:`broadcast`.
        """
        return self._set(broadcast=value)

    def getBroadcast(self):
        """
        Gets the value of :py:attr:`broadcast` or its default value.
        """
        return self.getOrDefault(self.broadcast)

    def setCaching(self, value):
        """
        Sets the value of :py:attr:`caching`.
        """
        return self._set(caching=value)

    def getCaching(self):
        """
        Gets the value of :py:attr:`caching` or its default value.
        """
        return self.getOrDefault(self.caching)

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        assert value > 0, "k must be greater than 0"
        return self._set(k=value)

    def getK(self):
        """
        Gets the value of k or its default value.
        """
        return self.getOrDefault(self.k)

    def _fit(self, datasets):
        """
        Computes multiple CCs using a deflation scheme.
        :param datasets: A list of numpy.arrays.
        """
        standardization = self.getStandardization()
        n_ccs = self.getK()
        maxIter = self.getMaxIter()
        rhos = self.getRhos()
        tol = self.getTol()
        colsPerBlock = self.getColsPerBlock()
        broadcast = self.getBroadcast()
        caching = self.getCaching()
        verbose = self.getVerbose()

        if standardization:
            datasets = [X - np.mean(X, axis=0) for X in datasets]

        d = len(datasets)

        if len(rhos) == 1:
            rhos = rhos * d
        else:
            assert len(rhos) == d,\
                "Please provide one regularization parameter per dataset."
        if len(colsPerBlock) == 1:
            colsPerBlock = colsPerBlock * d
        else:
            assert len(colsPerBlock) == d,\
                "Please provide one colsPerBlock value per dataset."

        # cross covariance matrices
        CC = [
            makeColBlockMatrix(datasets[i], colsPerBlock[i], caching)
            .leftMultiply(datasets[j].T, broadcast)
            for i in range(d)
            for j in (x for y in (range(i), range(i+1, d))
                      for x in y)  # all indices other than i
        ]

        # matrices of sparse loadings
        ZZ = [lil_matrix((n_ccs, X.shape[1])) for X in datasets]
        Sigma = []

        for k in range(n_ccs):

            z_init = []  # initial estimates of sparse canonical loadings
            pattern = []  # sparsity patterns

            for i in range(d):

                # 1. Get the cov matrices involving dataset i. For any
                #    covariance matrix involving a dataset that was
                #    already processed, reduce the rows using the sparsity
                #    pattern that was computed.
                # 2. Call singleL1 one time for all of those cov matrices,
                #    returning a z for each one.
                # 3. Add the z's up from step 2 and threshold and standardize
                #    the resulting vector. This is z initial for dataset i.

                z = np.zeros(datasets[i].shape[1])
                max_rho = 0

                # indices of datasets other than i
                not_i = [x for y in (range(i), range(i+1, d)) for x in y]

                # indices of covariance matrices we'll need for this iteration
                for j in range(i*(d - 1), (i + 1)*(d - 1)):

                    # get the index of the left dataset in CC[j]
                    left_idx = not_i[j % (d - 1)]

                    if left_idx < i:
                        p = pattern[left_idx]
                        if np.size(p) == 0:
                            print("Canonical loading vector " + str(k) +
                                  " of dataset " + str(left_idx) +
                                  " is all zeros. Moving on to next dataset.")
                            # move on to next not_i dataset
                            import pdb; pdb.set_trace()
                            continue
                    else:
                        p = None  # keep all columns of left dataset

                    if k == 0:
                        lZ = None
                        rZ = None
                    else:
                        lZ = (ZZ[left_idx][0:k, p] if p is not None
                              else ZZ[left_idx][0:k, :]).tocsr()
                        sigma_ij = np.diag(
                            [Sigma[cc][i, left_idx] for cc in range(k)]
                        )
                        rZ = csr_matrix.dot(
                            sigma_ij,
                            ZZ[i][0:k, :].tocsr()
                        ) ## TODO: check

                    z_term, this_rho = CCA._singleL1(
                        CC[j], datasets[left_idx][:, p] if p is not None
                        else datasets[left_idx], datasets[i],
                        rhos[i], tol, maxIter, lZ, rZ, p, verbose
                    )
                    z += z_term
                    if this_rho > max_rho:
                        max_rho = this_rho

                z = _thresholding(z, max_rho)
                norm_z = np.linalg.norm(z)
                if norm_z > 0:
                    z = z / norm_z

                pattern.append(z.nonzero()[0])
                z_init.append(z)

            zz, sigma = CCA._coeffEstSingleComp(
                datasets, z_init, pattern, k, maxIter, tol, ZZ, Sigma, verbose
            )
            for i in range(d):
                ZZ[i][k, pattern[i]] = zz[i]
            Sigma.append(sigma)
            Sigma[k] += np.tril(Sigma[k]).T

            for i in range(d):
                for j in (x for y in (range(i), range(i+1, d)) for x in y):
                    # C_ij - z_i.T C_ij z_j * z_i z_j.T
                    # subtractOuter mutates its object, so no need to reassign to CC
                    CC[i*(d - 1) + (j if j < i else j - 1)].subtractOuter(
                        Sigma[k][i, j] * ZZ[j][k, :].todense().getA1(),
                        ZZ[i][k, :].todense().getA1()
                    )
        ## done, unpersist the ColBlockMatrices
        if caching:
            for cbm in CC:
                cbm.unpersist()
        del CC
        gc.collect()
        return CCAModel(k=n_ccs, ZZ=[Z.tocsr() for Z in ZZ], Sigma=Sigma,
                        standardization=standardization, rhos=rhos)

    def fit(self, datasets, params=None):
        """
        Overrides pyspark.ml.Estimator's fit method.
        """
        # TODO: assert all datasets are np.array
        assert len(datasets) > 1, "need at least 2 datasets"
        assert np.all(
            np.array([X.shape[0] for X in datasets]) == datasets[0].shape[0]),\
            "all datasets must have same number of observations"
        if params is None:
            params = dict()
        if isinstance(params, (list, tuple)):
            models = [None] * len(params)
            for index, model in self.fitMultiple(datasets, params):
                models[index] = model
            return models
        elif isinstance(params, dict):
            if params:
                return self.copy(params)._fit(datasets)
            else:
                return self._fit(datasets)
        else:
            raise ValueError("Params must be either a param map or a list/"
                             "tuple of param maps, but got %s." % type(params))

    @staticmethod
    def _coeffEstSingleComp(datasets, z_init, pattern, k, maxIter, tol,
                            ZZ=None, Sigma=None, verbose=False):
        """
        """
        d = len(datasets)
        z_final = []

        for i in range(d):

            # get reduced versions of the ith dataset and initial loadings
            # vector using the ith sparsity pattern
            p_i = pattern[i]
            if p_i.size == 0:
                continue
            X_i = datasets[i][:, p_i]
            z_i = z_init[i][p_i]

            nIter = 1
            f = []

            while not _checkState(f, nIter, maxIter, tol, verbose) in (-1, 1, 2):

                tmp = np.zeros(p_i.size)

                # all indices other than i
                for j in (x for y in (range(i), range(i+1, d)) for x in y):

                    p_j = pattern[j]
                    if np.size(p_j) == 0:
                        continue
                    X_j = datasets[j][:, p_j]
                    if j < i:
                        z_j = z_final[j]  # already shrunken
                        tmp += np.matmul(  # X_i.T @ X_j @ z_j
                            X_i.T, np.matmul(
                                X_j, z_j
                            )
                        )
                        if k > 0:
                            tmp -= Sigma[k-1][i, j] * csr_matrix.dot(
                                ZZ[j][k-1, p_j].tocsr(), z_j
                            ) * ZZ[i][k-1, p_i].tocsr().transpose().\
                                todense().getA1()
                    else:
                        tmp += np.matmul(  # X_i.T @ X_j @ X_j.T @ X_i @ z_i
                            X_i.T, np.matmul(
                                X_j, np.matmul(
                                    X_j.T, np.matmul(X_i, z_i)
                                )
                            )
                        )
                        if k > 0:
                            sig = Sigma[k-1][i, j]
                            zz_i = ZZ[i][k-1, p_i].tocsr()  # row vector
                            zz_j = ZZ[j][k-1, p_j].tocsr()  # row vector
                            tmp -= sig * np.dot(
                                csr_matrix.dot(
                                    zz_j, X_j.T
                                ),
                                np.matmul(X_i, z_i)
                            ) * zz_i.transpose().todense().getA1() +\
                                sig * csr_matrix.dot(zz_i, z_i) *\
                                np.matmul(
                                    X_i.T, csr_matrix.dot(
                                        X_j, zz_j.transpose()
                                    )
                                ).ravel() - sig**2 * csr_matrix.dot(
                                    zz_j, zz_j.transpose()
                                ).todense().getA1() * csr_matrix.dot(
                                    zz_i, z_i
                                ) * zz_i.transpose().todense().getA1()

                norm_tmp = np.linalg.norm(tmp)
                f.append(-2*norm_tmp)
                if norm_tmp > 0:
                    z_i = tmp / norm_tmp
                else:
                    z_i = tmp
                nIter += 1

            z_final.append(z_i.ravel())

        sigma = np.zeros((d, d))
        for i in range(d):
            p_i = pattern[i]
            X_i = datasets[i][:, p_i]
            z_i = z_final[i]
            if k > 0:
                z_i_prev = ZZ[i][k-1, p_i].todense().getA1()  # row vector
            for j in range(i):
                p_j = pattern[j]
                if p_i.size == 0 or p_j.size == 0:
                    sigma[i, j] = 0
                    continue
                X_j = datasets[j][:, p_j]
                z_j = z_final[j]
                if k == 0:
                    # z_i.T @ X_i.T X_j z_j for j < i
                    sigma[i, j] = np.dot(
                        np.matmul(X_i, z_i),
                        np.matmul(X_j, z_j)
                    )
                else:
                    z_j_prev = ZZ[j][k-1, p_j].todense().getA1()
                    # z_i.T @ (X_i.T @ X_j - Sigma[k-1][i, j] *
                    #     ZZ[i][k-1,p_i] @ ZZ[j][k-1,p_j].T) @ z_j
                    sigma[i, j] = np.dot(
                        np.matmul(X_i, z_i),
                        np.matmul(X_j, z_j)
                    ) - Sigma[k-1][i, j] *\
                        np.dot(z_i, z_i_prev) * np.dot(z_j, z_j_prev)

        return (z_final, sigma)

    @staticmethod
    def _singleL1(covBlock, lX, rX, rho, tol, maxIter,
                  lZ=None, rZ=None, p=None, verbose=False):
        """
        Using an off-diagonal block of the sample covariance matrix,
        `covBlock`, return an initial estimate of the canonical loading
        vector corresponding to the `right` matrix of `covBlock`.

        :param covBlock: A sparkle.distributed.ColBlockMatrix formed by
                       multiplying the transpose of the `lX` matrix by
                       the `rX` matrix.
        :param lX: The left matrix in the sample covariance block.
        :param rX: The right matrix in the sample covariance block.
        :param lZ: A scipy.sparse.csr_matrix which contains the loadings
                   corresponding to the left matrix (optional).
        :param rZ: A scipy.sparse.csr_matrix which contains the loadings
                   corresponding to the right matrix (optional).
        :param rho: Sparsity parameter.
        :param tol: Convergence tolerance.
        :param maxIter: Maxiumum number of iterations.
        :param p: Optional sparsity pattern.
        """
        rho_max, x = covBlock.maxL2NormCol(p)
        x = x / rho_max
        rho = rho * rho_max

        f = []
        nIter = 1

        while not _checkState(f, nIter, maxIter, tol, verbose):

            z = np.matmul(rX.T, np.matmul(lX, x))
            if lZ is not None:
                z = z - rZ.transpose().dot(lZ.dot(x))
            z = _thresholding(z, rho)

            f.append(np.sum(z**2))

            x = np.matmul(lX.T, np.matmul(rX, z))
            if lZ is not None:
                x = x - lZ.transpose().dot(rZ.dot(z))
            norm_x = np.linalg.norm(x)
            if norm_x > 0:
                x = x / norm_x

            nIter += 1

        z = np.matmul(rX.T, np.matmul(lX, x))
        if lZ is not None:
            z = z - rZ.transpose().dot(lZ.dot(x))
        return z, rho

    def suggest_colsPerBlock(datasets, n_cores=4, partitions_per_core=3):
        """
        :param partitions_per_core: Spark recommends 2-4 partitions per CPU.
          See https://spark.apache.org/docs/latest/rdd-programming-guide.html
        """
        nCols = np.array([X.shape[1] for X in datasets])
        nPartitions = n_cores * partitions_per_core
        return (nCols / nPartitions).astype(int)


class CCAModel(Model, HasStandardization):

    k = Param(Params._dummy(), "k", "the number of canonical vectors",
              typeConverter=TypeConverters.toInt)

    rhos = Param(Params._dummy(), "rhos", "the sparsity parameters in [0,1]",
                 typeConverter=TypeConverters.toList)

    @keyword_only
    def __init__(self, k=1, ZZ=None, Sigma=None, standardization=True,
                 rhos=[0.1]):
        super(CCAModel, self).__init__()
        self._setDefault(k=1, standardization=True, rhos=[0.1])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, k=1, ZZ=None, Sigma=None, standardization=True,
                  rhos=[0.1]):
        self.setK(k)
        self.setStandardization(standardization)
        self.setRhos(rhos)
        self.__ZZ = ZZ
        self.__Sigma = Sigma

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        assert value > 0, "k must be greater than 0"
        return self._set(k=value)

    def getK(self):
        """
        Gets the value of k or its default value.
        """
        return self.getOrDefault(self.k)

    def setRhos(self, value):
        """
        Sets the value of :py:attr:`rhos`.
        """
        assert min(value) >= 0 and max(value) <= 1,\
            "all rhos must be in [0, 1]"
        return self._set(rhos=value)

    def getRhos(self):
        """
        Gets the value of :py:attr:`rhos` or its default value.
        """
        return self.getOrDefault(self.rhos)

    @property
    def ZZ(self):
        """
        A np.array of scipy.sparse.csr_matrix with first :py:attr:`k` canonical
        weight vectors.
        """
        return self.__ZZ

    @property
    def Sigma(self):
        """
        A np.array of scipy.sparse.csr_matrix with first :py:attr:`k` canonical
        weights on the diagonal.
        """
        return self.__Sigma

    def canonicalCorrelations(self, datasets = None):
        """
        Take :py:attr:`k` datasets and return the :py:attr:`k` canonical correlations.
        :param datasets: A list of numpy.arrays.
        """
        if datasets is not None:
            XZ = [csr_matrix.dot(X, Z.transpose()) for X, Z in
                  zip(datasets, self.ZZ)]
            k = self.getK()
            corrs = np.zeros((k, len(XZ), len(XZ)))
            for i in range(len(XZ)):
                corrs[:, i, i] = np.ones(k)
                for j in range(i + 1, len(XZ)):
                    for cc in range(k):
                        corrs[cc, j, i] = np.corrcoef(
                            XZ[i][:, cc], XZ[j][:, cc], rowvar=False)[0, 1]
            for cc in range(k):
                # fill in the upper triangles with the transpose of the lower
                corrs[cc, :, :] = corrs[cc, :, :] + np.tril(
                    corrs[cc, :, :], -1).T
        else:
            corrs = None  # TODO: save canonical correlations during CCA.fit
        return np.nan_to_num(corrs)

    def _transform(self, datasets, outcome_index=None):
        # TODO: use 'outcome_index' to allow user to use d - 1 datasets to predict the
        # remaining one
        ZZ = self.ZZ
        d = len(ZZ)
        assert len(datasets) == d,\
            "number of datasets should be len(self.ZZ)"
        standardization = self.getStandardization()
        if standardization:
            datasets = [X - np.mean(X, axis=0) for X in datasets]

        if outcome_index is not None:
            assert outcome_index >= 0 and outcome_index < d,\
                "outcome_index is not a valid index for datasets"
            XZ = [csr_matrix.dot(datasets[j], ZZ[j].transpose()) for j in
                  [x for x in range(d) if x != outcome_index]]
            XZ_sums = np.sum(XZ, axis=0)
            prediction = np.dot(XZ_sums, pinv(ZZ[outcome_index].todense()).transpose())
        else:
            XZ = [csr_matrix.dot(X, Z.transpose()) for X, Z in zip(datasets, ZZ)]
            XZ_sums = []
            for i in range(d):
                for j in range(d):
                    if j != i:
                        if len(XZ_sums) == i:
                            XZ_sums.append(XZ[j])
                        else:
                            XZ_sums[i] += XZ[j]
            prediction = [np.dot(XZ_sums[i], pinv(ZZ[i].todense()).transpose())
                          for i in range(d)]
        return prediction

    def transform(self, datasets, params=None, outcome_index=None):
        """
        Overrides pyspark.ml.Transformer's transform method.
        :param datasets: A list of np.arrays corresponding to self.ZZ
        :param params: an optional param map that overrides embedded params.
        :returns: predictions
        """
        if params is None:
            params = dict()
        if isinstance(params, dict):
            if params:
                return self.copy(params)._transform(datasets, outcome_index)
            else:
                return self._transform(datasets, outcome_index)
        else:
            raise ValueError("Params must be a param map but got %s." % type(params))

    def save(self, path):
        """Save this CCAModel instance to the given path"""
        _save(self, path)

    def load(path):
        """Load a CCAModel instance from the given path"""
        return _load(path)


class CCACanonicalCorrelationEvaluator(Evaluator):

    def _evaluate(self, canCorrs):
        # TODO: extract upper triangles
        return np.mean(canCorrs)

    def evaluate(self, model, datasets, params=None):
        """
        Overrides pyspark.ml.evaluation.Evaluator's evaluate method.
        :param canCorrs: the result of a call to CCAModel.canonicalCorrelations
        :returns: average correlation between the canonical variables
        """
        if params is None:
            params = dict()
        if isinstance(params, dict):
            if params:
                return self.copy(params)._evaluate(
                    model.canonicalCorrelations(datasets)
                )
            else:
                return self._evaluate(
                    model.canonicalCorrelations(datasets)
                )
        else:
            raise ValueError("Params must be a param map but got %s."
                             % type(params))


class CCAPredictionEvaluator(Evaluator):

    outcome_index = Param(Params._dummy(), "outcome_index",
                          "the index of the outcome dataset",
                          typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, outcome_index=0):
        super(Evaluator, self).__init__()
        self._setDefault(outcome_index=0)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, outcome_index=0):
        self.setOutcomeIndex(outcome_index)

    def setOutcomeIndex(self, value):
        """
        Sets the value of :py:attr:`outcome_index`.
        """
        return self._set(outcome_index=value)

    def getOutcomeIndex(self):
        """
        Gets the value of :py:attr:`outcome_index`.
        """
        return self.getOrDefault(self.outcome_index)

    def _evaluate(self, prediction, dataset):
        corrs = np.zeros(prediction.shape[1])
        for i in range(prediction.shape[1]):
            corrs[i] = np.corrcoef(prediction[:, i], dataset[:, i])[0, 1]
        return np.mean(np.nan_to_num(corrs))

    def evaluate(self, model, datasets, params=None):
        """
        Overrides pyspark.ml.evaluation.Evaluator's evaluate method.
        :param canCorrs: the result of a call to CCAModel.canonicalCorrelations
        :returns: average correlation between the canonical variables
        """
        if params is None:
            params = dict()
        if isinstance(params, dict):
            if params:
                return self.copy(params)._evaluate(
                    model.transform(datasets, outcome_index=self.getOutcomeIndex()),
                    datasets[self.getOutcomeIndex()]
                )
            else:
                return self._evaluate(
                    model.transform(datasets, outcome_index=self.getOutcomeIndex()),
                    datasets[self.getOutcomeIndex()]
                )
        else:
            raise ValueError("Params must be a param map but got %s."
                             % type(params))


class CCACrossValidator(CrossValidator):
    """
    Borrows heavily from pyspark.ml.tuning.CrossValidator.
    """
    verbose = Param(Params._dummy(), "verbose", "if True, print progress to STDOUT")

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None,
                 numFolds=3, seed=None, parallelism=1, verbose=False):
        super(CCACrossValidator, self).__init__(
            estimator=estimator,
            estimatorParamMaps=estimatorParamMaps,
            evaluator=evaluator,
            numFolds=numFolds, seed=seed,
            parallelism=parallelism
        )
        self._setDefault(verbose=False)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None,
                  numFolds=3, seed=None, parallelism=1, verbose=False):
        self.setEstimator(estimator)
        self.setEstimatorParamMaps(estimatorParamMaps)
        self.setEvaluator(evaluator)
        self.setNumFolds(numFolds)
        self.setSeed(seed)
        self.setParallelism(parallelism)
        self.setVerbose(verbose)

    def setVerbose(self, verbose):
        """
        Sets the value of :py:attr:`verbose`.
        """
        return self._set(verbose=verbose)

    def getVerbose(self):
        """
        Gets the value of verbose or its default value.
        """
        return self.getOrDefault(self.verbose)

    def _fit(self, datasets):
        """
        Overrides pyspark.ml.tuning.Crossvalidator's _fit method.
        """
        verbose = self.getVerbose()
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        h = 1.0 / nFolds
        randUnif = np.random.uniform(size = datasets[0].shape[0])
        metrics = np.zeros(numModels)

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))

        for i in range(nFolds):
            if verbose:
                print("Fold " + str(i + 1) + " of " + str(nFolds))
            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (randUnif >= validateLB) & (randUnif < validateUB)
            validate = [X[condition,:] for X in datasets]
            train = [X[~condition,:] for X in datasets]

            tasks = self._parallelFitTasks(est, train, eva, validate, epm, verbose)
            for j, metric in pool.imap_unordered(lambda f: f(), tasks):
                metrics[j] += (metric / nFolds)

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(datasets, epm[bestIndex])
        if verbose:
            print("Best model: " + str(i + 1) + " of " + str(nFolds))
            print("k: {0}".format(bestModel.getK()))
            print("rhos: {0}".format(bestModel.getRhos()))
            print("Avg. correlation across " + str(nFolds) + " folds: " +
                  str(metrics[bestIndex]))
        return CCACrossValidatorModel(bestModel)


    def fit(self, datasets, params=None):
        """
        Overrides pyspark.ml.Estimator's fit method, which is inherited from
        pyspark.ml.tuning.CrossValidator.
        """
        if params is None:
            params = dict()
        if isinstance(params, (list, tuple)):
            models = [None] * len(params)
            for index, model in self.fitMultiple(datasets, params):
                models[index] = model
            return models
        elif isinstance(params, dict):
            if params:
                return self.copy(params)._fit(datasets)
            else:
                return self._fit(datasets)
        else:
            raise ValueError("Params must be either a param map or a list/tuple of"
                             "param maps, but got %s." % type(params))

    @staticmethod
    def _parallelFitTasks(est, train, eva, validation, epm, verbose=False):
        """
        Creates a list of callables which can be called from different threads to fit and evaluate
        an estimator in parallel. Each callable returns an `(index, metric)` pair. Borrows heavily
        from pyspark.ml.tuning._parallelFitTasks.

        :param est: Estimator, the estimator to be fit.
        :param train: DataFrame, training data set, used for fitting.
        :param eva: Evaluator, used to compute `metric`
        :param validation: DataFrame, validation data set, used for evaluation.
        :param epm: Sequence of ParamMap, params maps to be used during fitting & evaluation.
        :return: (int, float), an index into `epm` and the associated metric value.
        """
        modelIter = est.fitMultiple(train, epm)

        def singleTask():
            index, model = next(modelIter)
            metric = eva.evaluate(model, validation)
            if verbose:
                print("k: {0}".format(model.getK()))
                print("rhos: {0}".format(model.getRhos()))
                print("Avg. correlation: " + str(metric))
            return index, metric

        return [singleTask] * len(epm)


class CCACrossValidatorModel(CrossValidatorModel):
    """
    Borrows heavily from pyspark.ml.tuning.CrossValidatorModel.
    """
    def __init__(self, bestModel):
        super(CCACrossValidatorModel, self).__init__(bestModel)

    def _transform(self, datasets):
        return self.bestModel.transform(datasets)

    def transform(self, datasets, params=None):
        """
        Overrides pyspark.ml.Transformer's transform method.
        """
        if params is None:
            params = dict()
        if isinstance(params, dict):
            if params:
                return self.copy(params)._transform(datasets)
            else:
                return self._transform(datasets)
        else:
            raise ValueError("Params must be a param map but got %s." % type(params))
        
    def save(self, path):
        """Save this CCACrossValidatorModel instance to the given path"""
        _save(self, path)

    def load(path):
        """Load a CCACrossValidatorModel instance from the given path"""
        return _load(path)
