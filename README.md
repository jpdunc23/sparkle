# SparKLe

SparKLe implements the methods of Solari et al. (2019) for large-scale sparse kernel canonical correlation analysis with the aim of inferring non-linear, yet interpretable, associations between multiple sets of high-dimensional covariates from observations on matching subjects.

SparKLe uses the Python API of Apache Spark, an open-source distributed data analytics framework  in  the MapReduce  lineage  of  computational  paradigms that  was  originally developed at the University of California, Berkeley’s AMPLab. Spark is often used in settings with large numbers of "observations", whereas SparKLe is designed for the  challenge of enormously large numbers of parameters across multiple datasets. Common settings in which SparKLe works well include multi-omics, fMRI studies, imaging data, and other settings where high dimensional data is encountered.

## Installation

### Prerequisites

SparKLe requires:

* Apache Spark (>= 2.3.0)
* Python (>= 3.6.8)
* NumPy (>= 1.15.4)
* SciPy (>= 1.2.0)

### User Installation from this Repo

```
pip install --user -e git+https://github.com/jpdunc23/sparkle#egg=sparkle
```

## Usage

### The Basics

```python
from sparkle.cca import *
from numpy import random

# generate some random data
data = [random.randn(100, p) for p in [300, 500, 700, 1000]]

# instantiate a new CCA object with hyperparameters k and rhos
cca = CCA(k = 1, rhos = (0.1, 0.3, 0.7, 0.4))

# train on four datasets, returning a CCAModel instance
cca_model = cca.fit(data)

# the four canonical loadings matrices are stored the ZZ
# field of the CCAModel as a list of scipy.sparse.csr_matrix
cca_model.ZZ

# See what the pairwise canonical correlations were between
# the four datasets
cca_model.canonicalCorrelations(data)

# we can use the transform method to produce predictions for each
# dataset using the three other datasets
cca_model.transform(data)

# save the CCAModel instance for later use
cca_model.save("/path/to/project")

# load a saved CCAModel using the static method CCAModel.load
cca_model = CCAModel.load("/path/to/project")
```

### Cross Validation Workflow

```python
from pyspark.ml.tuning import ParamGridBuilder

# split data into training and validation sets
training = [d[0:80, :] for d in data]
testing = [d[80:100, :] for d in data]

# instantiate CCA and CCAEvaluator objects
cca = CCA()
evaluator = CCAPredictionEvaluator()

# create a ParamGridBuilder instance with a grid of hyperparameters
paramGrid = ParamGridBuilder() \
    .baseOn({cca.broadcast: True, cca.k: 1}) \
    .addGrid(cca.rhos, [[0.1], [0.5], [0.9]]) \
    .build()
  
# create a CCACrossValidator instance
cv = CCACrossValidator(parallelism = 4) \
    .setEstimator(cca) \
    .setEvaluator(evaluator) \
    .setEstimatorParamMaps(paramGrid) \
    .setNumFolds(10) \
    .setVerbose(True)

# run the cross validation on training data
cv_fit = cv.fit(training)

# see the best tuning parameters
cv_fit.bestModel.getRhos()

# see the canonical correlations for training and test data
cv_fit.bestModel.canonicalCorrelations(training)
cv_fit.bestModel.canonicalCorrelations(testing)

# see average correlation between predicted features and true features
evaluator.evaluate(cv_fit.bestModel, testing)

# save the CV fit for later use
cv_fit.save("/path/to/project")
cv_fit = CCACrossValidatorModel.load("/path/to/project")
```

## Authors

* **Omid Shams Solari** - *Method development*
* **James P.C. Duncan** - *Package development*

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [Apache Spark](https://spark.apache.org/)
* [pyrcca](https://github.com/gallantlab/pyrcca)
