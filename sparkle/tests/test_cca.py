"""
Unit tests for sparkle.cca module.
"""
import unittest
import numpy as np

from sparkle.cca import *
from sparkle.util import *
from sparkle.distributed import makeColBlockMatrix
from sparkle.tests.test_util import SparkTestCase


class DefaultValuesTests(unittest.TestCase):

    def test_CCA_param_defaults(self):
        cca = CCA()
        self.assertEqual(cca.getK(), 1)
        self.assertEqual(cca.getMaxIter(), 1000)
        self.assertEqual(cca.getRhos(), [0.1])
        self.assertEqual(cca.getColsPerBlock(), [1024])
        self.assertTrue(cca.getBroadcast())
        self.assertTrue(cca.getStandardization())
        self.assertEqual(cca.getTol(), 1e-4)


class CCATests(SparkTestCase):

    def test_singleL1(self):
        X1 = np.array([
            [0.005146443, 0.022261778,-1.539553e-04,-0.004892040, 0.034279120,
             0.019231456, 0.0033527404,-0.007459772, 0.007264192, -0.024504849],
            [-0.004373749,-0.018919362, 1.308402e-04, 0.004157542,-0.029132402,
             -0.016344016,-0.0028493550, 0.006339751,-0.006173536, 0.020825654],
            [0.002724898, 0.011786992,-8.151504e-05,-0.002590199, 0.018149839,
             0.010182520, 0.0017751827,-0.003949742, 0.003846187, -0.012974635],
            [-0.001299829,-0.005622623, 3.888425e-05, 0.001235575,-0.008657825,
             -0.004857259,-0.0008467965, 0.001884103,-0.001834706, 0.006189152],
            [0.003203071, 0.013855403,-9.581951e-05,-0.003044734, 0.021334819,
             0.011969375, 0.0020866962,-0.004642852, 0.004521126, -0.015251457]
        ])
        X2 = np.array([
            [-0.019976273,0.017104130,0.019716665,-3.778085e-04,0.0030775849,
             -0.010618072,-0.012964638],
            [0.016977005,-0.014536090,-0.016756376,3.210838e-04,-0.0026155117,
             0.009023859,0.011018108],
            [-0.010576880,0.009056160,0.010439425,-2.000391e-04,0.0016294955,
             -0.005621974, -0.006864415],
            [0.005045377,-0.004319964,-0.004979808,9.542251e-05,-0.0007773009,
             0.002681790,0.003274459],
            [-0.012432938,0.010645358,0.012271362,-2.351424e-04,0.0019154435,
             -0.006608532, -0.008069000]
        ])
        # MuleR.singleL1.pattern(t(X1) %*% X2, 0.1, centre=FALSE)
        z2_R = [0.55190880,-0.46373971,-0.54393937,0.00000000,-0.03315256,
                0.26463059, 0.33666552]
        # MuleR.singleL1.pattern(t(X2[,z2 != 0]) %*% X1, 0.1, centre=FALSE)
        z1_R = [0.03808095,0.41733973,0.00000000,-0.03244362,0.68363205,
                0.35019080,0.00000000,-0.08934201,0.08500815,-0.46704394]

        X1_CBM = makeColBlockMatrix(X1, 2)
        X2_CBM = makeColBlockMatrix(X2, 2)
        C12 = X2_CBM.leftMultiply(X1.T)
        C21 = X1_CBM.leftMultiply(X2.T)
        z2, rho2 = CCA._singleL1(C12, X1, X2, 0.1, 1e-4, 1000)
        z2 = _thresholding(z2, rho2)
        norm_z2 = np.linalg.norm(z2)
        if norm_z2 > 0:
            z2 = z2 / norm_z2
        p2 = z2.nonzero()[0]
        X2_red = X2[:,p2]
        z1, rho1 = CCA._singleL1(
            C21, X2_red, X1, 0.1, 1e-4, 1000, p=p2
        )
        z1 = _thresholding(z1, rho1)
        norm_z1 = np.linalg.norm(z1)
        if norm_z1 > 0:
            z1 = z1 / norm_z1
        self.assertTrue(np.all(np.abs(z1 - z1_R) < 1e-7))
        self.assertTrue(np.all(np.abs(z2 - z2_R) < 1e-7))

    def test_CCA(self):
        datasets = [np.loadtxt('data/X%d.tsv' % i) for i in range(1, 4)]
        ZZ = [np.loadtxt('data/Z%d.tsv' % i) for i in range(1, 4)]
        cca_model = CCA(k = 1).fit(datasets)
        for i in range(3):
            self.assertTrue(np.all(
                np.abs((cca_model.ZZ[i].todense() - ZZ[i][:, 0:1].T).getA1()) < 1e-8
            ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
