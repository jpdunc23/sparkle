"""
Shared classes for unit testing.
"""
import unittest
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

class SparkTestCase(unittest.TestCase):

    def setUp(self):
        class_name = self.__class__.__name__
        conf = SparkConf().setAppName(class_name).setMaster('local[1]')
        self.sc = SparkContext.getOrCreate(conf)
        self.spark = SparkSession(self.sc)

    def tearDown(self):
        pass
