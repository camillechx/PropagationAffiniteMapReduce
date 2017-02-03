import os
os.environ['SPARK_HOME']="C:/Users/Camille/Documents/Spark"
from pyspark import SparkContext
sc = SparkContext.getOrCreate()