import csv
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark import SparkContext
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors

spark = SparkSession.builder.appName("DataFrame").getOrCreate()
data = spark.read.csv('Pharmacies.csv', sep=',',inferSchema=True, header=True)
sample_data = spark.read.csv('Sample.csv', sep=',',inferSchema=True, header=True)

rdd = data.rdd
rdd_rem_dup = rdd.distinct()
cleaned_rdd = rdd_rem_dup.filter(lambda x: x is not None)

sample_rdd = sample_data.rdd
srdd_rem_dup = sample_rdd.distinct()
scleaned_rdd = srdd_rem_dup.filter(lambda x: x is not None)

column_index = 9
extracted_rdd = cleaned_rdd.map(lambda x: float(x[column_index]))
sextracted_rdd = scleaned_rdd.map(lambda x: float(x[column_index]))

# Apply one-hot encoding (if needed) and standardization
encoded_rdd = extracted_rdd.map(lambda x: Vectors.dense(x))  # No one-hot encoding here
scaler = StandardScaler().fit(encoded_rdd)
scaled_rdd = scaler.transform(encoded_rdd)

sencoded_rdd = sextracted_rdd.map(lambda x: Vectors.dense(x))  # No one-hot encoding here
sscaler = StandardScaler().fit(sencoded_rdd)
sscaled_rdd = scaler.transform(sencoded_rdd)

# Define the number of clusters
k = 3

# Train K-Means model
model = KMeans.train(scaled_rdd, k)

# Get cluster centers
centers = model.clusterCenters

# Get cluster assignments for each record
cluster_assignments = model.predict(sscaled_rdd)

# Print cluster centers and assignments
print("Cluster centers:")
for center in centers:
    print(center)

print("Cluster assignments:")
for assignment in cluster_assignments.collect():
    print(assignment)

