# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:20:17 2024

@author: Lenovo G40-45
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
#X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

data = pd.read_excel('E:\IZZATUL\SKRIPSI 2\Data Penelitian\Clean_Debit.xlsx')

# Menghitung inersia untuk jumlah kluster 1 hingga 10
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

# Plot Elbow method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Metode Elbow')
plt.xlabel('Jumlah Kluster')
plt.ylabel('Inersia')
plt.show()
