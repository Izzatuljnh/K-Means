# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 02:21:23 2023

@author: Lenovo G40-45
"""

#from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import cdist


dataset = pd.read_excel('E:\IZZATUL\SKRIPSI 2\Data Penelitian\Clean_2018.xlsx', usecols=['Hari', 'Debit 2018', 'TMA 2018'])
data = np.array (dataset)
df = pd.DataFrame(data)
#k = 3


# Inisialisasi model K-Means
kmeans = KMeans(n_clusters=4, random_state=40)

# Latih model dengan data
kmeans.fit(data)

# Prediksi klaster untuk setiap sampel
labels = kmeans.fit_predict(data)

# Mendapatkan koordinat pusat klaster
centroids = kmeans.cluster_centers_

# Menghitung jarak Euclidean antara dua set data
distances = cdist(data, centroids)

# Tambahkan label ke dataframe
df['cluster'] = labels

# Print hasil cluster
print(df)

# Visualisasi klaster hasil
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')
plt.xlabel('Hari')
plt.ylabel('Debit')
plt.title('K-Means Clustering 2018')
plt.show()

#%%
# Hitung Akurasi
Silhouette = silhouette_score(dataset, labels)

dbi_score = davies_bouldin_score(dataset, labels)
