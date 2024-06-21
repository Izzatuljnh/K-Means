# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:59:25 2024

@author: Lenovo G40-45
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# Memuat dataset
dataset = pd.read_excel('E:\IZZATUL\SKRIPSI 2\Data Penelitian\Clean_Debit.xlsx', usecols=['Hari', 'Debit', 'TMA'])

# Normalisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(dataset)

# Inisialisasi model K-Means
kmeans = KMeans(n_clusters=4, random_state=40)

# Latih model dengan data yang sudah dinormalisasi
kmeans.fit(data_scaled)

# Prediksi klaster untuk setiap sampel
labels = kmeans.predict(data_scaled)

# Mendapatkan koordinat pusat klaster
centroids = kmeans.cluster_centers_

# Menghitung jarak Euclidean antara dua set data
distances = cdist(data_scaled, centroids)

# Visualisasi klaster hasil
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')
plt.xlabel('Hari')
plt.ylabel('Debit')
plt.title('K-Means Clustering')
plt.show()

# Cetak koordinat centroid
print("Centroid koordinat (dalam skala yang dinormalisasi):")
print(centroids)

# Jika ingin melihat centroid dalam skala asli:
centroids_original_scale = scaler.inverse_transform(centroids)
print("\nCentroid koordinat (dalam skala asli):")
print(centroids_original_scale)

#%%
# Hitung Akurasi
Silhouette = silhouette_score(dataset, labels)

dbi_score = davies_bouldin_score(dataset, labels)

