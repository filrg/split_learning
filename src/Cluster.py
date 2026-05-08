from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

def clustering_algorithm(label_counts, num_cluster):
    return K_Means(label_counts, num_cluster)

def K_Means(label_counts, num_cluster):

    infor_cluster = []

    label_counts = normalize(label_counts, norm='l1', axis=1)
    kmeans = KMeans(n_clusters= num_cluster, random_state=42)
    kmeans.fit(label_counts)
    labels = kmeans.labels_

    counts = np.bincount(labels)
    for count in counts:
        infor_cluster.append([count])

    return labels, infor_cluster
