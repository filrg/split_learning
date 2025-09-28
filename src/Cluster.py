from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.preprocessing import normalize
import numpy as np

def clustering_algorithm(label_counts, client_cluster_config):
    if client_cluster_config["cluster"] == "AffinityPropagation":
        return Affinity_Propagation(label_counts, client_cluster_config["AffinityPropagation"])
    elif client_cluster_config["cluster"] == "KMeans":
        return K_Means(label_counts, client_cluster_config["KMeans"])

def Affinity_Propagation(label_counts, config):
    infor_cluster = []
    damping = config['damping']
    max_iter = config['max_iter']
    label_counts = normalize(label_counts, norm='l1', axis=1)
    ap = AffinityPropagation(damping=damping, max_iter=max_iter, random_state=42)
    ap.fit(label_counts)

    labels = ap.labels_
    counts = np.bincount(labels)
    for count in counts:
        infor_cluster.append([count])

    num_cluster = len(np.unique(labels))

    return labels, infor_cluster, num_cluster

def K_Means(label_counts, config):

    infor_cluster = []

    num_cluster = config['num-cluster']
    label_counts = normalize(label_counts, norm='l1', axis=1)
    kmeans = KMeans(n_clusters= num_cluster, random_state=42)
    kmeans.fit(label_counts)
    labels = kmeans.labels_

    counts = np.bincount(labels)
    for count in counts:
        infor_cluster.append([count])

    return labels, infor_cluster, num_cluster
