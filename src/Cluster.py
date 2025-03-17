from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score
import copy
import numpy as np
from src.Utils import num_client_in_cluster


def clustering_algorithm(list_performance, num, client_cluster_config, partition):
    return clustering_AffinityPropagation(list_performance, num, client_cluster_config["AffinityPropagation"], partition)


def clustering_AffinityPropagation(list_performance, num, config, partition=None):
    if partition is None:
        label_counts = []
        for i in list_performance:
            if i != -1:
                label_counts.append(i)
        damping = config['damping']
        max_iter = config['max_iter']
        affinity_propagation = AffinityPropagation(damping=damping, max_iter=max_iter)
        affinity_propagation.fit(np.array(label_counts).reshape(-1, 1))

        cluster_centers_indices = affinity_propagation.cluster_centers_indices_
        labels = affinity_propagation.labels_
        labels = labels.tolist()

        return list_performance, None, None, None
    else:
        list_cut_layer = partition["cut-layers"]
        num_cluster = partition["num-cluster"]
        infor_cluster = partition["infor-cluster"]
        return list_performance, infor_cluster, num_cluster, list_cut_layer
