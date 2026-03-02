import numpy as np
import torch

# KHÔNG DÙNG NỮA DO CONFIG THỦ CÔNG
"""
def flatten_weights(state_dict):
    vectors = []
    for key in sorted(state_dict.keys()):
        vectors.append(state_dict[key].cpu().numpy().flatten())
    return np.concatenate(vectors)

def simple_kmeans(data, k, max_iters=100):
    n_samples = data.shape[0]
    if n_samples <= k:
        return np.arange(n_samples)
    
    # Khởi tạo centroids ngẫu nhiên
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[indices]
    
    clusters = np.zeros(n_samples)
    
    for _ in range(max_iters):
        # Tính khoảng cách Euclidean
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # Gán cụm
        new_clusters = np.argmin(distances, axis=1)
        
        if np.array_equal(clusters, new_clusters):
            break
        
        clusters = new_clusters
        
        # Cập nhật centroids
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        centroids = new_centroids
        
    return clusters.astype(int)

def perform_out_clustering(client_weights_dict, num_clusters):
    client_ids = list(client_weights_dict.keys())
    flat_weights = []
    
    for cid in client_ids:
        flat_weights.append(flatten_weights(client_weights_dict[cid]))
    
    data = np.stack(flat_weights)
    cluster_labels = simple_kmeans(data, num_clusters)
    
    # Kết quả mapping: {client_id: cluster_id}
    result = {client_ids[i]: int(cluster_labels[i]) for i in range(len(client_ids))}
    return result

def perform_in_clustering(client_speeds_dict, num_clusters):
    
    client_ids = list(client_speeds_dict.keys())
    # Chuyển tốc độ thành vector 1D (N, 1) để dùng simple_kmeans
    speeds = np.array([client_speeds_dict[cid] for cid in client_ids]).reshape(-1, 1)
    
    cluster_labels = simple_kmeans(speeds, num_clusters)
    
    # Kết quả mapping: {client_id: in_cluster_id}
    result = {client_ids[i]: int(cluster_labels[i]) for i in range(len(client_ids))}
    return result
"""
