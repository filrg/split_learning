from sklearn.mixture import GaussianMixture
import numpy as np
import heapq

def auto_threshold(performance, n_init=9):
    if len(performance) > 1:
        X = np.log(np.array(performance)).reshape(-1, 1)

        gm  = GaussianMixture(n_components=2, n_init=n_init, covariance_type='full').fit(X)

        mu  = np.sort(gm.means_.flatten())          # μ1 < μ2
        var = gm.covariances_.flatten()[np.argsort(gm.means_.flatten())]
        w   = gm.weights_[np.argsort(gm.means_.flatten())]

        a = var[0] - var[1]
        b = 2*(var[1]*mu[0] - var[0]*mu[1])
        c = var[0]*mu[1]**2 - var[1]*mu[0]**2 + 2*var[0]*var[1]*np.log((var[1]*w[0])/(var[0]*w[1]))
        roots = np.roots([a, b, c])
        thresh_log = np.real(roots[(roots>mu[0]) & (roots<mu[1])][0])
        return np.exp(thresh_log)
    else:
        return 0

def lpt(layer2, layer1, num_cluster):

    layer2 = sorted(layer2, key=lambda x:x[1], reverse=True)
    clusters = [(0, []) for _ in range(num_cluster)]
    heapq.heapify(clusters)

    for performance in layer2:
        total, cluster = heapq.heappop(clusters)
        cluster.append(performance[0])
        total += performance[1]
        heapq.heappush(clusters, (total, cluster))

    clusters = [heapq.heappop(clusters) for _ in range(len(clusters))]
    for idx, (total, infor) in enumerate(clusters):
        min_index = layer1.index(min(layer1))
        clusters[idx] = (min_index, infor)
        layer1[min_index] = max(layer1) + 1

    return clusters

# layer1 = [1]
# layer2 = [
#         ("A", 9),
#         ("B", 7),
#         ("C", 5),
#         ("D", 3),
#         ("E", 2),
#         ("F", 1)
#     ]
# num = 1
# a = lpt(layer2, layer1, num)
# print(a)