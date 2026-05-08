from sklearn.mixture import GaussianMixture
import numpy as np

def auto_threshold(performance, n_init=9):
    performance = np.array(performance, dtype=float)
    if performance.size <= 1:
        return 0.0

    X = np.log(performance).reshape(-1, 1)
    gm = GaussianMixture(
        n_components=2,
        n_init=n_init,
        covariance_type='full',
        random_state=0
    ).fit(X)
    mu_raw = gm.means_.flatten()
    order = np.argsort(mu_raw)
    mu  = mu_raw[order]
    var = gm.covariances_.reshape(-1)[order]
    w   = gm.weights_[order]

    a = var[0] - var[1]
    b = 2 * (var[1]*mu[0] - var[0]*mu[1])
    c = (var[0]*mu[1]**2
         - var[1]*mu[0]**2
         + 2*var[0]*var[1]*np.log((var[1]*w[0])/(var[0]*w[1])))

    if np.isclose(a, 0):
        if np.isclose(b, 0):
            thresh_log = np.mean(mu)
        else:
            root = -c / b
            if mu[0] < root < mu[1]:
                thresh_log = root
            else:
                thresh_log = np.mean(mu)
    else:
        roots = np.roots([a, b, c])
        real_roots = roots[np.isreal(roots)].real
        candidates = real_roots[(real_roots > mu[0]) & (real_roots < mu[1])]

        if candidates.size > 0:
            mid = np.mean(mu)
            thresh_log = candidates[np.argmin(np.abs(candidates - mid))]
        else:
            thresh_log = np.mean(mu)

    return float(np.exp(thresh_log))
