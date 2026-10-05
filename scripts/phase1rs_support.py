"""The frozen Phase 1R empirical support-radius gate."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def support_radii(endpoints: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    """Return endpoint NN distances and target LOO-30NN radius q99.

    This preserves Phase 1R's definition: fit only the empirical target cloud,
    remove each target row's own index, take its 30th other neighbor, then use
    the 99th percentile as the unchanged support boundary.
    """
    targets = np.asarray(targets, dtype=np.float64)
    endpoints = np.asarray(endpoints, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=31, algorithm="kd_tree", n_jobs=1).fit(targets)
    loo_dist, loo_ind = nn.kneighbors(targets, n_neighbors=32)
    loo30 = np.empty(len(targets), dtype=np.float64)
    for i, (distances, indices) in enumerate(zip(loo_dist, loo_ind)):
        other = distances[indices != i]
        if len(other) < 30:
            raise ValueError("Support reference has fewer than 30 non-self neighbours")
        loo30[i] = other[29]
    radius99 = float(np.quantile(loo30, 0.99))
    predicted, _ = nn.kneighbors(endpoints, n_neighbors=1)
    return predicted[:, 0], radius99
