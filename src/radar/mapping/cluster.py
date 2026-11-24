# cluster.py — reduce embedding vectors, group them and name the groups. No database, no model.

import numpy as np

NAMING_METHODS = ['centroid', 'frequent']


def reduce(vectors, n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42, **options):
    """Project the vectors with UMAP to few dimensions before clustering (for every method).

    Density and distances are hard to tell apart in ~4000 dimensions; UMAP keeps the
    neighbourhoods. With random_state set, the projection is reproducible.
    Too few points for UMAP: the vectors are returned unchanged.
    """
    if len(vectors) <= n_components + 1:
        return vectors
    import umap  # heavy import (numba), only when needed
    reducer = umap.UMAP(n_components=n_components, n_neighbors=min(n_neighbors, len(vectors) - 1),
                        min_dist=min_dist, metric=metric, random_state=random_state, **options)
    return reducer.fit_transform(vectors)


def cluster(vectors, method, threshold = None, params = None):
    """Cluster id per point (row of vectors). Outliers become a cluster of their own.

    method 'agglomerative': threshold is the distance at which clusters stop merging,
        params are passed to sklearn's AgglomerativeClustering (e.g. {'linkage': 'average'}).
    method 'hdbscan': params are passed to sklearn's HDBSCAN (e.g. {'min_cluster_size': 8});
        its outliers (-1) are split up. threshold is not used: cluster_selection_epsilon
        crashed sklearn's HDBSCAN on our data (embedding playground, 20.11.).
    """
    params = {} if params is None else params
    if len(vectors) < 2:
        return [0] * len(vectors)

    if method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        if threshold is None:
            raise ValueError("method 'agglomerative' needs a threshold")
        options = {'metric': 'cosine', 'linkage': 'average', **params}
        labels = AgglomerativeClustering(n_clusters = None, distance_threshold = threshold, **options).fit_predict(vectors)
    elif method == 'hdbscan':
        from sklearn.cluster import HDBSCAN
        if threshold is not None:
            raise ValueError("method 'hdbscan' takes no threshold; tune min_cluster_size and min_samples")
        options = dict(params)
        options['min_cluster_size'] = min(options.get('min_cluster_size', 5), len(vectors))
        labels = HDBSCAN(**options).fit_predict(vectors)
    else:
        raise ValueError(f'unknown clustering method: {method!r}')

    # every outlier (-1) becomes a cluster of its own
    labels = [int(label) for label in labels]
    next_label = max(labels) + 1
    for point, label in enumerate(labels):
        if label == -1:
            labels[point] = next_label
            next_label += 1
    return labels


def name_cluster(points, vectors, counts, labels, method='centroid'):
    """Name of one cluster: one of its labels.

    :param points: indices of the cluster's points
    :param vectors: full embeddings (not the UMAP projection), length 1 per row
    :param counts: number of mentions per point
    :param labels: label text per point
    method 'centroid': the label closest to the mention-weighted mean, so a frequent label
        weighs more than a rare variant; 'frequent': the most mentioned label.
    Ties are broken alphabetically, so the name is reproducible.
    """
    points = sorted(points, key=lambda point: labels[point])  # max() keeps the first of equals
    if method == 'frequent':
        return labels[max(points, key=lambda point: counts[point])]
    if method == 'centroid':
        weights = np.asarray([counts[point] for point in points], dtype=float)
        mean = (vectors[points] * weights[:, None]).sum(axis=0) / weights.sum()
        similarity = vectors[points] @ mean  # cosine up to a constant factor (rows have length 1)
        return labels[points[int(np.argmax(similarity))]]
    raise ValueError(f'unknown naming method {method!r}, known: {", ".join(NAMING_METHODS)}')
