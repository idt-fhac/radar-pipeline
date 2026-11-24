import numpy as np
import pytest

from radar.mapping.cluster import cluster, name_cluster, reduce
from radar.mapping.embed import render


def unit(rows):
    rows = np.asarray(rows, dtype=float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_render_replaces_placeholders_and_keeps_braces():
    assert render('{topic}: {label}', 'E-Auto', 'Preis {netto}') == 'E-Auto: Preis {netto}'


def test_hdbscan_outliers_become_clusters_of_their_own():
    rng = np.random.default_rng(0)
    group_a = rng.normal(0, 0.01, (10, 2))
    group_b = rng.normal(5, 0.01, (10, 2))
    outlier = np.array([[50.0, -50.0]])
    labels = cluster(np.vstack([group_a, group_b, outlier]), 'hdbscan', params={'min_cluster_size': 5})
    assert len(set(labels[:10])) == 1 and len(set(labels[10:20])) == 1
    assert labels[20] not in labels[:20]  # the outlier is kept, as its own cluster


def test_hdbscan_rejects_threshold():
    with pytest.raises(ValueError, match='no threshold'):
        cluster(np.zeros((3, 2)), 'hdbscan', threshold=0.3)


def test_agglomerative_needs_threshold():
    with pytest.raises(ValueError, match='threshold'):
        cluster(np.ones((3, 2)), 'agglomerative')


def test_reduce_leaves_too_few_points_unchanged():
    vectors = np.ones((4, 10))
    assert reduce(vectors, n_components=5) is vectors


def test_centroid_name_is_weighted_by_mentions():
    vectors = unit([[1, 0], [0.9, 0.1], [0, 1]])
    labels = ['Reichweite', 'Reichweite Winter', 'Ladezeit']
    # rare 'Ladezeit' far away, 'Reichweite' mentioned 50 times: the weighted mean sits near it
    assert name_cluster([0, 1, 2], vectors, [50, 1, 1], labels, 'centroid') == 'Reichweite'
    assert name_cluster([0, 1, 2], vectors, [1, 1, 50], labels, 'centroid') == 'Ladezeit'


def test_frequent_name_breaks_ties_alphabetically():
    vectors = unit([[1, 0], [0, 1]])
    assert name_cluster([0, 1], vectors, [3, 3], ['Preis', 'Kosten'], 'frequent') == 'Kosten'
    assert name_cluster([0, 1], vectors, [3, 5], ['Preis', 'Kosten'], 'frequent') == 'Kosten'
    with pytest.raises(ValueError, match='naming'):
        name_cluster([0], vectors, [1, 1], ['a', 'b'], 'llm')
