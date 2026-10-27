from otter.test_files import test_case
import numpy as np

OK_FORMAT = False
name = "q5"
points = 1

import pandas as pd
from sklearn.cluster import KMeans

@test_case(points=None, hidden=False)
def test_clusters(X, listeners, genres, kmeans, cluster_means):
    assert isinstance(kmeans, KMeans), "Fit a KMeans model."
    assert isinstance(kmeans.n_clusters, (int, np.integer)) and 2 <= kmeans.n_clusters <= 8, "Choose 2–8 clusters from your elbow plot and explain your choice."
    assert kmeans.n_init == 10 and kmeans.random_state == 42, "Keep n_init=10 and random_state=42."
    assert kmeans.cluster_centers_.shape == (kmeans.n_clusters, len(genres)), "Fit on all ten genres."
    assert np.array_equal(kmeans.labels_, kmeans.predict(X)), "Fit kmeans on X."
    assert np.array_equal(listeners['cluster'].to_numpy(), kmeans.labels_), "Use the fitted labels for the cluster column."
    expected = listeners.groupby('cluster')[genres].mean()
    assert isinstance(cluster_means, pd.DataFrame), "Return a DataFrame of means."
    pd.testing.assert_frame_equal(cluster_means.sort_index(), expected.sort_index(), check_dtype=False, check_exact=False)
