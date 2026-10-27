from otter.test_files import test_case
import numpy as np

OK_FORMAT = False
name = "q4"
points = 1

from sklearn.cluster import KMeans

@test_case(points=None, hidden=False)
def test_elbow(X, ks, inertias):
    assert list(ks) == list(range(1, 9)), "Try k=1 through k=8."
    assert len(inertias) == 8, "Record one inertia per k."
    expected = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
    assert np.allclose(inertias, expected), "Fit each model on all ten genres and record inertia_."
