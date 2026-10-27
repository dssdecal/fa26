from otter.test_files import test_case
import numpy as np

OK_FORMAT = False
name = "q6"
points = 1

from sklearn.decomposition import PCA

@test_case(points=None, hidden=False)
def test_variance(X, explained, cumulative, n_for_80):
    expected = PCA().fit(X).explained_variance_ratio_
    assert np.shape(explained) == (X.shape[1],), "Keep one variance ratio per component."
    assert np.allclose(explained, expected), "Use PCA's explained_variance_ratio_."
    assert np.allclose(cumulative, np.cumsum(expected)), "Compute the running total in component order."
    assert isinstance(n_for_80, (int, np.integer)), "The component count should be an integer."
    assert n_for_80 == np.flatnonzero(np.cumsum(expected) >= .8)[0] + 1, "Choose the smallest count reaching 80%; positions start at zero."
