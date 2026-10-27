from otter.test_files import test_case
import numpy as np

OK_FORMAT = False
name = "q7"
points = 1

from sklearn.decomposition import PCA

@test_case(points=None, hidden=False)
def test_projection(X, pca_2d, X_2d):
    assert isinstance(pca_2d, PCA) and pca_2d.n_components == 2, "Create PCA with two components."
    assert np.shape(X_2d) == (len(X), 2), "Keep one row per listener and two components."
    assert np.allclose(X_2d, pca_2d.transform(X)), "Use fit_transform on X."
    expected = PCA(n_components=2).fit(X)
    assert np.allclose(pca_2d.explained_variance_ratio_, expected.explained_variance_ratio_), "Fit PCA on all ten genre columns."
    assert np.allclose(pca_2d.components_.T @ pca_2d.components_, expected.components_.T @ expected.components_), "Keep the first two principal directions."
