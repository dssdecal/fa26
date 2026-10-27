from otter.test_files import test_case
import numpy as np

OK_FORMAT = False
name = "q2"
points = 1

@test_case(points=None, hidden=False)
def test_assignment(X2, centroids, distances, assignments):
    expected = np.linalg.norm(X2[:, None, :] - centroids[None, :, :], axis=2)
    assert np.shape(distances) == (len(X2), 3), "One row per listener and one column per centroid."
    assert np.allclose(distances, expected), "Compute Euclidean distances across the two feature columns."
    assert np.array_equal(assignments, np.argmin(expected, axis=1)), "Choose the nearest centroid for every listener."
