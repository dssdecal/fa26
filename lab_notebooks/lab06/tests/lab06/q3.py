from otter.test_files import test_case
import numpy as np

OK_FORMAT = False
name = "q3"
points = 1

@test_case(points=None, hidden=False)
def test_update(X2, assignments, new_centroids):
    expected = np.array([X2[np.asarray(assignments) == j].mean(axis=0) for j in range(3)])
    assert np.shape(new_centroids) == (3, 2), "Keep two coordinates for each of three centroids."
    assert np.allclose(new_centroids, expected), "Average only the listeners assigned to each centroid, column by column."
