from otter.test_files import test_case
import numpy as np
from sklearn.model_selection import train_test_split

OK_FORMAT = False

name = "q3"
points = 1

@test_case(points=None, hidden=False)
def test_cv_mses(cv_mses, alphas):
    assert len(cv_mses) == len(alphas), "Append one mean CV MSE for every alpha."
    assert np.all(np.isfinite(cv_mses)) and np.all(np.asarray(cv_mses) > 0), "Store ordinary positive MSE values, not scikit-learn's negative scores."

@test_case(points=None, hidden=False)
def test_best_alpha(best_alpha, cv_mses, alphas):
    assert best_alpha == alphas[int(np.argmin(cv_mses))], "best_alpha should be the alpha with the smallest mean CV MSE."
