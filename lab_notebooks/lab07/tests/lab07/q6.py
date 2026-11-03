from otter.test_files import test_case
import numpy as np
import torch
from torch import nn

OK_FORMAT = False
name = "q6"
points = 1

@test_case(points=None, hidden=False)
def test_evaluation(baseline, majority_digit, X_train, y_train, best_model, X_test_t, y_test, test_pred, test_acc):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    assert isinstance(baseline, LogisticRegression) and baseline.max_iter == 1000
    reference = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    assert np.allclose(baseline.coef_, reference.coef_), "Fit the baseline on training data only."
    assert majority_digit == np.bincount(y_train).argmax()
    best_model.eval()
    with torch.no_grad():
        expected = best_model(X_test_t).argmax(dim=1).numpy()
    assert np.shape(test_pred) == np.shape(y_test) and np.array_equal(test_pred, expected)
    assert np.isclose(test_acc, accuracy_score(y_test, expected)), "Compute the share of correct test predictions."
