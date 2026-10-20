from otter.test_files import test_case
import numpy as np
from sklearn.model_selection import train_test_split

OK_FORMAT = False

name = "q5"
points = 1

@test_case(points=None, hidden=False)
def test_refund_rate(refund_rate, y_train_c):
    assert np.isclose(refund_rate, np.mean(y_train_c)), "refund_rate should be the share of training labels equal to 1."

@test_case(points=None, hidden=False)
def test_always_no(always_no_pred, always_no_accuracy, y_val_c):
    assert np.array_equal(always_no_pred, np.zeros(len(y_val_c), dtype=int)), "always_no_pred should contain a 0 for every validation row."
    assert np.isclose(always_no_accuracy, np.mean(always_no_pred == y_val_c)), "always_no_accuracy should be the share of correct validation predictions."
