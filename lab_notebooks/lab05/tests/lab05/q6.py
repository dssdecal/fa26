from otter.test_files import test_case
import numpy as np
from sklearn.model_selection import train_test_split

OK_FORMAT = False

name = "q6"
points = 1

@test_case(points=None, hidden=False)
def test_confusion_counts(tp, tn, fp, fn, pred_05, y_val_c):
    expected = (
        np.sum((pred_05 == 1) & (y_val_c == 1)),
        np.sum((pred_05 == 0) & (y_val_c == 0)),
        np.sum((pred_05 == 1) & (y_val_c == 0)),
        np.sum((pred_05 == 0) & (y_val_c == 1)),
    )
    assert (tp, tn, fp, fn) == expected, "Recheck which actual/predicted combinations define TP, TN, FP, and FN."

@test_case(points=None, hidden=False)
def test_precision_recall(precision_05, recall_05, tp, fp, fn):
    assert np.isclose(precision_05, tp / (tp + fp)), "Precision is TP / (TP + FP)."
    assert np.isclose(recall_05, tp / (tp + fn)), "Recall is TP / (TP + FN)."
