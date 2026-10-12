from otter.test_files import test_case
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

OK_FORMAT = False
name = "q8"


@test_case(points=1, success_message="Your classification metrics are correct.")
def test_classification_metrics(majority_class, majority_acc, log_acc, log_precision, log_recall, g_train, g_test, log_pred):
    expected_class = g_train.mode()[0]
    assert majority_class == expected_class, "Use the most common training label."
    assert np.isclose(majority_acc, (g_test == expected_class).mean()), "Check majority_acc."
    assert np.isclose(log_acc, accuracy_score(g_test, log_pred)), "Check log_acc."
    assert np.isclose(log_precision, precision_score(g_test, log_pred)), "Check log_precision."
    assert np.isclose(log_recall, recall_score(g_test, log_pred)), "Check log_recall."
