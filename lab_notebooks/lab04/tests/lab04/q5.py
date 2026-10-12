from otter.test_files import test_case
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

OK_FORMAT = False
name = "q5"


@test_case(points=1, success_message="Your regression metrics are correct.")
def test_regression_metrics(baseline_mse, lin_mse, lin_r2, y_train, y_test, lin_pred):
    expected_baseline = np.mean((y_test - y_train.mean()) ** 2)
    assert np.isclose(baseline_mse, expected_baseline), "Check baseline_mse."
    assert np.isclose(lin_mse, mean_squared_error(y_test, lin_pred)), "Check lin_mse."
    assert np.isclose(lin_r2, r2_score(y_test, lin_pred)), "Check lin_r2."
