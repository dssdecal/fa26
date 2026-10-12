from otter.test_files import test_case
import numpy as np
from sklearn.linear_model import LogisticRegression

OK_FORMAT = False
name = "q7"


@test_case(points=1, success_message="Your logistic regression predictions and probabilities are correct.")
def test_logistic_model(log_model, log_pred, log_proba, X_train, X_test, g_train):
    expected_model = LogisticRegression(max_iter=1000).fit(X_train, g_train)
    assert isinstance(log_model, LogisticRegression), "log_model should be LogisticRegression."
    np.testing.assert_array_equal(log_pred, expected_model.predict(X_test))
    np.testing.assert_allclose(log_proba, expected_model.predict_proba(X_test)[:, 1], rtol=1e-8, atol=1e-8)
