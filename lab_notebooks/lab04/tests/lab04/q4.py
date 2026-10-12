from otter.test_files import test_case
import numpy as np
from sklearn.linear_model import LinearRegression

OK_FORMAT = False
name = "q4"


@test_case(points=1, success_message="Your linear regression model and predictions are correct.")
def test_linear_model(lin_model, lin_pred, X_train, X_test, y_train):
    expected = LinearRegression().fit(X_train, y_train).predict(X_test)
    assert isinstance(lin_model, LinearRegression), "lin_model should be a LinearRegression model."
    np.testing.assert_allclose(lin_pred, expected, rtol=1e-8, atol=1e-8)
