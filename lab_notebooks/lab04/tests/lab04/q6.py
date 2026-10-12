from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q6"


@test_case(points=1, success_message="Your labeled coefficient Series is correct.")
def test_coefficients(lin_coefs, lin_model, X):
    expected = pd.Series(lin_model.coef_, index=X.columns)
    assert isinstance(lin_coefs, pd.Series), "lin_coefs should be a pandas Series."
    pd.testing.assert_series_equal(lin_coefs, expected)
