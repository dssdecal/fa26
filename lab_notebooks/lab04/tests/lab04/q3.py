from otter.test_files import test_case
import pandas as pd
from sklearn.model_selection import train_test_split

OK_FORMAT = False
name = "q3"


@test_case(points=1, success_message="Your train/test split is correct and aligned.")
def test_split(X, tips, X_train, X_test, y_train, y_test, g_train, g_test):
    expected = train_test_split(
        X, tips["tip"], tips["generous"], test_size=0.2, random_state=42
    )
    actual = (X_train, X_test, y_train, y_test, g_train, g_test)
    for got, want in zip(actual, expected):
        if isinstance(want, pd.DataFrame):
            pd.testing.assert_frame_equal(got, want)
        else:
            pd.testing.assert_series_equal(got, want)
