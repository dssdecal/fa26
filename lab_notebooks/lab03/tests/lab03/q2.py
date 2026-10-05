from otter.test_files import test_case
import pandas as pd

OK_FORMAT = False
name = "q2"


@test_case(points=1, success_message="You separated the training feature and target correctly.")
def test_feature_and_target(X_train, y_train):
    homes = pd.read_csv("house_prices.csv")
    train = homes.loc[homes["split"] == "train"].copy()
    size_mean = train["size_k_sqft"].mean()
    size_std = train["size_k_sqft"].std(ddof=0)
    expected_x = pd.DataFrame({"size_scaled": (train["size_k_sqft"] - size_mean) / size_std})
    expected_y = train["price_100k"]
    assert isinstance(X_train, pd.DataFrame), "X_train should be a one-column DataFrame."
    assert isinstance(y_train, pd.Series), "y_train should be a Series."
    pd.testing.assert_frame_equal(X_train, expected_x, obj="X_train")
    pd.testing.assert_series_equal(y_train, expected_y, obj="y_train")
