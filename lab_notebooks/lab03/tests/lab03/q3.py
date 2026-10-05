from otter.test_files import test_case
import numpy as np
import pandas as pd

OK_FORMAT = False
name = "q3"


@test_case(points=1, success_message="Your baseline predicts the training mean for every house.")
def test_baseline(baseline_train_pred):
    homes = pd.read_csv("house_prices.csv")
    prices = homes.loc[homes["split"] == "train", "price_100k"].to_numpy()
    assert isinstance(baseline_train_pred, np.ndarray), "Use np.full to make a NumPy array."
    np.testing.assert_allclose(
        baseline_train_pred,
        np.full(len(prices), prices.mean()),
        rtol=1e-8,
        atol=1e-8,
        err_msg="Each baseline prediction should equal the mean training price.",
    )
