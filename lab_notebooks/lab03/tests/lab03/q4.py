from otter.test_files import test_case
import numpy as np
import pandas as pd

OK_FORMAT = False
name = "q4"


@test_case(points=1, success_message="Your MAE and MSE correctly summarize the baseline errors.")
def test_baseline_losses(baseline_mae, baseline_mse):
    homes = pd.read_csv("house_prices.csv")
    prices = homes.loc[homes["split"] == "train", "price_100k"].to_numpy()
    errors = prices - prices.mean()
    assert np.isclose(baseline_mae, np.mean(np.abs(errors)), rtol=1e-8, atol=1e-8), "Check the MAE formula."
    assert np.isclose(baseline_mse, np.mean(errors**2), rtol=1e-8, atol=1e-8), "Check the MSE formula."
