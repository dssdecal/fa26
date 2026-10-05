from otter.test_files import test_case
import numpy as np
import pandas as pd

OK_FORMAT = False
name = "q5"


@test_case(points=1, success_message="Your gradient and parameter update are correct.")
def test_gradient_step(current_pred, grad_w, grad_b, new_w, new_b):
    homes = pd.read_csv("house_prices.csv")
    train = homes.loc[homes["split"] == "train"]
    x = ((train["size_k_sqft"] - train["size_k_sqft"].mean()) / train["size_k_sqft"].std(ddof=0)).to_numpy()
    y = train["price_100k"].to_numpy()
    b = y.mean()
    expected_pred = np.full(len(y), b)
    expected_grad_w = -2 * np.mean(x * (y - expected_pred))
    expected_grad_b = -2 * np.mean(y - expected_pred)
    assert np.shape(current_pred) == np.shape(expected_pred), "Make one prediction per training home."
    np.testing.assert_allclose(current_pred, expected_pred, rtol=1e-8, atol=1e-8)
    assert np.isclose(grad_w, expected_grad_w, rtol=1e-8, atol=1e-8), "Check the slope gradient."
    assert np.isclose(grad_b, expected_grad_b, rtol=1e-8, atol=1e-8), "Check the intercept gradient."
    assert np.isclose(new_w, -0.1 * expected_grad_w, rtol=1e-8, atol=1e-8), "Step opposite the slope gradient."
    assert np.isclose(new_b, b - 0.1 * expected_grad_b, rtol=1e-8, atol=1e-8), "Step opposite the intercept gradient."
