from otter.test_files import test_case
import numpy as np
import pandas as pd

OK_FORMAT = False
name = "q6"


@test_case(points=1, success_message="Your updated line and validation MSE are correct.")
def test_updated_predictions(updated_train_pred, updated_train_mse, validation_pred, validation_mse):
    homes = pd.read_csv("house_prices.csv")
    train = homes.loc[homes["split"] == "train"]
    validation = homes.loc[homes["split"] == "validation"]
    size_mean = train["size_k_sqft"].mean()
    size_std = train["size_k_sqft"].std(ddof=0)
    x = ((train["size_k_sqft"] - size_mean) / size_std).to_numpy()
    y = train["price_100k"].to_numpy()
    x_val = ((validation["size_k_sqft"] - size_mean) / size_std).to_numpy()
    y_val = validation["price_100k"].to_numpy()
    b = y.mean()
    grad_w = -2 * np.mean(x * (y - b))
    grad_b = -2 * np.mean(y - b)
    w_new, b_new = -0.1 * grad_w, b - 0.1 * grad_b
    expected_train = w_new * x + b_new
    expected_validation = w_new * x_val + b_new
    assert np.shape(updated_train_pred) == np.shape(expected_train), "Make one prediction per training home."
    assert np.shape(validation_pred) == np.shape(expected_validation), "Make one prediction per validation home."
    np.testing.assert_allclose(updated_train_pred, expected_train, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(validation_pred, expected_validation, rtol=1e-8, atol=1e-8)
    assert np.isclose(updated_train_mse, np.mean((y - expected_train) ** 2), rtol=1e-8, atol=1e-8), "Check training MSE."
    assert np.isclose(validation_mse, np.mean((y_val - expected_validation) ** 2), rtol=1e-8, atol=1e-8), "Check validation MSE."
