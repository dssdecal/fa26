from otter.test_files import test_case
import numpy as np
from sklearn.model_selection import train_test_split

OK_FORMAT = False

name = "q2"
points = 1

@test_case(points=None, hidden=False)
def test_mae(val_mae, actual, val_pred):
    assert np.isclose(val_mae, np.mean(np.abs(actual - val_pred))), "Compute MAE as the mean absolute prediction error."

@test_case(points=None, hidden=False)
def test_rmse(val_rmse, actual, val_pred):
    assert np.isclose(val_rmse, np.sqrt(np.mean((actual - val_pred) ** 2))), "Compute RMSE as the square root of mean squared error."

@test_case(points=None, hidden=False)
def test_r2(val_r2, actual, val_pred):
    expected = 1 - np.sum((actual - val_pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
    assert np.isclose(val_r2, expected), "Compute R² as 1 minus SSE divided by the total sum of squares."
