from otter.test_files import test_case
import numpy as np
import torch
from torch import nn

OK_FORMAT = False
name = "q2"
points = 1

@test_case(points=None, hidden=False)
def test_splits(X, y, X_dev, X_train, X_val, X_test, y_dev, y_train, y_val, y_test,
                X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, SEED):
    from sklearn.model_selection import train_test_split
    dev, test, yd, yt = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    train, val, ytr, yv = train_test_split(dev, yd, test_size=0.25, random_state=SEED, stratify=yd)
    for actual, expected in zip([X_dev, X_test, y_dev, y_test, X_train, X_val, y_train, y_val],
                                [dev, test, yd, yt, train, val, ytr, yv]):
        assert np.array_equal(actual, expected), "Use stratified 60/20/20 splits and SEED in both calls."
    for tensor, array in [(X_train_t, X_train), (X_val_t, X_val), (X_test_t, X_test)]:
        assert tensor.dtype == torch.float32 and np.allclose(tensor.numpy(), array)
    for tensor, array in [(y_train_t, y_train), (y_val_t, y_val)]:
        assert tensor.dtype == torch.long and np.array_equal(tensor.numpy(), array)
