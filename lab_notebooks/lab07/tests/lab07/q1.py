from otter.test_files import test_case
import numpy as np
import torch
from torch import nn

OK_FORMAT = False
name = "q1"
points = 1

@test_case(points=None, hidden=False)
def test_pixels(X, images):
    assert isinstance(X, np.ndarray) and X.shape == (len(images), 64), "One row per image, 64 pixel columns."
    assert np.allclose(X, images.reshape(len(images), -1) / 16), "Flatten images and scale by the known maximum of 16."
