from otter.test_files import test_case
import numpy as np
import torch
from torch import nn

OK_FORMAT = False
name = "q3"
points = 1

@test_case(points=None, hidden=False)
def test_architecture(make_mlp):
    for width in [8, 32, 64]:
        network = make_mlp(hidden_size=width)
        assert isinstance(network, nn.Sequential) and len(network) == 3, "Use Linear, ReLU, Linear."
        assert isinstance(network[0], nn.Linear) and isinstance(network[1], nn.ReLU) and isinstance(network[2], nn.Linear)
        assert (network[0].in_features, network[0].out_features) == (64, width)
        assert (network[2].in_features, network[2].out_features) == (width, 10)
        assert network(torch.zeros(5, 64)).shape == (5, 10)
