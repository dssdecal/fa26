from otter.test_files import test_case
import numpy as np
import torch
from torch import nn

OK_FORMAT = False
name = "q5"
points = 1

@test_case(points=None, hidden=False)
def test_selection(candidates, width_results, best_width, best_model, X_val_t, y_val_t, X_train_t, y_train_t):
    assert set(candidates) == {8, 32, 64}
    assert sorted(width_results['hidden_units'].tolist()) == [8, 32, 64]
    for row in width_results.itertuples():
        network = candidates[row.hidden_units]
        network.eval()
        with torch.no_grad():
            val = (network(X_val_t).argmax(1) == y_val_t).float().mean().item()
            train = (network(X_train_t).argmax(1) == y_train_t).float().mean().item()
        assert np.isclose(row.val_acc, val), "Compare validation predictions with validation labels."
        assert np.isclose(row.train_acc, train)
        assert row.parameters == sum(p.numel() for p in network.parameters())
    expected = int(width_results.sort_values(['val_acc', 'hidden_units'], ascending=[False, True]).iloc[0]['hidden_units'])
    assert best_width == expected and best_model is candidates[expected], "Select on validation accuracy, breaking ties toward fewer units."
