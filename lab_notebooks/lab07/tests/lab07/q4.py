from otter.test_files import test_case
import numpy as np
import torch
from torch import nn

OK_FORMAT = False
name = "q4"
points = 1

@test_case(points=None, hidden=False)
def test_learning(make_mlp, train_model, model, history, X_val_t, y_val_t):
    assert set(history) == {"train_loss", "val_loss", "val_acc"}
    assert all(len(values) == 30 and np.isfinite(values).all() for values in history.values())
    assert all(0 <= value <= 1 for value in history['val_acc'])
    model.eval()
    with torch.no_grad():
        scores = model(X_val_t)
        expected_loss = nn.CrossEntropyLoss()(scores, y_val_t).item()
        expected_acc = (scores.argmax(1) == y_val_t).float().mean().item()
    assert np.isclose(history['val_loss'][-1], expected_loss), "Record cross-entropy on validation logits."
    assert np.isclose(history['val_acc'][-1], expected_acc)
    probe = make_mlp(8)
    before = [p.detach().clone() for p in probe.parameters()]
    result = train_model(probe, epochs=2)
    assert all(len(values) == 2 for values in result.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, probe.parameters())), "Training must update weights."
    assert all(value > 0 for value in result['train_loss'])
