"""Pure-math regression tests for the pinball (quantile) loss in probabilistic.py --
deterministic, no trained checkpoint or dataset needed, so these run in a few
milliseconds in CI on every push.

The tau=0.5 case is checked against the textbook fact that pinball loss at the median
quantile reduces exactly to (half of) mean absolute error -- not just "some loss went
down during training", which is a much weaker and less useful thing to assert.
"""
import torch

from forecasting.probabilistic import pinball_loss, QUANTILES


def test_pinball_loss_tau_half_equals_half_mae():
    preds = torch.tensor([[10.0, 10.0, 10.0], [4.0, 4.0, 4.0]])
    target = torch.tensor([12.0, 4.0])  # errors: +2 (under-predicted), 0 (exact)

    # Isolate tau=0.5 (index 1 of QUANTILES = (0.1, 0.5, 0.9)) by comparing against a
    # loss computed with only that one quantile.
    tau_half_idx = QUANTILES.index(0.5)
    single_quantile_preds = preds[:, tau_half_idx : tau_half_idx + 1]
    loss = pinball_loss(single_quantile_preds, target, quantiles=(0.5,))

    mae = torch.mean(torch.abs(target - single_quantile_preds.squeeze(-1)))
    assert torch.isclose(loss, 0.5 * mae, atol=1e-6)


def test_pinball_loss_hand_computed_value():
    """Reproduces the exact hand-computed case verified earlier in this project (a single
    tau=0.1 prediction of 10 against an actual of 12): underestimating (actual > pred)
    costs tau * error = 0.1 * 2 = 0.2; the mean over this one sample is 0.2 itself."""
    preds = torch.tensor([[10.0]])
    target = torch.tensor([12.0])
    loss = pinball_loss(preds, target, quantiles=(0.1,))
    assert torch.isclose(loss, torch.tensor(0.2), atol=1e-6)


def test_pinball_loss_is_zero_for_perfect_prediction():
    preds = torch.tensor([[5.0, 5.0, 5.0]])
    target = torch.tensor([5.0])
    loss = pinball_loss(preds, target, quantiles=QUANTILES)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)