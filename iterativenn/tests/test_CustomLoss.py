import pytest
import torch

from iterativenn.nn_modules.CustomLoss import AdjustedCrossEntropyLoss, AdjustedMSELoss


@pytest.fixture
def input_and_target():
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)
    target = torch.tensor([0, 1, 2], dtype=torch.long)
    return input, target


def test_cross_entropy_loss(input_and_target):
    criterion = AdjustedCrossEntropyLoss(n_vars=10)
    input, target = input_and_target
    loss = criterion(input, target)

    expected_loss = 1.407 / 10
    print(f"input: {input}")
    print(f"target: {target}")
    print(f"computed loss: {loss}")
    print(f"expected loss: {expected_loss}")
    torch.testing.assert_allclose(loss, expected_loss, rtol=1e-3, atol=1e-3)


def test_mse_loss(input_and_target):
    criterion = AdjustedMSELoss(n_vars=10)
    input, target = input_and_target
    loss = criterion(input, target)

    expected_loss = 22 / 10
    print(f"input: {input}")
    print(f"target: {target}")
    print(f"computed loss: {loss}")
    print(f"expected loss: {expected_loss}")
    torch.testing.assert_allclose(loss, expected_loss, rtol=1e-6, atol=1e-6)
