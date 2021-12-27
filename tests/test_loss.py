import pytest
import torch
from tft_torch import loss as tft_loss


def test_instance_wise_loss_computation():
    """ Test the function compute_quantile_loss_instance_wise()"""

    num_samples = 100
    num_horizons = 10
    num_quantiles = 5

    outputs = torch.rand((num_samples, num_horizons, num_quantiles), dtype=torch.float32)
    targets = torch.rand(num_samples, num_horizons, dtype=torch.float32)
    desired_quantiles, _ = torch.rand(num_quantiles, dtype=torch.float32).sort()

    losses_array = tft_loss.compute_quantile_loss_instance_wise(outputs=outputs,
                                                                targets=targets,
                                                                desired_quantiles=desired_quantiles)

    assert all([a == b for a, b in zip(losses_array.shape, [num_samples, num_horizons, num_quantiles])])


def test_qloss_and_qrisk_computation():
    """ Test the function get_quantiles_loss_and_q_risk()"""

    num_samples = 100
    num_horizons = 10
    num_quantiles = 5

    outputs = torch.rand((num_samples, num_horizons, num_quantiles), dtype=torch.float32)
    targets = torch.rand(num_samples, num_horizons, dtype=torch.float32)
    desired_quantiles, _ = torch.rand(num_quantiles, dtype=torch.float32).sort()

    result = tft_loss.get_quantiles_loss_and_q_risk(outputs=outputs,
                                                    targets=targets,
                                                    desired_quantiles=desired_quantiles)

    assert isinstance(result, tuple)
    assert len(result) == 3

    q_loss, q_risk, losses_array = result

    assert len(q_loss.shape) == 0
    assert len(q_risk.shape) == 1 and q_risk.shape[0] == num_quantiles
    assert all([a == b for a, b in zip(losses_array.shape, [num_samples, num_horizons, num_quantiles])])
