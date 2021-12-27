import pytest
import torch
import torch.nn as nn
from tft_torch import base_blocks


def test_time_distribution_2d_input():
    """ Test the TimeDistributed module"""

    batch_size = 100
    input_dim = 20
    output_dim = 5

    base_module = nn.Linear(input_dim, output_dim)
    wrapped_module = base_blocks.TimeDistributed(base_module, batch_first=True, return_reshaped=True)

    # 2d input
    input_2d = torch.rand(batch_size, input_dim, dtype=torch.float32)
    # should act the same even when wrapped
    assert torch.equal(base_module(input_2d), wrapped_module(input_2d))


def test_time_distribution_temporal_input():
    """ Test the TimeDistributed module"""

    batch_size = 100
    input_dim = 20
    time_steps = 7
    output_dim = 5

    base_module = nn.Linear(input_dim, output_dim)

    # 3d input
    input_3d = torch.rand(batch_size, time_steps, input_dim, dtype=torch.float32)

    not_reshaped = base_blocks.TimeDistributed(base_module, batch_first=True, return_reshaped=False)(input_3d)
    assert len(not_reshaped.shape) == 2
    assert all([a == b for a, b in zip(not_reshaped.shape, [batch_size * time_steps, output_dim])])

    reshaped = base_blocks.TimeDistributed(base_module, batch_first=True, return_reshaped=True)(input_3d)
    assert len(reshaped.shape) == 3
    assert all([a == b for a, b in zip(reshaped.shape, [batch_size, time_steps, output_dim])])
