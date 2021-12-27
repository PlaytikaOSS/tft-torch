import pytest
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tft_torch import tft


def test_tft():
    """ Test the TemporalFusionTransformer module"""
    data_props = {'num_historical_numeric': 4,
                  'num_historical_categorical': 6,
                  'num_static_numeric': 10,
                  'num_static_categorical': 11,
                  'num_future_numeric': 2,
                  'num_future_categorical': 3,
                  'historical_categorical_cardinalities': (1 + np.random.randint(10, size=6)).tolist(),
                  'static_categorical_cardinalities': (1 + np.random.randint(10, size=11)).tolist(),
                  'future_categorical_cardinalities': (1 + np.random.randint(10, size=3)).tolist(),
                  }

    configuration = {
        'model':
            {
                'dropout': 0.05,
                'state_size': 64,
                'output_quantiles': [0.1, 0.5, 0.9],
                'lstm_layers': 2,
                'attention_heads': 4
            },
        # these arguments are related to possible extensions of the model class
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props
    }

    model = tft.TemporalFusionTransformer(OmegaConf.create(configuration))

    # create batch
    batch_size = 256
    historical_steps = 90
    future_steps = 30

    batch = {
        'static_feats_numeric': torch.rand(batch_size, data_props['num_static_numeric'],
                                           dtype=torch.float32),
        'static_feats_categorical': torch.stack([torch.randint(c, size=(batch_size,)) for c in
                                                 data_props['static_categorical_cardinalities']],
                                                dim=-1).type(torch.LongTensor),
        'historical_ts_numeric': torch.rand(batch_size, historical_steps, data_props['num_historical_numeric'],
                                            dtype=torch.float32),
        'historical_ts_categorical': torch.stack([torch.randint(c, size=(batch_size, historical_steps)) for c in
                                                  data_props['historical_categorical_cardinalities']],
                                                 dim=-1).type(torch.LongTensor),
        'future_ts_numeric': torch.rand(batch_size, future_steps, data_props['num_future_numeric'],
                                        dtype=torch.float32),
        'future_ts_categorical': torch.stack([torch.randint(c, size=(batch_size, future_steps)) for c in
                                              data_props['future_categorical_cardinalities']],
                                             dim=-1).type(torch.LongTensor),

    }

    batch_outputs = model(batch)

    assert isinstance(batch_outputs, dict)
    assert all([k in batch_outputs for k in ['predicted_quantiles',
                                             'static_weights',
                                             'historical_selection_weights',
                                             'future_selection_weights',
                                             'attention_scores']])

    num_quantiles = len(configuration['model']['output_quantiles'])
    num_static = data_props['num_static_numeric'] + data_props['num_static_categorical']
    num_historical = data_props['num_historical_numeric'] + data_props['num_historical_categorical']
    num_future = data_props['num_future_numeric'] + data_props['num_future_categorical']

    assert len(batch_outputs['predicted_quantiles'].shape) == 3 and all(
        [a == b for a, b in zip(batch_outputs['predicted_quantiles'].shape,
                                [batch_size, future_steps, num_quantiles])])

    assert len(batch_outputs['static_weights'].shape) == 2 and all(
        [a == b for a, b in zip(batch_outputs['static_weights'].shape,
                                [batch_size, num_static])])

    assert len(batch_outputs['historical_selection_weights'].shape) == 3 and all(
        [a == b for a, b in zip(batch_outputs['historical_selection_weights'].shape,
                                [batch_size, historical_steps, num_historical])])

    assert len(batch_outputs['future_selection_weights'].shape) == 3 and all(
        [a == b for a, b in zip(batch_outputs['future_selection_weights'].shape,
                                [batch_size, future_steps, num_future])])

    assert len(batch_outputs['attention_scores'].shape) == 3 and all(
        [a == b for a, b in zip(batch_outputs['attention_scores'].shape,
                                [batch_size, future_steps, historical_steps + future_steps])])
