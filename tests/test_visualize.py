import pytest
import numpy as np
import pandas as pd
from tft_torch import visualize


def test_weights_agg():
    """ Test the aggregate_weights function"""
    feat_names = ['A', 'B', 'C', 'D', 'E', 'F']
    num_attributes = len(feat_names)
    prctiles = [25, 50, 75]
    num_samples = 100

    # non-temporal channel
    weights = np.random.random((num_samples, num_attributes))
    agg = visualize.aggregate_weights(output_arr=weights,
                                      prctiles=prctiles,
                                      feat_names=feat_names)

    assert isinstance(agg, pd.DataFrame)
    assert all([a == b for a, b in zip(agg.shape, [num_attributes, len(prctiles)])])
    assert all([a == b for a, b in zip(agg.index.values.tolist(), feat_names)])
    assert all([a == b for a, b in zip(list(agg.columns), prctiles)])
