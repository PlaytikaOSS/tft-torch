from typing import Dict, List, Union, Callable, Optional
from IPython.display import display
import numpy as np
import pandas as pd
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True,
                 'figure.figsize': [10, 5],
                 'font.size': 17})
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def aggregate_weights(output_arr: np.ndarray,
                      prctiles: List[float],
                      feat_names: List[str]) -> pd.DataFrame:
    """
    Implements a utility function for aggregating selection weights for a set (array) of observations,
    whether these selection weights are associated with the static input attributes, or with a set of temporal selection
    weights.
    The aggregation of the weights is performed through the computation of several percentiles (provided by the caller)
    for describing the distribution of the weights, for each attribute.

    Parameters
    ----------
    output_arr: np.ndarray
        A 2D or 3D array containing the selection weights output by the model. A 3D tensor will imply selection
        weights associated with temporal inputs.
    prctiles: List[float]
        A list of percentiles according to which the distribution of selection weights will be described.
    feat_names: List[str]
        A list of strings associated with the relevant attributes (according to the their order).

    Returns
    -------
    agg_df: pd.DataFrame
        A pandas dataframe, indexed with the relevant feature names, containing the aggregation of selection weights.

    """

    prctiles_agg = []  # a list to contain the computation for each percentile
    for q in prctiles:  # for each of the provided percentile
        # infer whether the provided weights are associated with a temporal input channel
        if len(output_arr.shape) > 2:
            # lose the temporal dimension and then describe the distribution of weights
            flatten_time = output_arr.reshape(-1, output_arr.shape[-1])
        else:  # if static - take as is
            flatten_time = output_arr
        # accumulate
        prctiles_agg.append(np.percentile(flatten_time, q=q, axis=0))

    # combine the computations and index according to feature names
    agg_df = pd.DataFrame({prctile: aggs for prctile, aggs in zip(prctiles, prctiles_agg)})
    agg_df.index = feat_names

    return agg_df


def display_selection_weights_stats(outputs_dict: Dict[str, np.ndarray],
                                    prctiles: List[float],
                                    mapping: Dict,
                                    sort_by: Optional[float] = None,
                                    ):
    """
    Implements a utility function for displaying the selection weights statistics of multiple input channels according
    to the outputs provided by the model for a set of input observations.
    It requires a mapping which specifies which output key corresponds to each input channel, and the associated list
    of attributes.

    Parameters
    ----------
    outputs_dict: Dict[str,np.ndarray]
        A dictionary of numpy arrays containing the outputs of the model for a set of observations.
    prctiles: List[float]
        A list of percentiles according to which the distribution of selection weights will be described.
    mapping: Dict
        A dictionary specifying the output key corresponding to which input channel and the associated feature names.
    sort_by: Optional[float]
        The percentile according to which the weights statistics will be sorted before displaying (Must be included as
        part of ``prctiles``).
    """

    # if ``sort_by`` was not provided, the first percentile provided will be used for sorting.
    if not sort_by:
        sort_by = prctiles[0]
    else:
        # make sure the percentile according to which we wish to sort is included in the set of percentiles to compute.
        assert sort_by in prctiles, "Cannot sort by a percentile which was not listed"

    # for each input channel included in the mapping
    for name, config in mapping.items():
        # perform weight aggregation according to the provided configuration
        weights_agg = aggregate_weights(output_arr=outputs_dict[config['arr_key']],
                                        prctiles=prctiles,
                                        feat_names=config['feat_names'])
        print(name)
        print('=========')
        # display the computed statistics, sorted, and color highlighted according to the value.
        display(weights_agg.sort_values([sort_by], ascending=False).style.background_gradient(cmap='viridis'))


def display_attention_scores(attention_scores: np.ndarray,
                             horizons: Union[int, List[int]],
                             prctiles: Union[float, List[float]],
                             unit: Optional[str] = 'Units'):
    """
    Implements a utility function for displaying the statistics of attention scores according
    to the outputs provided by the model for a set of input observations.
    The statistics of the scores will be described using specified percentiles, and for specified horizons.

    Parameters
    ----------
    attention_scores: np.ndarray
        A numpy array containing the attention scores for the relevant dataset.
    horizons: Union[int, List[int]]
        A list or a single horizon, specified in time-steps units, for which the statistics will be computed.
        If more than one horizon was configured, then only a single percentile computation will be allowed.
    prctiles: Union[int, List[int]]
        A list or a single percentile to compute as a distribution describer for the scores.
        If more than percentile was configured, then only a single horizon will be allowed.
    unit: Optional[str]
        The units associated with the time-steps. This variable is used for labeling the corresponding axes.
    """

    # if any of ``horizons`` or ``prctiles`` is provided as int, transform into a list.
    if not isinstance(horizons, list):
        horizons = [horizons]
    if not isinstance(prctiles, list):
        prctiles = [prctiles]

    # make sure only maximum one of ``horizons`` and ``prctiles`` has more than one element.
    assert len(prctiles) == 1 or len(horizons) == 1

    # compute the configured percentiles of the attention scores, for each percentile separately
    attn_stats = {}
    for prctile in prctiles:
        attn_stats[prctile] = np.percentile(attention_scores, q=prctile, axis=0)

    fig, ax = plt.subplots(figsize=(20, 5))

    if len(prctiles) == 1:  # in case only a single percentile was configured
        relevant_prctile = prctiles[0]
        title = f"Multi-Step - Attention ({relevant_prctile}% Percentile)"
        scores_percentile = attn_stats[relevant_prctile]
        for horizon in horizons:  # a single line for each horizon
            # infer the corresponding x_axis according to the shape of the scores array
            siz = scores_percentile.shape
            x_axis = np.arange(siz[0] - siz[1], siz[0])
            ax.plot(x_axis, scores_percentile[horizon - 1], lw=1, label=f"t + {horizon} scores", marker='o')

    else:
        title = f"{horizons[0]} Steps Ahead - Attention Scores"
        for prctile, scores_percentile in attn_stats.items():  # for each percentile
            # infer the corresponding x_axis according to the shape of the scores array
            siz = scores_percentile.shape
            x_axis = np.arange(siz[0] - siz[1], siz[0])
            ax.plot(x_axis, scores_percentile[0], lw=1, label=f"{prctile}%", marker='o')

    ax.axvline(x=0, lw=1, color='r', linestyle='--')
    ax.grid(True)
    ax.set_xlabel(f"Relative Time-step [{unit}]")
    ax.set_ylabel('Attention Scores')
    ax.set_title(title)
    ax.legend()
    plt.show()


def display_target_trajectory(signal_history: np.ndarray,
                              signal_future: np.ndarray,
                              model_preds: np.ndarray,
                              observation_index: int,
                              model_quantiles: List[float],
                              transformation: Optional[Callable] = None,
                              unit: Optional[str] = 'Units'):
    """
    Implements a utility function for displaying, on a single observation level,
    the historical trajectory of the target variable, together with its future values, and the corresponding quantiles
    predicted by the model for each future time-step.
    In some cases, the target signal is transformed prior to training. For such cases, in order to allow visualization
    in the original scale of the target signal, a transformation function can be provided, for transforming the provided
    signals to the original scale.

    Parameters
    ----------
    signal_history: np.ndarray
        A numpy array containing the historical values of the target signal.
    signal_future: np.ndarray
        A numpy array containing the future values of the target signal (corresponding to the "labels").
    model_preds: np.ndarray
        A numpy array containing the predicted values by the model for each quantile.
    observation_index: int
        The index corresponding to the observation for which the visualization will be generated.
    model_quantiles: List[float]
        The list of quantiles configured in the trained model associated with the provided predictions.
    transformation: Callable
        If provided, the function will be treated as a transformation, that will be applied to the provided signal
        before creating the visualization.
    unit: Optional[str]
        The units associated with the time-steps. This variable is used for labeling the corresponding axes.
    """

    # take the relevant record from each of the provided arrays, using the specified index
    past = signal_history[observation_index, ...]
    future = signal_future[observation_index, ...]
    preds = model_preds[observation_index, ...]

    # infer the length of the history window, and the maximal horizon
    win_len = past.shape[0]
    max_horizon = future.shape[0]

    # if the transformation was not configured, use an identity transformation
    if transformation is None:
        transformation = lambda x: x

    fig, ax = plt.subplots(figsize=(20, 5))

    # generate temporal axes, according to which the signals will be displayed
    past_x = np.arange(1 - win_len, 1)  # historical x_axis steps
    fut_x = np.arange(1, max_horizon + 1)  # futuristic x_axis steps

    ax.plot(past_x, transformation(past[np.newaxis, ...]).reshape(-1), lw=3, label='observed', marker='o')
    ax.plot(fut_x, transformation(future[np.newaxis, ...]).reshape(-1), lw=3, label='target', marker='o')

    # for each predicted quantile, plot the quantile prediction
    for idx, quantile in enumerate(model_quantiles):
        ax.plot(fut_x, transformation(preds[np.newaxis, ..., idx]).reshape(-1), linestyle='--', lw=2, marker='s',
                label=f"predQ={quantile}")
    # display a sleeve between the lowest and highest quantiles, assuming ordered.
    ax.fill_between(fut_x,
                    transformation(preds[np.newaxis, ..., 0]).reshape(-1),
                    transformation(preds[np.newaxis, ..., -1]).reshape(-1),
                    color='gray', alpha=0.3, label=None)
    # add a line at time 0
    ax.axvline(x=0.5, linestyle='--', lw=3, label=None, color='k')
    ax.grid(True)
    ax.set_xlabel(f"Relative Time-Step [{unit}]")
    ax.set_ylabel('Target Variable')
    ax.legend()
    plt.show()


def display_sample_wise_attention_scores(attention_scores: np.ndarray,
                                         observation_index: int,
                                         horizons: Union[int, List[int]],
                                         unit: Optional[str] = None):
    """
    Implements a utility function for displaying, on a single observation level,
    the attention scores output by the model, for, possibly, a multitude of horizons.

    Parameters
    ----------
    attention_scores: np.ndarray
        A numpy array containing the attention scores for the relevant dataset.
    observation_index: int
        The index with the dataset, corresponding to the observation for which the visualization will be generated.
    horizons: Union[int, List[int]]
        A list or a single horizon, specified in time-steps units, for which the scores will be displayed.
    unit:Optional[str]
        The units associated with the time-steps. This variable is used for labeling the corresponding axes.
    """
    # if ``horizons`` is provided as int, transform into a list.
    if isinstance(horizons, int):
        horizons = [horizons]

    # take the relevant record from  the provided array, using the specified index
    sample_attn_scores = attention_scores[observation_index, ...]

    fig, ax = plt.subplots(figsize=(25, 10))

    # infer the corresponding x_axis according to the shape of the scores array
    attn_shape = sample_attn_scores.shape
    x_axis = np.arange(attn_shape[0] - attn_shape[1], attn_shape[0])

    # for each horizon, plot the associated attention score signal for all the steps
    for step in horizons:
        ax.plot(x_axis, sample_attn_scores[step - 1], marker='o', lw=3, label=f"t+{step}")

    ax.axvline(x=-0.5, lw=1, color='k', linestyle='--')
    ax.grid(True)
    ax.legend()

    ax.set_xlabel('Relative Time-Step ' + (f"[{unit}]" if unit else ""))
    ax.set_ylabel('Attention Score')
    ax.set_title('Attention Mechanism Scores - Per Horizon')
    plt.show()


def display_sample_wise_selection_stats(weights_arr: np.ndarray,
                                        observation_index: int,
                                        feature_names: List[str],
                                        top_n: Optional[int] = None,
                                        title: Optional[str] = '',
                                        historical: Optional[bool] = True,
                                        rank_stepwise: Optional[bool] = False):
    """
    Implements a utility function for displaying, on a single observation level, the selection weights output by the
    model. This function can handle selection weights of both temporal input channels and static input channels.


    Parameters
    ----------
    weights_arr: np.ndarray
        A 2D or 3D array containing the selection weights output by the model. A 3D tensor will implies selection
        weights associated with temporal inputs.
    observation_index: int
        The index with the dataset, corresponding to the observation for which the visualization will be generated.
    feature_names: List[str]
        A list of strings associated with the relevant attributes (according to the their order).
    top_n: Optional[int]
        An integer specifying the quantity of the top weighted features to display.
    title: Optional[str]
        A string which will be used when creating the title for the visualization.
    historical: Optional[bool]
        Specifies whether the corresponding input channel contains historical data or future data. Relevant only for
        temporal input channels, and used for display purposes.
    rank_stepwise: Optional[bool]
        Specifies whether to rank the features according to their weights, on each time-step separately, or simply
        display the raw selection weights output by the model. Relevant only for
        temporal input channels, and used for display purposes.
    """

    # a-priori assume non-temporal input channel
    num_temporal_steps = None

    # infer number of attributes according to the shape of the weights array
    weights_shape = weights_arr.shape
    num_features = weights_shape[-1]
    # infer whether the input channel is temporal or not
    is_temporal: bool = len(weights_shape) > 2

    # bound maximal number of features to display by the total amount of features available (in case provided)
    top_n = min(num_features, top_n) if top_n else num_features

    # take the relevant record from  the provided array, using the specified index
    sample_weights = weights_arr[observation_index, ...]

    if is_temporal:
        # infer number of temporal steps
        num_temporal_steps = weights_shape[1]
        # aggregate the weights (by averaging) across all the time-steps
        sample_weights_trans = sample_weights.T
        weights_df = pd.DataFrame({'weight': sample_weights_trans.mean(axis=1)}, index=feature_names)
    else:
        # in case the input channel is not temporal, just use the weights as is
        weights_df = pd.DataFrame({'weight': sample_weights}, index=feature_names)

    # ========================
    # Aggregative Barplot
    # ========================
    fig, ax = plt.subplots(figsize=(20, 10))
    weights_df.sort_values('weight', ascending=False).iloc[:top_n].plot.bar(ax=ax)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(11)
        tick.label.set_rotation(45)

    ax.grid(True)
    ax.set_xlabel('Feature Name')
    ax.set_ylabel('Selection Weight')
    ax.set_title(title + (" - " if title != "" else "") + \
                 f"Selection Weights " + ("Aggregation " if is_temporal else "") + \
                 (f"- Top {top_n}" if top_n < num_features else ""))
    plt.show()

    if is_temporal:
        # ========================
        # Temporal Display
        # ========================
        # infer the order of the features, according to the average selection weight across time
        order = sample_weights_trans.mean(axis=1).argsort()[::-1]

        # order the weights sequences as well as their names accordingly
        ordered_weights = sample_weights_trans[order]
        ordered_names = [feature_names[i] for i in order.tolist()]

        if rank_stepwise:
            # the weights are now considered to be the ranking after ordering the features in each time-step separately
            ordered_weights = np.argsort(ordered_weights, axis=0)

        fig, ax = plt.subplots(figsize=(30, 20))

        # create a corresponding x-axis, going forward/backwards, depending on the configuration
        if historical:
            map_x = {idx: val for idx, val in enumerate(np.arange(- num_temporal_steps, 1))}
        else:
            map_x = {idx: val for idx, val in enumerate(np.arange(1, num_temporal_steps + 1))}

        def format_fn(tick_val, tick_pos):
            if int(tick_val) in map_x:
                return map_x[int(tick_val)]
            else:
                return ''

        # display the weights as images
        im = ax.pcolor(ordered_weights, edgecolors='gray', linewidths=2)
        # feature names displayed to the left
        ax.yaxis.set_ticks(np.arange(len(ordered_names)))
        ax.set_yticklabels(ordered_names)

        ax2 = ax.twiny()
        ax2.set_xticks([])
        ax2.xaxis.set_ticks_position('top')
        ax.set_xlabel(('Historical' if historical else 'Future') + ' Time-Steps')
        ax2.set_xlabel(('Historical' if historical else 'Future') + ' Time-Steps')

        ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
        fig.colorbar(im, orientation="horizontal", pad=0.05, ax=ax2)
        plt.show()
