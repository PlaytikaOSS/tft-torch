from typing import Tuple
import torch


def compute_quantile_loss_instance_wise(outputs: torch.tensor,
                                        targets: torch.tensor,
                                        desired_quantiles: torch.tensor) -> torch.Tensor:
    """
    This function compute the quantile loss separately for each sample,time-step,quantile.

    Args:
        outputs: the outputs of the model [num_samples x num_horizons x num_quantiles]
        targets: the observed target for each horizon [num_samples x num_horizons]
        desired_quantiles: a tensor representing the desired quantiles, of shape (num_quantiles,)

    Returns:
        losses_array: a tensor [num_samples x num_horizons x num_quantiles] containing the quantile loss for each
            sample,time-step,quantile
    """
    # compute the actual error between the observed target and each predicted quantile
    errors = targets.unsqueeze(-1) - outputs
    # Dimensions:
    # errors: [num_samples x num_horizons x num_quantiles]

    # compute the loss separately for each sample,time-step,quantile
    losses_array = torch.max((desired_quantiles - 1) * errors, desired_quantiles * errors)
    # Dimensions:
    # losses_array: [num_samples x num_horizons x num_quantiles]

    return losses_array


def get_quantiles_loss_and_q_risk(outputs: torch.tensor,
                                  targets: torch.tensor,
                                  desired_quantiles: torch.tensor) -> Tuple[torch.tensor, ...]:
    """
    This function computes quantile loss and q-risk metric.

    Args:
        outputs: the outputs of the model [num_samples x num_horizons x num_quantiles]
        targets: the observed target for each horizon [num_samples x num_horizons]
        desired_quantiles: a tensor representing the desired quantiles, of shape (num_quantiles,)

    Returns:
        q_loss: a scalar representing the quantile loss across all samples,horizons and quantiles
        q_risk: a tensor (shape=(num_quantiles,)) with q-risk metric for each quantile separately
        losses_array: a tensor [num_samples x num_horizons x num_quantiles] containing the quantile loss for each
            sample,time-step,quantile

    """
    losses_array = compute_quantile_loss_instance_wise(outputs=outputs,
                                                       targets=targets,
                                                       desired_quantiles=desired_quantiles)

    # sum losses over quantiles and average across time and observations
    q_loss = (losses_array.sum(dim=-1)).mean(dim=-1).mean()  # a scalar (shapeless tensor)

    # compute q_risk for each quantile
    q_risk = 2 * (losses_array.sum(dim=1).sum(dim=0)) / (targets.abs().sum().unsqueeze(-1))

    return q_loss, q_risk, losses_array
