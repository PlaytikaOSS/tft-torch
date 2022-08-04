import torch
from torch import nn


class TimeDistributed(nn.Module):
    """
    This module can wrap any given module and stacks the time dimension with the batch dimension of the inputs
    before applying the module.
    Borrowed from this fruitful `discussion thread
    <https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4>`_.

    Parameters
    ----------
    module : nn.Module
        The wrapped module.
    batch_first: bool
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    return_reshaped: bool
        A boolean indicating whether to return the output in the corresponding original shape or not.
    """

    def __init__(self, module: nn.Module, batch_first: bool = True, return_reshaped: bool = True):
        super(TimeDistributed, self).__init__()
        self.module: nn.Module = module  # the wrapped module
        self.batch_first: bool = batch_first  # indicates the dimensions order of the sequential data.
        self.return_reshaped: bool = return_reshaped

    def forward(self, x):

        # in case the incoming tensor is a two-dimensional tensor - infer no temporal information is involved,
        # and simply apply the module
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and time-steps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * time-steps, input_size)
        # apply the module on each time-step separately
        y = self.module(x_reshape)

        # reshaping the module output as sequential tensor (if required)
        if self.return_reshaped:
            if self.batch_first:
                y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, time-steps, output_size)
            else:
                y = y.view(-1, x.size(1), y.size(-1))  # (time-steps, samples, output_size)

        return y


class NullTransform(nn.Module):
    def __init__(self):
        super(NullTransform, self).__init__()

    @staticmethod
    def forward(empty_input: torch.tensor):
        return []
