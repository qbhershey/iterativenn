from typing import Optional

import torch
from torch import Tensor, nn


class AdjustedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Normalizes cross entropy loss by number of parameters in the network"""
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None,
                 reduction: str = 'mean', label_smoothing: float = 0.0, n_vars: int = 0, batch_size: int = 0,
                 data_size: int = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self.n_vars = n_vars

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.n_vars == 0:
            return super().forward(input, target)
        else:
            out = super().forward(input, target)
            out = torch.div(out, self.n_vars)
            return out


class AdjustedMSELoss(nn.MSELoss):
    """Normalizes mse loss by number of parameters in the network"""
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', n_vars: int = 0,
                 batch_size: int = 0, data_size: int = 0) -> None:
        super().__init__(size_average, reduce, reduction)
        self.n_vars = n_vars

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.n_vars == 0:
            return super().forward(input, target)
        else:
            out = super().forward(input, target)
            out = torch.div(out, self.n_vars)
            return out



