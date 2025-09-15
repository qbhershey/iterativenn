from typing import List

import torch
import iterativenn.nn_modules.MaskedLinear as MaskedLinear
from iterativenn.utils.gpt_config_utils import sparse_sequential2D_from_masked_linear

class HadamardIdentity(torch.nn.Module):
    def __init__(self, in_features: int, bias: bool = True, device=None, dtype=None) -> None:
        """Compute the Hadamard Identity.

        Args:
            in_features (int): number of features
            bias (bool, optional): keep bias. Defaults to True.
            device ([type], optional): cpu/gpu. Defaults to None.
        """
        super(HadamardIdentity, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.b1 = torch.nn.Parameter(torch.zeros(in_features, device=device, dtype=dtype)) if bias else None
        self.masked_linear1 = sparse_sequential2D_from_masked_linear(dim_z=in_features, dim_out=in_features, hidden_dim=30)
        self.b2 = torch.nn.Parameter(torch.zeros(in_features, device=device, dtype=dtype)) if bias else None
        self.masked_linear2 = sparse_sequential2D_from_masked_linear(dim_z=in_features, dim_out=in_features, hidden_dim=30)
        torch.nn.init.ones_(self.b2)
        torch.nn.init.zeros_(self.b1)

    def forward(self, x):
        output1 = self.masked_linear1(x) + self.b1
        output2 = self.masked_linear2(x) + self.b2
        return output1 * output2

    def number_of_trainable_parameters(self):
        return (self.masked_linear2.number_of_trainable_parameters() + self.masked_linear1.number_of_trainable_parameters() + self.b2.numel() + self.b1.numel())


