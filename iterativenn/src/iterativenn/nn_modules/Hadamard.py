from typing import List

import torch


class Hadamard(torch.nn.Module):
    def __init__(self, modules: List):
        """Compute the Hadamard product of a list of modules.

        Args:
            modules (List): a list of modules to be multiplied together.
        """
        super(Hadamard, self).__init__()
        self.modules = torch.ModuleList(modules)

    def forward(self, x):
        output = torch.ones_like(x)
        for module in self.modules:
            output *= module(x)
        return output

