from typing import List

import torch


class HeterogeneousActivation(torch.nn.Module):
    def __init__(self, activations: List, sizes: List):
        super(HeterogeneousActivation, self).__init__()
        self.activations = activations
        self.sizes = sizes

    def forward(self, x):
        start=0
        activated = []
        for activation, size in zip(self.activations, self.sizes):
            end = start + size
            activated.append(activation(x[:, start:end]))
            start = end
        # Note, I do it this way to avoid issues with gradients of inplace modified tensors
        x = torch.cat(activated, dim=1)
        return x

    def add_activation_block(self, activation_fn: torch.nn.Module, block_size: int):
        self.activations.append(activation_fn)
        self.sizes.append(block_size)
