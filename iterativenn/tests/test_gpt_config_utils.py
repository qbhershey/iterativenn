
import torch

from iterativenn.utils.gpt_config_utils import dense_sequential2D, sparse_sequential2D


def test_sparse_sequential2D():
    # Define input and hidden dimensions
    dim = 100
    hidden_dim = 10

    # Create the sparse_sequential2D module
    seq2d_module = sparse_sequential2D(dim, hidden_dim=hidden_dim)

    # Test input/output dimensions
    input_tensor = torch.randn(1, dim)
    output_tensor = seq2d_module(input_tensor)
    assert input_tensor.shape == output_tensor.shape, "Input and output dimensions do not match."

    # Test forward pass
    assert torch.is_tensor(output_tensor), "Output is a torch tensor."

def test_dense_sequential2D():
    # Define input and hidden dimensions
    dim = 100
    hidden_dim = 10

    # Create the sparse_sequential2D module
    seq2d_module = dense_sequential2D(dim, hidden_dim=hidden_dim)

    # Test input/output dimensions
    input_tensor = torch.randn(1, dim)
    output_tensor = seq2d_module(input_tensor)
    assert input_tensor.shape == output_tensor.shape, "Input and output dimensions do not match."

    # Test forward pass
    assert torch.is_tensor(output_tensor), "Output is a torch tensor."

