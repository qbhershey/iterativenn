""""

# From https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2

date: Sep 2021
author: rcpaffenroth, hpathak

"""
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def trivial_collate_fn(batch):
    return batch

def fetch_data(parameters):
    data_dir = '/opt/ml/data/mnist/'

    # From https://github.com/pytorch/examples/blob/master/mnist/main.py
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNIST(data_dir, download=True, transform=transform)
    MNIST_train, MNIST_val, _ = random_split(dataset, [parameters['train_size'], parameters['test_size'],
                                                       60000 - (parameters['train_size'] + parameters['test_size'])])
    return MNIST_train, MNIST_val

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()