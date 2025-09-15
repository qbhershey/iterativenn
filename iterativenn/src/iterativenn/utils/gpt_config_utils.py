from typing import List

import omegaconf

from iterativenn.nn_modules.Sequential2D import Sequential2D
from iterativenn.nn_modules.nlp import GPT2Block, GPT2LMHead


def sparse_sequential2D_from_masked_linear(dim_z, dim_out=None, hidden_dim=30):
    """
       Args:
           dim_z: total dimension this function mapping inputs
           dim_out: dim of output and if not provided this will be set to dim_z
           hidden_dim: hidden dimension

       Returns:
           A Sparse Sequential2D Module made using config and Masked Linear

    """
    n = 3
    assert (n * hidden_dim) < dim_z, f"negative dim error {dim_z, hidden_dim}"
    if dim_out is None:
        dim_out = dim_z

    #with initialize(version_base=None, config_path="../../../scripts/conf/model", job_name="test_app"):

    cfg = omegaconf.OmegaConf.load("../conf/model/sequential2D_sparse.yaml") # local
    # cfg = omegaconf.OmegaConf.load("/home/hnpatha/dev/iterativenn/scripts/conf/model/sequential2D_sparse.yaml")
    # cfg = compose(config_name="sequential2D_sparse.yaml")
    cfg.in_features_list = [hidden_dim, hidden_dim, hidden_dim, dim_z - (n * hidden_dim)]
    cfg.out_features_list = [hidden_dim, hidden_dim, hidden_dim, dim_out - (n * hidden_dim)]
    seq2d_module = Sequential2D.from_config(cfg)
    return seq2d_module



def sparse_sequential2D(dim_z, dim_out=None, hidden_dim=30):
    """
    Args:
        dim_z: total dimension this function mapping inputs
        dim_out: dim of output and if not provided this will be set to dim_z
        hidden_dim: hidden dimension

    Returns:
        A Sparse Sequential2D Module made using config

    """
    n = 2
    assert (n * hidden_dim) < dim_z, f"negative dim error {dim_z, hidden_dim}"
    if dim_out is None:
        dim_out = dim_z
    cfg = {
            "in_features_list": [hidden_dim, hidden_dim, dim_z - (n * hidden_dim)],
            "out_features_list": [hidden_dim, hidden_dim, dim_out - (n * hidden_dim)],
            "block_types": [
                ['Linear', 'Linear', None],
                ['Linear', None, None],
                [None, None, 'Linear'],
            ],
            "block_kwargs": [
                [None, None, None],
                [None, None, None],
                [None, None, None],
            ]
    }
    seq2d_module = Sequential2D.from_config(cfg)
    return seq2d_module


def dense_sequential2D(dim_z, dim_out=None, hidden_dim=30):
    """

    Args:
        dim_z: total dimension this function mapping inputs
        dim_out: dim of output and if not provided this will be set to dim_z
        hidden_dim: hidden dimension

    Returns:
        A Dense Sequential2D Module made using config

    """
    n = 2
    assert (n * hidden_dim) < dim_z, f"negative dim error {dim_z, hidden_dim}"
    if dim_out is None:
        dim_out = dim_z
    cfg = {
            "in_features_list": [hidden_dim, hidden_dim, dim_z - (n * hidden_dim)],
            "out_features_list": [hidden_dim, hidden_dim, dim_out - (n * hidden_dim)],
            "block_types": [
                ['Linear', 'Linear', 'Linear'],
                ['Linear', 'Linear', 'Linear'],
                ['Linear', 'Linear', 'Linear'],
            ],
            "block_kwargs": [
                [None, None, None],
                [None, None, None],
                [None, None, None],
            ]
    }
    seq2d_module = Sequential2D.from_config(cfg)
    return seq2d_module

class GPT2BlockUpdater:
    """
    This class is used to update the GPT2 blocks in the config
    """
    def __init__(self, size: int = 12):
        self.size = size
        self.block_types = [[None for _ in range(size+1)] for _ in range(size+1)]
        self.block_kwargs = [[None for _ in range(size+1)] for _ in range(size+1)]
        self.lr = 1e-6
        self.blocks_lr = [[self.lr for _ in range(size + 1)] for _ in range(size + 1)]

    def get_block_types(self) -> List:
        return self.block_types

    def get_block_kwargs(self) -> List:
        return self.block_kwargs

    def get_block_type(self, i: int, j: int) -> str:
        return self.block_types[i][j]

    def get_block_kwarg(self, i: int, j: int) -> dict:
        return self.block_kwargs[i][j]

    def set_block_types(self, value: List):
        self.block_types = value

    def set_block_kwargs(self, value: List):
        self.block_kwargs = value

    def update_block_types(self, i: int, j: int, value: str):
        self.block_types[i][j] = value

    def update_block_kwargs(self, i: int, j: int, value: dict):
        self.block_kwargs[i][j] = value

    def init_gpt2_blocks(self, mode=None, start=0):
        """
        Updates the blocks next to the diagonol i.e. j-i == 1 to GPT decoder blocks
        Example: (1, 2), (2, 3), (6, 7) etc will be set to GPT2 decoder blocks
        what is the difference between GPT2Block and GPT2LMHead?
        - GPT2Block is the decoder block, it has a self attention layer and a feed forward layer.
        - GPT2LMHead is the decoder block with a linear layer on top
        - trainables are set to True/False for all blocks.
        """
        for i in range(start, self.size+1):
            for j in range(start, self.size+1):
                if j-i == 1:
                    self.block_types[i][j] = 'Module'
                    if i != self.size-1:
                        self.block_kwargs[i][j] = {'module': GPT2Block.from_pretrained(block_idx=i), 'trainable': True}
                    else:
                        if mode == 'block_predict':
                            self.block_kwargs[i][j] = {'module': GPT2Block.from_pretrained(block_idx=i), 'trainable': True}
                        else:
                            self.block_kwargs[i][j] = {'module': GPT2LMHead.from_pretrained(), 'trainable': True}


    def init_gpt2_blocks_insert_hidentity(self, operator, insert_pos: tuple = (0, 1)):
        """
        Updates the blocks next to the diagonol i.e. j-i == 1 to GPT decoder blocks
        Example: (1, 2), (2, 3), (6, 7) etc will be set to GPT2 decoder blocks
        What is hidentity?
        - hidentity is a module that does element wise multiplication of the input with a trainable parameter
        - hidentity is used to insert a new block in the middle of the network
        what happens to other blocks after insertion?
        - the blocks after insertion are shifted to down by 1 in lower subdiagonal
        """
        from iterativenn.nn_modules.h_identity import HadamardIdentity
        counter = 0
        for i in range(self.size+1):
            for j in range(self.size+1):
                if j-i == 1:
                    self.block_types[i][j] = 'Module'
                    if i != self.size-1:
                        if (i, j) == insert_pos:
                            self.block_kwargs[i][j] = {'module': operator}
                            print(f"inserted {i, j}")
                        else:
                            self.block_kwargs[i][j] = {'module': GPT2Block.from_pretrained(block_idx=counter), 'trainable': True}
                            counter += 1
                    else:
                        self.block_kwargs[i][j] = {'module': GPT2LMHead.from_pretrained(), 'trainable': True}

    def get_block_lr(self) -> List:
        return self.blocks_lr

    def update_block_lrs(self, i: int, j: int, value: float):
        self.blocks_lr[i][j] = value





if __name__ == '__main__':

    seq2d = sparse_sequential2D_from_masked_linear(100)
    print(seq2d)