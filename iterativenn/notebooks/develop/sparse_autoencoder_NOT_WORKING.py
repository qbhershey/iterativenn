# %%
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MNIST autoencoder example.

To run:
python autoencoder.py --trainer.max_epochs=50
"""

# %%

import matplotlib.pylab as py

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from torchvision import transforms
from torchvision.datasets import MNIST


# %%


def createBlockSparseMatrix(row_sizes, col_sizes, block_types, initialization_types):
    """
    Create a block sparse matrix.
    Args:
        rows_sizes (list of int): The number of rows in each block

        cols_sizes (list of int): The number of columns in each block

        block_types (2D-array of dict): A 2D-array that describe each block.
           Each dict has a key named "type" and perhaps other keys to describe the block
           0  : A block of zeros with no gradient.
                This is special in that this block takes up no space or time.
           "W": A dense block with all entries trainable.
           "D": A block with all off-diagonal entries 0 and only the diagonal entries are trainable.
                Note, this is fast and does not *not* touch the whole block.
           "R=0.5": A sparse block with randomly placed 0s with no gradients.
                The probability an entry is trainable is 0.5 in this case.
                Note, this can be slow since a random number is drawn for each entry in the block.
           "S=15": A sparse block with randomly placed 0s with no gradients.
                This draws, for example, 15 random indices in the block and makes those entries trainable.
                Note, this is fast and does not *not* touch the whole block.
                However, you might get less that 15 since if you happen to draw the same indices twice
                then they will get coalesced into a single entry (see https://pytorch.org/docs/stable/sparse.html for details)
           "Row=15": A sparse block with randomly placed 0s with no gradients.
                This draws, for example, 15 random indices *per row* in the block and makes those entries trainable.
                Note, this is fast and does not *not* touch the whole block (though it does touch every row).
                However, you might get less that 15 entries in each row
                since if you happen to draw the same indices twice then they will get
                coalesced into a single entry (see https://pytorch.org/docs/stable/sparse.html for details)

        initialization_types (2D-array of dict): A 2D-array of dicts that describes the initialization of the
           trainable parameters in each block.
           Each dict has a key named "type" and perhaps other keys to describe the block
           0 : Initialization each trainable entry with 0
           1 : Initialization each trainable entry with 1
           "C=0.3": Initialize each trainable entry with the value 0.3
           "G": Initalize each trainable entry with draw from Gaussian mu=0.0,sigma=1.0
           "G=0.2,0.7": Initalize each trainable entry with draw from Gaussian mu=0.2,sigma=0.7
           "U": Initalize each trainable entry with draw from Uniform min=-1.0,max=1.0
           "U=-0.5,0.5": Initalize each trainable entry with draw from Uniform min=-0.5,max=0.5
           T : Initalize with the given tensor.
    """
    row_sizes = torch.tensor(row_sizes)
    col_sizes = torch.tensor(col_sizes)

    total_rows = torch.sum(row_sizes)
    total_cols = torch.sum(col_sizes)
    rows = []
    cols = []
    vals = []

    for current_row in range(len(row_sizes)):
        for current_column in range(len(col_sizes)):
            # These offsets are used below
            row_offset = torch.sum(row_sizes[:current_row])
            col_offset = torch.sum(col_sizes[:current_column])

            # Need an initialization function I can call below to initialize
            initialization_type = initialization_types[current_row][current_column]
            if torch.is_tensor(initialization_type):
                # FIXME:  This is likely very slow.
                def initialize(i, j, W=initialization_type):
                    return W[i, j]
            elif initialization_type == 0:
                def initialize(i, j):
                    return 0.0
            elif initialization_type == 1:
                def initialize(i, j):
                    return 1.0
            elif initialization_type[0] == "C":
                val = float(initialization_type[2:])

                def initialize(i, j, val=val):
                    return val
            elif initialization_type[0] == "G":
                if len(initialization_type[0]) == 1:
                    mu = 0.0
                    sigma = 1.0
                else:
                    mu = float(initialization_type[2:].split(',')[0])
                    sigma = float(initialization_type[2:].split(',')[1])

                def initialize(i, j, mu=mu, sigma=sigma):
                    return np.random.randn()*sigma + mu
            elif initialization_type[0] == "U":
                if len(initialization_type[0]) == 1:
                    min = -1.0
                    max = 1.0
                else:
                    min = float(initialization_type[2:].split(',')[0])
                    max = float(initialization_type[2:].split(',')[1])

                def initialize(i, j, min=min, max=max):
                    return np.random.rand()*(max-min) + min
            else:
                assert False, "Unknown initialization type"

            # The implementations of the various block types
            block_type = block_types[current_row][current_column]
            if block_type == 0:
                pass
            elif block_type[0] == "W":
                for k in range(row_sizes[current_row]):
                    for l in range(col_sizes[current_column]):
                        rows.append(row_offset+k)
                        cols.append(col_offset+l)
                        vals.append(initialize(k, l))
            elif block_type[0] == "D":
                n = torch.min(row_sizes[current_row], col_sizes[current_column])
                for k in range(n):
                    rows.append(row_offset+k)
                    cols.append(col_offset+k)
                    vals.append(initialize(k, k))
            elif block_type[0:3] == "Row":
                n = int(block_type[4:])
                for k in range(row_sizes[current_row]):
                    for l in range(n):
                        v = np.random.randint(0, int(col_sizes[current_column]))
                        rows.append(row_offset+k)
                        cols.append(col_offset+v)
                        vals.append(initialize(k, v))
            elif block_type[0] == "R":
                p = float(block_type[2:])
                for k in range(row_sizes[current_row]):
                    for l in range(col_sizes[current_column]):
                        if torch.rand(1)[0] < p:
                            rows.append(row_offset+k)
                            cols.append(col_offset+l)
                            vals.append(initialize(k, l))
            elif block_type[0] == "S":
                n = int(block_type[2:])
                for k in range(n):
                    u = np.random.randint(0, int(row_sizes[current_row]))
                    v = np.random.randint(0, int(col_sizes[current_column]))
                    rows.append(row_offset+u)
                    cols.append(col_offset+v)
                    vals.append(initialize(u, v))
            else:
                assert False, "Unknown block type"
    return torch.sparse_coo_tensor([rows, cols], vals, [total_rows, total_cols])


# %%
row_sizes = [20, 30, 15]
col_sizes = [40, 50]
block_types = [[0,     'D'],
               ['W',   'R=0.1'],
               ['S=5', 'Row=2']
               ]

T = torch.randn((30, 40))

initializaztion_types = [[0,            'U'],
                         [T,            'G'],
                         ['U=-1.0,1.0', 'C=-1.0']
                         ]
S = createBlockSparseMatrix(row_sizes, col_sizes, block_types, initializaztion_types)

# %%
py.spy(S.to_dense())

# %%
py.imshow(S.to_dense())

# %%


class SparseLinear(nn.Module):
    def __init__(self, S, bias=True):
        super(SparseLinear, self).__init__()
        self.weight = nn.Parameter(S)
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                [1, self.weight.shape[1]], requires_grad=True))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # addm does b + W @ input, so we need to transpose
        # everything to get the neural network standard input @ W

        # FIXME:  This is the sparse version of the affine transformation.
        # You need this if you want to actually use sparse tensors.
        # y = torch.sparse.addmm(torch.transpose(self.bias, 1, 0),
        #                        torch.transpose(self.weight, 1, 0),
        #                        torch.transpose(input, 1, 0))
        # return torch.transpose(y, 1, 0)

        y = torch.addmm(torch.transpose(self.bias, 1, 0),
                        torch.transpose(self.weight, 1, 0),
                        torch.transpose(input, 1, 0))
        return torch.transpose(y, 1, 0)

# %%


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # FIXME:  This is where you can insert the "sparse" weights, though they
        # are, in this case, actually dense (purely for testing).
        # W1 = createBlockSparseMatrix([28*28], [hidden_dim], [['W']], [['G=0.0,0.1']])
        # W2 = createBlockSparseMatrix([hidden_dim], [3], [['W']], [['G=0.0,0.1']])
        # W3 = createBlockSparseMatrix([3], [hidden_dim], [['W']], [['G=0.0,0.1']])
        # W4 = createBlockSparseMatrix([hidden_dim], [28*28], [['W']], [['G=0.0,0.1']])

        W1 = torch.randn([28*28, hidden_dim], requires_grad=True)*0.1
        W2 = torch.randn([hidden_dim, 3], requires_grad=True)*0.1
        W3 = torch.randn([3, hidden_dim], requires_grad=True)*0.1
        W4 = torch.randn([hidden_dim, 28*28], requires_grad=True)*0.1

        L1 = SparseLinear(W1)
        L2 = SparseLinear(W2)
        L3 = SparseLinear(W3)
        L4 = SparseLinear(W4)

        # The normal pytorch linear layers, just for testing.
        # L1 = nn.Linear(28*28, hidden_dim)
        # L2 = nn.Linear(hidden_dim, 3)
        # L3 = nn.Linear(3, hidden_dim)
        # L4 = nn.Linear(hidden_dim, 28*28)

        self.encoder = nn.Sequential(
            L1,
            nn.ReLU(),
            L2,
        )
        self.decoder = nn.Sequential(
            L3,
            nn.ReLU(),
            L4,
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss, on_step=True)

    def configure_optimizers(self):
        # FIXME:  SGD seems to be the only optimizer that works with sparse tensors, and
        # even it has a issue with momentum.  Straight SGD doe note converge.
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

# %%


class MyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 32,
        train_size: int = 100,
        test_size: int = 100
    ):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNIST('../data/external',
                        train=True,
                        download=True,
                        transform=self.transform)
        self.mnist_test = MNIST('../data/external',
                                train=False,
                                download=True,
                                transform=self.transform)
        self.mnist_train, self.mnist_val, _ = random_split(dataset, [train_size, test_size,
                                                                     60000-(train_size+test_size)])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

# %%


def main():
    # init model
    autoencoder = LitAutoEncoder()
    dm = MyDataModule()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(autoencoder, dm)


# %%
main()

# %%

# %%
