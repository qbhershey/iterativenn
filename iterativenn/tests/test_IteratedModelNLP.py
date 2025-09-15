import pytest
import torch
import torch.utils.data as data

from iterativenn.lit_modules.IteratedModelBatch import ConfigCallbacks
from iterativenn.lit_modules.IteratedModelBatch import IteratedModelNLP
from iterativenn.nn_modules.Sequential2D import Sequential2D


# Create a dataset
class DummyDataset(data.Dataset):
    def __init__(self):
        self.x = torch.randn(8, 3, 768)
        self.y = torch.randn(8, 3, 768)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

config_vocab = 10000
embed_len = 768
cfg = {
    "sequential2D": {
        "in_features_list": [768, 10, embed_len],
        "out_features_list": [768, 10, embed_len],


        "block_types": [
            [None, 'Linear', None],
            [None, None, 'Linear'],
            [None, None, None],
        ],
        "block_kwargs": [
            [None, None, None],
            [None, None, None],
            [None, None, None],
        ]
    },
    "callbacks": {
        "loss": {
            "func": "MSELoss",
            "idx_list" : range(768+10, 768+10+embed_len),
            "sequence_position": 'all',
        },
        "initialization": {
            "func": "zeros",
            "size":768+10+embed_len,
            "seq_len": 3,
            "batch_size": 4,
        },
        "data": {
            "func": "insert",
            "idx_list": range(768),
            "flatten_input": False,
        },
        "output": {
            "func": "all",
            "idx_list" : range(768+10, 768+10+embed_len)
        },
    }
}


@pytest.fixture
def input_model():
    model = Sequential2D.from_config(cfg["sequential2D"])
    # Create callbacks object
    callbacks = ConfigCallbacks(cfg["callbacks"])
    return model, callbacks


def test_IteratedModelNLP(input_model):
    # Create a PyTorch model
    model, callbacks = input_model
    # Test initialization of the model with valid inputs
    model_nlp = IteratedModelNLP(model, callbacks, normalize_loss=False, optimizer='Adam')
    assert model_nlp.model == model
    assert model_nlp.callbacks == callbacks
    assert model_nlp.normalize_loss == False
    assert model_nlp.optimizer == 'Adam'


def test_forward(input_model):
    # Test the forward method of the model
    model, callbacks = input_model
    x_batch = torch.randn(4, 3, 768)
    model_nlp = IteratedModelNLP(model, callbacks, normalize_loss=False, optimizer='Adam')
    y_batch = model_nlp.forward(x_batch)
    assert y_batch.shape == torch.Size([4, 3, 768])


def test_training_step(input_model):
    # Create a dataloader
    dataset = DummyDataset()
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
    model, callbacks = input_model
    # Test the training_step method of the model
    model_nlp = IteratedModelNLP(model, callbacks, normalize_loss=False, optimizer='Adam')
    for batch_idx, batch in enumerate(dataloader):
        x_batch, y_batch = batch
        loss = model_nlp.training_step(batch, batch_idx, do_logging=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


def test_validation_step(input_model):
    # Create a dataloader
    dataset = DummyDataset()
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
    model, callbacks = input_model
    # Test the training_step method of the model
    model_nlp = IteratedModelNLP(model, callbacks, normalize_loss=False, optimizer='Adam')
    for batch_idx, batch in enumerate(dataloader):
        x_batch, y_batch = batch
        loss = model_nlp.validation_step(batch, batch_idx, do_logging=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0