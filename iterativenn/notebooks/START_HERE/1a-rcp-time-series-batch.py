# coding: utf-8
# # Load modules

# +
import functools
import time
from pprint import pprint
import shutil

import pandas as pd
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from iterativenn.lit_modules.IteratedModel import IteratedModel, ConfigCallbacks
from iterativenn.utils.model_wrappers import IterativeRNN, IterativeGRU, IterativeLSTM
from iterativenn.nn_modules.Sequential2D import Sequential2D
from iterativenn.nn_modules.Sequential1D import Sequential1D
from iterativenn.nn_modules.HeterogeneousActivation import HeterogeneousActivation

from iterativenn.utils.DatasetUtils import MemoryCopySequence
from iterativenn.utils.DataModules import SequenceModule

import click

# In this notebook we do a batch processing version of the notebook 1-rcp-time-series.ipynb.  The idea is that you can use this to do larger problems.
# We also introduce two new features:
#  "click" for handling command line arguments
#  "wandb" for doing distributed logging to wandb

# We define a few helper functions to make the code a bit cleaner.
def get_callbacks(x_size, y_size, h_size):
    loss_dict = { 'func': 'MSELoss', 'idx_list': range(x_size, x_size+y_size), 'sequence_position': 'all'}
    initialization_dict = { 'func': 'zeros', 'size': x_size+y_size+h_size}
    data_dict = { 'func': 'insert', 'idx_list': range(0, x_size), 'flatten_input': True}
    output_dict = { 'func': 'all_series', 'idx_list': range(x_size, x_size+y_size)}
    callback_dict = {'loss': loss_dict, 'initialization': initialization_dict, 'data': data_dict, 'output': output_dict}
    return ConfigCallbacks(callback_dict)

def RNN(x_size, y_size, h_size):
    callbacks = get_callbacks(x_size, y_size, h_size)
    model = IterativeRNN(input_size=x_size,
                         hidden_size=y_size+h_size)
    iteratedModel = IteratedModel(model=model, callbacks=callbacks, batch_optimize=True)
    return iteratedModel

def GRU(x_size, y_size, h_size):
    callbacks = get_callbacks(x_size, y_size, h_size)
    model = IterativeGRU(input_size=x_size,
                         hidden_size=y_size+h_size)
    iteratedModel = IteratedModel(model=model, callbacks=callbacks, batch_optimize=True)
    return iteratedModel

def LSTM(x_size, y_size, h_size):
    callbacks = get_callbacks(x_size, y_size, h_size)
    model = IterativeLSTM(input_size=x_size,
                          hidden_size=h_size,
                          output_size=y_size,
                          y_before_h=True)
    iteratedModel = IteratedModel(model=model, callbacks=callbacks, batch_optimize=True)
    return iteratedModel

def DenseINN(x_size, y_size, h_size):
    default_model = 'MaskedLinear.from_description'
    default_kwargs = {'block_type':'W', 'initialization_type':'G=0.0,0.01', 'trainable':True, 'bias':False}

    cfg = {
            "in_features_list": [x_size, y_size, h_size], 
            "out_features_list": [x_size, y_size, h_size], 
            "block_types": [
                    ['Identity', default_model, default_model],
                    [None,       default_model, default_model],
                    [None,       default_model, default_model]
                ],
            "block_kwargs": [
                    [None, default_kwargs, default_kwargs],
                    [None, default_kwargs, default_kwargs],
                    [None, default_kwargs, default_kwargs]
                ],
        }

    # We use the above config to create a Sequential2D model.  Note, this can also be done by hand, and not using the config factory.
    # However, this is a bit cleaner and easier to get started with.
    base_model = Sequential2D.from_config(cfg)

    # There are our chosen activations for each layer.  Note, you can choose these as you like based upon your problem.
    # for example, you might want to make the activations for $y$ something like a sigmoid if you are doing classification.
    base_activations = [torch.nn.Identity(), torch.nn.Identity(), torch.nn.ELU()]
    activation = HeterogeneousActivation(base_activations, [x_size, y_size, h_size])


    # We create a Sequential1D which applied the activations to the output of the Sequential2D model.
    model = Sequential1D(base_model, activation, 
                        in_features=base_model.in_features, 
                        out_features=base_model.out_features)

    # There are a few callbacks we need to use to make this work as described above.
    callbacks = get_callbacks(x_size, y_size, h_size)

    # Note, the batch_optimize=True collects a batch of training sequences and sends them to the model at once.  It should
    # be faster for large models.  
    iteratedModel = IteratedModel(model=model, callbacks=callbacks, batch_optimize=True)
    return iteratedModel

def SparseINN(x_size, y_size, h_size):
    default_model = 'MaskedLinear.from_description'
    default_kwargs = {'block_type':'R=0.2', 'initialization_type':'G=0.0,0.01', 'trainable':'non-zero', 'bias':False}

    cfg = {
            "in_features_list": [x_size, y_size, h_size], 
            "out_features_list": [x_size, y_size, h_size], 
            "block_types": [
                    ['Identity', default_model, default_model],
                    [None,       default_model, default_model],
                    [None,       default_model, default_model]
                ],
            "block_kwargs": [
                    [None, default_kwargs, default_kwargs],
                    [None, default_kwargs, default_kwargs],
                    [None, default_kwargs, default_kwargs]
                ],
        }

    base_model = Sequential2D.from_config(cfg)
    base_activations = [torch.nn.Identity(), torch.nn.Identity(), torch.nn.ELU()]

    activation = HeterogeneousActivation(base_activations, [x_size, y_size, h_size])
    model = Sequential1D(base_model, activation, 
                        in_features=base_model.in_features, 
                        out_features=base_model.out_features)
    callbacks = get_callbacks(x_size, y_size, h_size)
    iteratedModel = IteratedModel(model=model, callbacks=callbacks, batch_optimize=True)
    return iteratedModel

def VariableINN(x_size, y_size, h_size):
    default_model = 'MaskedLinear.from_description'
    default1_kwargs = {'block_type':'R=0.1', 'initialization_type':'G=0.0,0.01', 'trainable':'non-zero', 'bias':False}
    default2_kwargs = {'block_type':'R=0.2', 'initialization_type':'G=0.0,0.01', 'trainable':'non-zero', 'bias':True}

    cfg = {
            "in_features_list": [x_size, y_size, h_size//2, h_size//2], 
            "out_features_list": [x_size, y_size, h_size//2, h_size//2], 
            "block_types": [
                    ['Identity', default_model, default_model, None         ],
                    [None,       default_model, None,          default_model],
                    [None,       None         , default_model, default_model],
                    [None,       default_model, default_model, None         ],
                ],
            "block_kwargs": [
                    [None,       default1_kwargs, default2_kwargs, None          ],
                    [None,       default2_kwargs, None,            default2_kwargs],
                    [None,       None,            default1_kwargs, default1_kwargs],
                    [None,       default1_kwargs, default2_kwargs, None          ],
                ],
        }

    base_model = Sequential2D.from_config(cfg)
    base_activations = [torch.nn.Identity(), torch.nn.Identity(), torch.nn.ELU(), torch.nn.Tanh()]

    activation = HeterogeneousActivation(base_activations, [x_size, y_size, h_size//2, h_size//2])
    model = Sequential1D(base_model, activation, 
                        in_features=base_model.in_features, 
                        out_features=base_model.out_features)

    # Note, in this case the sizes of the callbacks are different than the sizes of the model!  This is perfectly fine.
    callbacks = get_callbacks(x_size, y_size, h_size)
    iteratedModel = IteratedModel(model=model, callbacks=callbacks, batch_optimize=True)
    return iteratedModel

models = {
    'RNN': {'factory': RNN, 'h_size': 100},
    'GRU': {'factory': GRU, 'h_size': 55},
    'LSTM': {'factory': LSTM, 'h_size': 400},
    'DenseINN': {'factory': DenseINN, 'h_size': 100},
    'SparseINN': {'factory': SparseINN, 'h_size': 230},
    'VariableINN': {'factory': VariableINN, 'h_size': 316},
}

@click.command()
@click.option('--model_name', default='LSTM', help=f'The name of the model to use. Valid options are: {models.keys()}.')
@click.option('--seed', default=None, help='The random seed to use.')
@click.option('--dimension', default=2, help='The dimension of the input and output.')
@click.option('--min_sequence_size', default=2, help='The minimum sequence size.')
@click.option('--max_sequence_size', default=4, help='The maximum sequence size.')
@click.option('--copies', default=2, help='The number of copies of each element in the sequence.')
@click.option('--train_size', default=1024, help='The number of training examples.')
@click.option('--val_size', default=1024, help='The number of validation examples.')
@click.option('--test_size', default=1024, help='The number of test examples.')
@click.option('--batch_size', default=1024, help='The batch size.')
@click.option('--max_epochs', default=5, help='The number of epochs to train for.')
@click.option('--accelerator', default='cpu', help='The accelerator to use.  Valid options are: cpu, gpu, and auto.')
@click.option('--num_threads', default=1, help='The number of threads to use.')
def run_experiment(**kwargs):
    if kwargs['seed'] is not None:
        torch.manual_seed(kwargs['seed'])
    # Set the number of threads
    torch.set_num_threads(kwargs['num_threads'])

    # Create the dataset
    dataset_factory = functools.partial(MemoryCopySequence, 
                                        min_sequence_size=kwargs['min_sequence_size'], 
                                        max_sequence_size=kwargs['max_sequence_size'], 
                                        dimension=kwargs['dimension'], 
                                        copies=kwargs['copies'])

    # And we wrap the data is a pytorch lightning data module.   This is not necessary, but it makes it easier to use the data in the training loop.
    data_module = SequenceModule(dataset=dataset_factory, 
                                    batch_size=kwargs['batch_size'], 
                                    train_size=kwargs['train_size'], 
                                    val_size=kwargs['val_size'], 
                                    test_size=kwargs['test_size'], 
                                    num_workers=1)

    # Create the model
    model = models[kwargs['model_name']]['factory'](x_size=kwargs['dimension'], y_size=kwargs['dimension'], h_size=models[kwargs['model_name']]['h_size'])

    kwargs['number_of_trainable_parameters'] = model.number_of_trainable_parameters()

    # You need a Weight and Biases account to use this!
    # In fact, you also need to be logged in to wandb for this to work.
    # see https://wandb.ai
    logger = WandbLogger(name=kwargs['model_name'], 
                         save_dir="logs/1a-rcp-time-series-batch",
                         project="rcp-time-series-batch")

    trainer=pl.Trainer(logger=logger, enable_checkpointing=False, 
                        devices=1 if torch.cuda.is_available() else None,
                        accelerator=kwargs['accelerator'],
                        max_epochs=kwargs['max_epochs'], callbacks=[RichProgressBar()])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    run_experiment()


