from pytorch_lightning import Trainer

import torch

from iterativenn.utils.DataModules import MNISTRepeatedSequenceDataModule
from iterativenn.nn_modules.Sequential2D import Sequential2D
from iterativenn.lit_modules import IteratedModel
from iterativenn.utils.logger_factory import LoggerFacade

import warnings

global_max_epochs = 10
global_optimizer = 'SGD'

def factory_run(cfg):
    sequential2D = Sequential2D.from_config(cfg["sequential2D"])
    callbacks = IteratedModel.ConfigCallbacks(cfg["callbacks"])
    model = IteratedModel.IteratedModel(sequential2D, 
                                        callbacks,
                                        normalize_loss=True,
                                        optimizer=global_optimizer)
    data_module = MNISTRepeatedSequenceDataModule(min_copies=2, max_copies=2, seed=1234)
    # Initialize a trainer
    trainer = Trainer(
        accelerator='auto',
        devices=1, #if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=global_max_epochs,
        logger = LoggerFacade(None, 'console', 'info'),
        log_every_n_steps=1,
        enable_progress_bar=False,
    )

    with torch.no_grad():
        data_module.prepare_data()
        data_module.setup('fit')
        batch = next(iter(data_module.train_dataloader()))
        loss_before = float(model.training_step(batch, 0, do_logging=False))

    with warnings.catch_warnings():
        # There are warning that I dont' care about at this moment and are not relevant to the example.
        warnings.simplefilter("ignore")
        trainer.fit(model, data_module)

    with torch.no_grad():
        data_module.prepare_data()
        data_module.setup('fit')
        batch = next(iter(data_module.train_dataloader()))
        loss_after = float(model.training_step(batch, 0, do_logging=False))

    return sequential2D, loss_before, loss_after

def test_Sequential2D_training_1():
    cfg = {
        "sequential2D": {
            "in_features_list": [28*28, 100, 10], 
            "out_features_list": [28*28, 100, 10], 
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
                "func": "CrossEntropyLoss",
                "idx_list" : range(28*28+100, 28*28+100+10),
            },
            "initialization": {
                "func": "zeros",
                "size": 28*28+100+10,
            },
            "data": {
                "func": "insert",
                "idx_list": range(28*28),
                "flatten_input": True,
            },
            "output": {
                "func": "max",
                "idx_list" : range(28*28+100, 28*28+100+10)
            },
        }
    }
    _, loss_before, loss_after = factory_run(cfg)
    assert loss_before > loss_after, "Loss should decrease after training."


def test_Sequential2D_training_2():
    cfg1 = {
        "sequential2D": {
            "in_features_list": [28*28, 100, 10], 
            "out_features_list": [28*28, 100, 10], 
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
                "func": "CrossEntropyLoss",
                "idx_list" : range(28*28+100, 28*28+100+10),
            },
            "initialization": {
                "func": "zeros",
                "size": 28*28+100+10,
            },
            "data": {
                "func": "insert",
                "idx_list": range(28*28),
                "flatten_input": True,
            },
            "output": {
                "func": "max",
                "idx_list" : range(28*28+100, 28*28+100+10)
            },
        }
    }

    previous_model, previous_loss_before, previous_loss_after = factory_run(cfg1)
    assert previous_loss_before > previous_loss_after, f"Loss should decrease after training but is before {previous_loss_before} and after {previous_loss_after}."

    cfg2 = {
        "sequential2D": {
            "in_features_list": [28*28+100+10, 10], 
            "out_features_list": [28*28+100+10, 10], 
            "block_types": [
                ['Module', None],
                [None, None],
            ],

            "block_kwargs": [
                [{'module':previous_model}, None],
                [None, None],
            ],
        },
        "callbacks": {
            "loss": {
                "func": "CrossEntropyLoss",
                "idx_list" : range(28*28+100, 28*28+100+10),
            },
            "initialization": {
                "func": "zeros",
                "size": 28*28+100+10+10,
            },
            "data": {
                "func": "insert",
                "idx_list": range(28*28),
                "flatten_input": True,
            },
            "output": {
                "func": "max",
                "idx_list" : range(28*28+100, 28*28+100+10)
            },
        }
    }

    _, loss_before, loss_after = factory_run(cfg2)
    assert torch.isclose(torch.tensor(loss_before), torch.tensor(previous_loss_after)), f"Loss should be the same as the previous model. {loss_before} != {previous_loss_after}"
    assert loss_before > loss_after, "Loss should decrease after training."
    assert previous_loss_before > loss_after, "Loss should decrease after training."

def test_Sequential2D_training_3():
    cfg1 = {
        "sequential2D": {
            "in_features_list": [28*28, 100, 10], 
            "out_features_list": [28*28, 100, 10], 
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
                "func": "CrossEntropyLoss",
                "idx_list" : range(28*28+100, 28*28+100+10),
            },
            "initialization": {
                "func": "zeros",
                "size": 28*28+100+10,
            },
            "data": {
                "func": "insert",
                "idx_list": range(28*28),
                "flatten_input": True,
            },
            "output": {
                "func": "max",
                "idx_list" : range(28*28+100, 28*28+100+10)
            },
        }
    }

    previous_model, previous_loss_before, previous_loss_after = factory_run(cfg1)
    assert previous_loss_before > previous_loss_after, "Loss should decrease after training."

    default_block_kwargs = {'block_type':'W', 'initialization_type':'G=0.0,0.0', 'trainable':True, 'bias':False}

    cfg2 = {
        "sequential2D": {
            "in_features_list": [28*28+100+10, 10], 
            "out_features_list": [28*28+100+10, 10], 
            "block_types": [
                ['Module', 'MaskedLinear.from_description'],
                ['MaskedLinear.from_description', 'MaskedLinear.from_description'],
            ],
            "block_kwargs": [
                [{'module':previous_model}, default_block_kwargs],
                [default_block_kwargs, default_block_kwargs],
            ],
        },
        "callbacks": {
            "loss": {
                "func": "CrossEntropyLoss",
                "idx_list" : range(28*28+100, 28*28+100+10),
            },
            "initialization": {
                "func": "zeros",
                "size": 28*28+100+10+10,
            },
            "data": {
                "func": "insert",
                "idx_list": range(28*28),
                "flatten_input": True,
            },
            "output": {
                "func": "max",
                "idx_list" : range(28*28+100, 28*28+100+10)
            },
        }
    }

    _, loss_before, loss_after = factory_run(cfg2)
    assert loss_before > loss_after, "Loss should decrease after training."
    assert previous_loss_before > loss_after, "Loss should decrease after training."


