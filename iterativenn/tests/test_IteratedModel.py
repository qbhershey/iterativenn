import torch
import functools
from pytorch_lightning import Trainer

from iterativenn.lit_modules import IteratedModel
from iterativenn.utils.model_wrappers import IterativeRNN
from iterativenn.utils.logger_factory import LoggerFactory
from iterativenn.nn_modules.Sequential2D import Sequential2D
from iterativenn.utils.DataModules import MNISTRepeatedSequenceDataModule
from iterativenn.utils.logger_factory import LoggerFacade
from iterativenn.utils.DatasetUtils import MemoryCopySequence
from iterativenn.utils.DataModules import SequenceModule

def test_IteratedModel():
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

    sequential2D = Sequential2D.from_config(cfg["sequential2D"])
    callbacks = IteratedModel.ConfigCallbacks(cfg["callbacks"])

    # Test both the optimized and non-optimized versions of IteratedModel
    model = IteratedModel.IteratedModel(sequential2D, 
                                        callbacks,
                                        optimizer='Adam')
    # The min_copies and max_copies are set to 2 so that the model can iterate.  This
    # is important, and I tend to forget to set it.
    data_module = MNISTRepeatedSequenceDataModule(min_copies=2, max_copies=2, seed=1234, )

    trainer = Trainer(
        accelerator = 'auto',
        devices = 1, #if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs = 10,
        logger = LoggerFacade(None, 'console', 'info'),
        enable_progress_bar = False)
    trainer.fit(model, data_module)   

    batch = next(iter(data_module.train_dataloader()))
    loss = model.training_step(batch, 0, do_logging=False)
    # Note, the default settings for MNISTRepeatedSequenceDataModule are a
    # very easy problem, so we should be able to get a loss of less than 0.01.
    assert loss < 0.1, f"IteratedModel should have a loss less than 0.1, but is {loss}"

def test_IteratedModel2():
    dimension = 2
    dataset = functools.partial(MemoryCopySequence, min_sequence_size=2, max_sequence_size=4, dimension=dimension, copies=2)
    data_module = SequenceModule(dataset=dataset, batch_size=4, train_size=8, val_size=8, test_size=8)

    hidden_size = 10
    loss_dict = { 'func': 'MSELoss', 'idx_list': range(hidden_size, hidden_size+dimension), 'sequence_position': 'all'}
    initialization_dict = { 'func': 'zeros', 'size': hidden_size+dimension}
    data_dict = { 'func': 'insert', 'idx_list': range(0, dimension), 'flatten_input': True}
    output_dict = { 'func': 'all_series', 'idx_list': range(hidden_size, hidden_size+dimension)}
    callback_dict = {'loss': loss_dict, 'initialization': initialization_dict, 'data': data_dict, 'output': output_dict}
    callbacks = IteratedModel.ConfigCallbacks(callback_dict)

    model_params = {'input_size': dimension, 'hidden_size': hidden_size, 'output_size': hidden_size}
    assert model_params['output_size'] == model_params['hidden_size'], 'RNN output size must be equal to hidden size'
    
    loss = []
    torch.manual_seed(0)
    model = IterativeRNN(input_size=model_params['input_size'],
                            hidden_size=model_params['hidden_size'])
    iteratedModel = IteratedModel.IteratedModel(model=model, callbacks=callbacks, batch_optimize=True)
    torch.set_float32_matmul_precision('medium')
    trainer=Trainer(logger=False, enable_checkpointing=False, 
                        accelerator='cpu', devices=1, max_epochs=3)
    trainer.fit(iteratedModel, data_module)

    batch = next(iter(data_module.train_dataloader()))
    loss.append(iteratedModel.training_step(batch, 0, do_logging=False))

if __name__ == '__main__':
    test_IteratedModel()