import torch
import pytorch_lightning as pl
import numpy as np

from iterativenn.utils.model_factory import ModelFactory_callback
from iterativenn.utils.data_factory import DataFactory
from iterativenn.lit_modules.IteratedModel import IteratedModel, ConfigCallbacks
from iterativenn.nn_modules.Sequential1D import Sequential1D
from iterativenn.nn_modules.MaskedLinear import MaskedLinear

import pytest

@pytest.mark.long
def test_MLPvsINN():
    globals = {}
    globals['layer_sizes'] = [2500, 500, 200, 10]
    globals['activation'] = 'ELU'
    globals['accelerator'] = 'auto'
    globals['optimizer'] = 'Adam'
    globals['learning rate'] = 0.01
    globals['bias'] = False
    globals['initialization mean'] = 0.0
    globals['initialization std'] = 0.01


    globals['do_MLP'] = True
    globals['do_sequence_MLP'] = True
    globals['do_INN'] = True
    globals['do_sparse_INN'] = True
    globals['saved models'] = {}

    globals['number_of_runs'] = 1
    globals['data size'] = 16
    globals['batch size'] = 8
    globals['epochs'] = 1
    
    def MLP_data_loader():
        data_cfg = {
            'dataset': 'MNIST',
            'transform': 'both',
            'train_size': globals['data size'],
            'validation_size': globals['data size'],
            'test_size': globals['data size'],
            'batch_size': globals['batch size'],
        }

        data = DataFactory(data_cfg)
        return data

    def MLP_sequence_data_loader():
        data_cfg = {
            'dataset': 'MNIST',
            'transform': 'both',
            'train_size': globals['data size'],
            'validation_size': globals['data size'],
            'test_size': globals['data size'],
            'batch_size': globals['batch size'],
            'sequence_dict': {
                'type': 'uniform',
                'min_copies': 1,
                'max_copies': 1,
                'evaluate_loss': 'last'
            }
        }
        data = DataFactory(data_cfg)
        return data

    def INN_data_loader():
        data_cfg = {
            'dataset': 'MNIST',
            'transform': 'both',
            'train_size': globals['data size'],
            'validation_size': globals['data size'],
            'test_size': globals['data size'],
            'batch_size': globals['batch size'],
            'sequence_dict': {
                'type': 'uniform',
                'min_copies': 3,
                'max_copies': 3,
                'evaluate_loss': 'last'
            }
        }
        data = DataFactory(data_cfg)
        return data

    class LitModule(pl.LightningModule):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.losses = []
        
        def forward(self, X):
            return self.module(X)

        def training_step(self, batch, batch_idx):
            X,Y = batch
            Y_hat = self.module(X.view(X.shape[0], -1))
            loss = torch.nn.CrossEntropyLoss(reduction='sum')(Y_hat, Y)
            self.losses.append(loss)
            return loss
        
        def configure_optimizers(self):
            if globals['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(params=self.parameters(), lr=globals['learning rate'])
            elif globals['optimizer'] == 'SGD':
                return torch.optim.SGD(params=self.parameters(), lr=globals['learning rate'])
            return optimizer


    def train_MLP():
        data = MLP_data_loader()
        # Sequential1D is just a wrapper around nn.Sequential that
        # records the input and output sizes of the network.  This is useful
        # for the INN solution below as the MLP can be used as a block in an INN.
        # However, the as far as this part is concerned, it is identical to nn.Sequential.
        MLP = Sequential1D(in_features=globals['layer_sizes'][0],
                        out_features=globals['layer_sizes'][-1]) 

        for i in range(len(globals['layer_sizes'])-1):
            linear_module = torch.nn.Linear(globals['layer_sizes'][i], globals['layer_sizes'][i+1], bias=globals['bias'])
            with torch.no_grad():
                torch.nn.init.normal_(linear_module.weight, 
                                    globals['initialization mean'], 
                                    globals['initialization std'])
            MLP.add_module(f'linear{i}', linear_module)
            # always end with a linear layer
            if i < len(globals['layer_sizes'])-2:      
                if globals['activation'] == 'ELU':
                    MLP.add_module(f'activation{i}', torch.nn.ELU())      
                else:
                    raise ValueError(f'Unknown activation {globals["activation"]}')

        MLP_module = LitModule(MLP)

        trainer = pl.Trainer(max_epochs=globals['epochs'],
                            devices=1 if torch.cuda.is_available() else None,
                            accelerator=globals['accelerator'])
        trainer.fit(MLP_module, data)
        return MLP_module

    if globals['do_MLP']:
        name = 'MLP'
        globals['saved models'][name] = {}
        # run all the models and save them
        globals['saved models'][name]['models'] = [train_MLP() for i in range(globals['number_of_runs'])]
        # save the number of trainable parameters
        globals['saved models'][name]['average trainable params'] = np.mean([sum(p.numel() for p in model.parameters() if p.requires_grad) for model in globals['saved models'][name]['models']])
        # save the losses
        losses = [torch.tensor(model.losses).detach().numpy() for model in globals['saved models'][name]['models']]
        globals['saved models'][name]['average losses'] = np.mean(losses, axis=0)
        globals['saved models'][name]['std losses'] = np.std(losses, axis=0)

    def train_sequence_MLP():
        data = MLP_sequence_data_loader()
        # Sequential1D is just a wrapper around nn.Sequential that
        # records the input and output sizes of the network.  This is useful
        # for the INN solution below as the MLP can be used as a block in an INN.
        # However, the as far as this part is concerned, it is identical to nn.Sequential.
        MLP = Sequential1D(in_features=globals['layer_sizes'][0],
                        out_features=globals['layer_sizes'][-1]) 

        for i in range(len(globals['layer_sizes'])-1):

            # You can use a masked linear layer if you want
            linear_module = MaskedLinear(globals['layer_sizes'][i], 
                                            globals['layer_sizes'][i+1], 
                                            bias=globals['bias'])
            with torch.no_grad():
                torch.nn.init.normal_(linear_module.weight_0, 
                                    globals['initialization mean'], 
                                    globals['initialization std'])

            MLP.add_module(f'linear{i}', linear_module)
            # always end with a linear layer
            if i < len(globals['layer_sizes'])-2:      
                if globals['activation'] == 'ELU':
                    MLP.add_module(f'activation{i}', torch.nn.ELU())      
                else:
                    raise ValueError(f'Unknown activation {globals["activation"]}')

        cfg = {
            "callbacks": {
                "loss": {
                    "func": "CrossEntropyLoss",
                    "idx_list" : range(0, globals['layer_sizes'][-1]),
                },
                "initialization": {
                    "func": "zeros",
                    "size": globals['layer_sizes'][0],
                },
                "data": {
                    "func": "insert",
                    "idx_list": range(0, globals['layer_sizes'][0]),
                    "flatten_input": True,
                },
                "output": {
                    "func": "max",
                    "idx_list" : range(0, globals['layer_sizes'][-1]),
                },
            }
        }

        MLP_callbacks = ConfigCallbacks(cfg['callbacks'])

        MLP_module = IteratedModel(MLP, MLP_callbacks, normalize_loss=False, 
                                learning_rate=globals['learning rate'], optimizer=globals['optimizer'])
        trainer = pl.Trainer(max_epochs=globals['epochs'],
                            devices=1 if torch.cuda.is_available() else None,
                            accelerator=globals['accelerator'])
        trainer.fit(MLP_module, data)
        return MLP_module

    if globals['do_sequence_MLP']:
        name = 'sequence_MLP'
        globals['saved models'][name] = {}
        # run all the models and save them
        globals['saved models'][name]['models'] = [train_sequence_MLP() for i in range(globals['number_of_runs'])]
        # save the number of trainable parameters
        globals['saved models'][name]['average trainable params'] = np.mean([model.number_of_trainable_parameters() for model in globals['saved models'][name]['models']])
        # save the losses
        losses = [torch.tensor(model.losses).detach().numpy() for model in globals['saved models'][name]['models']]
        globals['saved models'][name]['average losses'] = np.mean(losses, axis=0)
        globals['saved models'][name]['std losses'] = np.std(losses, axis=0)

    def train_INN():
        data = INN_data_loader()
        default_module = 'MaskedLinear.from_description'
        default_module_kwargs = {'block_type': 'W', 
                                'initialization_type': f'G={globals["initialization mean"]},{globals["initialization std"]}', 
                                'trainable': True, 'bias': globals['bias']}
        
        cfg = {
            "model_type": "sequential2D",   
            "in_features_list": globals['layer_sizes'], 
            "out_features_list": globals['layer_sizes'], 

            "block_types": [
                ['Identity', default_module, None,           None],
                [None,       None,           default_module, None],
                [None,       None,           None,           default_module],
                [None,       None,           None,          None],
            ],
            "block_kwargs": [
                [None, default_module_kwargs, None, None],
                [None, None, default_module_kwargs, None],
                [None, None, None, default_module_kwargs],
                [None, None, None, None],
            ],

            "activations" : ['Identity', 'ELU', 'ELU', 'Identity'],
            "activation_sizes" : globals['layer_sizes'],
            "callbacks": {
                "loss": {
                    "func": "CrossEntropyLoss",
                    "idx_list" : range(sum(globals['layer_sizes'][:-1]), sum(globals['layer_sizes'])),
                },
                "initialization": {
                    "func": "zeros",
                    "size": sum(globals['layer_sizes']),
                },
                "data": {
                    "func": "insert",
                    "idx_list": range(0, globals['layer_sizes'][0]),
                    "flatten_input": True,
                },
                "output": {
                    "func": "max",
                    "idx_list" : range(sum(globals['layer_sizes'][:-1]), sum(globals['layer_sizes']))
                },
            }
        }
        # This is an IteratedModule, which is a wrapper around a module
        module = ModelFactory_callback(cfg)
        INN_module = IteratedModel(module.model, module.callbacks, normalize_loss=False, 
                                learning_rate=globals['learning rate'], optimizer=globals['optimizer'])
        trainer = pl.Trainer(max_epochs=globals['epochs'],
                            devices=1 if torch.cuda.is_available() else None,
                            accelerator=globals['accelerator'])
        trainer.fit(INN_module, data)
        return INN_module


    if globals['do_INN']:
        name = 'INN'
        globals['saved models'][name] = {}
        # run all the models and save them
        globals['saved models'][name]['models'] = [train_INN() for i in range(globals['number_of_runs'])]
        # save the number of trainable parameters
        globals['saved models'][name]['average trainable params'] = np.mean([model.number_of_trainable_parameters() for model in globals['saved models'][name]['models']])
        # save the losses
        losses = [torch.tensor(model.losses).detach().numpy() for model in globals['saved models'][name]['models']]
        globals['saved models'][name]['average losses'] = np.mean(losses, axis=0)
        globals['saved models'][name]['std losses'] = np.std(losses, axis=0)

    def train_sparse_INN():    
        data = INN_data_loader()
        default_module = 'SparseLinear.from_singleBlock'
        default_module_kwargs = {'block_type': 'R=0.4', 'initialization_type': f'G={globals["initialization mean"]},{globals["initialization std"]}', 'bias': globals["bias"]}
        # default_module = 'MaskedLinear.from_description'
        # default_module = 'SparseLinear.from_description'
        # default_module_kwargs = {'block_type': 'R=0.6', 'initialization_type': f'G={globals["initialization mean"]},{globals["initialization std"]}', 'trainable': 'non-zero', 'bias': globals["bias"]}

        cfg = {
            "model_type": "sequential2D",   
            "in_features_list": globals['layer_sizes'], 
            "out_features_list": globals['layer_sizes'], 

            "block_types": [
                ['Identity', default_module, default_module, default_module],
                [None,       default_module, default_module, default_module],
                [None,       default_module, default_module, default_module],
                [None,       default_module, default_module, default_module],
            ],
            "block_kwargs": [
                [None, default_module_kwargs, default_module_kwargs, default_module_kwargs],
                [None, default_module_kwargs, default_module_kwargs, default_module_kwargs],
                [None, default_module_kwargs, default_module_kwargs, default_module_kwargs],
                [None, default_module_kwargs, default_module_kwargs, default_module_kwargs],
            ],

            "activations" : ['Identity', 'ELU', 'ELU', 'Identity'],
            "activation_sizes" : globals['layer_sizes'],
            "callbacks": {
                "loss": {
                    "func": "CrossEntropyLoss",
                    "idx_list" : range(sum(globals['layer_sizes'][:-1]), sum(globals['layer_sizes'])),
                },
                "initialization": {
                    "func": "zeros",
                    "size": sum(globals['layer_sizes']),
                },
                "data": {
                    "func": "insert",
                    "idx_list": range(0, globals['layer_sizes'][0]),
                    "flatten_input": True,
                },
                "output": {
                    "func": "max",
                    "idx_list" : range(sum(globals['layer_sizes'][:-1]), sum(globals['layer_sizes']))
                },
            }
        }
        # This is an IteratedModule, which is a wrapper around a module
        module = ModelFactory_callback(cfg)
        INN_module = IteratedModel(module.model, module.callbacks, normalize_loss=False, 
                                learning_rate=globals['learning rate'], optimizer=globals['optimizer'])
        trainer = pl.Trainer(max_epochs=globals['epochs'],
                            devices=1 if torch.cuda.is_available() else None,
                            accelerator=globals['accelerator'])
        trainer.fit(INN_module, data)
        return INN_module

    if globals['do_sparse_INN']:
        name = 'sparse INN'
        globals['saved models'][name] = {}
        # run all the models and save them
        globals['saved models'][name]['models'] = [train_sparse_INN() for i in range(globals['number_of_runs'])]
        # save the number of trainable parameters
        globals['saved models'][name]['average trainable params'] = np.mean([model.number_of_trainable_parameters() for model in globals['saved models'][name]['models']])
        # save the losses
        losses = [torch.tensor(model.losses).detach().numpy() for model in globals['saved models'][name]['models']]
        globals['saved models'][name]['average losses'] = np.mean(losses, axis=0)
        globals['saved models'][name]['std losses'] = np.std(losses, axis=0)


    for name in globals['saved models']:
        average_loss = globals['saved models'][name]['average losses']
        print(f'{name} average loss: {np.mean(average_loss)}')



