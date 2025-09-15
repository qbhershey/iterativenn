import torch
import pytorch_lightning as pl

from iterativenn.nn_modules.HeterogeneousActivation import HeterogeneousActivation
from iterativenn.nn_modules.MaskedLinear import MaskedLinear

from iterativenn.lit_modules.IteratedModel import IteratedModel, ConfigCallbacks

from iterativenn.utils.model_wrappers import IterativeLSTM, IterativeGRU, IterativeRNN
from iterativenn.nn_modules.Sequential2D import Sequential2D
from iterativenn.nn_modules.Sequential1D import Sequential1D

def ModelFactory(cfg):
    if 'factory_type' in cfg.keys() and cfg['factory_type'] == 'callbacks':
        return ModelFactory_callback(cfg)
    else:
        raise ValueError('Unknown factory type %s' % cfg['factory_type'])   

def ModelFactory_callback(cfg):
    if cfg['model_type'] == 'description':
        model = MaskedLinear.from_config(cfg)
    elif cfg['model_type'] == 'sequential2D':
        model = Sequential2D.from_config(cfg)
    elif cfg['model_type'] == 'MLP':
        model = MaskedLinear.from_MLP(sizes=cfg['sizes'], 
                                      bias=cfg['bias'],
                                      )
    elif cfg['model_type'] == 'LSTM':
        model = IterativeLSTM(input_size=cfg['input_size'],
                              hidden_size=cfg['hidden_size'],
                              output_size=cfg['output_size'])
    elif cfg['model_type'] == 'RNN':
        assert cfg['output_size'] == cfg['hidden_size'], 'RNN output size must be equal to hidden size'
        model = IterativeRNN(input_size=cfg['input_size'],
                             hidden_size=cfg['hidden_size'])
    elif cfg['model_type'] == 'GRU':
        assert cfg['output_size'] == cfg['hidden_size'], 'GRU output size must be equal to hidden size'
        model = IterativeGRU(input_size=cfg['input_size'],
                             hidden_size=cfg['hidden_size'])
    else:
        raise ValueError('Unknown model type %s' % cfg['model_type'])

    # Not all models have activations.  For example, the LSTM does not.
    if not cfg['activations'] is False:
        activations = []
        for activation in cfg['activations']:
            if activation == 'ReLU':
                activations.append(torch.nn.ReLU())
            elif activation == 'Sigmoid':
                activations.append(torch.nn.Sigmoid())
            elif activation == 'ELU':
                activations.append(torch.nn.ELU())
            elif activation == 'Identity':
                activations.append(torch.nn.Identity())
            else:
                raise ValueError('Unknown activation %s' % activation)

        activation = HeterogeneousActivation(activations, cfg['activation_sizes'])
        model = Sequential1D(model, activation, 
                             in_features=model.in_features, 
                             out_features=model.out_features)

    callbacks = ConfigCallbacks(cfg['callbacks'])

    if cfg.get('learning_rate', False):
        learning_rate = cfg['learning_rate']
    else: learning_rate = 0.02

    return IteratedModel(model=model, callbacks=callbacks, learning_rate=learning_rate)