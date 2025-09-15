import sys
import torch
import numpy as np
import pytorch_lightning as pl
import hydra
import omegaconf
import wandb

sys.path.insert(0, './iterativenn/src/')

from iterativenn.RunnerUtils import get_cuda, train, log_sample_data, finalize
from iterativenn.utils.model_factory import ModelFactory
from iterativenn.utils.logger_factory import LoggerFactory
from Data.RandomAnomaly.DataFactoryRA import random_anomaly

data_module = random_anomaly()

def runner_main(cfg):
    default_root_dir = None
    model = ModelFactory(cfg['model'])
    logger = LoggerFactory(cfg['logger'], model, default_root_dir)
    log_sample_data(cfg['logger'], logger, data_module)
    cfg_dict = omegaconf.OmegaConf.to_container(cfg)
    logger.log_hyperparams(cfg_dict)
    accelerator, devices = get_cuda(cfg['base'], logger)
    if hasattr(model, "number_of_trainable_parameters"):
        total_parameters = model.number_of_trainable_parameters()
        logger.console_logger.info(f'total parameters {total_parameters:,d}')
        logger.log_hyperparams({'total_parameters': total_parameters})
    else:
        total_parameters = 0
        for p in model.parameters():
            total_parameters += p.numel()
        logger.console_logger.info(f'total parameters {total_parameters:,d}')
        logger.log_hyperparams({'total_parameters': total_parameters})
    if cfg['base']['max_epochs'] > 0:
        trainer=train(cfg['base'], logger, model, data_module, default_root_dir, accelerator, devices)
        wandb.run.summary['Sparsity'] = float(cfg['model']['block_types'][1][1][2:])
        wandb.run.summary['Hidden'] = cfg['model']['row_sizes'][1]
        finalize(logger, model, cfg)
    wandb.finish()

def random_update(cfg, _lr=True, _hidden=True, _sparsity=True, _initialization=True, _dist='uniform', _size=None):
    rng = np.random.default_rng()
    indim, outdim = 2500, 10

    if _dist=='uniform':
        new_learning_rate = float(np.round(rng.uniform(0.00001, 0.001), 5))     #(0.000001, 0.01), 6)
        new_hidden = int(rng.uniform(1,1000))                                   #(1, 2000)
        new_sparsities = np.round(-1*rng.uniform(-1,0,(2,3)),2)
        new_means = np.round(rng.uniform(-0.005, 0.005, (2,3)), 3)                #(-0.05, 0.05, (2,3)), 4) #rng.uniform(-0.025, 0.025, (2,3))
        new_stds = np.round(rng.uniform(1e-6, 0.1, (2,3)), 3)                      #(0, 0.1, (2,3)), 4)

    if _dist=='uniform_size':
        cfg['logger']['project'] = 'random_anomaly_'+_dist+'_'+str(_size)
        new_sparsities = np.round(-1*rng.uniform(-1,0,(2,3)),2)
        new_hidden = int(np.round((-(new_sparsities[0,0]*indim + new_sparsities[0,2]*outdim + new_sparsities[1,1]*outdim) + np.sqrt((new_sparsities[0,0]*indim + new_sparsities[0,2]*outdim + new_sparsities[1,1]*outdim)**2 - 4*new_sparsities[0,1]*(new_sparsities[1,0]*indim*outdim + new_sparsities[1,2]*outdim**2-_size))) / (2*new_sparsities[0,1]), 0))
        new_learning_rate = float(np.round(rng.uniform(0.00001, 0.001), 5))     #(0.000001, 0.01), 6)
        new_means = np.round(rng.uniform(-0.005, 0.005, (2,3)), 3)                #(-0.05, 0.05, (2,3)), 4) #rng.uniform(-0.025, 0.025, (2,3))
        new_stds = np.round(rng.uniform(1e-6, 0.1, (2,3)), 3)                      #(0, 0.1, (2,3)), 4)

    if _dist=='normal_size':
        cfg['logger']['project'] = 'random_anomaly_'+_dist+'_'+str(_size)
        new_sparsities = np.round(np.clip(abs(rng.normal(0.5, 0.25, (2,3))), a_min=0.02, a_max=1), 2)
        new_hidden = int(np.round((-(new_sparsities[0,0]*indim + new_sparsities[0,2]*outdim + new_sparsities[1,1]*outdim) + np.sqrt((new_sparsities[0,0]*indim + new_sparsities[0,2]*outdim + new_sparsities[1,1]*outdim)**2 - 4*new_sparsities[0,1]*(new_sparsities[1,0]*indim*outdim + new_sparsities[1,2]*outdim**2-_size))) / (2*new_sparsities[0,1]), 0))
        new_learning_rate = float(np.round(np.clip(rng.normal(1e-3, 4e-4), a_min=1e-6, a_max=None), 5))
        new_means = np.round(rng.normal(0, 0.0025, (2,3)), 3)                    #(0, 0.3, (2,3))  #(0, 0.0125, (2,3)) #(0, 0.005, (2,3))
        new_stds = np.round(np.clip(rng.normal(0.05, 0.015, (2,3)), a_min=0.0001, a_max=None), 3)

    elif _dist=='normal':
        cfg['logger']['project'] = 'random_anomaly_'+_dist
        new_learning_rate = float(np.round(np.clip(rng.normal(1e-3, 4e-4), a_min=1e-6, a_max=None), 5))
        new_hidden = int(np.clip(rng.normal(500, 200), a_min=10, a_max=None))
        new_sparsities = np.round(np.clip(abs(rng.normal(0.5, 0.25, (2,3))), a_min=0.02, a_max=1), 2)
        new_means = np.round(rng.normal(0, 0.0025, (2,3)), 3)                    #(0, 0.3, (2,3))  #(0, 0.0125, (2,3)) #(0, 0.005, (2,3)) 
        new_stds = np.round(np.clip(rng.normal(0.05, 0.015, (2,3)), a_min=0.0001, a_max=None), 3)             

    if _lr:
        cfg['model']['learning_rate'] = new_learning_rate

    if _hidden:
        if cfg['model'].get('in_features_list', False):
            cfg['model']['in_features_list'][1] = new_hidden
            cfg['model']['out_features_list'][1] = new_hidden
        else: 
            cfg['model']['row_sizes'][1] = new_hidden
            cfg['model']['col_sizes'][1] = new_hidden
        cfg['model']['activation_sizes'][1] = new_hidden
        start = sum(cfg['model']['activation_sizes'][:2])
        end = sum(cfg['model']['activation_sizes'][:3])
        newrange = "asteval::range("+str(start)+", "+str(end)+")"
        cfg['model']['callbacks']['loss']['idx_list'] = newrange
        cfg['model']['callbacks']['initialization']['size'] = end
        cfg['model']['callbacks']['output']['idx_list'] = newrange

    if _sparsity:
        sparsities = cfg['model']['block_types']
        for row in range(1, len(sparsities)):
            for col in range(len(sparsities[row])):
                sparsities[row][col] = 'R='+str(new_sparsities[row-1][col])

    if _initialization:
        initializations = cfg['model']['initialization_types']
        for row in range(1, len(initializations)):
            for col in range(len(initializations[row])):
                #initializations[row][col] = 'G='+str(new_means[row-1][col])+',0.05' #only vary mean
                #initializations[row][col] = 'G=0,'+str(new_stds[row-1][col]) #only vary std
                initializations[row][col] = 'G='+str(new_means[row-1][col])+','+str(new_stds[row-1][col])

    return cfg

runs = [["base=medium", "logger=ra_uniform", "model=ra_test_111111", "data=random_anomaly"]]

for _ in range(50):
    try:
        for r in runs:
            with hydra.initialize(version_base=None, config_path="conf"):
                cfg = hydra.compose(config_name="config.yaml", overrides=r)
                cfg = random_update(cfg, _lr=True, _hidden=True, _sparsity=True, _initialization=True, _dist='normal_size', _size=50000)
            runner_main(cfg)
    except:
        wandb.finish()