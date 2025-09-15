import sys
import torch
import numpy as np
import pytorch_lightning as pl
import hydra
import omegaconf
import wandb
import minari
import gymnasium as gym

sys.path.insert(0, './iterativenn/src/')

from iterativenn.RunnerUtils import get_cuda, train, log_sample_data, finalize
from iterativenn.utils.model_factory import ModelFactory
from iterativenn.utils.logger_factory import LoggerFactory
from Data.AdroitHammer.DataFactoryAH import AdroitHammer, score

data_module = AdroitHammer()

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
        average_model_reward, test_runs = score(model, 100) 
        wandb.log({'average_model_reward': average_model_reward, 'test_runs': test_runs})
        #wandb.run.summary['Sparsity'] = float(cfg['model']['block_types'][1][1][2:])
        #wandb.run.summary['Hidden'] = cfg['model']['row_sizes'][1]
        finalize(logger, model, cfg)
    wandb.finish()


runs = [["base=medium", "logger=AHVaried", "model=400AH14", "data=adroit_hammer"],
        ["base=medium", "logger=AHVaried", "model=400AH15", "data=adroit_hammer"],
        ["base=medium", "logger=AHVaried", "model=400AH16", "data=adroit_hammer"],
        ["base=medium", "logger=AHVaried", "model=400AH17", "data=adroit_hammer"]]


for _ in range(2):
    for r in runs:
        with hydra.initialize(version_base=None, config_path="conf"):
            cfg = hydra.compose(config_name="config.yaml", overrides=r)
            #cfg['logger']['name'] = 'H'+str(cfg['model']['row_sizes'][1])+'R'+str(int(float(cfg['model']['block_types'][1][1][2:])*100))
        runner_main(cfg)