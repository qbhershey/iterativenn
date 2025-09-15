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
        finalize(logger, model, cfg)
    wandb.finish()


runs = [["base=medium", "logger=ra_test", "model=ra_test_111111", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_S2D_dense", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_S2D_mldense", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_S2D_dense_mb", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_S2D_mldense_mb", "data=random_anomaly"]]

runs = [["base=medium", "logger=ra_test", "model=ra_test_balanced_0", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_balanced_1", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_balanced_2", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_balanced_3", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_balanced_4", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_balanced_5", "data=random_anomaly"],
        ["base=medium", "logger=ra_test", "model=ra_test_balanced_6", "data=random_anomaly"]]

runs = [["base=medium", "logger=ra_test", "model=ra_test_balanced_8", "data=random_anomaly"]]



hiddens = [29, 105, 176]
hiddens = [216]
hiddens = [208]
hiddens = [145]

def hidden_update(cfg, hidden):
    if cfg['model'].get('in_features_list', False):
        cfg['model']['in_features_list'][1] = hidden
        cfg['model']['out_features_list'][1] = hidden
    else: 
        cfg['model']['row_sizes'][1] = hidden
        cfg['model']['col_sizes'][1] = hidden
    cfg['model']['activation_sizes'][1] = hidden
    start = sum(cfg['model']['activation_sizes'][:2])
    end = sum(cfg['model']['activation_sizes'][:3])
    newrange = "asteval::range("+str(start)+", "+str(end)+")"
    cfg['model']['callbacks']['loss']['idx_list'] = newrange
    cfg['model']['callbacks']['initialization']['size'] = end
    cfg['model']['callbacks']['output']['idx_list'] = newrange
    return cfg

for _ in range(2):
    for r in runs:
        #for hidden in hiddens:
            with hydra.initialize(version_base=None, config_path="conf"):
                cfg = hydra.compose(config_name="config.yaml", overrides=r)
            #cfg = hidden_update(cfg, hidden)
            runner_main(cfg)