import torch
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
import wandb
import pandas as pd
import pathlib
import json
import time
import os
from collections import defaultdict

from matplotlib import pyplot as plt

import logging
logger = logging.getLogger(__name__)

from iterativenn.lit_modules.IteratedModel import IteratedModel

class LoggerFacade(Logger):
    def __init__(self, pl_logger, logger_type, console_level):
        self.pl_logger = pl_logger
        self.logger_type = logger_type

        self.console_logger = logger
        # This bears some explanation.  We want to be able to control the
        # the logging level even when the logger is running in a separate
        # process.  For example, this happens when the logger is run
        # through Dask.
        if console_level == 'debug':
            self.console_logger.setLevel(logging.DEBUG)
        elif console_level == 'info':
            self.console_logger.setLevel(logging.INFO)
        elif console_level == 'error':
            self.console_logger.setLevel(logging.ERROR)
        else:
            assert False, f"Unknown console level {console_level}"    
        self.console_logger.info(f"logger_type {self.logger_type}")

        def defaultdict_of_list():
            return defaultdict(list)
        self.log_z_data = defaultdict(defaultdict_of_list)

    @property
    def name(self):
        return "LoggerFacade"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        if self.pl_logger is not None:
            self.pl_logger.log_hyperparams(params)
        #self.console_logger.info(f"params {params}")

    # Some logging only happens on rank 0, but some things we want to log
    # on all ranks.  
    #@rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        if self.pl_logger is not None:
            self.pl_logger.log_metrics(metrics, step=step)
        else:
            self.console_logger.info(f"metrics {metrics} step {step}")
        # self.console_logger.info(f"metrics {metrics} step {step}")

    @rank_zero_only
    def log_table(self, name, columns, data):
        if self.logger_type == 'wandb':
            table = wandb.Table(columns=columns)
            for row in data:
                new_row = []
                for item in row:
                    # If the data is 1D, then we just log it.
                    if len(item.shape) == 1:
                        new_row.append(item.detach().cpu().numpy())
                    # If the data is 2D or 3D, then we log it as an image.
                    else:      
                        new_row.append(wandb.Image(item))
                table.add_data(*new_row)
            self.pl_logger.experiment.log({name: table})

    # Some logging only happens on rank 0, but some things we want to log
    # on all ranks.  
    #@rank_zero_only
    def log_z(self, name, z, sequence_idx, batch_idx, step):
        if self.logger_type == 'wandb':
            # if batch_idx==0 and step%10 == 0:
            if batch_idx==0:
                # This saves the lot as a non-interactive image.
                z = z.detach().cpu().numpy()
                self.log_z_data[name]['z'].append(wandb.Image(z))
                self.log_z_data[name]['sequence_idx'].append(sequence_idx)
                self.log_z_data[name]['batch_idx'].append(batch_idx)
                self.log_z_data[name]['step'].append(step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        if self.pl_logger is not None:
            self.pl_logger.save()
        else:
            self.console_logger.debug(f"save not implemented for logger type {self.logger_type}") 

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        # if self.pl_logger is not None:
        #     self.pl_logger.finalize(status)
        # This doesn't seem to break anything, though it does lead to error messages.
        # Also, it doesn't seen to be necessary.
        # if self.logger_type == 'wandb':
        #     wandb.finish()
        for name in self.log_z_data.keys():
            columns = self.log_z_data[name].keys()
            data = list(zip(*self.log_z_data[name].values()))
            df = pd.DataFrame(data, columns=columns)
            self.pl_logger.experiment.log({name: wandb.Table(dataframe=df)})

    @rank_zero_only
    def save_model(self, model, save_model_name, run_path_file):
        if self.logger_type == 'wandb':
            # This is nice.  The wandb logger gives us a path to save the model
            # i.e., everything in this directory will get uploaded to wandb.
            if isinstance(model, IteratedModel):
                # model is a SeqeuenceModel, and I really want the model inside.
                torch.save(model.model, pathlib.Path(self.pl_logger.experiment.dir) / save_model_name)
                self.console_logger.info(f"Saved model path {self.pl_logger.experiment.path}")
                save_dict ={}
                save_dict['save_model_name'] = save_model_name
                save_dict['path'] = self.pl_logger.experiment.path
                save_dict['time'] = time.ctime()
                json.dump(save_dict, open(run_path_file, 'w'))
            else:
                # model is not a IteratedModel.  This could be implemented, but I don't need it right now.
                self.console_logger.warning(f"save_model only implemented for IteratedModel, not {type(model)}") 
        else:
            self.console_logger.warning(f"save_model not implemented for logger type {self.logger_type}") 

def LoggerFactoryParallelSetup(cfg):
    if cfg['type'] == 'wandb':
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.setup()

def LoggerFactory(cfg=None, model=None, default_root_dir=None):
    if cfg is None:
        cfg = {}
        cfg['type'] = 'console'
        cfg['console_level'] = 'info'

    if cfg['type'] == 'wandb':
        # RCP: There is a lot to say here.  I have been struggling with
        # getting wandb to work with pytorch lightning and dask (as well as 
        # multiprocessing).  I have tried a lot of things, and this is the
        # best I have come up with. In particular, the reinit=True is
        # magic.  It just so happens that the pytorch lightning wandb logger
        # doesn't support this, so I have to do it myself.  
        # Note the following two linear appear in 
        # https://docs.wandb.ai/guides/track/log/distributed-training
        # #hanging-at-the-beginning-of-training
        # But the don't seem to be necessary for the current version of wandb.
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.setup()
        wandb.init(project=cfg['project'], 
                   entity=cfg['entity'], 
                   name=cfg['name'],
                   dir=default_root_dir,
                   reinit=True)

        # Some of the following parameters are taken care of above.
        logger = WandbLogger(
                            # save_dir=default_root_dir,
                            offline=cfg['offline'],
                            #
                            #project=cfg['project'],
                            #entity=cfg['entity'],
                            #name=cfg['name'],
                            #
                            log_model=cfg['log_model'],
                            )
        logger.watch(model, log_graph=False, log_freq=100)
        logger = LoggerFacade(logger, cfg['type'], cfg['console_level'])
    elif cfg['type'] == 'tensorboard':
        logger = TensorBoardLogger(save_dir=default_root_dir,
                                   # As said in the docs, 
                                   # https://pytorch-lightning.readthedocs.io/en/stable/
                                   # extensions/generated/
                                   # pytorch_lightning.loggers.TensorBoardLogger.html
                                   # Enables a placeholder metric with key hp_metric 
                                   # when log_hyperparams is called without a metric 
                                   # (otherwise calls to log_hyperparams without a 
                                   # metric are ignored).
                                   default_hp_metric=True,
                                   name=cfg['name']
                                  )
        logger = LoggerFacade(logger, cfg['type'], cfg['console_level'])
    elif cfg['type'] == 'console':
        logger = LoggerFacade(None, cfg['type'], cfg['console_level'])
    else:
        raise ValueError(f"Unknown logger type: {cfg['type']}")
    
    logger.console_logger.info(f"All logging it going to {default_root_dir}")
    return logger