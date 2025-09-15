# "Weights and Biases" is a really nice logging library/web pag.
# See here for details:  https://wandb.ai/
# If you want to use this then you need a wandb account, which is free!  Just go to https://wandb.ai/signup to get started.

import pathlib

import omegaconf
# This is where things get interesting.  We are going to use the PyTorch Lightning library.
import pytorch_lightning as pl
import torch

import numpy as np

from iterativenn.utils.data_factory import DataFactory
from iterativenn.utils.model_factory import ModelFactory
from iterativenn.utils.logger_factory import LoggerFactory, LoggerFactoryParallelSetup

import warnings

# This uses the 'base' part of the top-level config
def get_cuda(cfg, logger):
    # We detect if we are running on a GPU or not.  If we are, we will use it.
    cuda_params = {}

    if torch.cuda.is_available() and cfg['use_cuda']:
        cuda_params['available'] = True
        cuda_params['current_device'] = torch.cuda.current_device()
        cuda_params['device'] = torch.cuda.device(0)
        cuda_params['device_count'] = torch.cuda.device_count()
        cuda_params['device_name'] = torch.cuda.get_device_name(0)
        # Note, devices > 1 is not supported by dask, since it
        # makes additional threads.
        devices = 1
        accelerator = 'cuda'
    else:
        cuda_params['available'] = False
        cuda_params['current_device'] = None
        cuda_params['device'] = None
        cuda_params['device_count'] = None
        cuda_params['device_name'] = None
        # Note, devices > 1 is not supported by dask, since it
        # makes additional threads.
        devices = 1
        accelerator = 'auto'

    logger.console_logger.info(f"CUDA available: {cuda_params['available']}")
    logger.console_logger.info(f"CUDA name: {cuda_params['device_name']}")
    logger.log_hyperparams(cuda_params)
    return accelerator, devices 

def train(cfg, logger, model, data_module, default_root_dir, accelerator, devices):
    with warnings.catch_warnings():
        # We intentionally want to have num_workers=0 for the dataloaders.
        # This is to keep the process in a single thread.  This is because
        # Dask, Joblin, etc. are already managing the parallelism and get
        # confused if we have multiple threads. 
        warnings.filterwarnings("ignore", ".*does not have many workers.*") 
        # We often have a GPU, but want to use the CPU for doing hyperparameter
        # sweeps.  This is a warning that we are not using the GPU.  
        warnings.filterwarnings("ignore", ".*GPU available but not used.*") 
        trainer = pl.Trainer(# This seems to be the default sage way to do this.
                            accelerator=accelerator,
                            devices=devices,
                            # This is the code for that many pytorch examples use:
                            # accelerator='auto',
                            # devices=1 if torch.cuda.is_available() else None,
                            # This is a Facade pattern.  It is a wrapper around the logger.
                            logger=logger,
                            # In run this in hydra and I don't want a bunch of progress bars.
                            enable_progress_bar=False,
                            # This turns off the model summary (which gives the size
                            # of the model).  Note, in may cases for our work this is
                            # incorrect when MaskedLinear is used is anything but
                            # a trivial case.
                            enable_model_summary=False,
                            # FIXME:  I don't understand the interaction between this and the logger.
                            #         Especially in the case of the Facade pattern I use. So, I turn this
                            #         off for now.               
                            enable_checkpointing=False,
                            default_root_dir=default_root_dir,
                            log_every_n_steps=cfg['log_every_n_steps'],
                            profiler=cfg['profiler'],
                            max_epochs=cfg['max_epochs'])
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
    return trainer

# This is a simple replacement for the below.  It doesn't have most of the functionality, but does
# at least provide a way to visualize the data for the model.
def log_sample_data(logger_cfg, logger, data_module):
    # log sample minibatch of data to wandb
    if logger_cfg['type'] == 'wandb':
        dl_train = data_module.train_dataloader()
        # Each minibatch will contain a list of sequences
        for sequence_minibatch in iter(dl_train):
            # We log each sequence as a separate table, since they might be different lengths.
            for i, sequence in enumerate(sequence_minibatch):
                columns = []
                data = []
                for j, x in enumerate(sequence['x']):
                    # Each data item will be a column in the table
                    columns.append(f'x_{j}')
                    data.append(x)
                logger.log_table(f'training sequence {i}', columns, [data])
                if i > logger_cfg['sample_data_size']:
                    break
            break

def finalize(logger, model, cfg):
    if cfg['logger']['type'] == 'wandb' and cfg['logger']['save_model']:
        logger.save_model(model, 
                          cfg['logger']['save_model_name'],
                          cfg['logger']['run_path_file']
                          )
    logger.finalize('success')

def runner_parallel_init(cfg):
    # Normally not needed, but if you have multiple
    # runners in the same process, then this lets you
    # set that up.

    # At the moment, this is only needed for two things.
    # First, we set the number of threads.
    # IMPORTANT:  This is to make pytorch not use multiple threads,
    # which IS THE DEFAULT!  Of course, this can mess up
    # if we are controlling the parallelism with Dask, Joblib, etc.
    torch.set_num_threads(cfg['base']['num_threads'])

    # Second, we might need to setup the logger.  For example
    # wandb needs to be setup for parallel processing.
    LoggerFactoryParallelSetup(cfg['logger'])

def runner_main(cfg):
    if cfg['base']['base_dir']=='default':
        base_dir = pathlib.Path().home() / 'iterativenn_logs'
    else:
        base_dir = pathlib.Path(cfg['base']['base_dir'])

    default_root_dir = (base_dir / cfg['logger']['name'])
    default_root_dir.mkdir(parents=True, exist_ok=True)

    if cfg['base']['seed'] != 'None':
        torch.manual_seed(cfg['base']['seed'])

    # Get the pytorch lightning data module
    data_module = DataFactory(cfg['data'])

    # Get the pytorch lightning model
    model = ModelFactory(cfg['model'])

    # This is the pytorch lightning logger
    logger = LoggerFactory(cfg['logger'], model, default_root_dir)

    # Log a minibatch of data to the logger
    log_sample_data(cfg['logger'], logger, data_module)

    # Put the hyper parameters into the logger
    cfg_dict = omegaconf.OmegaConf.to_container(cfg)
    logger.log_hyperparams(cfg_dict)

    # Log the CUDA setup
    accelerator, devices = get_cuda(cfg['base'], logger)

    # Check to see if the model has a function to count the number of trainable parameters
    # This is from 
    # https://stackoverflow.com/questions/5268404/what-is-the-fastest-way-to-check-if-a-class-has-a-function-defined
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

    # Sometimes we just want to see in the setup works correctly.
    # This also let's us just get the number of parameters
    if cfg['base']['max_epochs'] > 0:
        trainer=train(cfg['base'], logger, model, 
                    data_module, default_root_dir, accelerator, devices)

        # See the comments above.
        # test_model(cfg['model'], cfg['data'], trainer, logger, data_module)

        finalize(logger, model, cfg)
