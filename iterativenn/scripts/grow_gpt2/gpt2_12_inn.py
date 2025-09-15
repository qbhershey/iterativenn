import gc
import pathlib
from dataclasses import asdict

import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.strategies import DDPStrategy

from iterativenn.lit_modules.IteratedModuleGPT2 import ConfigCallbacks
from iterativenn.lit_modules.IteratedModuleGPT2 import IteratedModelGPT2, IteratedModelGPT2CachedGrow
from iterativenn.nn_modules.Sequential2D import Sequential2D
from iterativenn.nn_modules.nlp import GPT2ModelEmbeddings, GPTConfig
from iterativenn.utils.DataModules import LMDataModuleGPT2
from iterativenn.utils.gpt_config_utils import (GPT2BlockUpdater, sparse_sequential2D_from_masked_linear)
from iterativenn.utils.logger_factory import LoggerFactory
from iterativenn.lit_callbacks.cache import CustomWriter, CacheReader

import torch
import os


def runner_main(cfg_logger):
    torch.set_float32_matmul_precision('medium')
    # 3 for testing purpose
    # 13 gives performance same as GPT2 hugging face model

    gpt_config = GPTConfig() # Initialize GPTConfig
    GPT_LAYERS_START = 8
    GPT_LAYERS_END = gpt_config.n_layer+1
    GPT_LAYERS_WINDOW = GPT_LAYERS_END - GPT_LAYERS_START # plus 1 for language model head # 3

    gpt_dim = gpt_config.n_embd*gpt_config.seq_len  # usually 768*1024
    gpt_out_dim = gpt_config.vocab_size * gpt_config.seq_len # usually 50257*1024 for wiki103

    """
    Next, 
    - First, we are initializing the GPT2ModelEmbeddings from pretrained
    - Then we are initializing the GPT2BlockUpdater with size of GPT_LAYERS
    - Idea is to insert the operator_module to Sequential2D-GPT module at the block operator_ids
    - Then we are initializing the Sequential2D module with 3 layers of hidden_dim=30 and dim_z=gpt_dim
    - Then we are updating the block types, kwargs and lr for the operator_module
    - Then gpt2_block_updater is initialized with the gpt2 decoders 
    - And then we insert the operator_module to the gpt2_block_updater at the block operator_ids
    """

    # load pretrained embeddings
    gpt_embed = GPT2ModelEmbeddings.from_pretrained() # Initialize GPT2ModelEmbeddings from pretrained
    gpt2_block_updater = GPT2BlockUpdater(size=GPT_LAYERS_END) # Initialize GPT2BlockUpdater with size of GPT_LAYERS
    # what is sparse_sequential2D_from_masked_linear doing?
    # it is creating a Sequential2D module with 3 layers of hidden_dim=30 and dim_z=gpt_dim
    operator_module = sparse_sequential2D_from_masked_linear(dim_z=gpt_dim) # Initialize Sequential2D with dim_z=gpt_dim

    # Here we are updating the block types, kwargs and lr for the operator_module
    operator_ids = [(2,1)] # (2, 9) is the block where we are adding the operator_module which is sparse sequential2D
    operator_lrs = [(1e-2)] # learning rate for the operator_module is set to 1e-2

    gpt2_block_updater.init_gpt2_blocks(start=8) # Initialize GPT2BlockUpdater with GPT2 blocks in subdiagonal format
    for i in range(len(operator_ids)):
        operator_idx, operator_idy = operator_ids[i]
        gpt2_block_updater.update_block_types(operator_idx, operator_idy, 'Module')
        gpt2_block_updater.update_block_kwargs(operator_idx, operator_idy, {'module': operator_module}) # Inside-Mini
        gpt2_block_updater.update_block_lrs(operator_idx, operator_idy, operator_lrs[i])
        nparams = operator_module.number_of_trainable_parameters()
        print(f"operator params {nparams}")

    block_types = gpt2_block_updater.get_block_types()
    block_kwargs = gpt2_block_updater.get_block_kwargs()
    blocks_lr = gpt2_block_updater.get_block_lr()

    del gpt2_block_updater
    gc.collect()


    print(blocks_lr)
    in_features_list = [gpt_dim]*GPT_LAYERS_WINDOW + [gpt_out_dim]
    out_features_list = [gpt_dim]*GPT_LAYERS_WINDOW + [gpt_out_dim]
    cfg = {
        "sequential2D": {
            "in_features_list": in_features_list,
            "out_features_list": out_features_list,
            "input_hparams": gpt_config,
            "block_types": block_types,
            "block_kwargs": block_kwargs
        },
        "callbacks": {
            "loss": {
                "func": "CrossEntropyLoss",
                "idx_list": range(gpt_dim*GPT_LAYERS_WINDOW, (gpt_dim*GPT_LAYERS_WINDOW)+gpt_out_dim),
                "sequence_position": 'all',
                "logits_shape": (gpt_config.batch_size, gpt_config.seq_len, gpt_config.vocab_size)
            },
            "initialization": {
                "func": "zeros",
                "size": gpt_dim*GPT_LAYERS_WINDOW+gpt_out_dim,
                "seq_len": gpt_config.seq_len,
                "batch_size": gpt_config.batch_size,
            },
            "data": {
                "func": "insert",
                "idx_list": range(gpt_dim),
                "flatten_input": True,
                "batch_size": gpt_config.batch_size,
            },
            "output": {
                "func": "all",
                "idx_list": range(gpt_dim*GPT_LAYERS_WINDOW, (gpt_dim*GPT_LAYERS_WINDOW) + gpt_out_dim)
            },
            "optimization":{
                "func": "customLR",
                "block_lr": blocks_lr
            }
        }
    }
    """
    Next,
    - First, Sequential2D from config is initialized with cfg["sequential2D"]
    - Then, ConfigCallbacks is initialized with cfg["callbacks"]
    - Then, IteratedModelGPT2 is initialized with model, callbacks, gpt_embed, normalize_loss=False, optimizer='Adam_customLR', iterations=GPT_LAYERS
    - LMDataModuleGPT2 is initialized , which is Data module for trainer 
    - Trainer is initialized and then trained
    """

    model = Sequential2D.from_config(cfg["sequential2D"])
    callbacks = ConfigCallbacks(cfg["callbacks"])
    model_nlp = IteratedModelGPT2CachedGrow(model, callbacks, gpt_embed, normalize_loss=False, optimizer='Adam_customLR', iterations=GPT_LAYERS_WINDOW)
    npm = model_nlp.number_of_trainable_parameters()
    base_dir = pathlib.Path().home() / 'iterativenn_logs'
    default_root_dir = (base_dir / cfg_logger['name'])
    default_root_dir.mkdir(parents=True, exist_ok=True)
    logger = LoggerFactory(cfg_logger, model, default_root_dir)
    logger.log_hyperparams({'gpt2_layers': GPT_LAYERS_WINDOW})
    logger.log_hyperparams({'operatorsAt': operator_ids})
    logger.log_hyperparams({'total_parameters': model_nlp.number_of_trainable_parameters()})
    logger.log_hyperparams(asdict(gpt_config))

    data_module = LMDataModuleGPT2(
        model_name_or_path="gpt2",
        pad_to_max_length=True,
        preprocessing_num_workers=8,
        overwrite_cache=False,
        max_seq_length=gpt_config.seq_len,
        mlm_probability=0.15,
        train_batch_size=gpt_config.batch_size,
        val_batch_size=gpt_config.batch_size,
        dataloader_num_workers=8,
    )
    pred_writer = CustomWriter(output_dir="outputs", write_interval="batch")

    # TODO: make this a part of config
    root_train = "/Users/hpathak/research/iterativenn/scripts/grow_gpt2/outputs/0"
    root_val = root_train
    root_test = root_train
    cache_reader = CacheReader(root_train, root_val, root_test, train_cache_size=16, mode="fixed_cache_size")

    trainer = Trainer(
        accelerator='auto',
        devices =1, #2 if torch.cuda.is_available() else 1,  # limiting got iPython runs
        max_epochs=20,
        logger=logger,
        enable_progress_bar=True,
        strategy='auto',#DDPStrategy(find_unused_parameters=True),
        check_val_every_n_epoch=20,
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=2), pred_writer, cache_reader]
    )

    trainer.fit(model_nlp, data_module)
    # trainer.validate(model_nlp, data_module)
    # trainer.predict(model_nlp, data_module)



@hydra.main(version_base=None, config_name="wandb_hp.yaml", config_path="../conf/logger")
def main(cfg_logger):
    runner_main(cfg_logger)

if __name__ == '__main__':
    main()