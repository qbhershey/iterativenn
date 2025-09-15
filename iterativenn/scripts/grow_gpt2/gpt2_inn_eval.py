import gc
import pathlib
from dataclasses import asdict

import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from iterativenn.lit_modules.IteratedModuleGPT2 import ConfigCallbacks
from iterativenn.lit_modules.IteratedModuleGPT2 import IteratedModelGPT2
from iterativenn.nn_modules.Sequential2D import Sequential2D
from iterativenn.nn_modules.nlp import GPT2ModelEmbeddings, GPTConfig
from iterativenn.utils.DataModules import LMDataModuleGPT2
from iterativenn.utils.gpt_config_utils import (GPT2BlockUpdater, sparse_sequential2D_from_masked_linear)
from iterativenn.utils.logger_factory import LoggerFactory


def runner_main(cfg_logger):
    torch.set_float32_matmul_precision('medium')
    # 3 for testing purpose
    # 13 gives performance same as GPT2 hugging face model
    gpt_config = GPTConfig()
    GPT_LAYERS = gpt_config.n_layer+1 # plus 1 for language model head # 3

    gpt_dim = gpt_config.n_embd*gpt_config.seq_len
    gpt_out_dim = gpt_config.vocab_size * gpt_config.seq_len

    # load pretrained embeddings
    gpt_embed = GPT2ModelEmbeddings.from_pretrained()
    gpt2_block_updater = GPT2BlockUpdater(size=GPT_LAYERS)
    operator_module = sparse_sequential2D_from_masked_linear(dim_z=gpt_dim)
    operator_ids = [(1, 11)]
    operator_lrs = [1e-2]

    gpt2_block_updater.init_gpt2_blocks()
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
    in_features_list = [gpt_dim]*GPT_LAYERS + [gpt_out_dim]
    out_features_list = [gpt_dim]*GPT_LAYERS + [gpt_out_dim]
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
                "idx_list": range(gpt_dim*GPT_LAYERS, (gpt_dim*GPT_LAYERS)+gpt_out_dim),
                "sequence_position": 'all',
                "logits_shape": (gpt_config.batch_size, gpt_config.seq_len, gpt_config.vocab_size)
            },
            "initialization": {
                "func": "zeros",
                "size": gpt_dim*GPT_LAYERS+gpt_out_dim,
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
                "idx_list": range(gpt_dim*GPT_LAYERS, (gpt_dim*GPT_LAYERS) + gpt_out_dim)
            },
            "optimization":{
                "func": "customLR",
                "block_lr": blocks_lr
            }
        }
    }

    model = Sequential2D.from_config(cfg["sequential2D"])

    checkpoint_path = "/home/hnpatha/dev/iterativenn/LoggerFacade/0.1/checkpoints"
    state_dict = torch.load(f'{checkpoint_path}/epoch=19-step=9640-v2.ckpt')
    state_dict_tmp = {}
    for k, v in state_dict['state_dict'].items():
        if "input" in k:
            pass
        else:
            state_dict_tmp.update({str(k).lstrip("model.") : v})

    model.load_state_dict(state_dict_tmp)

    callbacks = ConfigCallbacks(cfg["callbacks"])
    model_nlp = IteratedModelGPT2(model, callbacks, gpt_embed, normalize_loss=False, optimizer='Adam', iterations=GPT_LAYERS)
    base_dir = pathlib.Path().home() / 'iterativenn_logs'
    default_root_dir = (base_dir / cfg_logger['name'])
    default_root_dir.mkdir(parents=True, exist_ok=True)
    logger = LoggerFactory(cfg_logger, model, default_root_dir)
    logger.log_hyperparams({'gpt2_layers': GPT_LAYERS})
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

    trainer = Trainer(
        accelerator='auto',
        devices =[1] if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=20,
        logger=logger,
        enable_progress_bar=True,
        strategy=DDPStrategy(find_unused_parameters=True),
        check_val_every_n_epoch=2,
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=2)]
    )

    trainer.validate(model_nlp, data_module)



@hydra.main(version_base=None, config_name="wandb_hp.yaml", config_path="../conf/logger")
def main(cfg_logger):
    runner_main(cfg_logger)

if __name__ == '__main__':
    main()
