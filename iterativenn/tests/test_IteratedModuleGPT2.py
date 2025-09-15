import gc

import pytest
import torch

from iterativenn.lit_modules.IteratedModuleGPT2 import ConfigCallbacks
from iterativenn.lit_modules.IteratedModuleGPT2 import IteratedModelGPT2
from iterativenn.nn_modules.nlp import GPT2ModelEmbeddings, GPTConfig, Sequential2D
from iterativenn.utils.DataModules import LMDataModuleGPT2
from iterativenn.utils.gpt_config_utils import GPT2BlockUpdater, sparse_sequential2D

GPT_LAYERS = 3 # 13 gives performance same as GPT2 hugging face model
gpt_config = GPTConfig()
gpt_dim = gpt_config.n_embd*gpt_config.seq_len
gpt_out_dim = gpt_config.vocab_size * gpt_config.seq_len

gpt_embed = GPT2ModelEmbeddings.from_pretrained()
operator_module = sparse_sequential2D(dim_z=gpt_dim)
gpt2_block_updater = GPT2BlockUpdater(size=GPT_LAYERS)
gpt2_block_updater.update_block_types(0, 2, 'Module')
gpt2_block_updater.update_block_kwargs(0, 2, {'module': operator_module}) # Inside-Mini
gpt2_block_updater.init_gpt2_blocks()
block_types = gpt2_block_updater.get_block_types()
block_kwargs = gpt2_block_updater.get_block_kwargs()

del gpt2_block_updater
gc.collect()

in_features_list = [gpt_dim]*GPT_LAYERS + [gpt_out_dim]
out_features_list = [gpt_dim]*GPT_LAYERS + [gpt_out_dim]
cfg = {
    "sequential2D": {
        "in_features_list": in_features_list,
        "out_features_list": out_features_list,
        "input_layer": {'model': gpt_embed},
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
    }
}


@pytest.fixture
def input_model():
    model = Sequential2D.from_config(cfg["sequential2D"])
    # Create callbacks object
    callbacks = ConfigCallbacks(cfg["callbacks"])
    return model, callbacks


def test_iterated_model_gpt2(input_model):
    # Create a PyTorch model
    model, callbacks = input_model
    # Test initialization of the model with valid inputs
    model_nlp = IteratedModelGPT2(model, callbacks, gpt_embed, normalize_loss=False, optimizer='Adam', iterations=GPT_LAYERS)
    assert model_nlp.model == model
    assert model_nlp.callbacks == callbacks
    assert model_nlp.normalize_loss == False
    assert model_nlp.optimizer == 'Adam'


def test_forward(input_model):
    # Test the forward method of the model
    model, callbacks = input_model

    data_module = LMDataModuleGPT2(
        model_name_or_path="gpt2",
        pad_to_max_length=True,
        preprocessing_num_workers=4,
        overwrite_cache=False,
        max_seq_length=gpt_config.seq_len,
        mlm_probability=0.15,
        train_batch_size=gpt_config.batch_size,
        val_batch_size=gpt_config.batch_size,
        dataloader_num_workers=4,
    )
    data_module.setup()

    train_dataloader = data_module.train_dataloader()

    for batch in train_dataloader:
        model_nlp = IteratedModelGPT2(model, callbacks, gpt_embed, normalize_loss=False, optimizer='Adam', iterations=GPT_LAYERS)
        y_batch = model_nlp.forward(batch)
        assert y_batch.shape == torch.Size([gpt_config.batch_size, gpt_out_dim])
        break


def test_training_step(input_model):
    # Create a dataloader

    model, callbacks = input_model

    data_module = LMDataModuleGPT2(
        model_name_or_path="gpt2",
        pad_to_max_length=True,
        preprocessing_num_workers=4,
        overwrite_cache=False,
        max_seq_length=gpt_config.seq_len,
        mlm_probability=0.15,
        train_batch_size=gpt_config.batch_size,
        val_batch_size=gpt_config.batch_size,
        dataloader_num_workers=4,
    )
    data_module.setup()

    val_dataloader = data_module.train_dataloader()

    for batch_idx, batch in enumerate(val_dataloader):
        model_nlp = IteratedModelGPT2(model, callbacks, gpt_embed, normalize_loss=False, optimizer='Adam', iterations=GPT_LAYERS)
        loss = model_nlp.training_step(batch, batch_idx, do_logging=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        print(loss)
        # TODO: handle in case batch size is not equal 16
        break


def test_validation_step(input_model):
    model, callbacks = input_model

    data_module = LMDataModuleGPT2(
        model_name_or_path="gpt2",
        pad_to_max_length=True,
        preprocessing_num_workers=1,
        overwrite_cache=False,
        max_seq_length=gpt_config.seq_len,
        mlm_probability=0.15,
        train_batch_size=gpt_config.batch_size,
        val_batch_size=gpt_config.batch_size,
        dataloader_num_workers=1,
    )
    data_module.setup()

    val_dataloader = data_module.val_dataloader()

    for batch_idx, batch in enumerate(val_dataloader):
        model_nlp = IteratedModelGPT2(model, callbacks, gpt_embed, normalize_loss=False, optimizer='Adam', iterations=GPT_LAYERS)
        loss = model_nlp.validation_step(batch, batch_idx, do_logging=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        # TODO: handle in case batch size is not equal 16
        break