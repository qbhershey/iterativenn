from iterativenn.nn_modules.nlp import Transformer


def test_all_zeros_input():
    input_tensor = torch.zeros((32, 10, 768))
    model = Transformer()
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape


def test_random_input():
    input_tensor = torch.randn((16, 20, 768))
    model = Transformer(num_layers=2, dropout=0.2)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape


def test_all_positive_input():
    input_tensor = torch.abs(torch.randn((8, 30, 768)))
    model = Transformer(num_heads=4, hidden_dim=1024)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape


def test_all_negative_input():
    input_tensor = torch.randn((4, 25, 768)) * -1
    model = Transformer(num_layers=3, dropout=0.3)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape


def test_large_input():
    input_tensor = torch.randn((2, 15, 768)) * 10
    model = Transformer(num_heads=16, hidden_dim=4096)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape


import torch
from iterativenn.nn_modules.nlp import GPT2Block, GPT2ModelEmbeddings, GPT2LMHead
from transformers import GPT2LMHeadModel

def test_from_pretrained_lm_head():
    # Hugging face weights
    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    model_hf_lm_head_state_dict = model_hf.lm_head.state_dict()

    # NanoGPT weights
    block = GPT2LMHead.from_pretrained()
    # check when block is loaded it has state dict upto date with hugging face weights
    block_state_dict = block.state_dict()

    assert torch.equal(block_state_dict["lm_head.weight"], model_hf_lm_head_state_dict["weight"])

def test_from_pretrained_embed():
    # Hugging face weights
    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    model_hf_wte_state_dict = model_hf.transformer.wte.state_dict()
    model_hf_wpe_state_dict = model_hf.transformer.wpe.state_dict()


    # NanoGPT weights
    block = GPT2ModelEmbeddings.from_pretrained()
    block_state_dict = block.state_dict()

    assert torch.equal(block_state_dict["word_embeddings.weight"], model_hf_wte_state_dict["weight"])
    assert torch.equal(block_state_dict["position_embeddings.weight"], model_hf_wpe_state_dict["weight"])



def test_from_pretrained_block():
    # Hugging face weights
    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    model_hf_state_dict = model_hf.transformer.h[-1].state_dict()
    model_hf_state_dict_keys = model_hf_state_dict.keys()

    # NanoGPT weights
    block = GPT2Block.from_pretrained(block_idx=-1)
    block_state_dict = block.state_dict()
    block_state_dict_keys = block_state_dict.keys()

    # Compare the two state dicts
    print(set(block_state_dict_keys), set(model_hf_state_dict_keys))

    for key in block_state_dict_keys:
        block_tensor = block_state_dict[key]
        hf_tensor = model_hf_state_dict[key]
        print(key)
        if key.endswith('.weight') and key.split('.')[-2] in ['c_attn', 'c_fc', 'c_proj']:
            # transpose the tensor since the Hugging Face implementation uses a Conv1D module
            hf_tensor = hf_tensor.transpose(-2, -1)
        assert torch.equal(block_tensor, hf_tensor)