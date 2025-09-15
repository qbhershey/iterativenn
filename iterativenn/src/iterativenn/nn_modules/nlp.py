import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from iterativenn.nn_modules.Sequential2D import Sequential2D


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self, x) -> Dict:
        """
        Instead of from hugging face's signature as below; x is used to get input batch

        #         input_ids: Optional[torch.LongTensor] = None,
        #         attention_mask: Optional[torch.LongTensor] = None,
        #         labels: Optional[torch.LongTensor] = None,
        #         token_type_ids: Optional[torch.LongTensor] = None,
        #         position_ids: Optional[torch.LongTensor] = None,
        #         inputs_embeds: Optional[torch.FloatTensor] = None,
        #         past_key_values_length: int = 0,
        """

        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        labels = x['labels']

        token_type_ids = None
        position_ids = None
        inputs_embeds = None
        past_key_values_length = 0

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # embeddings = torch.unsqueeze(embeddings, 1)
        # embeddings = torch.squeeze(embeddings)  # input -> (batch, 1(seq_len), embed) output-> (batch, embed)
        # for a seq len export a list of tensors

        # embeddings <batch, seq_len, embed_dim>
        # embeddings <batch, seq_len, embed_dim>
        return embeddings, labels

class Transformer(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, hidden_dim=2048, num_layers=6, dropout=0.1):
        super(Transformer, self).__init__()

        # Parameters
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads, dim_feedforward=self.hidden_dim, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, x):
        # Reshape input tensor to (seq_len, batch, input_dim)
        x = x.permute(1, 0, 2)
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        # Reshape output tensor to (batch, seq_len, input_dim)
        x = x.permute(1, 0, 2)
        return x

@dataclass
class GPTConfig:
    """
    This class contains the configuration for GPT.
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    seq_len: int = 16
    batch_size: int = 16
    embd_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-05
    pad_token_id: int = -100
    max_position_embeddings: int = 1024


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class GPT2DenseLayer(nn.Module):
    """ Feed-forward and residual MLP layer with normalization."""

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class GPT2Block(nn.Module):
    """ Pre-activation residual block."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPT2DenseLayer(config)
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.n_embd = config.n_embd
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # additional features for sequential 2d
        self.in_features = config.seq_len * config.n_embd
        self.out_features = config.seq_len * config.n_embd

    def forward(self, x):
        # B, s * 768
        # here we reshape twice so that it can be used as is with seq 2d
        x = x.view(self.batch_size, self.seq_len, self.n_embd)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = x.view(self.batch_size, self.seq_len * self.n_embd)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, block_idx=0):
        from transformers import GPT2LMHeadModel
        gpt_config = GPTConfig()
        block = cls(gpt_config)
        block_state_dict = block.state_dict()
        block_state_dict_keys = block.state_dict().keys()
        # discard this mask / buffer, not a param
        block_state_dict_keys = [k for k in block_state_dict_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        print(f"loading block {block_idx}")
        # Hugging face weights
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="gpt_tmp/")
        model_hf_state_dict = model_hf.transformer.h[block_idx].state_dict()
        model_hf_state_dict_keys = model_hf_state_dict.keys()


        # Load the Hugging Face weights into the NanoGPT block
        for k in block_state_dict_keys:
            if any(k.endswith(w) for w in transposed):
                assert block_state_dict[k].shape[::-1] == model_hf_state_dict[
                    k].shape, f"mismatched shape: {block_state_dict[k].shape[::-1]} != {model_hf_state_dict[k].shape}"
                with torch.no_grad():
                    block_state_dict[k].copy_(model_hf_state_dict[k].t())
            else:
                assert block_state_dict[k].shape == model_hf_state_dict[
                    k].shape, f"mismatched shape: {block_state_dict[k].shape} != {model_hf_state_dict[k].shape}"
                with torch.no_grad():
                    block_state_dict[k].copy_(model_hf_state_dict[k])

        return block


class GPT2LMHead(nn.Module):
    """
    Language Model Head for the transformer
    """
    def __init__(self, config):
        super().__init__()
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # init all weights
        self.apply(self._init_weights)
        self.in_features = config.seq_len * config.n_embd
        self.out_features = config.seq_len * config.vocab_size
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.n_embd = config.n_embd

    def forward(self, x):
        x = x.view(self.batch_size, self.seq_len, self.n_embd)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # logits <B, seq, vocab_size>
        logits = logits.view(self.batch_size, self.out_features)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @classmethod
    def from_pretrained(cls):
        from transformers import GPT2LMHeadModel
        gpt_config = GPTConfig()
        gpt_lm_head = cls(gpt_config)
        lm_state_dict = gpt_lm_head.state_dict()
        lm_state_dict_keys = gpt_lm_head.state_dict().keys()

        # Hugging face weights
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="gpt_tmp/")
        model_hf_lm_head_state_dict = model_hf.lm_head.state_dict()
        model_hf_lnf_head_state_dict = model_hf.transformer.ln_f.state_dict()

        for k in lm_state_dict_keys:
            if "lm_head" in k:
                with torch.no_grad():
                    lm_state_dict[k].copy_(model_hf_lm_head_state_dict["weight"])
                    print(f"loaded: {k}")
            elif "ln_f.weight" in k:
                with torch.no_grad():
                    lm_state_dict[k].copy_(model_hf_lnf_head_state_dict["weight"])
                    print(f"loaded: {k}")
            elif "ln_f.bias" in k:
                with torch.no_grad():
                    lm_state_dict[k].copy_(model_hf_lnf_head_state_dict["bias"])
                    print(f"loaded: {k}")

        return gpt_lm_head


class GPT2ModelEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.n_embd)

        #self.LayerNorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = position_ids.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else position_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeds
        #embeddings = self.LayerNorm(embeddings)
        #embeddings = self.dropout(embeddings)

        return embeddings

    @classmethod
    def from_pretrained(cls):
        from transformers import GPT2LMHeadModel
        gpt_config = GPTConfig()
        gpt_embed = cls(gpt_config)
        block_state_dict = gpt_embed.state_dict()
        block_state_dict_keys = gpt_embed.state_dict().keys()

        # Hugging face weights
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='gpt_tmp/')
        model_hf_wte_state_dict = model_hf.transformer.wte.state_dict()
        model_hf_wpe_state_dict = model_hf.transformer.wpe.state_dict()
        #model_hf_ln_f_state_dict = model_hf.transformer.ln_f.state_dict()

        # Load the Hugging Face weights into the NanoGPT block
        for k in block_state_dict_keys:
            if "word_embeddings" in k:
                with torch.no_grad():
                    block_state_dict[k].copy_(model_hf_wte_state_dict["weight"])
                    print(f"loaded: {k}")
            elif "position_embeddings" in k:
                with torch.no_grad():
                    block_state_dict[k].copy_(model_hf_wpe_state_dict["weight"])
                    print(f"loaded: {k}")
            else:
                print(f"passed: {k}")
        return gpt_embed
