from typing import Optional
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.util import lagged_sequence_values, unsqueeze_expand
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import StudentTOutput

llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=2048),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}



def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Derived from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L86
    https://github.com/huggingface/transformers/blob/main/LICENSE

    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    
    return alibi.reshape(batch_size, num_heads, 1, seq_length).to(dtype)
    # HF original output
    # return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alibi: torch.Tensor,
    beta: float,
    attn_mask: torch.Tensor=None,
    dropout_p: float=0.0,
    is_causal: bool=False,
    scale: int=None
) -> torch.Tensor:
    """
    Derived from: https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L4902
    """

    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=attn_bias.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1)
    attn_weight = beta * alibi + scale_factor * attn_weight

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


@dataclass
class LTSMConfig:
    feature_size: int = 3 + 6  # target + loc + scale + time features
    block_size: int = 2048
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096


class Block(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, alibi: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), alibi)
        x = x + self.mlp(self.rms_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache: Optional[torch.Tensor] = None

        # Alibi beta term
        # TODO: Move it to model config
        self.beta = 1

    def forward(self, x: torch.Tensor, alibi: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = scaled_dot_product_attention(
            q, k, v, alibi, self.beta, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        # keep RMSNorm in float32
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class LagGPTAlibiModel(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        distr_output=StudentTOutput(),
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()
        self.lags_seq = sorted(
            list(
                set(
                    get_lags_for_frequency(freq_str="Q", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="M", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="W", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="D", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="H", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="T", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="S", num_default_lags=1)
                )
            )
        )
        config = LTSMConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            block_size=context_length + prediction_length,
            feature_size=input_size * (len(self.lags_seq)) + 2,
        )
        self.n_head = n_head
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_parallel_samples = num_parallel_samples

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_args_proj(config.n_embd)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.feature_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def prepare_input(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ):
        scaled_past_target, loc, scale = self.scaler(past_target, past_observed_values)

        if future_target is not None:
            future_length = future_target.shape[1]
            input = torch.cat(
                (
                    scaled_past_target[..., -self.context_length :],
                    (future_target[..., : future_length - 1] - loc) / scale,
                ),
                dim=-1,
            )
        else:
            input = scaled_past_target[..., -self.context_length :]

        prior_input = (past_target[..., : -self.context_length] - loc) / scale
        lags = lagged_sequence_values(self.lags_seq, prior_input, input, dim=-1)

        static_feat = torch.cat((loc.abs().log1p(), scale.log()), dim=-1)
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=lags.shape[-2]
        )

        return torch.cat((lags, expanded_static_feat), dim=-1), loc, scale

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
        )

        # forward the LLaMA model itself
        x = self.transformer.wte(
            transformer_input
        )  # token embeddings of shape (b, t, n_embd)
        
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        # Create attention mask for alibi
        attention_mask = torch.ones((B, T), device=transformer_input.device)

        # Compute alibi tensor for given attention and n_head
        alibi = build_alibi_tensor(attention_mask, self.n_head, dtype=transformer_input .dtype)

        for block in self.transformer.h:
            x = block(x, alibi)
        x = self.transformer.ln_f(x)

        params = self.param_proj(x)
        return params, loc, scale
