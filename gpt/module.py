from typing import Optional
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import StudentTOutput

llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=2048),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head,
                dtype=x.dtype,
                device=x.device,
            )

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
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
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)

    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.transpose(1, 2).type_as(x)


class GPTModel(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        num_feat_dynamic_real: int = 0,
        distr_output=StudentTOutput(),
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()
        config = LTSMConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            block_size=context_length + prediction_length - 1,
            feature_size=input_size + 2 + num_feat_dynamic_real,
        )
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

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        scaled_past, loc, scale = self.scaler(past_target, past_observed_values)
        log_abs_loc = loc.abs().log1p()
        log_scale = scale.log()
        static_feat = torch.cat((log_abs_loc, log_scale), dim=1)
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, past_time_feat.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, past_time_feat), dim=-1)
        transformer_input = torch.cat((scaled_past.unsqueeze(-1), features), dim=-1)

        # forward the LLaMA model itself
        x = self.transformer.wte(
            transformer_input
        )  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        params = self.param_proj(x)
        return params, loc, scale
