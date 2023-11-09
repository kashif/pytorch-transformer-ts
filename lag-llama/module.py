from typing import Optional, List
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from gluonts.torch.util import lagged_sequence_values, unsqueeze_expand
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import StudentTOutput


@dataclass
class LTSMConfig:
    feature_size: int = 3 + 6  # target + loc + scale + time features
    block_size: int = 2048
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    rope_scaling: Optional[dict] = None


class Block(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

        self.y_cache = None

    def forward(self, x: torch.Tensor, is_test: bool) -> torch.Tensor:
        if is_test and self.y_cache is not None:
            # Only use the most recent one, rest is in cache
            x = x[:, -1:]

        x = x + self.attn(self.rms_1(x), is_test)
        y = x + self.mlp(self.rms_2(x))

        if is_test:
            if self.y_cache is None:
                self.y_cache = y  # Build cache
            else:
                self.y_cache = torch.cat([self.y_cache, y], dim=1)[
                    :, 1:
                ]  # Update cache
        return y


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, device, dtype, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # query projections for all heads, but in a batch
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # key, value projections
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

        self.rope_scaling = config.rope_scaling
        self._rope_scaling_validation()

        self._init_rope()
        self.kv_cache = None

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.n_embd // self.n_head, max_position_embeddings=self.block_size
            )
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "nope":
                self.rotary_emb = None
            elif scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.n_embd // self.n_head,
                    max_position_embeddings=self.block_size,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.n_embd // self.n_head,
                    max_position_embeddings=self.block_size,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
            "linear",
            "dynamic",
            "nope",
        ]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_type in ["linear", "dynamic"]:
            if (
                rope_scaling_factor is None
                or not isinstance(rope_scaling_factor, float)
                or rope_scaling_factor < 1.0
            ):
                raise ValueError(
                    f"`rope_scaling`'s factor field must be an float >= 1, got {rope_scaling_factor}"
                )

    def forward(self, x: torch.Tensor, is_test: bool) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if is_test:
            # Optimized for single next prediction
            q = self.q_proj(x[:, -1:])

            if self.kv_cache is not None:
                # Update cache
                k, v = self.kv_proj(x[:, -1:]).split(self.n_embd, dim=2)
                k = torch.cat([self.kv_cache[0], k], dim=1)[:, 1:]
                v = torch.cat([self.kv_cache[1], v], dim=1)[:, 1:]
                self.kv_cache = k, v
            else:
                # Build cache
                k, v = self.kv_proj(x).split(self.n_embd, dim=2)
                self.kv_cache = k, v
        else:
            # Business as usual
            q = self.q_proj(x)
            k, v = self.kv_proj(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, -1, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, -1, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, -1, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(device=v.device, dtype=v.dtype, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

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


class LagLlamaModel(nn.Module):
    def __init__(
        self,
        max_context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        lags_seq: List[int],
        rope_scaling=None,
        distr_output=StudentTOutput(),
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()
        self.lags_seq = lags_seq

        config = LTSMConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            block_size=max_context_length,
            feature_size=input_size * (len(self.lags_seq)) + 2 * input_size + 6,
            rope_scaling=rope_scaling,
        )
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

    def prepare_input(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ):
        scaled_past_target, loc, scale = self.scaler(past_target, past_observed_values)

        if future_target is not None:
            input = torch.cat(
                (
                    scaled_past_target[..., max(self.lags_seq) :],
                    (future_target[..., :-1] - loc) / scale,
                ),
                dim=-1,
            )
        else:
            input = scaled_past_target[..., max(self.lags_seq) :]

        time_feat = (
            torch.cat(
                (
                    past_time_feat[..., max(self.lags_seq) :, :],
                    future_time_feat[..., :-1, :],
                ),
                dim=1,
            )
            if future_time_feat is not None
            else past_time_feat[..., max(self.lags_seq) :, :]
        )

        prior_input = (past_target[..., : max(self.lags_seq)] - loc) / scale
        lags = lagged_sequence_values(self.lags_seq, prior_input, input, dim=-1)

        static_feat = torch.cat((loc.abs().log1p(), scale.log()), dim=-1)
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=lags.shape[-2]
        )

        return torch.cat((lags, expanded_static_feat, time_feat), dim=-1), loc, scale

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
        is_test: bool = False,
    ) -> torch.Tensor:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )

        # forward the LLaMA model itself
        x = self.transformer.wte(
            transformer_input
        )  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x, is_test)
        x = self.transformer.ln_f(x)

        params = self.param_proj(x)
        return params, loc, scale

    def reset_cache(self) -> None:
        """
        Resets all cached key-values in attention.
        Has to be called after prediction loop in predictor
        """
        for block in self.transformer.h:
            block.y_cache = None
            block.attn.kv_cache = None
