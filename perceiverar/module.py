from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import einsum

from einops import rearrange, repeat

from gluonts.core.component import validated
from gluonts.torch.scaler import MeanScaler, NOPScaler
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.util import lagged_sequence_values

# helper functions
def exists(val):
    return val is not None


# feedforward
def FeedForward(dim, mult=4, dropout=0.0):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias=False),
    )


# attention
class CausalAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device=x.device, dtype=torch.bool).triu(
            j - i + 1
        )
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CausalPrefixAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        max_heads_process=2,
        dropout=0.0,
        cross_attn_dropout=0.0
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.max_heads_process = max_heads_process

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.cross_attn_dropout = cross_attn_dropout  # they drop out a percentage of the prefix during training, shown to help prevent overfitting

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, context_mask=None):
        batch, context_len, device = x.shape[0], context.shape[-2], x.device

        # take care of cross attention dropout
        if self.training and self.cross_attn_dropout > 0.0:
            rand = torch.zeros((batch, context_len), device=device).uniform_()
            keep_context_len = context_len - int(context_len * self.cross_attn_dropout)
            keep_indices = rand.topk(keep_context_len, dim=-1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()

            context = rearrange(context[keep_mask], "(b n) d -> b n d", b=batch)

            if exists(context_mask):
                context_mask = rearrange(
                    context_mask[keep_mask], "(b n) -> b n", b=batch
                )

        # normalization
        x = self.norm(x)
        context = self.context_norm(context)

        # derive queries, keys, values
        q = self.to_q(x)

        k_input, v_input = self.to_kv(x).chunk(2, dim=-1)
        k_context, v_context = self.to_kv(context).chunk(2, dim=-1)

        k = torch.cat((k_context, k_input), dim=1)
        v = torch.cat((v_context, v_input), dim=1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        q = q * self.scale

        # take care of masking
        i, j = q.shape[-2], k.shape[-2]
        mask_value = -torch.finfo(q.dtype).max

        if exists(context_mask):
            mask_len = context_mask.shape[-1]
            context_mask = F.pad(context_mask, (0, max(j - mask_len, 0)), value=True)
            context_mask = rearrange(context_mask, "b j -> b 1 1 j")

        causal_mask = torch.ones((i, j), device=x.device, dtype=torch.bool).triu(
            j - i + 1
        )

        # process in chunks of heads
        out = []
        max_heads = self.max_heads_process
        for q_chunk, k_chunk, v_chunk in zip(
            q.split(max_heads, dim=1),
            k.split(max_heads, dim=1),
            v.split(max_heads, dim=1),
        ):
            sim = einsum("b h i d, b h j d -> b h i j", q_chunk, k_chunk)

            if exists(context_mask):
                sim = sim.masked_fill(~context_mask, mask_value)

            sim = sim.masked_fill(causal_mask, mask_value)

            attn = sim.softmax(dim=-1)
            attn = self.dropout(attn)

            out_chunk = einsum("b h i j, b h j d -> b h i d", attn, v_chunk)
            out.append(out_chunk)

        # concat all the heads together
        out = torch.cat(out, dim=1)

        # merge heads and then combine with linear
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class PerceiverARModel(nn.Module):
    """
    Module implementing the PerceiverAR model.

    Parameters
    ----------
    freq
        String indicating the sampling frequency of the data to be processed.
    context_length
        Length of the RNN unrolling prior to the forecast date.
    prediction_length
        Number of time points to predict.
    num_feat_dynamic_real
        Number of dynamic real features that will be provided to ``forward``.
    num_feat_static_real
        Number of static real features that will be provided to ``forward``.
    num_feat_static_cat
        Number of static categorical features that will be provided to
        ``forward``.
    cardinality
        List of cardinalities, one for each static categorical feature.
    embedding_dimension
        Dimension of the embedding space, one for each static categorical
        feature.
    num_layers
        Number of layers in the RNN.
    hidden_size
        Size of the hidden layers in the RNN.
    dropout_rate
        Dropout rate to be applied at training time.
    distr_output
        Type of distribution to be output by the model at each time step
    lags_seq
        Indices of the lagged observations that the RNN takes as input. For
        example, ``[1]`` indicates that the RNN only takes the observation at
        time ``t-1`` to produce the output for time ``t``; instead,
        ``[1, 25]`` indicates that the RNN takes observations at times ``t-1``
        and ``t-25`` as input.
    scaling
        Whether to apply mean scaling to the observations (target).
    num_parallel_samples
        Number of samples to produce when unrolling the RNN in the prediction
        time range.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        depth: int,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        embedding_dimension: Optional[List[int]] = None,
        input_size: int = 1,
        perceive_depth: int = 1,
        heads: int = 2,
        perceive_max_heads_process: int = 2,
        ff_mult: int = 1,
        hidden_size: int = 32,
        dropout_rate: float = 0.1,
        cross_attn_dropout: float = 0.1,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.target_shape = distr_output.event_shape
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.past_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        dim_head = input_size * len(self.lags_seq) + self._number_of_features

        self.perceive_layers = nn.ModuleList([])
        for _ in range(perceive_depth):
            self.perceive_layers.append(
                nn.ModuleList(
                    [
                        CausalPrefixAttention(
                            dim=dim_head,
                            dim_head=hidden_size,
                            heads=heads,
                            max_heads_process=perceive_max_heads_process,
                            dropout=dropout_rate,
                            cross_attn_dropout=cross_attn_dropout,
                        ),
                        FeedForward(dim_head, mult=ff_mult, dropout=dropout_rate),
                    ]
                )
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CausalAttention(
                            dim=dim_head, dim_head=hidden_size, heads=heads
                        ),
                        FeedForward(dim_head, mult=ff_mult, dropout=dropout_rate),
                    ]
                )
            )

        self.param_proj = distr_output.get_args_proj(dim_head)

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + self.input_size  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def lagged_perciever(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Applies the underlying RNN to the provided target data and covariates.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length, *target_shape)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        future_target
            (Optional) tensor of future target values,
            shape: ``(batch_size, prediction_length, *target_shape)``.

        Returns
        -------
        Tuple
            A tuple containing, in this order:
            - Parameters of the output distribution
            - Scaling factor applied to the target
            - Raw output of the RNN
            - Static input to the RNN
            - Output state from the RNN
        """
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, loc, scale = self.scaler(context, observed_context)

        prior_input = past_target[:, : -self.context_length] / scale
        input = (
            torch.cat((context, future_target[:, :-1]), dim=1) / scale
            if future_target is not None
            else context / scale
        )

        embedded_cat = self.embedder(feat_static_cat)
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, log_scale),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, input.shape[1], -1)

        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, -self.context_length + 1 :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_time_feat is not None
            else past_time_feat[:, -self.context_length + 1 :, ...]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)
        lags = lagged_sequence_values(
            self.lags_seq, prior_input, input, dim=-1
        )
        perciever_input = torch.cat((lags, features), dim=-1)

        prefix, x = (
            perciever_input[:, : self.context_length - 1, ...],
            perciever_input[:, self.context_length - 1 :, ...],
        )

        # initial perceiver attention and feedforward (one cross attention)
        for cross_attn, ff in self.perceive_layers:
            x = cross_attn(x, prefix) + x
            x = ff(x) + x

        # layers
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        # output
        params = self.param_proj(x)
        return (params, scale, static_feat, perciever_input)

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        """
        Instantiate the output distribution

        Parameters
        ----------
        params
            Tuple of distribution parameters.
        scale
            (Optional) scale tensor.
        trailing_n
            If set, the output distribution is created only for the last
            ``trailing_n`` time points.

        Returns
        -------
        torch.distributions.Distribution
            Output distribution from the model.
        """
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Invokes the model on input data, and produce outputs future samples.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length, *target_shape)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        num_parallel_samples
            How many future samples to produce.
            By default, self.num_parallel_samples is used.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        params, scale, static_feat, prefix = self.lagged_perciever(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_prefix = prefix.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_params = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=0) for s in params
        ]
        distr = self.output_distribution(
            repeated_params, trailing_n=1, scale=repeated_scale
        )
        next_sample = distr.sample()
        future_samples = [next_sample]

        # greedy sampling
        for k in range(1, self.prediction_length):
            scaled_next_sample = next_sample / repeated_scale
            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )
            next_lags = lagged_sequence_values(
                self.lags_seq,
                repeated_past_target,
                scaled_next_sample,
                dim=-1
            )

            next_x = torch.cat((next_lags, next_features), dim=-1)

            x = next_x
            for cross_attn, ff in self.perceive_layers:
                x = cross_attn(x, repeated_prefix) + x
                x = ff(x) + x

            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x

            repeated_prefix = torch.cat((repeated_prefix, next_x), dim=1)
            repeated_past_target = torch.cat(
                (repeated_past_target, scaled_next_sample), dim=1
            )
            params = self.param_proj(x)
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            future_samples.append(next_sample)

        future_samples_concat = torch.cat(future_samples, dim=1)

        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length) + self.target_shape,
        )
