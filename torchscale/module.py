from typing import List, Optional, Dict, Any
import math

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler

from apex.normalization import FusedLayerNorm as LayerNorm

from torchscale.architecture.config import EncoderDecoderConfig
from torchscale.component.relative_position_bias import RelativePositionBias
from torchscale.architecture.encoder import EncoderLayer
from torchscale.architecture.decoder import DecoderLayer
from torchscale.component.multiway_network import MultiwayWrapper
from torchscale.architecture.utils import init_bert_params


class Encoder(nn.Module):
    def __init__(self, args, is_moe_layer=False, is_encoder_decoder=True):
        super().__init__()

        self.dropout_module = torch.nn.Dropout(args.dropout, inplace=True)

        embed_dim = args.encoder_embed_dim

        self.layers = nn.ModuleList([])
        moe_freq = args.moe_freq
        for i in range(args.encoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_encoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                    is_encoder_decoder=is_encoder_decoder,
                )
            )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = MultiwayWrapper(args, LayerNorm(embed_dim))
        else:
            self.layer_norm = None

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.encoder_attention_heads,
            )
        else:
            self.relative_position = None

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = (
                    math.pow(
                        math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
                    )
                    / 1.15
                )
            else:
                init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if args.subln:
            if is_encoder_decoder:
                init_scale = math.sqrt(
                    math.log(3 * args.decoder_layers)
                    * math.log(2 * args.encoder_layers)
                    / 3
                )
            else:
                init_scale = math.sqrt(math.log(args.encoder_layers * 2))
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.mul_(init_scale)

    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = EncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        return layer

    def forward(self, enc_input, encoder_padding_mask=None):
        x = enc_input.transpose(0, 1)  # (B, T, C) -> (T, B, C)

        rel_pos_bias = None
        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(
                batch_size=x.size(1), qlen=x.size(0), klen=x.size(0)
            )

        for layer in self.layers:
            x, _ = layer(
                x, encoder_padding_mask=encoder_padding_mask, rel_pos=rel_pos_bias
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x  # (T, B, C)


class Decoder(nn.Module):
    def __init__(self, args, is_encoder_decoder=True):
        super().__init__()

        embed_dim = args.decoder_embed_dim

        self.dropout_module = torch.nn.Dropout(args.dropout, inplace=True)

        if args.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.decoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_decoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                    is_encoder_decoder=is_encoder_decoder,
                )
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.self_attn_relative_position = None
        self.cross_attn_relative_position = None

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.self_attn_relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.decoder_attention_heads,
            )
            if is_encoder_decoder:
                self.cross_attn_relative_position = RelativePositionBias(
                    num_buckets=args.rel_pos_buckets,
                    max_distance=args.max_rel_pos,
                    n_heads=args.decoder_attention_heads,
                )

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = math.pow(12.0 * args.decoder_layers, 0.25)
            else:
                init_scale = math.pow(8.0 * args.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if args.subln:
            if is_encoder_decoder:
                init_scale = math.sqrt(math.log(args.decoder_layers * 3))
            else:
                init_scale = math.sqrt(math.log(args.decoder_layers * 2))
            for name, p in self.named_parameters():
                if "encoder_attn" in name:
                    continue
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.mul_(init_scale)

    def build_decoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )

        return layer

    def forward(self, dec_input, encoder_out, incremental_state=None):
        x = dec_input.transpose(0, 1)  # (B, T, C) -> (T, B, C)

        # relative position
        self_attn_rel_pos_bias = None
        slen = dec_input.size(1)
        if self.self_attn_relative_position is not None:
            self_attn_rel_pos_bias = self.self_attn_relative_position(
                batch_size=x.size(1), qlen=slen, klen=slen
            )
            if incremental_state is not None:
                self_attn_rel_pos_bias = self_attn_rel_pos_bias[:, -1:, :]
        cross_attn_rel_pos_bias = None
        if self.cross_attn_relative_position is not None:
            cross_attn_rel_pos_bias = self.cross_attn_relative_position(
                batch_size=x.size(1),
                qlen=slen,
                klen=encoder_out["encoder_out"].size(0),
            )
            if incremental_state is not None:
                cross_attn_rel_pos_bias = cross_attn_rel_pos_bias[:, -1:, :]

        # decoder layers
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = torch.triu(
                    torch.zeros([x.size(0), x.size(0)])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(x),
                    1,
                )
            else:
                self_attn_mask = None
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, _, _, _ = layer(
                x,
                encoder_out,
                None,
                incremental_state[idx] if incremental_state is not None else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=None,
                self_attn_rel_pos=self_attn_rel_pos_bias,
                cross_attn_rel_pos=cross_attn_rel_pos_bias,
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x.transpose(0, 1)  # (T, B, C) -> (B, T, C)


class TorchscaleModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        # torchscale config
        enc_dec_config: Dict[str, Any],
        input_size: int = 1,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 1,
    ) -> None:
        super().__init__()

        self.input_size = input_size

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
        self.history_length = context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        # total feature size
        d_model = self.input_size * len(self.lags_seq) + self._number_of_features

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model)

        config = EncoderDecoderConfig(**enc_dec_config)
        config.encoder_embed_dim = d_model
        config.decoder_embed_dim = d_model

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

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

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.
        shift: int
            shift the lags by this amount back.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = [l - shift for l in self.lags_seq]

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def create_network_inputs(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        # time feature
        time_feat = (
            past_time_feat[:, self._past_length - self.context_length :, ...]
            if future_time_feat is None or future_target is None
            else torch.cat(
                (
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
        )

        # target
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        # weights = torch.linspace(0.0001, 1, steps=observed_context.size(-1), device=observed_context.device)
        _, scale = self.scaler(context, observed_context)

        inputs = (
            torch.cat((past_target, future_target), dim=1) / scale
            if future_target is not None
            else past_target / scale
        )

        inputs_length = (
            self._past_length + self.prediction_length
            if future_target is not None
            else self._past_length
        )
        assert inputs.shape[1] == inputs_length

        subsequences_length = (
            self.context_length
            if future_time_feat is None or future_target is None
            else self.context_length + self.prediction_length
        )

        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, log_scale),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, time_feat.shape[1], -1
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        # self._check_shapes(prior_input, inputs, features)
        # sequence = torch.cat((prior_input, inputs), dim=1)

        lagged_sequence = self.get_lagged_subsequences(
            sequence=inputs,
            subsequences_length=subsequences_length,
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], -1
        )

        if features is None:
            transformer_inputs = reshaped_lagged_sequence
        else:
            transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        return transformer_inputs, scale, static_feat

    def output_params(self, transformer_inputs):
        enc_input = transformer_inputs[:, : self.context_length, ...]
        dec_input = transformer_inputs[:, self.context_length :, ...]

        enc_out = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_out)

        return self.param_proj(dec_output)

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)

    # for prediction
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
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        encoder_inputs, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
        )

        enc_out = self.encoder(encoder_inputs)

        params = self.param_proj(enc_out.transpose(0, 1))  # (B, T, D)
        distr = self.output_distribution(params, trailing_n=1)

        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=1
        )

        future_samples = []

        for k in range(self.prediction_length):
            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )

            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )

            decoder_input = torch.cat((reshaped_lagged_sequence, next_features), dim=-1)

            output = self.decoder(decoder_input, repeated_enc_out)

            params = self.param_proj(output)
            distr = self.output_distribution(params)
            next_sample = distr.sample()

            repeated_past_target = torch.cat((repeated_past_target, next_sample), dim=1)
            future_samples.append(next_sample)

        unscaled_future_samples = torch.cat(future_samples, dim=1) * repeated_scale
        return unscaled_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length) + self.target_shape,
        )
