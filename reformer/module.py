import re
from typing import List, Optional

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from reformer_pytorch.reformer_pytorch import Reformer

ENC_PREFIX = "enc_"
DEC_PREFIX = "dec_"


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return bool(re.match(f"^{prefix}", str))


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)


def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        lambda x: string_begins_with(prefix, x), d
    )
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items()))
    )
    return kwargs_without_prefix, kwargs


def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs


def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if "input_mask" in enc_kwargs:
        dec_kwargs.setdefault("context_mask", enc_kwargs["input_mask"])
    return enc_kwargs, dec_kwargs, kwargs


class ReformerEncDec(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)

        assert (
            "dim" not in dec_kwargs and "dim" not in enc_kwargs
        ), "you must set the dim for both encoder and decoder"

        enc_kwargs["dim"] = dec_kwargs["dim"] = dim
        dec_kwargs["causal"] = True

        enc_kwargs.setdefault("bucket_size", 64)
        dec_kwargs.setdefault("bucket_size", enc_kwargs["bucket_size"] * 2)

        self.encoder = Reformer(**enc_kwargs)
        self.decoder = Reformer(**dec_kwargs)


class ReformerModel(nn.Module):
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
        # Reformer arguments
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.1,
        # univariate input
        input_size: int = 1,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: Optional[str] = "std",
        num_parallel_samples: int = 100,
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
        if scaling == "mean" or scaling is True:
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        # total feature size
        self.embed = nn.Linear(self.input_size * len(self.lags_seq) + self._number_of_features, d_model, bias=False)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model)

        # Reformer enc-decoder and mask initializer
        self.reformer = ReformerEncDec(
            dim=d_model,
            enc_heads=nhead,
            dec_heads=nhead,
            enc_depth=num_encoder_layers,
            dec_depth=num_decoder_layers,
            # enc_bucket_size=self.prediction_length // 2,
            # dec_bucket_size=(self.context_length + self.prediction_length) // 2,
            enc_bucket_size=self.context_length // 2,
            dec_bucket_size=self.prediction_length // 2,
            enc_lsh_dropout=dropout,
            enc_ff_dropout=dropout,
            enc_post_attn_dropout=dropout,
            enc_layer_dropout=dropout,
            dec_lsh_dropout=dropout,
            dec_ff_dropout=dropout,
            dec_post_attn_dropout=dropout,
            dec_layer_dropout=dropout,
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + self.input_size * 2  # the log(scale) and log(abs(loc)) features
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
        indices = [lag - shift for lag in self.lags_seq]

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

    def _check_shapes(
        self,
        prior_input: torch.Tensor,
        inputs: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> None:
        assert len(prior_input.shape) == len(inputs.shape)
        assert (
            len(prior_input.shape) == 2 and self.input_size == 1
        ) or prior_input.shape[2] == self.input_size
        assert (len(inputs.shape) == 2 and self.input_size == 1) or inputs.shape[
            -1
        ] == self.input_size
        assert (
            features is None or features.shape[2] == self._number_of_features
        ), f"{features.shape[2]}, expected {self._number_of_features}"

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
            torch.cat(
                (
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.context_length :, ...]
        )

        # target
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, loc, scale = self.scaler(context, observed_context)

        inputs = (
            (torch.cat((past_target, future_target), dim=1) - loc) / scale
            if future_target is not None
            else (past_target - loc) / scale
        )

        inputs_length = (
            self._past_length + self.prediction_length
            if future_target is not None
            else self._past_length
        )
        assert inputs.shape[1] == inputs_length

        subsequences_length = (
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
        )

        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        log_abs_loc = (
            loc.sign() * loc.abs().log1p()
            if self.input_size == 1
            else loc.squeeze(1).sign() * loc.squeeze(1).abs().log1p()
        )
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, log_abs_loc, log_scale),
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

        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        return transformer_inputs, loc, scale, static_feat

    def output_params(self, Reformer_inputs):
        projected_inputs = self.embed(Reformer_inputs)
        enc_input = projected_inputs[:, : self.context_length, ...]
        dec_input = projected_inputs[:, self.context_length :, ...]

        enc_out = self.reformer.encoder(enc_input)
        dec_output = self.reformer.decoder(dec_input, keys=enc_out)

        return self.param_proj(dec_output)

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, loc=loc, scale=scale)

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

        encoder_inputs, loc, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
        )

        enc_out = self.reformer.encoder(self.embed(encoder_inputs))

        repeated_loc = loc.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_feat.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        future_samples = []

        # greedy decoding
        for k in range(self.prediction_length):
            # self._check_shapes(repeated_past_target, next_sample, next_features)
            # sequence = torch.cat((repeated_past_target, next_sample), dim=1)

            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )

            # concat a tensor of zeros of seq len prediction_len - k + 1 to decoder_input
            B, _, L = reshaped_lagged_sequence.shape

            zero_pad = torch.zeros(
                (B, self.prediction_length - k - 1, L),
                device=reshaped_lagged_sequence.device,
            )

            reshaped_lagged_sequence = torch.cat(
                [reshaped_lagged_sequence, zero_pad], dim=1
            )

            # concat the full feat to this padded reshaped_lagged_seq
            decoder_input = torch.cat(
                (reshaped_lagged_sequence, repeated_features), dim=-1
            )

            output = self.reformer.decoder(self.embed(decoder_input), keys=repeated_enc_out)

            params = self.param_proj(output[:, k : k + 1])
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()

            repeated_past_target = torch.cat(
                (repeated_past_target, (next_sample - repeated_loc) / repeated_scale),
                dim=1,
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length) + self.target_shape,
        )
