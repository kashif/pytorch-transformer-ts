from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.torch.scaler import MeanScaler, NOPScaler
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder as BaseFeatureEmbedder


class FeatureEmbedder(BaseFeatureEmbedder):
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        concat_features = super(FeatureEmbedder, self).forward(features=features)

        if self._num_features > 1:
            features = torch.chunk(concat_features, self._num_features, dim=-1)
        else:
            features = [concat_features]

        return features


class GatedResidualNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        d_input: Optional[int] = None,
        d_output: Optional[int] = None,
        d_static: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        d_input = d_input or d_hidden
        d_static = d_static or 0
        if d_output is None:
            d_output = d_input
            self.add_skip = False
        else:
            if d_output != d_input:
                self.add_skip = True
                self.skip_proj = nn.Linear(in_features=d_input, out_features=d_output)
            else:
                self.add_skip = False

        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_input + d_static, out_features=d_hidden),
            nn.ELU(),
            nn.Linear(in_features=d_hidden, out_features=d_hidden),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_output * 2),
            nn.GLU(),
        )

        self.lnorm = nn.LayerNorm(d_output)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.add_skip:
            skip = self.skip_proj(x)
        else:
            skip = x

        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.mlp(x)
        x = self.lnorm(x + skip)
        return x


class VariableSelectionNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        n_vars: int,
        dropout: float = 0.0,
        add_static: bool = False,
    ):
        super().__init__()
        self.weight_network = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_input=d_hidden * n_vars,
            d_output=n_vars,
            d_static=d_hidden if add_static else None,
            dropout=dropout,
        )

        self.variable_network = nn.ModuleList(
            [
                GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)
                for _ in range(n_vars)
            ]
        )

    def forward(
        self, variables: List[torch.Tensor], static: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flatten = torch.cat(variables, dim=-1)
        if static is not None:
            static = static.expand_as(variables[0])
        weight = self.weight_network(flatten, static)
        weight = torch.softmax(weight.unsqueeze(-2), dim=-1)

        var_encodings = [net(var) for var, net in zip(variables, self.variable_network)]
        var_encodings = torch.stack(var_encodings, dim=-1)

        var_encodings = torch.sum(var_encodings * weight, dim=-1)

        return var_encodings, weight


class TemporalFusionEncoder(nn.Module):
    @validated()
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=d_input, hidden_size=d_hidden, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=d_input, hidden_size=d_hidden, batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        if d_input != d_hidden:
            self.skip_proj = nn.Linear(in_features=d_input, out_features=d_hidden)
            self.add_skip = True
        else:
            self.add_skip = False

        self.lnorm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        ctx_input: torch.Tensor,
        tgt_input: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
    ):
        ctx_encodings, states = self.encoder_lstm(ctx_input, states)

        if tgt_input is not None:
            tgt_encodings, _ = self.decoder_lstm(tgt_input, states)
            encodings = torch.cat((ctx_encodings, tgt_encodings), dim=1)
            skip = torch.cat((ctx_input, tgt_input), dim=1)
        else:
            encodings = ctx_encodings
            skip = ctx_input

        if self.add_skip:
            skip = self.skip_proj(skip)
        encodings = self.gate(encodings)
        encodings = self.lnorm(skip + encodings)
        return encodings


class TemporalFusionDecoder(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_hidden: int,
        d_var: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.enrich = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_static=d_var,
            dropout=dropout,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.att_net = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        self.att_lnorm = nn.LayerNorm(d_hidden)

        self.ff_net = nn.Sequential(
            GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        self.ff_lnorm = nn.LayerNorm(d_hidden)

        self.register_buffer(
            "attn_mask",
            self._generate_subsequent_mask(
                prediction_length, prediction_length + context_length
            ),
        )

    @staticmethod
    def _generate_subsequent_mask(
        target_length: int, source_length: int
    ) -> torch.Tensor:
        mask = (torch.triu(torch.ones(source_length, target_length)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        expanded_static = static.expand_as(x)
        # static.repeat((1, self.context_length + self.prediction_length, 1))

        skip = x[:, self.context_length :, ...]
        x = self.enrich(x, expanded_static)

        # does not work on GPU :-(
        # mask_pad = torch.ones_like(mask)[:, 0:1, ...]
        # mask_pad = mask_pad.repeat((1, self.prediction_length))
        # key_padding_mask = torch.cat((mask, mask_pad), dim=1).bool()

        query_key_value = x
        attn_output, _ = self.attention(
            query=query_key_value[:, self.context_length :, ...],
            key=query_key_value,
            value=query_key_value,
            # key_padding_mask=key_padding_mask,
            attn_mask=self.attn_mask if causal else None,
        )
        att = self.att_net(attn_output)

        x = x[:, self.context_length :, ...]
        x = self.att_lnorm(x + att)
        x = self.ff_net(x)
        x = self.ff_lnorm(x + skip)

        return x


class TFTModel(nn.Module):
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
        # TFT inputs
        num_heads: int,
        embed_dim: int,
        variable_dim: int,
        dropout: float,
        # univariate input
        input_size: int = 1,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        self.input_size = input_size

        self.target_shape = distr_output.event_shape
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real

        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.history_length = context_length + max(self.lags_seq)

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=[variable_dim] * num_feat_static_cat,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output

        # projection networks
        self.target_proj = nn.Linear(
            in_features=input_size * len(self.lags_seq), out_features=variable_dim
        )

        self.dynamic_proj = nn.Linear(
            in_features=num_feat_dynamic_real, out_features=variable_dim
        )

        self.static_feat_proj = nn.Linear(
            in_features=num_feat_static_real + input_size, out_features=variable_dim
        )

        # variable selection networks
        self.past_selection = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=2,  # target and time features
            dropout=dropout,
            add_static=True,
        )

        self.future_selection = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=2,  # target and time features
            dropout=dropout,
            add_static=True,
        )

        self.static_selection = VariableSelectionNetwork(
            d_hidden=variable_dim,
            n_vars=2,  # cat, static_feat
            dropout=dropout,
        )

        # Static Gated Residual Networks
        self.selection = GatedResidualNetwork(
            d_hidden=variable_dim,
            dropout=dropout,
        )

        self.enrichment = GatedResidualNetwork(
            d_hidden=variable_dim,
            dropout=dropout,
        )

        self.state_h = GatedResidualNetwork(
            d_hidden=variable_dim,
            d_output=embed_dim,
            dropout=dropout,
        )

        self.state_c = GatedResidualNetwork(
            d_hidden=variable_dim,
            d_output=embed_dim,
            dropout=dropout,
        )

        # Encoder and Decoder network
        self.temporal_encoder = TemporalFusionEncoder(
            d_input=variable_dim,
            d_hidden=embed_dim,
        )
        self.temporal_decoder = TemporalFusionDecoder(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            d_hidden=embed_dim,
            d_var=variable_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # distribution output
        self.param_proj = distr_output.get_args_proj(embed_dim)

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

        # calculate scale
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, loc, scale = self.scaler(context, observed_context)

        # scale the target and create lag features of targets
        target = (
            torch.cat((past_target, future_target), dim=1) / scale
            if future_target is not None
            else past_target / scale
        )
        subsequences_length = (
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
        )

        lagged_target = self.get_lagged_subsequences(
            sequence=target,
            subsequences_length=subsequences_length,
        )
        lags_shape = lagged_target.shape
        reshaped_lagged_target = lagged_target.reshape(lags_shape[0], lags_shape[1], -1)

        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat(
            (feat_static_real, log_scale),
            dim=1,
        )

        # return the network inputs
        return (
            reshaped_lagged_target,  # target
            time_feat,  # dynamic real covariates
            scale,  # scale
            embedded_cat,  # static covariates
            static_feat,
        )

    def output_params(self, target, time_feat, embedded_cat, static_feat):
        target_proj = self.target_proj(target)

        past_target_proj = target_proj[:, : self.context_length, ...]
        future_target_proj = target_proj[:, self.context_length :, ...]

        time_feat_proj = self.dynamic_proj(time_feat)
        past_time_feat_proj = time_feat_proj[:, : self.context_length, ...]
        future_time_feat_proj = time_feat_proj[:, self.context_length :, ...]

        static_feat_proj = self.static_feat_proj(static_feat)

        static_var, _ = self.static_selection(embedded_cat + [static_feat_proj])
        static_selection = self.selection(static_var).unsqueeze(1)
        static_enrichment = self.enrichment(static_var).unsqueeze(1)

        past_selection, _ = self.past_selection(
            [past_target_proj, past_time_feat_proj], static_selection
        )

        future_selection, _ = self.future_selection(
            [future_target_proj, future_time_feat_proj], static_selection
        )

        c_h = self.state_h(static_var)
        c_c = self.state_c(static_var)
        states = [c_h.unsqueeze(0), c_c.unsqueeze(0)]

        enc_out = self.temporal_encoder(past_selection, future_selection, states)

        dec_output = self.temporal_decoder(enc_out, static_enrichment)

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

        (
            target,
            time_feat,
            scale,
            embedded_cat,
            static_feat,
        ) = self.create_network_inputs(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
        )

        past_target_proj = self.target_proj(target)
        past_time_feat_proj = self.dynamic_proj(time_feat)
        future_time_feat_proj = self.dynamic_proj(future_time_feat)
        static_feat_proj = self.static_feat_proj(static_feat)

        static_var, _ = self.static_selection(embedded_cat + [static_feat_proj])
        static_selection = self.selection(static_var).unsqueeze(1)
        static_enrichment = self.enrichment(static_var).unsqueeze(1)

        past_selection, _ = self.past_selection(
            [past_target_proj, past_time_feat_proj], static_selection
        )

        c_h = self.state_h(static_var)
        c_c = self.state_c(static_var)
        states = [c_h.unsqueeze(0), c_c.unsqueeze(0)]

        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_time_feat_proj = future_time_feat_proj.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_past_selection = past_selection.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_selection = static_selection.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_enrichment = static_enrichment.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_states = [
            s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
            for s in states
        ]

        # greedy decoding
        future_samples = []
        for k in range(self.prediction_length):
            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1 + k,
                shift=1,
            )
            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )
            reshaped_lagged_sequence_proj = self.target_proj(reshaped_lagged_sequence)

            next_time_feat_proj = repeated_time_feat_proj[:, : k + 1]
            future_selection, _ = self.future_selection(
                [reshaped_lagged_sequence_proj, next_time_feat_proj],
                repeated_static_selection,
            )
            enc_out = self.temporal_encoder(
                repeated_past_selection, future_selection, repeated_states
            )

            dec_output = self.temporal_decoder(
                enc_out, repeated_static_enrichment, causal=False
            )

            params = self.param_proj(dec_output[:, -1:])
            distr = self.output_distribution(params, scale=repeated_scale)

            next_sample = distr.sample()
            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length) + self.target_shape,
        )
