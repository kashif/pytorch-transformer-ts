from typing import Optional, List

import numpy as np

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler


class ValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model):
        super(ValueEmbedding, self).__init__()
        self.value_proj = nn.Linear(feature_size, d_model, bias=False)

    def forward(self, x):
        return self.value_proj(x)


class PositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class LagTransformerModel(nn.Module):
    @validated()
    def __init__(
        self,
        lags_seq: List[int],
        # transformer arguments
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        scaling: str,
        max_context_length: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        # univariate input
        input_size: int = 1,
        distr_output: DistributionOutput = StudentTOutput(),
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.target_shape = distr_output.event_shape

        self.lags_seq = lags_seq
        self.num_parallel_samples = num_parallel_samples

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        # total feature size
        feature_size = self.input_size * len(self.lags_seq) + self._number_of_features
        self.enc_dec_embedding = ValueEmbedding(
            feature_size=feature_size, d_model=d_model
        )

        self.pos_embedding = PositionalEmbedding(max_context_length, d_model)

        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model)

        # transformer enc-decoder and mask initializer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

    @property
    def _number_of_features(self) -> int:
        return self.input_size * 2  # the log(scale) and log1p(loc)

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
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ):
        _past_length = past_target.shape[1]
        prediction_length = future_target.shape[1] if future_target is not None else 0
        context_length = _past_length - max(self.lags_seq)

        # target
        context = past_target[:, max(self.lags_seq) :]
        observed_context = past_observed_values[:, max(self.lags_seq) :]
        _, loc, scale = self.scaler(context, observed_context)

        inputs = (
            (torch.cat((past_target, future_target), dim=1) - loc) / scale
            if future_target is not None
            else (past_target - loc) / scale
        )

        inputs_length = _past_length + prediction_length
        assert inputs.shape[1] == inputs_length
        subsequences_length = context_length + prediction_length
        lagged_sequence = self.get_lagged_subsequences(
            sequence=inputs,
            subsequences_length=subsequences_length,
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], -1
        )

        # embeddings
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()
        log1p_loc = (
            loc.abs().log1p() if self.input_size == 1 else loc.squeeze(1).abs().log1p()
        )

        static_feat = torch.cat((log1p_loc, log_scale), dim=1)
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, lags_shape[1], -1)

        transformer_inputs = torch.cat(
            (reshaped_lagged_sequence, expanded_static_feat), dim=-1
        )

        return transformer_inputs, loc, scale, static_feat

    def output_params(self, transformer_inputs, context_length: int):
        enc_input = self.enc_dec_embedding(transformer_inputs[:, :context_length, ...])
        enc_pos = self.pos_embedding(enc_input.size())

        dec_input = self.enc_dec_embedding(transformer_inputs[:, context_length:, ...])
        prediction_length = dec_input.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            prediction_length
        ).to(dec_input.device)
        dec_pos = self.pos_embedding(
            dec_input.size(), past_key_values_length=context_length
        )

        enc_out = self.transformer.encoder(enc_input + enc_pos)
        dec_output = self.transformer.decoder(
            dec_input + dec_pos, enc_out, tgt_mask=tgt_mask
        )

        return self.param_proj(dec_output)

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, loc=loc, scale=scale)
