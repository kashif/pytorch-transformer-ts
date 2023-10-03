# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.scaler import Scaler, MeanScaler, NOPScaler, StdScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import (
    lagged_sequence_values,
    repeat_along_dim,
    take_last,
    unsqueeze_expand,
)
from gluonts.itertools import prod
from gluonts.model import Input, InputSpec
from hflayers import Hopfield
from hflayers.transformer import HopfieldEncoderLayer

class HopfieldARModel(nn.Module):
    """
    Module implementing the HopfieldAR model, see [SFG17]_.

    *Note:* the code of this model is unrelated to the implementation behind
    `SageMaker's HopfieldAR Forecasting Algorithm
    <https://docs.aws.amazon.com/sagemaker/latest/dg/HopfieldAR.html>`_.

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
    default_scale
        Default scale that is applied if the context length window is
        completely unobserved. If not set, the scale in this case will be
        the mean scale in the batch.
    num_parallel_samples
        Number of samples to produce when unrolling the RNN in the prediction
        time range.
    nonnegative_pred_samples
        Should final prediction samples be non-negative? If yes, an activation
        function is applied to ensure non-negative. Observe that this is applied
        only to the final samples and this is not applied during training.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        num_feat_static_cat: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        beta: float = 2.0,
        activation: str = "relu",
        dim_feedforward: int = 31,
        dropout_rate: float = 0.1,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: Optional[str] = "std",
        default_scale: Optional[float] = None,
        num_parallel_samples: int = 100,
        nonnegative_pred_samples: bool = False,
    ) -> None:
        super().__init__()

        assert distr_output.event_shape == ()
        assert num_feat_dynamic_real > 0
        assert num_feat_static_real > 0
        assert num_feat_static_cat > 0
        assert len(cardinality) == num_feat_static_cat
        assert (
            embedding_dimension is None
            or len(embedding_dimension) == num_feat_static_cat
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model)

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.lags_seq = [l - 1 for l in self.lags_seq]
        self.num_parallel_samples = num_parallel_samples
        self.past_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling == "mean" or scaling == True:
            self.scaler = MeanScaler(keepdim=True, dim=1, default_scale=default_scale)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)


        self.embed = nn.Linear(len(self.lags_seq) + self._number_of_features, d_model, bias=False)

        encoder_association = Hopfield(
            input_size=d_model, num_heads=nhead, dropout=dropout_rate, scaling=beta
        )
        encoder_layer = HopfieldEncoderLayer(
            encoder_association,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation=activation,
        )
        encoder_layer.self_attn = encoder_layer.hopfield_association
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.register_buffer(
            "mask",
            nn.Transformer.generate_square_subsequent_mask(prediction_length + context_length),
        )

        # self.rnn = nn.LSTM(
        #     input_size=self.rnn_input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=dropout_rate,
        #     batch_first=True,
        # )

        self.nonnegative_pred_samples = nonnegative_pred_samples

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat),
                    dtype=torch.long,
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real),
                    dtype=torch.float,
                ),
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self._past_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "past_target": Input(
                    shape=(batch_size, self._past_length),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            },
            zeros_fn=torch.zeros,
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + 2  # the log(scale) and log1p(loc) of the target
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def prepare_encoder_input(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        context = past_target[..., -self.context_length :]
        observed_context = past_observed_values[..., -self.context_length :]

        input, loc, scale = self.scaler(context, observed_context)
        future_length = future_time_feat.shape[-2]
        if future_length > 1:
            assert future_target is not None
            input = torch.cat(
                (input, (future_target[..., : future_length - 1] -loc) / scale),
                dim=-1,
            )
        prior_input = (past_target[..., : -self.context_length] -loc)/ scale

        lags = lagged_sequence_values(
            self.lags_seq, prior_input, input, dim=-1
        )

        time_feat = torch.cat(
            (
                take_last(past_time_feat, dim=-2, num=self.context_length - 1),
                future_time_feat,
            ),
            dim=-2,
        )

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, loc.sign() * loc.abs().log1p(), scale.log()),
            dim=-1,
        )
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=time_feat.shape[-2]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        return torch.cat((lags, features), dim=-1), loc, scale, static_feat

    def unroll_lagged_encoder(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Applies the underlying encoder to the provided target data and covariates.

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
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            Tensor of dynamic real features in the future,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        future_target
            (Optional) tensor of future target values,
            shape: ``(batch_size, prediction_length)``.

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
        encoder_input, loc, scale, static_feat = self.prepare_encoder_input(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        embeded_input = self.embed(encoder_input)
        output = self.encoder(embeded_input, is_causal=self.training)

        params = self.param_proj(output)
        return params, loc, scale, output, static_feat

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
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
        return self.distr_output.distribution(sliced_params, loc=loc, scale=scale)

    def post_process_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Method to enforce domain-specific constraints on the generated samples.
        For example, we can enforce forecasts to be nonnegative.
        Parameters
        ----------
        samples
            Tensor of samples
        Returns
        -------
            Tensor of processed samples with the same shape.
        """

        if self.nonnegative_pred_samples:
            return torch.relu(samples)

        return samples

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
            shape: ``(batch_size, past_length)``.
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

        repeated_past_time_feat = past_time_feat.repeat_interleave(num_parallel_samples, 0)
        repeated_past_target = past_target.repeat_interleave(num_parallel_samples, 0)
        repeated_past_observed_values = past_observed_values.repeat_interleave(num_parallel_samples, 0)
        repeated_future_time_feat = future_time_feat.repeat_interleave(num_parallel_samples, 0)

        repeated_feat_static_real = feat_static_real.repeat_interleave(num_parallel_samples, 0)
        repeated_feat_static_cat = feat_static_cat.repeat_interleave(num_parallel_samples, 0)
        
        future_samples = []
        for k in range(self.prediction_length):
            params, loc, scale, _, _ = self.unroll_lagged_encoder(
                repeated_feat_static_cat,
                repeated_feat_static_real,
                repeated_past_time_feat,
                repeated_past_target,
                repeated_past_observed_values,
                repeated_future_time_feat[:, :1],
            )
            distr = self.output_distribution(params, loc=loc, scale=scale, trailing_n=1)
            next_sample = distr.sample()
            future_samples.append(self.post_process_samples(next_sample))

            repeated_past_target = torch.cat((repeated_past_target, next_sample), dim=1)
            repeated_past_observed_values = torch.cat((repeated_past_observed_values, torch.ones_like(next_sample)), dim=1)
            repeated_past_time_feat = torch.cat((repeated_past_time_feat, repeated_future_time_feat[:, k : k + 1, ...]), dim=1) 

        # params, loc, scale, _, static_feat = self.unroll_lagged_encoder(
        #     feat_static_cat,
        #     feat_static_real,
        #     past_time_feat,
        #     past_target,
        #     past_observed_values,
        #     future_time_feat[:, :1],
        # )

        # repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)   
        # repeated_scale = scale.repeat_interleave(
        #     repeats=num_parallel_samples, dim=0
        # )
        # repeated_static_feat = static_feat.repeat_interleave(
        #     repeats=num_parallel_samples, dim=0
        # ).unsqueeze(dim=1)
        # repeated_past_target = (
        #     (past_target.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc)
        #     / repeated_scale
        # )
        # repeated_time_feat = future_time_feat.repeat_interleave(
        #     repeats=num_parallel_samples, dim=0
        # )

        # repeated_params = [
        #     s.repeat_interleave(repeats=num_parallel_samples, dim=0)
        #     for s in params
        # ]
        # distr = self.output_distribution(
        #     repeated_params, trailing_n=1, loc=repeated_loc, scale=repeated_scale
        # )
        # next_sample = distr.sample()
        # future_samples = [next_sample]

        # for k in range(1, self.prediction_length):
        #     scaled_next_sample = (next_sample - repeated_loc) / repeated_scale
        #     next_features = torch.cat(
        #         (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
        #         dim=-1,
        #     )

        #     next_lags = lagged_sequence_values(
        #         self.lags_seq, repeated_past_target, scaled_next_sample, dim=-1
        #     )
        #     embedded_input = self.embed(torch.cat((next_lags, next_features), dim=-1))
        #     output = self.encoder(embedded_input)

        #     repeated_past_target = torch.cat(
        #         (repeated_past_target, scaled_next_sample), dim=1
        #     )

        #     params = self.param_proj(output)
        #     distr = self.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
        #     next_sample = distr.sample()
        #     future_samples.append(next_sample)

        future_samples_concat = torch.cat(future_samples, dim=1)

        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length)
        )

    def log_prob(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
    ) -> torch.Tensor:
        return -self.loss(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=torch.ones_like(future_target),
            loss=NegativeLogLikelihood(),
            future_only=False,
            aggregate_by=torch.sum,
        )

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        loss: DistributionLoss = NegativeLogLikelihood(),
        future_only: bool = False,
        aggregate_by=torch.mean,
    ) -> torch.Tensor:
        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]
        batch_shape = future_target.shape[: extra_dims + 1]

        repeats = prod(extra_shape)
        feat_static_cat = repeat_along_dim(feat_static_cat, 0, repeats)
        feat_static_real = repeat_along_dim(feat_static_real, 0, repeats)
        past_time_feat = repeat_along_dim(past_time_feat, 0, repeats)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_observed_values = repeat_along_dim(
            past_observed_values, 0, repeats
        )
        future_time_feat = repeat_along_dim(future_time_feat, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )

        params, loc, scale, _, _ = self.unroll_lagged_encoder(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target_reshaped,
        )

        if future_only:
            distr = self.output_distribution(
                params, loc=loc, scale=scale, trailing_n=self.prediction_length
            )
            loss_values = (
                loss(distr, future_target_reshaped) * future_observed_reshaped
            )
        else:
            distr = self.output_distribution(params, loc=loc, scale=scale)
            context_target = take_last(
                past_target, dim=-1, num=self.context_length - 1
            )
            target = torch.cat(
                (context_target, future_target_reshaped),
                dim=1,
            )
            context_observed = take_last(
                past_observed_values, dim=-1, num=self.context_length - 1
            )
            observed_values = torch.cat(
                (context_observed, future_observed_reshaped), dim=1
            )
            loss_values = loss(distr, target) * observed_values

        loss_values = loss_values.reshape(*batch_shape, *loss_values.shape[1:])

        return aggregate_by(
            loss_values,
            dim=tuple(range(extra_dims + 1, len(future_target.shape))),
        )
