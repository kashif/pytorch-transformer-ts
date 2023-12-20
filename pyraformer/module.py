from typing import List, Optional
import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.torch.scaler import MeanScaler, NOPScaler
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.util import lagged_sequence_values, unsqueeze_expand

from pyraformer.Layers import EncoderLayer, Predictor, Decoder
from pyraformer.Layers import (
    Bottleneck_Construct,
    Conv_Construct,
    MaxPooling_Construct,
    AvgPooling_Construct,
)
from pyraformer.Layers import (
    get_mask,
    refer_points,
    get_k_q,
    get_q_k,
    get_subsequent_mask,
)
from pyraformer.embed import SingleStepEmbedding, DataEmbedding, CustomEmbedding
import random

class EncoderSS(nn.Module):
    @validated()
    def __init__(
        self,
        covariate_size,
        num_seq,
        input_size,
        dropout,
        d_model,
        d_inner_hid,
        d_k,
        d_v,
        num_heads,
        n_layer,
        loss,
        window_size,
        inner_size,
        use_tvm,
        prediction_length,
        device
    ):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.mask, self.all_size = get_mask(input_size, window_size, inner_size, device)
        self.indexes = refer_points(self.all_size, window_size, device)

        if use_tvm:

            assert (
                len(set(self.window_size)) == 1
            ), "Only constant window size is supported."
            q_k_mask = get_q_k(input_size, inner_size, window_size[0], device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model,
                        d_inner_hid,
                        num_heads,
                        d_k,
                        d_v,
                        dropout=dropout,
                        normalize_before=False,
                        use_tvm=True,
                        q_k_mask=q_k_mask,
                        k_q_mask=k_q_mask,
                    )
                    for i in range(n_layer)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model,
                        d_inner_hid,
                        num_heads,
                        d_k,
                        d_v,
                        dropout=dropout,
                        normalize_before=False,
                    )
                    for i in range(n_layer)
                ]
            )

        self.embedding = SingleStepEmbedding(
            covariate_size, num_seq, d_model, input_size, device
        )

        self.conv_layers = Bottleneck_Construct(d_model, window_size, d_k)

    def forward(self, sequence):

        seq_enc = self.embedding(sequence)
        mask = self.mask.repeat(len(seq_enc), self.num_heads, 1, 1).to(sequence.device)
        
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(
            seq_enc.device
        )
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        all_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        return all_enc


class PyraformerSSModel(nn.Module):
    @validated()
    def __init__(
        self,
        # freq,
        context_length,
        prediction_length,
        covariate_size,
        num_seq,
        dropout,
        input_size,
        d_model,
        d_inner_hid,
        d_k,
        d_v,
        num_heads,
        n_layer,
        loss,
        window_size,
        inner_size,
        use_tvm,
        # lags_seq,
        # num_feat_dynamic_real,
        # num_feat_static_cat,
        # num_feat_static_real,
        # cardinality,
        # embedding_dimension,
        distr_output,
        # loss: DistributionLoss = NegativeLogLikelihood(),
        scaling,
        num_parallel_samples,
    )-> None:
        super().__init__()
        self.input_size = input_size

        self.target_shape = distr_output.event_shape
        self.context_length = context_length
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
        self.num_parallel_samples = num_parallel_samples
        self.history_length = context_length + max(self.lags_seq)
        if scaling == "mean" or scaling == True:
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)
        self.embed = nn.Linear(
        self.input_size * len(self.lags_seq) + self._number_of_features, d_model
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model*4)

        self.encoder = EncoderSS(
            covariate_size,
            num_seq,
            input_size,
            dropout,
            d_model,
            d_inner_hid,
            d_k,
            d_v,
            num_heads,
            n_layer,
            loss,
            window_size,
            inner_size,
            use_tvm,
            prediction_length,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )


        # convert hidden vectors into two scalar
        # self.mean_hidden = Predictor(4 * d_model, 1)
        # self.var_hidden = Predictor(4 * d_model, 1)

    @property
    def _number_of_features(self) -> int:
        return (
            # sum(self.embedding_dimension)
            # + self.num_feat_dynamic_real
            # + self.num_feat_static_real
            self.input_size * 2  # the log(scale) and log(abs(loc)) features
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

        return torch.cat((lags, expanded_static_feat), dim=-1), loc, scale, static_feat

    def output_params(self, transformer_inputs):
        enc_output = self.encoder(self.embed(transformer_inputs))
        return self.param_proj(enc_output)

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, loc=loc, scale=scale)

    #prediction
    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        # future_target: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:               
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples
        # past_time_feat = kwargs["past_time_feat"]
        # future_time_feat = kwargs["future_time_feat"]

        repeated_past_target = past_target.repeat_interleave(
            self.num_parallel_samples, 0
        )
        repeated_past_observed_values = past_observed_values.repeat_interleave(
            self.num_parallel_samples, 0
        )
        # repeated_past_time_feat = past_time_feat.repeat_interleave(
        #     self.model.num_parallel_samples, 0
        # )
        # repeated_future_time_feat = future_time_feat.repeat_interleave(
        #     self.model.num_parallel_samples, 0
        # )

        future_samples = []
        for t in range(self.prediction_length):
            encoder_inputs, loc, scale, static_feat = self.create_network_inputs(past_target=repeated_past_target, past_observed_values=repeated_past_observed_values)
            params = self.output_params(encoder_inputs)
            distr = self.output_distribution(params, loc=loc, scale=scale)
            # params, loc, scale = self.model(
            #     *args,
            #     past_time_feat=repeated_past_time_feat,
            #     future_time_feat=repeated_future_time_feat[..., : t + 1, :],
            #     past_target=repeated_past_target,
            #     past_observed_values=repeated_past_observed_values,
            #     is_test=False,
            # )
            sliced_params = [p[:, -1:] for p in params]
            # distr = self.model.distr_output.distribution(sliced_params, loc, scale)
            sample = distr.sample()
            future_samples.append(sample)

            repeated_past_target = torch.cat((repeated_past_target, sample), dim=1)
            repeated_past_observed_values = torch.cat(
                (repeated_past_observed_values, torch.ones_like(sample)), dim=1
            )

        # self.model.reset_cache()

        concat_future_samples = torch.cat(future_samples, dim=-1)
        return concat_future_samples.reshape(
            (-1, num_parallel_samples, self.prediction_length)
            + self.distr_output.event_shape,
        )



        # for t in range(self.prediction_length):
        #     encoder_inputs, loc, scale, static_feat = self.create_network_inputs(past_target, past_observed_values)
        #     params = self.output_params(encoder_inputs)
        #     distr = self.output_distribution(params, loc=loc, scale=scale)

        #     sliced_params = [p[:, -1:] for p in params]
        #     distr = self.output_distribution(sliced_params, loc, scale)
        #     past_target = torch.cat((past_target, distr.mean), dim=1)
        #     past_observed_values = torch.cat(
        #         (past_observed_values, torch.ones_like(distr.mean)), dim=1
        #     )

        # sliced_params = [p[:, -self.prediction_length :] for p in params]
        # distr = self.output_distribution(sliced_params, loc, scale)
        # sample = distr.sample((self.num_parallel_samples, self.prediction_length))
        # return sample.transpose(1, 0).reshape(
        #     (-1, self.num_parallel_samples, self.prediction_length)
        #     + self.target_shape,
        # )            

class EncoderLR(nn.Module):
    @validated()
    def __init__(
        self,
        # model,
        window_size,
        truncate,
        input_size,
        inner_size,
        decoder,
        d_model,
        d_k,
        d_v,
        d_inner_hid,
        dropout,
        n_layer,
        enc_in,
        covariate_size,
        seq_num,
        CSCM,
        d_bottleneck,
        num_head,
        use_tvm,
        embed_type,
        device,
    ):
        super().__init__()

        self.d_model = d_model
        # self.model_type = model
        self.window_size = window_size
        self.truncate = truncate
        if decoder == "attention":
            self.mask, self.all_size = get_mask(
                input_size, window_size, inner_size, device
            )
        else:
            self.mask, self.all_size = get_mask(
                input_size + 1, window_size, inner_size, device
            )
        self.decoder_type = decoder
        if decoder == "FC":
            self.indexes = refer_points(self.all_size, window_size, device)

        if use_tvm:
            assert (
                len(set(self.window_size)) == 1
            ), "Only constant window size is supported."
            padding = 1 if decoder == "FC" else 0
            q_k_mask = get_q_k(input_size + padding, inner_size, window_size[0], device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model,
                        d_inner_hid,
                        num_head,
                        d_k,
                        d_v,
                        dropout=dropout,
                        normalize_before=False,
                        use_tvm=True,
                        q_k_mask=q_k_mask,
                        k_q_mask=k_q_mask,
                    )
                    for i in range(n_layer)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model,
                        d_inner_hid,
                        num_head,
                        d_k,
                        d_v,
                        dropout=dropout,
                        normalize_before=False,
                    )
                    for i in range(n_layer)
                ]
            )

            if embed_type == "CustomEmbedding":
                self.enc_embedding = CustomEmbedding(
                    enc_in, d_model, covariate_size, seq_num, dropout
                )
            else:
                self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)

        self.conv_layers = eval(CSCM)(d_model, window_size, d_bottleneck)

    def forward(self, x_enc, x_mark_enc):
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == "FC":
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(
                seq_enc.device
            )
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == "attention" and self.truncate:
            seq_enc = seq_enc[:, : self.all_size[0]]

        return seq_enc


class PyraformerLRModel(nn.Module):
    @validated()
    def __init__(
        self,
        prediction_length,
        context_length,
        d_model,
        input_size,
        decoder,
        window_size,
        truncate,
        d_inner_hid,
        d_k,
        d_v,
        dropout,
        enc_in,
        covariate_size,
        seq_num,
        CSCM,
        d_bottleneck,
        num_head,
        n_layer,
        inner_size,
        use_tvm,
        
        # lags_seq,
        # num_feat_dynamic_real,
        # num_feat_static_cat,
        # num_feat_static_real,
        # cardinality,
        # embedding_dimension,
        num_parallel_samples,
        embed_type,
        distr_output,
        device,    
    )-> None:
        super().__init__()
        self.input_size = input_size

        self.target_shape = distr_output.event_shape
        self.context_length = context_length
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
        self.num_parallel_samples = num_parallel_samples
        self.history_length = context_length + max(self.lags_seq)
        if scaling == "mean" or scaling == True:
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)
        self.embed = nn.Linear(
        self.input_size * len(self.lags_seq) + self._number_of_features, d_model
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model*4)

        self.encoder = EncoderLR(
            window_size,
            truncate,
            input_size,
            inner_size,
            decoder,
            d_model,
            d_k,
            d_v,
            d_inner_hid,
            dropout,
            n_layer,
            enc_in,
            covariate_size,
            num_seq,
            CSCM,
            d_bottleneck,
            num_head,
            use_tvm,
            embed_type,
        )
        if decoder == 'attention':
            mask = get_subsequent_mask(input_size, window_size, self.prediction_length, truncate)
            self.decoder = Decoder(d_model, d_inner_hid, num_head, d_k, d_v, dropout, enc_in, covariate_size, seq_num, mask)
            self.predictor = Predictor(d_model, enc_in)
        elif opt.decoder == 'FC':
            self.predictor = Predictor(4 * d_model, prediction_length * enc_in)
        

        # convert hidden vectors into two scalar
        # self.mean_hidden = Predictor(4 * d_model, 1)
        # self.var_hidden = Predictor(4 * d_model, 1)

    @property
    def _number_of_features(self) -> int:
        return (
            # sum(self.embedding_dimension)
            # + self.num_feat_dynamic_real
            # + self.num_feat_static_real
            self.input_size * 2  # the log(scale) and log(abs(loc)) features
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

        return torch.cat((lags, expanded_static_feat), dim=-1), loc, scale, static_feat

    def output_params(self, transformer_inputs):
        projected_inputs = self.embed(transformer_inputs)
        enc_input = projected_inputs[:, : self.context_length, ...]
        dec_input = projected_inputs[:, self.context_length :, ...]
        if decoder == 'attention':
            enc_out, _ = self.encoder(enc_input)
            dec_output = self.decoder(dec_input, enc_out)

        return self.param_proj(dec_output)
        
        # enc_output = self.encoder(self.embed(transformer_inputs))
        # return self.param_proj(enc_output)

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, loc=loc, scale=scale)

    #prediction
    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        # future_target: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:               
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples
        # past_time_feat = kwargs["past_time_feat"]
        # future_time_feat = kwargs["future_time_feat"]

        repeated_past_target = past_target.repeat_interleave(
            self.num_parallel_samples, 0
        )
        repeated_past_observed_values = past_observed_values.repeat_interleave(
            self.num_parallel_samples, 0
        )
        # repeated_past_time_feat = past_time_feat.repeat_interleave(
        #     self.model.num_parallel_samples, 0
        # )
        # repeated_future_time_feat = future_time_feat.repeat_interleave(
        #     self.model.num_parallel_samples, 0
        # )

        future_samples = []
        for t in range(self.prediction_length):
            encoder_inputs, loc, scale, static_feat = self.create_network_inputs(past_target=repeated_past_target, past_observed_values=repeated_past_observed_values)
            params = self.output_params(encoder_inputs)
            distr = self.output_distribution(params, loc=loc, scale=scale)
            # params, loc, scale = self.model(
            #     *args,
            #     past_time_feat=repeated_past_time_feat,
            #     future_time_feat=repeated_future_time_feat[..., : t + 1, :],
            #     past_target=repeated_past_target,
            #     past_observed_values=repeated_past_observed_values,
            #     is_test=False,
            # )
            sliced_params = [p[:, -1:] for p in params]
            # distr = self.model.distr_output.distribution(sliced_params, loc, scale)
            sample = distr.sample()
            future_samples.append(sample)

            repeated_past_target = torch.cat((repeated_past_target, sample), dim=1)
            repeated_past_observed_values = torch.cat(
                (repeated_past_observed_values, torch.ones_like(sample)), dim=1
            )

        # self.model.reset_cache()

        concat_future_samples = torch.cat(future_samples, dim=-1)
        return concat_future_samples.reshape(
            (-1, num_parallel_samples, self.prediction_length)
            + self.distr_output.event_shape,
        )

    
    
    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain):
    #     """
    #     Return the hidden representations and predictions.
    #     For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
    #     Input: event_type: batch*seq_len;
    #            event_time: batch*seq_len.
    #     Output: enc_output: batch*seq_len*model_dim;
    #             type_prediction: batch*seq_len*num_classes (not normalized);
    #             time_prediction: batch*seq_len.
    #     """
    #     if self.decoder_type == 'attention':
    #         enc_output = self.encoder(x_enc, x_mark_enc)
    #         dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

    #         if pretrain:
    #             dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
    #             pred = self.predictor(dec_enc)
    #         else:
    #             pred = self.predictor(dec_enc)
    #     elif self.decoder_type == 'FC':
    #         enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
    #         pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)

    #     return pred

