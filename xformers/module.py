# +
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler

from xformers.factory.model_factory import xFormer, xFormerConfig


# -

class XformerModel(nn.Module):
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
        
        # xformer arguments
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        attention_args: Dict[str, Any],
        activation: str = "gelu",
        residual_norm_style: str = "pre",
        dropout: float = 0.1,
        reversible: bool = False,
        hidden_layer_multiplier: int = 2,
        use_rotary_embeddings: bool = False,

        # univariate input
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
        
        attention_args["dropout"] = dropout
        attention_args["causal"] = False
        attention_args["seq_len"] = self.context_length
        attention_args["num_rules"] = nhead
        attention_args["attention_query_mask"] = (torch.rand((context_length, 1)) < 0.5)
        
        
        xformer_config = [
            # A list of the encoder blocks which constitute the Transformer.
            # Note that a sequence of different encoder blocks can be used
            {
                "reversible": reversible,  # Optionally make these layers reversible, to save memory
                "block_type": "encoder",
                "num_layers": num_encoder_layers,  # Optional, this means that this config will repeat N times
                "dim_model": d_model,
                "residual_norm_style": residual_norm_style,  # Optional, pre/post
                "position_encoding_config": {
                    "name": "sine",
                    "dim_model": d_model,
                },
                "multi_head_config": {
                    "use_rotary_embeddings": use_rotary_embeddings,
                    "num_heads": nhead,
                    "residual_dropout": dropout,
                    "attention": attention_args,
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": dropout,
                    "activation": activation,
                    "hidden_layer_multiplier": hidden_layer_multiplier,
                    "dim_model": d_model,
                },
            },
        ]
        config = xFormerConfig(xformer_config)
        # xformer encoder
        self.encoder = xFormer.from_config(config)
        
        # causal vanilla transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, 
            nhead, 
            dim_feedforward=d_model*hidden_layer_multiplier, 
            dropout=dropout,
            activation=activation, 
            layer_norm_eps=1e-5, 
            batch_first=True, 
            norm_first=False,
        )
        decoder_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
            
        # causal decoder tgt mask for training
        self.register_buffer(
            "tgt_mask",
            nn.Transformer.generate_square_subsequent_mask(prediction_length),
        )

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
        self,
        sequence: torch.Tensor,
        subsequences_length: int,
        shift: int = 0
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
        
        #self._check_shapes(prior_input, inputs, features)
        #sequence = torch.cat((prior_input, inputs), dim=1)

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
        enc_input = transformer_inputs[:, :self.context_length, ...]
        dec_input = transformer_inputs[:, self.context_length:, ...]
        
        enc_out = self.encoder(src=enc_input)
        dec_output = self.decoder(dec_input, enc_out, tgt_mask=self.tgt_mask)
        
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
        
        enc_out = self.encoder(src=encoder_inputs)
        
        params = self.param_proj(enc_out)
        distr = self.output_distribution(params, trailing_n=1)
        
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(
                repeats=self.num_parallel_samples, dim=0
            )
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
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
            
            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample), dim=1
            )
            future_samples.append(next_sample)

        unscaled_future_samples = (
            torch.cat(future_samples, dim=1) * repeated_scale
        )
        return unscaled_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length)
            + self.target_shape,
        )


