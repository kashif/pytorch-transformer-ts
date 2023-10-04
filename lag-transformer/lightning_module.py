import random

import pytorch_lightning as pl
import torch
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

from module import LagTransformerModel
from aug import freq_mask, freq_mix


class LagTransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_kwargs: dict,
        context_length: int,
        prediction_length: int,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        aug_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = LagTransformerModel(**self.hparams.model_kwargs)
        self.context_length = self.hparams.context_length
        self.prediction_length = self.hparams.prediction_length

        self.loss = self.hparams.loss
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.aug_rate = self.hparams.aug_rate

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        if random.random() < self.aug_prob:
            if random.random() < 0.5:
                batch["past_target"], batch["future_target"] = freq_mask(
                    batch["past_target"], batch["future_target"], rate=self.aug_rate
                )
            else:
                batch["past_target"], batch["future_target"] = freq_mix(
                    batch["past_target"], batch["future_target"], rate=self.aug_rate
                )

        train_loss = self._compute_loss(batch)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        with torch.inference_mode():
            val_loss = self._compute_loss(batch)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _compute_loss(self, batch):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

        transformer_inputs, loc, scale, _ = self.model.create_network_inputs(
            past_target, past_observed_values, future_target
        )
        params = self.model.output_params(
            transformer_inputs, context_length=self.context_length
        )
        distr = self.model.output_distribution(params, loc=loc, scale=scale)

        loss_values = self.loss(distr, future_target)

        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)

    # greedy prediction
    def forward(self, *args, **kwargs):
        past_target = kwargs["past_target"]
        past_observed_values = kwargs["past_observed_values"]

        num_parallel_samples = self.model.num_parallel_samples

        encoder_inputs, loc, scale, static_feat = self.model.create_network_inputs(
            past_target, past_observed_values
        )
        enc_pos = self.model.pos_embedding(encoder_inputs.size())
        enc_out = self.model.transformer.encoder(
            self.model.enc_dec_embedding(encoder_inputs) + enc_pos
        )

        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        expanded_static_feat = repeated_static_feat.unsqueeze(1).expand(
            -1, self.prediction_length, -1
        )

        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale

        repeated_enc_out = enc_out.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        future_samples = []

        # greedy decoding
        for k in range(self.prediction_length):
            # self._check_shapes(repeated_past_target, next_sample, next_features)
            # sequence = torch.cat((repeated_past_target, next_sample), dim=1)

            lagged_sequence = self.model.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )

            decoder_input = torch.cat(
                (reshaped_lagged_sequence, expanded_static_feat[:, : k + 1]), dim=-1
            )

            dec_pos = self.model.pos_embedding(
                decoder_input.size(), past_key_values_length=self.context_length
            )
            output = self.model.transformer.decoder(
                self.model.enc_dec_embedding(decoder_input) + dec_pos, repeated_enc_out
            )

            params = self.model.param_proj(output[:, -1:])
            distr = self.model.output_distribution(
                params, scale=repeated_scale, loc=repeated_loc
            )
            next_sample = distr.sample()

            repeated_past_target = torch.cat(
                (repeated_past_target, (next_sample - repeated_loc) / repeated_scale),
                dim=1,
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, num_parallel_samples, self.prediction_length)
            + self.model.target_shape,
        )
