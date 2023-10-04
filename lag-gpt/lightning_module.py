import random

import pytorch_lightning as pl
import torch

from gluonts.core.component import validated
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import take_last, repeat_along_dim
from gluonts.itertools import prod

from module import LagGPTModel
from aug import freq_mask, freq_mix


class LagGPTLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``LagGPTLightningModule`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``LagGPTLightningModule`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``LagGPTLightningModule`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    """

    @validated()
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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context_length = self.hparams.context_length
        self.prediction_length = self.hparams.prediction_length
        self.model = LagGPTModel(**self.hparams.model_kwargs)
        self.loss = self.hparams.loss
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.aug_rate = self.hparams.aug_rate

    # greedy prediction
    def forward(self, *args, **kwargs):
        past_target = kwargs["past_target"]
        past_observed_values = kwargs["past_observed_values"]
        past_time_feat = kwargs["past_time_feat"]
        future_time_feat = kwargs["future_time_feat"]

        repeated_past_target = past_target.repeat_interleave(
            self.model.num_parallel_samples, 0
        )
        repeated_past_observed_values = past_observed_values.repeat_interleave(
            self.model.num_parallel_samples, 0
        )
        repeated_past_time_feat = past_time_feat.repeat_interleave(
            self.model.num_parallel_samples, 0
        )
        repeated_future_time_feat = future_time_feat.repeat_interleave(
            self.model.num_parallel_samples, 0
        )

        future_samples = []
        for t in range(self.prediction_length):
            params, loc, scale = self.model(
                *args,
                past_time_feat=repeated_past_time_feat,
                future_time_feat=repeated_future_time_feat[..., : t + 1, :],
                past_target=repeated_past_target,
                past_observed_values=repeated_past_observed_values,
                is_test=False,
            )
            sliced_params = [p[:, -1:] for p in params]
            distr = self.model.distr_output.distribution(sliced_params, loc, scale)
            sample = distr.sample()
            future_samples.append(sample)

            repeated_past_target = torch.cat((repeated_past_target, sample), dim=1)
            repeated_past_observed_values = torch.cat(
                (repeated_past_observed_values, torch.ones_like(sample)), dim=1
            )

        self.model.reset_cache()

        concat_future_samples = torch.cat(future_samples, dim=-1)
        return concat_future_samples.reshape(
            (-1, self.model.num_parallel_samples, self.prediction_length)
            + self.model.distr_output.event_shape,
        )

    # # beam-search? prediction
    # def forward(self, *args, **kwargs):
    #     past_time_feat = kwargs["past_time_feat"]
    #     past_target = kwargs["past_target"]
    #     past_observed_values = kwargs["past_observed_values"]
    #     future_time_feat = kwargs["future_time_feat"]

    #     future_samples = []
    #     for t in range(self.prediction_length):
    #         params, loc, scale = self.model.forward(
    #             *args,
    #             past_target=past_target,
    #             past_observed_values=past_observed_values,
    #             past_time_feat=past_time_feat,
    #         )
    #         sliced_params = [p[:, -1:] for p in params]
    #         distr = self.model.distr_output.distribution(sliced_params, loc, scale)
    #         sample = distr.sample((self.model.num_parallel_samples,))
    #         future_samples.append(sample.transpose(1, 0))

    #         past_target = torch.cat((past_target, distr.mean), dim=1)
    #         past_observed_values = torch.cat(
    #             (past_observed_values, torch.ones_like(distr.mean)), dim=1
    #         )
    #         past_time_feat = torch.cat(
    #             (past_time_feat, future_time_feat[:, t : t + 1, ...]),
    #             dim=1,
    #         )

    #     concat_future_samples = torch.cat(future_samples, dim=-1)
    #     return concat_future_samples.reshape(
    #         (-1, self.model.num_parallel_samples, self.prediction_length)
    #         + self.model.distr_output.event_shape,
    #     )

    # # mean prediction and then sample
    # def forward(self, *args, **kwargs):
    #     past_target = kwargs["past_target"]
    #     past_observed_values = kwargs["past_observed_values"]

    #     for t in range(self.prediction_length):
    #         params, loc, scale = self.model(
    #             *args,
    #             past_target=past_target,
    #             past_observed_values=past_observed_values,
    #         )
    #         sliced_params = [p[:, -1:] for p in params]
    #         distr = self.model.distr_output.distribution(sliced_params, loc, scale)
    #         past_target = torch.cat((past_target, distr.mean), dim=1)
    #         past_observed_values = torch.cat(
    #             (past_observed_values, torch.ones_like(distr.mean)), dim=1
    #         )

    #     sliced_params = [p[:, -self.prediction_length :] for p in params]
    #     distr = self.model.distr_output.distribution(sliced_params, loc, scale)
    #     sample = distr.sample((self.model.num_parallel_samples,))
    #     return sample.transpose(1, 0).reshape(
    #         (-1, self.model.num_parallel_samples, self.prediction_length)
    #         + self.model.distr_output.event_shape,
    #     )

    # train
    def _compute_loss(self, batch):
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        past_time_feat = batch["past_time_feat"]
        future_time_feat = batch["future_time_feat"]

        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]

        repeats = prod(extra_shape)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_observed_values = repeat_along_dim(past_observed_values, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )

        distr_args, loc, scale = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target_reshaped,
        )
        distr = self.model.distr_output.distribution(distr_args, loc, scale)

        context_target = take_last(past_target, dim=-1, num=self.context_length - 1)
        target = torch.cat(
            (context_target, future_target_reshaped),
            dim=1,
        )
        context_observed = take_last(
            past_observed_values, dim=-1, num=self.context_length - 1
        )
        observed_values = torch.cat((context_observed, future_observed_reshaped), dim=1)

        return (
            self.loss(distr, target) * observed_values
        ).sum() / observed_values.sum().clamp_min(1.0)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
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

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self._compute_loss(batch)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
