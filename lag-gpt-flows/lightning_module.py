import random

import pytorch_lightning as pl
import torch

from gluonts.core.component import validated
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import take_last, repeat_along_dim
from gluonts.itertools import prod

from module import LagGPTFlowsModel
from aug import freq_mask, freq_mix


class LagGPTFlowsLightningModule(pl.LightningModule):
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
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        aug_rate: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LagGPTFlowsModel(**self.hparams.model_kwargs)
        self.loss = self.hparams.loss
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.aug_rate = self.hparams.aug_rate

    # mean prediction and then sample
    def forward(self, *args, **kwargs):
        past_target = kwargs["past_target"]
        past_observed_values = kwargs["past_observed_values"]
        samples = torch.zeros(past_target.shape[0], self.model.prediction_length, self.model.num_parallel_samples).to(past_target.device)

        for t in range(self.model.prediction_length):
            transformer_output, scaled_input, loc, scale = self.model(
                *args,
                past_target=past_target,
                past_observed_values=past_observed_values,
            )
            # Get the encoding of the last timestep predicted so far
            pred_encoding = transformer_output[:, -1:] # Shape: [bsz, 1, dims]
            # Construct uniform random samples
            pred_samples = torch.rand(pred_encoding.shape[0], 1, self.model.num_parallel_samples, device=pred_encoding.device)
            # Transform to [min_u, max_u]
            pred_samples = self.model.min_u + (self.model.max_u - self.model.min_u) * pred_samples
            # Transform to the distribution of each token
            # Pass the encoding of the last token and predict the distribution of the next token
            pred_samples = self.model.marginal.inverse(
                pred_encoding,
                pred_samples
            ) # Shape: [bsz, 1, num_parallel_samples]
            # Renormalize the samples
            pred_samples = pred_samples * scale.unsqueeze(2) + loc.unsqueeze(2)
            samples[:, t] = pred_samples.squeeze(1)

            past_target = torch.cat((past_target, pred_samples.mean(-1)), dim=1)
            past_observed_values = torch.cat(
                (past_observed_values, torch.ones_like(pred_samples.mean(-1))), dim=1
            )

        return samples

    # train
    def _compute_loss(self, batch):
        past_target = batch["past_target"] # (bsz, model._past_length)
        past_observed_values = batch["past_observed_values"] # (bsz, model._past_length)
        future_target = batch["future_target"] # (bsz, model.prediction_length)
        future_observed_values = batch["future_observed_values"] # (bsz, model.prediction_length)

        extra_dims = len(future_target.shape) - len(past_target.shape) # usually 0
        extra_shape = future_target.shape[:extra_dims]

        repeats = prod(extra_shape)
        past_target = repeat_along_dim(past_target, 0, repeats) # (bsz, model._past_length)
        past_observed_values = repeat_along_dim(past_observed_values, 0, repeats) # (bsz, model._past_length)

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        ) # (bsz, model.prediction_length)
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        ) # (bsz, model.prediction_length)

        transformer_output, scaled_input, loc, scale = self.model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target_reshaped,
        ) # (bsz, )

        # TODO: We can also possibly use the history tokens in the objective since they are causally masked here?
        # NOTE: This is a decoder-only model 
        # So we train the normalizing flows to predict the marginals of the next token here 
        # The encoding given to the normalizing flow is the encoding of the kth token
        # NOTE: Maybe use prediction_length 
        # pred_encoding = transformer_output[:, -self.model.prediction_length:-1]
        pred_encoding = transformer_output[:, self.model.context_length:-1]
        pred_scaled_input = scaled_input[:, self.model.context_length+1:]

        pred_u, pred_logdet = self.model.marginal.forward_logdet(pred_encoding, pred_scaled_input) # (BSZ, )

        return -pred_logdet.sum() / pred_logdet.shape[0]

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
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
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
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
