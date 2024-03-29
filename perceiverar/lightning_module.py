import pytorch_lightning as pl
import torch

from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

from module import PerceiverARModel


class PerceiverARLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``PerceiverARModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``PerceiverARModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``PerceiverARModel`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    """

    def __init__(
        self,
        model: PerceiverARModel,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay

    def _compute_loss(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

        params, scale, _, _ = self.model.lagged_perciever(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        distr = self.model.output_distribution(params, scale)

        # context_target = past_target[:, -self.model.context_length + 1 :]
        # target = torch.cat(
        #     (context_target, future_target),
        #     dim=1,
        # )
        loss_values = self.loss(distr, future_target)

        # context_observed = past_observed_values[:, -self.model.context_length + 1 :]
        # observed_values = torch.cat((context_observed, future_observed_values), dim=1)

        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
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
