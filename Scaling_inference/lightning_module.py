import pytorch_lightning as pl
import torch
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average
from Scaling_inference.tft_model import TFTModel
from torch import optim
from Scaling_inference.utils import CosineWarmupScheduler


class TFTLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: TFTModel,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        max_iters: int = 100,
        warmup : int = 50
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iters = max_iters
        self.warmup = warmup

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        train_loss = self(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        #Sanity-Check for sample-output, comment-out to 
        # store mean and var-sample for plottiing (after training)

        # feat_static_cat = batch["feat_static_cat"]
        # feat_static_real = batch["feat_static_real"]
        # past_time_feat = batch["past_time_feat"]
        # past_target = batch["past_target"]
        # past_observed_values = batch["past_observed_values"]

        # future_time_feat = batch["future_time_feat"]
        # future_target = batch["future_target"]
        # future_observed_values = batch["future_observed_values"]

        
        # samples = self.model.forward(
        #     feat_static_cat=feat_static_cat,
        #     feat_static_real=feat_static_real,
        #     past_time_feat=past_time_feat,
        #     past_target=past_target,
        #     past_observed_values=past_observed_values,
        #     future_time_feat = future_time_feat,
            
        #     )
        # self.log(
        #     "mean_sample",
        #     samples.mean(),
        #     on_epoch=True,
        #     on_step=False,
        #     prog_bar=True,    
        # )
        # self.log(
        #     "var_samples",
        #     samples.std(),
        #     on_epoch=True,
        #     on_step=False,
        #     prog_bar=True,
        # )
        return train_loss
        

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        with torch.inference_mode():
            val_loss = self(batch)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                            warmup=self.warmup,
                                            max_iters=self.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
            
        

    def forward(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]

        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]

        (
            tft_target,
            time_feat,
            scale,
            embedded_cat,
            static_feat,
        ) = self.model.create_network_inputs(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )
        params = self.model.output_params(
            tft_target,
            time_feat=time_feat,
            embedded_cat=embedded_cat,
            static_feat=static_feat,
        )
        distr = self.model.output_distribution(params, scale)

        loss_values = self.loss(distr, future_target)
        

        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)

    #Sanity-Check for testing if there is any anomaly in
    # the output distribution
    # def test_step(self, batch, batch_idx) -> torch.Tensor:
    #     """
    #     Compute distribution outputs on a test batch.
    #     """
    #     feat_static_cat = batch["feat_static_cat"]
    #     feat_static_real = batch["feat_static_real"]
    #     past_time_feat = batch["past_time_feat"]
    #     past_target = batch["past_target"]
    #     past_observed_values = batch["past_observed_values"]

    #     future_time_feat = batch["future_time_feat"]
    #     future_target = batch["future_target"]
    #     future_observed_values = batch["future_observed_values"]

    #     (
    #         tft_target,
    #         time_feat,
    #         scale,
    #         embedded_cat,
    #         static_feat,
    #     ) = self.model.create_network_inputs(
    #         feat_static_cat=feat_static_cat,
    #         feat_static_real=feat_static_real,
    #         past_time_feat=past_time_feat,
    #         past_target=past_target,
    #         past_observed_values=past_observed_values,
    #         future_time_feat=future_time_feat,
    #         future_target=future_target,
    #     )
    #     params = self.model.output_params(
    #         tft_target,
    #         time_feat=time_feat,
    #         embedded_cat=embedded_cat,
    #         static_feat=static_feat,
    #     )
    #     distr = self.model.output_distribution(params, scale)
    #     self.log("test_distribution", distr)
    #     return distr