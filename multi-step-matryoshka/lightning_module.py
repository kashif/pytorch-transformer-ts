import random

from lightning import LightningModule
import torch

from gluonts.core.component import validated
from gluonts.torch.util import take_last, repeat_along_dim
from gluonts.itertools import prod

from module import LagLlamaModel
from aug import freq_mask, freq_mix


class LagLlamaLightningModule(LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``LagLlamaLightningModule`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``LagLlamaLightningModule`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``LagLlamaLightningModule`` to be trained.
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
        n_predictions: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        aug_rate: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context_length = self.hparams.context_length
        self.prediction_length = self.hparams.prediction_length
        self.n_predictions = self.hparams.n_predictions

        self.loss = None
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.aug_rate = self.hparams.aug_rate

        self.model = LagLlamaModel(**self.hparams.model_kwargs)

    # ancestral sampling next-step prediction
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

        repeated_future_time_feat = repeated_future_time_feat[:, : self.prediction_length, :]

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
            sliced_params = [p[:, -1:] for p in params[0]]
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

    # Self-speculative decoding
    def forward(self, *args, **kwargs):
        past_target = kwargs["past_target"]
        past_observed_values = kwargs["past_observed_values"]
        past_time_feat = kwargs["past_time_feat"]
        future_time_feat = kwargs["future_time_feat"]

        # Repeat inputs for parallel sampling
        repeated_past_target = past_target.repeat_interleave(self.model.num_parallel_samples, 0)
        repeated_past_observed_values = past_observed_values.repeat_interleave(self.model.num_parallel_samples, 0)
        repeated_past_time_feat = past_time_feat.repeat_interleave(self.model.num_parallel_samples, 0)
        repeated_future_time_feat = future_time_feat.repeat_interleave(self.model.num_parallel_samples, 0)
        repeated_future_time_feat = repeated_future_time_feat[:, :self.prediction_length, :]

        future_samples = []
        t = 0
        while t < self.prediction_length:
            # Determine number of steps to speculate (don't exceed prediction_length)
            steps_remaining = self.prediction_length - t
            n_spec_steps = min(self.n_predictions, steps_remaining)
            
            # Generate speculative predictions
            spec_samples = []
            for _ in range(n_spec_steps):
                params, loc, scale = self.model(
                    *args,
                    past_time_feat=repeated_past_time_feat,
                    future_time_feat=repeated_future_time_feat[..., :t + len(spec_samples) + 1, :],
                    past_target=repeated_past_target,
                    past_observed_values=repeated_past_observed_values,
                    is_test=False,
                )
                sliced_params = [p[:, -1:] for p in params[0]]
                distr = self.model.distr_output.distribution(sliced_params, loc, scale)
                sample = distr.sample()
                spec_samples.append(sample)

            # Verify speculative predictions
            verified_samples = []
            for i, sample in enumerate(spec_samples):
                # For first step, always accept the prediction
                if i == 0:
                    verified_samples.append(sample)
                    continue
                    
                # Verify each subsequent prediction
                temp_target = torch.cat([repeated_past_target] + verified_samples + [sample], dim=1)
                params, loc, scale = self.model(
                    *args,
                    past_time_feat=repeated_past_time_feat,
                    future_time_feat=repeated_future_time_feat[..., :t + i + 1, :],
                    past_target=temp_target,
                    past_observed_values=repeated_past_observed_values,
                    is_test=False,
                )
                
                sliced_params = [p[:, -1:] for p in params[0]]
                distr = self.model.distr_output.distribution(sliced_params, loc, scale)
                verified_sample = distr.sample()
                
                # If verification differs significantly, stop speculation
                if torch.abs(verified_sample - sample).mean() > 0.1:  # threshold can be tuned
                    break
                verified_samples.append(verified_sample)

            # Update state with verified predictions
            future_samples.extend(verified_samples)
            repeated_past_target = torch.cat([repeated_past_target] + verified_samples, dim=1)
            repeated_past_observed_values = torch.cat(
                [repeated_past_observed_values] + [torch.ones_like(s) for s in verified_samples], 
                dim=1
            )
            
            t += len(verified_samples)
            
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

    # train matryoshka loss
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
            future_time_feat=future_time_feat[:, : self.prediction_length, :],
            future_target=future_target_reshaped[:, :self.prediction_length],
        )
        
        losses = []
        for i, distr_arg in enumerate(distr_args):
            context_target = take_last(past_target, dim=-1, num=self.context_length - (1 + i))
            target = torch.cat(
                (context_target, future_target_reshaped[:, :self.prediction_length + i]),
                dim=1,
            )
            context_observed = take_last(
                past_observed_values, dim=-1, num=self.context_length - (1 + i)
            )
            observed_values = torch.cat((context_observed, future_observed_reshaped[:, :self.prediction_length + i]), dim=1)
            
            #distr = self.model.distr_output.distribution(distr_arg, loc, scale)
            
            losses.append(
                (self.model.distr_output.loss(target, distr_arg, loc=loc, scale=scale) * observed_values
            ).sum() / observed_values.sum().clamp_min(1.0))
        

        return torch.stack(losses).mean()


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
