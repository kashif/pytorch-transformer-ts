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

    # # ancestral sampling via next-step prediction using only the first distribution
    # def forward(self, *args, **kwargs):
    #     past_target = kwargs["past_target"]
    #     past_observed_values = kwargs["past_observed_values"]
    #     past_time_feat = kwargs["past_time_feat"]
    #     future_time_feat = kwargs["future_time_feat"]

    #     repeated_past_target = past_target.repeat_interleave(
    #         self.model.num_parallel_samples, 0
    #     )
    #     repeated_past_observed_values = past_observed_values.repeat_interleave(
    #         self.model.num_parallel_samples, 0
    #     )
    #     repeated_past_time_feat = past_time_feat.repeat_interleave(
    #         self.model.num_parallel_samples, 0
    #     )
    #     repeated_future_time_feat = future_time_feat.repeat_interleave(
    #         self.model.num_parallel_samples, 0
    #     )

    #     repeated_future_time_feat = repeated_future_time_feat[
    #         :, : self.prediction_length, :
    #     ]

    #     future_samples = []
    #     for t in range(self.prediction_length):
    #         params, loc, scale = self.model(
    #             *args,
    #             past_time_feat=repeated_past_time_feat,
    #             future_time_feat=repeated_future_time_feat[..., : t + 1, :],
    #             past_target=repeated_past_target,
    #             past_observed_values=repeated_past_observed_values,
    #             is_test=False,
    #         )
    #         # use only the first distribution for the ne
    #         sliced_params = [p[:, -1:] for p in params[0]]
    #         distr = self.model.distr_output.distribution(sliced_params, loc, scale)
    #         sample = distr.sample()
    #         future_samples.append(sample)

    #         repeated_past_target = torch.cat((repeated_past_target, sample), dim=1)
    #         repeated_past_observed_values = torch.cat(
    #             (repeated_past_observed_values, torch.ones_like(sample)), dim=1
    #         )

    #     self.model.reset_cache()

    #     concat_future_samples = torch.cat(future_samples, dim=-1)
    #     return concat_future_samples.reshape(
    #         (-1, self.model.num_parallel_samples, self.prediction_length)
    #         + self.model.distr_output.event_shape,
    #     )

    def forward(self, *args, **kwargs):
        """
        multi-step ancestral sampling, for each time step predict the next n_predictions - 1 steps and then continue for time step n_predictions till prediction_length
        """
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
        )[:, :self.prediction_length]

        future_samples = []
        cur_pos = 0

        while cur_pos < self.prediction_length:
            remaining_len = self.prediction_length - cur_pos
            steps_to_predict = min(self.n_predictions - 1, remaining_len)

            # Get multi-step predictions from the model
            params, loc, scale = self.model(
                past_time_feat=repeated_past_time_feat,
                future_time_feat=repeated_future_time_feat[..., :cur_pos + 1, :],
                past_target=repeated_past_target,
                past_observed_values=repeated_past_observed_values,
                is_test=False
            )
            
            # Sample proposed values from each distribution
            proposed_samples = []
            for i in range(steps_to_predict):
                # Get distribution for this step using parameters from the i-th prediction head
                sliced_params = [p[:, -1:] for p in params[i]]
                distr = self.model.distr_output.distribution(sliced_params, loc, scale)
                sample = distr.sample()
                proposed_samples.append(sample)
            # Concatenate sampled steps
            proposed_samples = torch.cat(proposed_samples, dim=1)
            
            # Append to future_samples
            future_samples.append(proposed_samples)

            # Update past_target and past_observed_values
            # Assuming the target dimension is at dim=1
            repeated_past_target = torch.cat(
                [repeated_past_target, proposed_samples], dim=1
            )
            repeated_past_observed_values = torch.cat(
                [repeated_past_observed_values, torch.ones_like(proposed_samples)],
                dim=1
            )

            # Update current position
            cur_pos += steps_to_predict


        self.model.reset_cache()

        # Concatenate and reshape samples
        concat_future_samples = torch.cat(future_samples, dim=1)[:, :self.prediction_length]
        return concat_future_samples.reshape(
            (-1, self.model.num_parallel_samples, self.prediction_length) 
            + self.model.distr_output.event_shape
        )

    # def forward(self, *args, **kwargs):
    #     """
    #     Self-speculative decoding for continuous distributions where model predicts 
    #     distribution parameters for multiple future steps.
    #     """
    #     past_target = kwargs["past_target"]
    #     past_observed_values = kwargs["past_observed_values"]
    #     past_time_feat = kwargs["past_time_feat"]
    #     future_time_feat = kwargs["future_time_feat"]

    #     repeated_past_target = past_target.repeat_interleave(
    #         self.model.num_parallel_samples, 0
    #     )
    #     repeated_past_observed_values = past_observed_values.repeat_interleave(
    #         self.model.num_parallel_samples, 0
    #     )
    #     repeated_past_time_feat = past_time_feat.repeat_interleave(
    #         self.model.num_parallel_samples, 0
    #     )
    #     repeated_future_time_feat = future_time_feat.repeat_interleave(
    #         self.model.num_parallel_samples, 0
    #     )[:, :self.prediction_length]

    #     future_samples = []
    #     cur_pos = 0

    #     while cur_pos < self.prediction_length:
    #         remaining_len = self.prediction_length - cur_pos
            
    #         # Get multi-step predictions from the model
    #         params, loc, scale = self.model(
    #             past_time_feat=repeated_past_time_feat,
    #             future_time_feat=repeated_future_time_feat[..., :cur_pos + 1, :],
    #             past_target=repeated_past_target,
    #             past_observed_values=repeated_past_observed_values,
    #             is_test=False
    #         )
            
    #         # Sample proposed values from each distribution
    #         proposed_samples = []
    #         for i in range(min(self.n_predictions -1, remaining_len)):
    #             # Get distribution for this step using parameters from the i-th prediction head
    #             sliced_params = [p[:, -1:] for p in params[i]]
    #             distr = self.model.distr_output.distribution(sliced_params, loc, scale)
    #             sample = distr.sample()
    #             proposed_samples.append(sample)
    #         proposed_samples = torch.cat(proposed_samples, dim=1)
            
    #         # verify the proposed samples by passing them to the model in parallel:
    #         proposed_params, proposed_loc, proposed_scale = self.model(
    #             past_time_feat=repeated_past_time_feat,
    #             future_time_feat=repeated_future_time_feat[..., :cur_pos + self.n_predictions - 1, :],
    #             past_target=repeated_past_target,
    #             past_observed_values=repeated_past_observed_values,
    #             future_target=proposed_samples,
    #         )

    #         # get the last "self.n_predictions - 1" parameters from  proposed_params[0]
    #         proposed_sliced_params = [p[:, -self.n_predictions + 1:] for p in proposed_params[0]]
    #         proposed_distr = self.model.distr_output.distribution(proposed_sliced_params, proposed_loc, proposed_scale)
    #         proposed_nll = - proposed_distr.log_prob(proposed_samples)

    #         import pdb; pdb.set_trace() 
    #         # TODO The rest of the code is not correct yet

    #         # Verify proposals using base model
    #         accepted_samples = []
    #         for i, proposal in enumerate(proposed_samples):
    #             # Add proposal to sequence temporarily
    #             test_target = torch.cat([repeated_past_target, 
    #                                 torch.cat(accepted_samples + [proposal], dim=1)], dim=1)
                
    #             # Get base model prediction for this position
    #             base_params, base_loc, base_scale = self.model(
    #                 past_time_feat=repeated_past_time_feat,
    #                 future_time_feat=repeated_future_time_feat[..., :cur_pos + i + 1, :],
    #                 past_target=test_target,
    #                 past_observed_values=repeated_past_observed_values,
    #                 is_test=False
    #             )
                
    #             # Get distribution from base model (first head/distribution)
    #             base_distr = self.model.distr_output.distribution(base_params[0], base_loc, base_scale)
                
    #             # Calculate log probability of proposal under base distribution
    #             log_prob = base_distr.log_prob(proposal)
                
    #             # Accept if log probability is above threshold
    #             if torch.all(log_prob > -5.0):  # Threshold can be tuned
    #                 accepted_samples.append(proposal)
    #             else:
    #                 break
                    
    #         if len(accepted_samples) == 0:
    #             # If no proposals accepted, sample one step from base distribution
    #             distr = self.model.distr_output.distribution(params[0], loc, scale)
    #             sample = distr.sample()
    #             future_samples.append(sample)
    #             repeated_past_target = torch.cat([repeated_past_target, sample], dim=1)
    #             cur_pos += 1
    #         else:
    #             # Add all accepted proposals
    #             future_samples.extend(accepted_samples)
    #             repeated_past_target = torch.cat([repeated_past_target] + accepted_samples, dim=1)
    #             cur_pos += len(accepted_samples)

    #         repeated_past_observed_values = torch.cat(
    #             (repeated_past_observed_values, torch.ones_like(repeated_past_target[:, -len(accepted_samples) or -1:])), 
    #             dim=1
    #         )

    #     self.model.reset_cache()

    #     # Concatenate and reshape samples
    #     concat_future_samples = torch.cat(future_samples, dim=1)
    #     return concat_future_samples.reshape(
    #         (-1, self.model.num_parallel_samples, self.prediction_length) 
    #         + self.model.distr_output.event_shape
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
            future_target=future_target_reshaped[:, : self.prediction_length],
        )

        losses = []
        # for all the multi-step predictions calculate the loss
        for i, distr_arg in enumerate(distr_args):
            context_target = take_last(
                past_target, dim=-1, num=self.context_length - (1 + i)
            )
            target = torch.cat(
                (
                    context_target,
                    future_target_reshaped[:, : self.prediction_length + i],
                ),
                dim=1,
            )
            context_observed = take_last(
                past_observed_values, dim=-1, num=self.context_length - (1 + i)
            )
            observed_values = torch.cat(
                (
                    context_observed,
                    future_observed_reshaped[:, : self.prediction_length + i],
                ),
                dim=1,
            )

            # distr = self.model.distr_output.distribution(distr_arg, loc, scale)

            losses.append(
                (
                    self.model.distr_output.loss(
                        target, distr_arg, loc=loc, scale=scale
                    )
                    * observed_values
                ).sum()
                / observed_values.sum().clamp_min(1.0)
            )

        # final loss is the mean of all the multi-step losses
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
