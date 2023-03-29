from pytorch_lightning.utilities.model_summary import summarize
from functools import lru_cache, partial
from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict, Any
from itertools import islice
from pandas.tseries.frequencies import to_offset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.common import ListDataset
from estimator import TransformerEstimator
from gluonts.dataset.util import to_pandas
import tqdm.auto as tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from datasets import load_dataset, interleave_datasets
from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset, ListDataset, DatasetCollection, ProcessDataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository.datasets import get_dataset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset, ListDataset, DatasetCollection, ProcessDataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from datasets import load_dataset, interleave_datasets
from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
    InstanceSampler,
)
from gluonts.torch.util import (
    IterableDataset,
)
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.util import weighted_average
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.time_feature import get_lags_for_frequency
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.common import ListDataset, Dataset, DatasetCollection, Cached
#Tuning GluonTS models with Optuna
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import torch
import json
#import stat
from gluonts.evaluation import Evaluator
import time
import random
import os
from pytorch_lightning import Trainer
from augmentation import TimeWrap#, RandomGuidedWarp
#from lightning.pytorch import Trainer, seed_everything
from deepspeed.profiling.flops_profiler import FlopsProfiler
import timeit
checkpoint_callback = ModelCheckpoint(filename='epoch{epoch:02d}_step{step:02d}',
          verbose=True,
          save_top_k=1,auto_insert_metric_name=False)

time_features=[
        MinuteOfHour(),
        HourOfDay(),
        DayOfWeek(),
        DayOfMonth(),
        DayOfYear(),
    ]

def add_time_feature(data, freq):
    length = len(data[FieldName.TARGET])
    start = pd.Period(data[FieldName.START], freq)
    index = pd.period_range(start, periods=length, freq=freq)

    data[FieldName.FEAT_TIME] = np.vstack(
        [feat(index) for feat in time_features]
    ).astype(np.float32)

    age = np.log10(2.0 + np.arange(length, dtype=np.float32))
    data[FieldName.FEAT_AGE] = age.reshape((1, length))
    return data

@lru_cache(10_000)
def as_period(val, freq):
    return pd.Period(val, freq)


@dataclass
class GluontsDataset(Dataset):
    def __init__(self, dataset, freq, prediction_length=24) -> None:
        super().__init__()
        transform = Chain([
             AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features,
                    pred_length=prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=prediction_length,
                    log_scale=True,
                ),
        ])

        self.dataset = Cached(transform.apply(dataset))
        self.freq = to_offset(freq)
        self.prediction_length = prediction_length

    def __iter__(self):
        for data in self.dataset:
            if len(data[FieldName.TARGET]) > self.prediction_length:
                yield {
                    FieldName.START: as_period(data[FieldName.START], self.freq),
                    FieldName.TARGET: data[FieldName.TARGET],
                    FieldName.FEAT_TIME: np.stack(data[FieldName.FEAT_TIME], 0),
                    FieldName.FEAT_AGE: np.stack(data[FieldName.FEAT_AGE], 0),
                    FieldName.ITEM_ID: data[FieldName.ITEM_ID],
                }

    def __len__(self):
        return len(self.dataset)


train_ds_list = []
#val_ds_list = []
test_ds_list = []
for i in ["electricity","traffic", "m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "solar-energy"]:
    print(i)
    dataset = get_dataset(i)
    train = GluontsDataset(dataset.train, dataset.metadata.freq, 24)
    test = GluontsDataset(dataset.test, dataset.metadata.freq, 24)
    train_ds_list.append(train)
    test_ds_list.append(test)

train_ds_size = np.array([len(ds) for ds in train_ds_list])
raw_weights = 1/train_ds_size
normalization_factor = 1/sum(raw_weights)
probablities = raw_weights * normalization_factor
train_ds = RandomlyCyclingMultiSourcesExamplesIterable(
     train_ds_list,
    # generator=np.random.default_rng(), 
     probabilities=probablities,
)
#train_ds = DatasetCollection(datasets=train_ds_list, interleave=True)
test_ds = DatasetCollection(datasets=test_ds_list, interleave=True)

class TransformerModel(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        
        # transformer arguments
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        activation: str = "gelu",
        dropout: float = 0.1,

        # univariate input
        input_size: int = 1,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        freq: Optional[str] = None,
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()
        
        assert (freq is not None) or (lags_seq is not None), "either freq or lags_seq must be given"
        
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
            
        # transformer enc-decoder and mask initializer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        
        # causal decoder tgt mask
        self.register_buffer(
            "tgt_mask",
            self.transformer.generate_square_subsequent_mask(prediction_length),
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
            torch.cat(
                (
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.context_length :, ...]
        )

        # target
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
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
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
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


        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)
        
        return transformer_inputs, scale, static_feat
    
    def output_params(self, transformer_inputs):
        enc_input = transformer_inputs[:, :self.context_length, ...]
        dec_input = transformer_inputs[:, self.context_length:, ...]
        
        enc_out = self.transformer.encoder(
            enc_input
        )
        dec_output = self.transformer.decoder(
            dec_input,
            enc_out,
            tgt_mask=self.tgt_mask
        )
        
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
        )
        
        enc_out = self.transformer.encoder(encoder_inputs)
        
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(
                repeats=self.num_parallel_samples, dim=0
            )
            / repeated_scale
        )
        
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_feat.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
       
        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        future_samples = []
        
        # greedy decoding
        for k in range(self.prediction_length):            
            #self._check_shapes(repeated_past_target, next_sample, next_features)
            #sequence = torch.cat((repeated_past_target, next_sample), dim=1)
            
            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1+k,
                shift=1, 
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )
            
            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k+1]), dim=-1)

            output = self.transformer.decoder(decoder_input, repeated_enc_out)
            
            params = self.param_proj(output[:,-1:])
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            
            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length)
            + self.target_shape,
        )
class TransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: TransformerModel= TransformerModel(activation="gelu", cardinality=[1], context_length=240, dim_feedforward=32, distr_output=StudentTOutput(), dropout=0.1, embedding_dimension=None, freq=None, input_size=1, lags_seq=[1, 2, 3, 4, 5, 6, 7, 24, 30], nhead=2, num_decoder_layers=6, num_encoder_layers=2, num_feat_dynamic_real=6, num_feat_static_cat=1, num_feat_static_real=1, num_parallel_samples=100, prediction_length=24, scaling=True),
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        time_warp: bool = True,
        # random_guided_warp: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # if model is None:
        #     self.model = None
        # else:   
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
     
        
#         self.random_guided_warp = random_guided_warp
        self.time_warp = time_warp

#         if self.random_guided_warp:
#             self.random_guided_warp_fn = RandomGuidedWarp()

        # if self.time_warp:
        # self.time_warp_fn = TimeWrap(p=0.5)
        self.batch_num = 0
        self.start_time = 0
        self.end_time = 0
    def freq_mask(self, x, y, rate=0.1, dim=1):
        x_len = x.shape[dim]
        y_len = y.shape[dim]
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate

        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)
        
        if x_len + y_len != xy.shape[dim]:
            xy = torch.cat([x[:,0:1, ...],  xy], 1)

        return torch.split(xy, [x_len, y_len], dim=dim)


    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
#         if self.random_guided_warp:
#             batch = self.random_guided_warp_fn(batch)
        
        # if self.time_warp:
        #     batch = self.time_warp_fn(batch)
        past_target, future_target = self.freq_mask(x=batch["past_target"], y=batch["future_target"])
        batch["past_target"] = past_target
        batch["future_target"] = future_target
        if batch_idx == 0:
            self.start_time  = timeit.default_timer()
        if batch_idx == 5:
            self.prof = FlopsProfiler(self)
            self.prof.start_profile()
        train_loss = self(batch)
        self.batch_num = batch_idx
        if batch_idx == 5:
            flops = self.prof.get_total_flops()
            params = self.prof.get_total_params()
            duration = self.prof.get_total_duration()
            print(flops, params)
            # flops = flops.replace("G", "")
            # flops = flops.replace(" ", "")

            # self.prof.print_model_profile(profile_step=profile_step)
            self.prof.end_profile()
            self.log( "training_flops", float(flops), on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
            
            self.log("throughput", float(4/duration), on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True, sync_dist=True
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
#         if self.random_guided_warp:
#             batch = self.random_guided_warp_fn(batch)
        
        # if self.time_warp:
        #     batch = self.time_warp_fn(batch)

        with torch.inference_mode():
            val_loss = self(batch)
        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True
        )
        return val_loss
    
    def on_train_epoch_end(self):
        self.end_time = timeit.default_timer()
        elapsed = self.end_time - self.start_time
        self.log("samples_per_sec", float((4*self.batch_num)/elapsed), on_epoch=True, on_step=False, prog_bar=True)

        
    def configure_optimizers(self):
        """Returns the optimizer to use"""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            #capturable=True,
        )
    
    def forward(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]
        
        transformer_inputs, scale, _ = self.model.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        params = self.model.output_params(transformer_inputs)
        distr = self.model.output_distribution(params, scale)

        loss_values = self.loss(distr, future_target)
    
        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class TransformerEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        prediction_length: int,

        # Transformer arguments
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        input_size: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1,

        context_length: Optional[int] = None,

        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        loss: DistributionLoss = NegativeLogLikelihood(),
        scaling: bool = True,
        freq: Optional[str] = None,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        trainer_kwargs = {
            "max_epochs": 100,
            **trainer_kwargs,
        }
        super().__init__(trainer_kwargs=trainer_kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.loss = loss

        self.input_size = input_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.lags_seq = lags_seq
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.num_feat_static_cat > 0
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
#                 AddTimeFeatures(
#                     start_field=FieldName.START,
#                     target_field=FieldName.TARGET,
#                     output_field=FieldName.FEAT_TIME,
#                     time_features=self.time_features,
#                     pred_length=self.prediction_length,
#                 ),
#                 AddAgeFeature(
#                     target_field=FieldName.TARGET,
#                     output_field=FieldName.FEAT_AGE,
#                     pred_length=self.prediction_length,
#                     log_scale=True,
#                 ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                ),
            ]
        )

    def _create_instance_splitter(
        self, module: TransformerLightningModule, mode: str
    ):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: TransformerLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "training"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    pin_memory=True,
                    persistent_workers=kwargs.get("num_workers", 0) > 0,
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: TransformerLightningModule,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "validation"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=kwargs.get("num_workers", 0) > 0,
            **kwargs,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: TransformerLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module.model,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    def create_lightning_module(self) -> TransformerLightningModule:
        model = TransformerModel(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=1 + self.num_feat_dynamic_real + len(self.time_features),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,

            # transformer arguments
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            activation=self.activation,
            dropout=self.dropout,
            dim_feedforward=self.dim_feedforward,

            # univariate input
            input_size=self.input_size,
            distr_output=self.distr_output,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )
        
        return TransformerLightningModule(model=model, loss=self.loss)


value=[]
seed_l = []

crps = []
#dataset = get_dataset("traffic")
params = []
dim = 3
prediction_length = 24
#freq = dataset.metadata.freq

seed_list = [0]#, 42, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
for seed in seed_list:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    all_value = []
    value = []
    for layer in [512]:#: [2,4,8 12,16, 20, 24, 28,32, 46, 64, 74, 76]:#, 20, 24, 28, 32, 46, 64, 74, 76, 1048576]:
        #seed_everything(seed)
        logger = CSVLogger("logs_aug", name="gpu_180_seed"+str(seed)+"layer"+str(layer))
        # st = os.stat("checkpoints/")
        # os.chmod("checkpoints/", st.st_mode | 0o222)
        # print(train_ds.device)
        estimator = TransformerEstimator(
            #freq=dataset.metadata.freq,
                prediction_length=24,
                context_length=24*10,
                lags_seq=[1,2,3,4,5,6,7,24,30],
                time_features=[
                    MinuteOfHour(),
                    HourOfDay(),
                    DayOfWeek(),
                    DayOfMonth(),
                    DayOfYear(),
                ],

                nhead=2,
                num_encoder_layers=layer,
                num_decoder_layers=6,
                dim_feedforward=32,
                activation="gelu",

                scaling=True,

                batch_size=4,
                num_batches_per_epoch=100,
                trainer_kwargs=dict(max_epochs=5, accelerator='gpu', gpus=6, num_nodes = 30, strategy="ddp",logger=logger,  gradient_clip_val=5, callbacks=[checkpoint_callback]))##
        
        predictor = estimator.train(training_data=train_ds, 
        # validation_data=val_ds,
        # num_workers=0,
        # shuffle_buffer_length=1024
        )

#         forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds, predictor=predictor)
#         forecasts = list(forecast_it)
#          # if layer == layers[0]:
#         tss = list(ts_it)

#         evaluator = Evaluator()
#         agg_metrics, _ = evaluator(iter(tss), iter(forecasts))
#         # agg_metrics["trainable_parameters"] = summarize(estimator.create_lightning_module()).trainable_parameters
#         value.append(agg_metrics['mean_wQuantileLoss'])
#         seed_l.append(seed)
#         # dataset_name.append(i)
#         print(layer)
#         all_value.append(agg_metrics)
#         params.append(summarize(estimator.create_lightning_module()).trainable_parameters)
#         df = pd.DataFrame()
#         df['seed'] = [seed]
#         df['crps'] = [agg_metrics['mean_wQuantileLoss']]
#         df['MAPE'] = [agg_metrics['MAPE']]
#         df['params'] = [summarize(estimator.create_lightning_module()).trainable_parameters] 

#         df.to_csv('AugSeed_10'+str(seed)+'_layer'+str(layer)+'.csv', index=False)
#         torch.cuda.empty_cache()

#     df = pd.DataFrame()
#     df['params'] = params
#     df['metrics'] = all_value
#     df['crps'] = value
#     df.to_csv('scaling_data'+str(seed)+'.csv', index = False)
#     print(layer)
#     crps.append(np.mean(value))

# df = pd.DataFrame()
# df['params'] = params
# df['crps'] = crps
# df.to_csv('scaling_elec.csv', index = False)









