from typing import List, Optional, Iterable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
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
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.time_feature import get_lags_for_frequency


class TransformerModel(nn.Module):
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
        embedding_dimension: Optional[List[int]] = None,
        # Added transformer arguments
        encoder=None,
        decoder=None,
        embeding=None,
        target_embed=None,
        generator=None,
        #############################
        dropout_rate: float = 0.1,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(hidden_size)
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
        self.history_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        # Added transformer enc-decoder and mask initializer
        self.encoder = encoder
        self.decoder = decoder
        self.embeding = embeding
        self.target_embed = target_embed
        self.generator = generator
        ########################

    # TODO
    # add method that does the forward for training

    """
    A build-in Encoder-Decoder architecture for TransformerModel class
    """

    def forward(self, src, tgt, mask_source, mask_target):
        "Take in and process masked sourcerc and target sequences."
        memory = self.encoder(self.embeding(src), mask_source)
        output = self.decoder(self.target_embed(tgt), memory, mask_source, mask_target)
        return output

    def encode(self, src, mask_source):
        return self.encoder(self.src_embed(src), mask_source)

    def decode(self, memory, mask_source, tgt, mask_target):
        return self.decoder(self.tgt_embed(tgt), memory, mask_source, mask_target)

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + 1  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

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

        # TODO
