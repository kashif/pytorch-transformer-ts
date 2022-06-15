from typing import Any, Dict, Iterable, List, Optional

import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.util import IterableDataset
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler
from lightning_module import PyraformerLightningModule
from module import PyraformerSSModel
from module import PyraformerLRModel
from torch.utils.data import DataLoader
from tools import SingleStepLoss as LossFactory
from torch.utils.data.sampler import RandomSampler

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


class PyraformerEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        # Train parameters
        inner_batch: int = 8,
        lr: float = 1e-5,
        visualize_fre: int = 2000,
        pretrain: bool = True,
        hard_sample_mining: bool = True,
        covariate_size: int = 3,
        # Model parameters
        num_seq: int = 370,  #
        decoder: str = "FC",  # selection: [FC, attention]
        context_length: Optional[int] = None,
        input_size: int = 1,
        dropout: float = 0.1,
        d_model: int = 512,
        d_inner_hid: int = 512,
        d_k: int = 128,
        d_v: int = 128,
        d_bottleneck: int = 128,
        num_heads: int = 4,
        n_layer: int = 4,
        enc_in: int = 1,  # depends on dataset used
        CSCM: str = "Bottleneck_Construct",  # [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
        embed_type: str = "CustomEmbedding",  # [DataEmbedding, CustomEmbedding]
        truncate: bool = False,
        # loss: DistributionLoss = LossFactory,
        ignore_zero: bool = True,
        single_step: bool = True,  # if False, Multistep=True
        inner_size: int = 3,
        use_tvm: bool = False,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        # loss: DistributionLoss = NegativeLogLikelihood(),
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        window_size: int = [4, 4, 4],
    ) -> None:
        trainer_kwargs = {
            "max_epochs": 10,
            **trainer_kwargs,
        }
        super().__init__(trainer_kwargs=trainer_kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.inner_batch = inner_batch
        self.lr = lr
        # self.visualize_fre = visualize_fre
        self.covariate_size = covariate_size
        self.num_seq = num_seq
        self.input_size = input_size
        self.dropout = dropout
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.d_k = d_k
        self.d_v = d_v
        self.d_bottleneck = d_bottleneck
        self.num_heads = num_heads
        self.n_layer = n_layer
        self.single_step = single_step
        self.ignore_zero = ignore_zero
        self.decoder = decoder
        self.enc_in = enc_in
        self.CSCM = CSCM
        self.embed_type = embed_type
        self.truncate = truncate
        self.loss = (
            LossFactory(self.ignore_zero)
            if self.single_step == True
            else torch.nn.MSELoss(reduction="none")
        )
        self.batch_size = batch_size
        self.distr_output = distr_output

        self.window_size = window_size  # [4,4,4]#window_size
        self.inner_size = inner_size
        self.use_tvm = use_tvm
        self.prediction_length = prediction_length
        # self.epochs = trainer_kwargs['max_epochs']
        # self.train_sampler = RandomSampler  or ExpectedNumInstanceSampler(num_instances=1.0, min_future=prediction_length)
        # self.validation_sampler = RandomSampler or ValidationSplitSampler(min_future=prediction_length)
        # self.test_sampler = RandomSampler

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
                [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
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
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
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

    def _create_instance_splitter(self, module: PyraformerLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]
        print(instance_sampler)
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
        module: PyraformerLightningModule,
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
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: PyraformerLightningModule,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "validation"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: PyraformerLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module.model,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def create_lightning_module(self) -> PyraformerLightningModule:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.single_step:
            model = PyraformerSSModel(
                freq=self.freq,
                covariate_size=self.covariate_size,
                num_seq=self.num_seq,
                input_size=self.input_size,
                dropout=self.dropout,
                d_model=self.d_model,
                d_inner_hid=self.d_inner_hid,
                d_k=self.d_k,
                d_v=self.d_v,
                num_heads=self.num_heads,
                n_layer=self.n_layer,
                loss=self.loss,
                window_size=self.window_size,
                inner_size=self.inner_size,
                use_tvm=self.use_tvm,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                lags_seq=self.lags_seq,
                num_feat_dynamic_real=self.num_feat_dynamic_real,
                num_feat_static_cat=self.num_feat_static_cat,
                num_feat_static_real=self.num_feat_static_real,
                cardinality=self.cardinality,
                embedding_dimension=self.embedding_dimension,
                distr_output=self.distr_output,
                scaling=self.scaling,
                num_parallel_samples=self.num_parallel_samples,
                device=device,
            )
        else:
            model = PyraformerLRModel(
                predict_step=self.prediction_length,
                d_model=self.d_model,
                input_size=self.input_size,
                decoder=self.decoder,
                window_size=self.window_size,
                truncate=self.truncate,
                d_inner_hid=self.d_inner_hid,
                d_k=self.d_k,
                d_v=self.d_v,
                dropout=self.dropout,
                enc_in=self.enc_in,
                covariate_size=self.covariate_size,
                seq_num=self.num_seq,
                CSCM=self.CSCM,
                d_bottleneck=self.d_bottleneck,
                num_head=self.num_heads,
                n_layer=self.n_layer,
                inner_size=self.inner_size,
                use_tvm=self.use_tvm,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                lags_seq=self.lags_seq,
                num_feat_dynamic_real=self.num_feat_dynamic_real,
                num_feat_static_cat=self.num_feat_static_cat,
                num_feat_static_real=self.num_feat_static_real,
                cardinality=self.cardinality,
                embedding_dimension=self.embedding_dimension,
                num_parallel_samples=self.num_parallel_samples,
                embed_type=self.embed_type,
                distr_output=self.distr_output,
                device=device,
            )
        return PyraformerLightningModule(model=model, loss=self.loss)
