from typing import Any, Dict, Iterable, List, Optional

import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    DummyValueImputation,
    # AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler

from lightning_module import PyraformerLightningModule
# from module import PyraformerSSModel
# from module import PyraformerLRModel
from tools import SingleStepLoss as LossFactory
from torch.utils.data.sampler import RandomSampler

# PREDICTION_INPUT_NAMES = [
#     "feat_static_cat",
#     "feat_static_real",
#     "past_time_feat",
#     "past_target",
#     "past_observed_values",
#     "future_time_feat",
# ]
PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]
TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class PyraformerEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        # freq: str,
        prediction_length: int,
        # Train parameters
        batch_size: int = 8,
        lr: float = 1e-5,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        aug_rate: float = 0.1,

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
        num_encoder_layers: int = 4,
        enc_in: int = 1,  # depends on dataset used
        CSCM: str = "Bottleneck_Construct",  # [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
        embed_type: str = "CustomEmbedding",  # [DataEmbedding, CustomEmbedding]
        truncate: bool = False,
        # loss: DistributionLoss = LossFactory,
        ignore_zero: bool = True,
        single_step: bool = True,  # if False, Multistep=True
        inner_size: int = 3,
        use_tvm: bool = False,
        # num_feat_dynamic_real: int = 0,
        # num_feat_static_cat: int = 0,
        # num_feat_static_real: int = 0,
        # cardinality: Optional[List[int]] = None,
        # embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        loss: DistributionLoss = NegativeLogLikelihood(),
        scaling: bool = True,
        # lags_seq: Optional[List[int]] = None,
        # time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        window_size: int = [2,2,2],#[4, 4, 4],
        ckpt_path: Optional[str] = None,
    ) -> None:
        trainer_kwargs = {
            "max_epochs": 10,
            **trainer_kwargs,
        }
        super().__init__(trainer_kwargs=trainer_kwargs)

        # self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.aug_prob = aug_prob
        self.aug_rate = aug_rate
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
        self.n_layer = num_encoder_layers
        self.single_step = single_step
        self.ignore_zero = ignore_zero
        self.decoder = decoder
        self.enc_in = enc_in
        self.CSCM = CSCM
        self.embed_type = embed_type
        self.truncate = truncate
        self.loss = loss
        # self.loss = (
        #     LossFactory(self.ignore_zero)
        #     if self.single_step == True
        #     else torch.nn.MSELoss(reduction="none")
        # )
        self.distr_output = distr_output

        self.window_size = window_size  # [4,4,4]#window_size
        self.inner_size = inner_size
        self.use_tvm = use_tvm
        self.prediction_length = prediction_length
        # self.epochs = trainer_kwargs['max_epochs']
        # self.train_sampler = RandomSampler  or ExpectedNumInstanceSampler(num_instances=1.0, min_future=prediction_length)
        # self.validation_sampler = RandomSampler or ValidationSplitSampler(min_future=prediction_length)
        # self.test_sampler = RandomSampler

        # self.num_feat_dynamic_real = num_feat_dynamic_real
        # self.num_feat_static_cat = num_feat_static_cat
        # self.num_feat_static_real = num_feat_static_real
        # self.cardinality = (
        #     cardinality if cardinality and num_feat_static_cat > 0 else [1]
        # )
        # self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        # self.lags_seq = lags_seq
        # self.time_features = (
        #     time_features
        #     if time_features is not None
        #     else time_features_from_frequency_str(self.freq)
        # )

        self.num_parallel_samples = num_parallel_samples
        self.num_batches_per_epoch = num_batches_per_epoch
        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )
        self.ckpt_path = ckpt_path

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    imputation_method=DummyValueImputation(0.0),
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
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                # FieldName.FEAT_TIME,
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
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: PyraformerLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
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
            prediction_length=self.prediction_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def create_lightning_module(self) -> PyraformerLightningModule:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = {
            "context_length":self.context_length,
            "prediction_length":self.prediction_length,
            "covariate_size":self.covariate_size,
            "num_seq":self.num_seq,
            "dropout":self.dropout,
            "input_size":self.input_size,
            "d_model":self.d_model,
            "d_inner_hid":self.d_inner_hid,
            "d_k":self.d_k,
            "d_v":self.d_v,
            "num_heads":self.num_heads,
            "n_layer":self.n_layer,
            "loss":self.loss,
            "window_size":self.window_size,
            "inner_size":self.inner_size,
            "use_tvm":self.use_tvm,
            "distr_output":self.distr_output,
            "scaling":self.scaling,
            "num_parallel_samples":self.num_parallel_samples,
        }
        if self.ckpt_path is not None:
            return PyraformerLightningModule.load_from_checkpoint(
                self.ckpt_path,
                model_kwargs=model_kwargs,
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                aug_prob=self.aug_prob,
                aug_rate=self.aug_rate,
            )
        else:
            return PyraformerLightningModule(
                model_kwargs=model_kwargs,
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                # context_length=self.context_length,
                # prediction_length=self.prediction_length,
                aug_prob=self.aug_prob,
                aug_rate=self.aug_rate,
            )

        # if self.single_step:
            # model = PyraformerSSModel(
            #     # freq=self.freq,
            #     covariate_size=self.covariate_size,
            #     num_seq=self.num_seq,
            #     input_size=self.input_size,
            #     dropout=self.dropout,
            #     d_model=self.d_model,
            #     d_inner_hid=self.d_inner_hid,
            #     d_k=self.d_k,
            #     d_v=self.d_v,
            #     num_heads=self.num_heads,
            #     n_layer=self.n_layer,
            #     loss=self.loss,
            #     window_size=self.window_size,
            #     inner_size=self.inner_size,
            #     use_tvm=self.use_tvm,
            #     prediction_length=self.prediction_length,
            #     context_length=self.context_length,
            #     distr_output=self.distr_output,
            #     scaling=self.scaling,
            #     num_parallel_samples=self.num_parallel_samples,
            #     device=device,
            # )
        # else:
        #     model = PyraformerLRModel(
        #         predict_step=self.prediction_length,
        #         d_model=self.d_model,
        #         input_size=self.input_size,
        #         decoder=self.decoder,
        #         window_size=self.window_size,
        #         truncate=self.truncate,
        #         d_inner_hid=self.d_inner_hid,
        #         d_k=self.d_k,
        #         d_v=self.d_v,
        #         dropout=self.dropout,
        #         enc_in=self.enc_in,
        #         covariate_size=self.covariate_size,
        #         seq_num=self.num_seq,
        #         CSCM=self.CSCM,
        #         d_bottleneck=self.d_bottleneck,
        #         num_head=self.num_heads,
        #         n_layer=self.n_layer,
        #         inner_size=self.inner_size,
        #         use_tvm=self.use_tvm,
        #         prediction_length=self.prediction_length,
        #         context_length=self.context_length,
        #         lags_seq=self.lags_seq,
        #         num_feat_dynamic_real=self.num_feat_dynamic_real,
        #         num_feat_static_cat=self.num_feat_static_cat,
        #         num_feat_static_real=self.num_feat_static_real,
        #         cardinality=self.cardinality,
        #         embedding_dimension=self.embedding_dimension,
        #         num_parallel_samples=self.num_parallel_samples,
        #         embed_type=self.embed_type,
        #         distr_output=self.distr_output,
        #         device=device,
            # )
        # return PyraformerLightningModule(model=model, loss=self.loss)
