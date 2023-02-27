from pytorch_lightning.utilities.model_summary import summarize
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.common import ListDataset
from estimator import TransformerEstimator
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset
from pytorch_lightning.loggers import CSVLogger
#Tuning GluonTS models with Optuna
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import json
import torch
from gluonts.evaluation import Evaluator
import time
import random


logger = CSVLogger("logs", name="vanilla")
# seed_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
seed_list = [0, 42, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
value=[]
seed_l = []
dataset_name = []
crps = []
dataset = get_dataset("solar-energy")
params = []
dim = 3
prediction_length = 24
freq = dataset.metadata.freq
seed_list = [32]#[0, 42, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
for layer in [2, 4, 8, 12, 16, 20, 24, 28]:

    value = []
    for seed in seed_list:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        estimator = TransformerEstimator(
            freq=dataset.metadata.freq,
            prediction_length=24,
            context_length=144,
            nhead=2,
            num_encoder_layers=layer,
            num_decoder_layers=6,
            dim_feedforward=16,
            activation="gelu",
            num_feat_static_cat=1,
            cardinality=[int(dataset.metadata.feat_static_cat[0].cardinality)],
            embedding_dimension=[dim],
            batch_size=192,
            num_batches_per_epoch=100,
            trainer_kwargs=dict(max_epochs=9, accelerator='auto', gpus=1, logger=logger))#
        predictor = estimator.train(
        training_data=dataset.train,
         # validation_data=val_ds,
         num_workers=8,
         # shuffle_buffer_length=1024
        )
        forecast_it, ts_it = make_evaluation_predictions(dataset=dataset.test, predictor=predictor)
        forecasts = list(forecast_it)
         # if layer == layers[0]:
        tss = list(ts_it)
        evaluator = Evaluator()
        agg_metrics, _ = evaluator(iter(tss), iter(forecasts))
        # agg_metrics["trainable_parameters"] = summarize(estimator.create_lightning_module()).trainable_parameters
        value.append(agg_metrics['mean_wQuantileLoss'])
        seed_l.append(seed)
        # dataset_name.append(i)
        print(seed)
    print(layer)
    crps.append(np.mean(value))
    params.append(summarize(estimator.create_lightning_module()).trainable_parameters)

df = pd.DataFrame()
df['params'] = params
df['crps'] = crps
df.to_csv('scaling_elec.csv', index = False)