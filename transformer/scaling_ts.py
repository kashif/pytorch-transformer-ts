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


import argparse

parser = argparse.ArgumentParser(description='scaling for time series')
parser.add_argument('--dataset', type=str, default='solar-energy', metavar='n', help='dataset name')
parser.add_argument('--seed_list', nargs="+", type=int, default=[3000], metavar='sl', help='a list of random seeds')
parser.add_argument('--layers', nargs="+", type=int, default=[3000], metavar='sl', help='a list of random seeds')
parser.add_argument('--max_epoch', type=int, default=30, metavar='sl', help='a list of random seeds')
args = parser.parse_args()
print(args)


logger = CSVLogger("logs", name="vanilla")

value=[]
seed_l = []
dataset_name = []
crps = []
dataset = get_dataset(args.dataset)
params = []
dim = 3
prediction_length = 24
freq = dataset.metadata.freq
seed_list = args.seed_list # [0]#, 42, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
layers = args.layers #[2]#, 4, 8, 12, 16, 20, 24, 28]
max_epoch = args.max_epoch #30

############# WanDB ###########

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project='ts_scaling') #name='Adam-32-0.001',
wandb_logger.experiment.config.update({
        "dataset": args.dataset,
        "seed_list": args.seed_list,
        "layers": args.layers,
        # "learning_rate": 0.02,
        # "architecture": "CNN",
        # "dataset": "CIFAR-100",
        "epochs": args.max_epoch,
        })
import wandb
import random

############# WanDB End ###########

for layer in layers:

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
            trainer_kwargs=dict(max_epochs=max_epoch, accelerator='auto', gpus=1, logger= wandb_logger)) #logger=logger))#
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
        wandb.log({"agg_metrics": agg_metrics })
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

print(df)
# log metrics to wandb
wandb.log({"df": df })
wandb.finish()



# These steps are required befor running this file 
# First install Wandb 
# pip install wandb 
# Second login to your account
# wandb login