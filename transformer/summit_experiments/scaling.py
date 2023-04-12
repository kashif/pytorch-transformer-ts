from pytorch_lightning.utilities.model_summary import summarize
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.common import ListDataset
from estimator import TransformerEstimator
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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

#logger = CSVLogger("logs", name="vanilla")
checkpoint_callback = ModelCheckpoint(filename='epoch{epoch:02d}_step{step:02d}',
          verbose=True,
          save_top_k=1,auto_insert_metric_name=False)
#checkpoint_callback.CHECKPOINT_JOIN_CHAR = "{epoch}_{step}"
# seed_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
# seed_list = [0, 42, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
value=[]
seed_l = []
dataset_name = []
crps = []
dataset = get_dataset("electricity")
params = []
dim = 3
prediction_length = 24
freq = dataset.metadata.freq
seed_list = [0]#, 42, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
for seed in seed_list:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    all_value = []
    value = []
    for layer in [4096]:#, 4, 8, 12, 16, 20, 24, 28, 32, 46, 64, 74]:
        # st = os.stat("checkpoints/")
        # os.chmod("checkpoints/", st.st_mode | 0o222)
        estimator = TransformerEstimator(
            freq=dataset.metadata.freq,
            prediction_length=24,
            context_length=56,
            nhead=2,
            num_encoder_layers=layer,
            num_decoder_layers=2,
            dim_feedforward=16,
            activation="gelu",

            num_feat_static_cat=1,
            cardinality=[int(dataset.metadata.feat_static_cat[0].cardinality)],
            embedding_dimension=[dim],

            batch_size=2,
            num_batches_per_epoch=10000,
            trainer_kwargs=dict(max_epochs=1, accelerator='gpu', gpus=6, logger=False, gradient_clip_val=5, strategy='ddp',num_nodes=20, callbacks=[checkpoint_callback]))#
        predictor = estimator.train(training_data=dataset.train,
         # validation_data=val_ds,
         #num_workers=0,
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
        print(layer)
        print(seed)
        all_value.append(agg_metrics)
        params.append(summarize(estimator.create_lightning_module()).trainable_parameters)

    df = pd.DataFrame()
    df['params'] = params
    df['metrics'] = all_value
    df['crps'] = value
    df.to_csv('scaling_elec_4096_'+str(seed)+'.csv', index = False)
    print(layer)
    crps.append(np.mean(value))

# df = pd.DataFrame()
# df['params'] = params
# df['crps'] = crps
# df.to_csv('scaling_elec.csv', index = False)

