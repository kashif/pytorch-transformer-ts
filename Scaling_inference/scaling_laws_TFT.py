import sys
import torch
sys.path.append('your_path')

from pytorch_lightning.utilities.model_summary import summarize
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.common import ListDataset
from Scaling_inference.estimator import TFTEstimator
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
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
import os
CHECKPOINT_PATH  ='you_path'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




logger = CSVLogger("logs", name="Temporal_Fusion")
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

CHECKPOINT_PATH = 'your_path'
def train_scaling(**model_parameters):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "scaling_TFT")
    os.makedirs(root_dir, exist_ok=True)
    estimator = TFTEstimator(**model_parameters, trainer_kwargs = dict(
                                                default_root_dir=root_dir,
                                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", 
                                                            monitor="val_acc")],
                                                accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                                                devices=1,
                                                max_epochs=100,
                                                gradient_clip_val=5, logger = logger),
                            )       

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "scaling_TFT.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = estimator.load_from_checkpoint(pretrained_filename)
    else:
        #model = TFTEstimator(max_iters=trainer.max_epochs*data_loader_train.length, **model_parameters)
        estimator = estimator.train(training_data=dataset.train,
                                        #num_workers=os.cpu_count(),
                                shuffle_buffer_length=1024)

    # Test best model on validation and test set
    forecast_it, ts_it = make_evaluation_predictions(
                            dataset=dataset.test, 
                            predictor=estimator
    )
    
    params = []    
    crps = []                          
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
    
    
    result = {"forecast_it": forecast_it, "ts_it": ts_it}

    
    return estimator, params, crps, result           


for layer in [2, 4, 8, 12, 16, 20, 24, 28]:

    value = []
    for seed in seed_list:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        estimator, params, crps, results = train_scaling(
                    freq = dataset.metadata.freq,
                    prediction_length = dataset.metadata.prediction_length,
                    num_feat_static_cat = 1,
                    cardinality  = [862],                                                    
                    scaling = False,
                    batch_size = 128,
                    num_batches_per_epoch = 5                                        
                                    )       
    
    print(layer)
    crps.append(np.mean(value))
    params.append(summarize(estimator.create_lightning_module()).trainable_parameters)

df = pd.DataFrame()
df['params'] = params
df['crps'] = crps
df.to_csv('scaling_elec.csv', index = False)


