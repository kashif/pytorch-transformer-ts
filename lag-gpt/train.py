import argparse
import json
import random
import numpy as np
import torch
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.evaluation._base import aggregate_valid
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.transform import ValidationSplitSampler

from estimator import LagGPTEstimator

TRAIN_DATASET_NAMES = [
    "airpassengers",
    "australian_electricity_demand",
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "electricity",
    "electricity_weekly",
    "exchange_rate",
    "fred_md",
    "hospital",
    "kaggle_web_traffic_weekly",
    "kdd_cup_2018_without_missing",
    "london_smart_meters_without_missing",
    "nn5_daily_with_missing",
    "nn5_weekly",
    "pedestrian_counts",
    "rideshare_without_missing",
    "saugeenday",
    "solar-energy",
    "solar_10_minutes",
    "solar_weekly",
    "taxi_30min",
    "temperature_rain_without_missing",
    "tourism_monthly",
    "uber_tlc_daily",
    "uber_tlc_hourly",
    "vehicle_trips_without_missing",
    "weather",
    "wiki-rolling_nips",
    "m4_daily",
    "m4_hourly",
    "m4_monthly",
    "m4_quarterly",
    "m4_yearly",
    "wind_farms_without_missing",
]

class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)

class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])

def create_sliding_window_dataset(name, window_size, is_train=True):
    # Splits each time series into non-overlapping sliding windows
    global_id = 0
    data = ListDataset([], freq=freq)

    freq = get_dataset(name).metadata.freq
    dataset = get_dataset(name).train if is_train else get_dataset(name).test

    for x in dataset:
        windows = []
        for i in range(0, len(x['target']), window_size):
            windows.append({
                'target': x['target'][i:i+window_size],
                'start': x['start'] + i,
                'item_id': str(global_id),
                'feat_static_cat': np.array([0]),
            })
            global_id += 1
        data += ListDataset(windows, freq=freq)
    return data

def create_test_dataset(name, window_size):
    # Similar to `create_sliding_window_dataset` but for test dataset
    dataset = get_dataset(name)
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length

    data = []
    for x in dataset.test:
        offset = len(x['target']) - window_size - prediction_length
        if offset > 0:
            target = x['target'][-(window_size + prediction_length):]
            data.append({
                'target': target,
                'start': x['start'] + offset,
            })
        else:
            data.append(x)
    return ListDataset(data, freq=freq), prediction_length

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = CSVLogger(
        save_dir="data-scaling-logs",
        name=f'context-{args.context_length}-layer-{args.n_layer}-seed-{args.seed}-aug-{args.aug_prob}-{args.aug_rate}',
        flush_logs_every_n_steps=1,
        version=0,
    )

    if Path(logger.log_dir).exists():
        ckpt_path = str(list((Path(logger.log_dir) / 'checkpoints').glob('*.ckpt'))[-1])
        print(f'Loading from checkpoing: {ckpt_path}')
    else:
        ckpt_path = None

    estimator = LagGPTEstimator(
        prediction_length=1,
        context_length=args.context_length,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        scaling="std",
        aug_prob=args.aug_prob,
        aug_rate=args.aug_rate,
        num_batches_per_epoch=args.num_batches_per_epoch,
        ckpt_path=ckpt_path,
        trainer_kwargs=dict(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=[args.gpu],
            limit_val_batches=args.limit_val_batches,
            logger=logger,
        ),
    )
    window_size = estimator.context_length + max(estimator.lags_seq) + estimator.prediction_length
    # Here we make a window slightly bigger so that instance sampler can sample from each window
    # An alternative is to have exact size and use different instance sampler (e.g. ValidationSplitSampler)
    window_size = 10 * window_size

    # We change ValidationSplitSampler to add min_past
    estimator.validation_sampler = ValidationSplitSampler(
        min_past=estimator.context_length + max(estimator.lags_seq),
        min_future=estimator.prediction_length,
    )

    if args.test:
        print('Testing only')
    else:
        # Create training data
        train_data, val_data = [], []
        for name in TRAIN_DATASET_NAMES:
            new_data = create_sliding_window_dataset(name, window_size)
            train_data.append(new_data)

            new_data = create_sliding_window_dataset(name, window_size, is_train=False)
            val_data.append(new_data)
        # Here weights are proportional to the number of time series (=sliding windows)
        weights = [len(x) for x in train_data]
        # Here weights are proportinal to the number of individual points in all time series
        # weights = [sum([len(x["target"]) for x in d]) for d in train_data]

        train_data = CombinedDataset(train_data, weights=weights)
        val_data = CombinedDataset(val_data, weights=weights)

        # Train
        # TODO: Depending on the stopping criterion, saved checkpoint will be based on validation
        # and the test set for these datasets will be the same (doesn't impact zero-shot experiment)
        train_output = estimator.train_model(
            training_data=train_data,
            validation_data=val_data,
            shuffle_buffer_length=2048,
        )

    # Evaluate
    if args.test:
        estimator.ckpt_path = ckpt_path
    else:
        estimator.ckpt_path = train_output.trainer.checkpokint_callback.best_model_path
    print(f'Use checkpoint: {estimator.ckpt_path}')

    for name in ['m4_weekly', 'traffic'] + TRAIN_DATASET_NAMES:
        print(f'Predict on {name}')
        test_data, prediction_length = create_test_dataset(name, window_size)

        # Adapt evaluator to new dataset
        estimator.prediction_length = prediction_length
        estimator.batch_size = max(30 // estimator.prediction_length, 1) # Some heuristic for GPU memory (TODO: change)
        predictor = estimator.create_predictor(
            estimator.create_transformation(),
            estimator.create_lightning_module(),
        )
        # Make evaluations
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,
            predictor=predictor,
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        evaluator = Evaluator(num_workers=1, aggregation_strategy=aggregate_valid)
        agg_metrics, _ = evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )

        with open(f'{logger.log_dir}/{name}.json', 'w') as f:
            json.dump(agg_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--limit_val_batches", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--test', action='store_true')
    # Model
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--aug_rate", type=float, default=0.1)
    parser.add_argument("--n_head", type=int, default=4)
    args = parser.parse_args()

    train(args)
