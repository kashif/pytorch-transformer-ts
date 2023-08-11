import argparse
import yaml
from pathlib import Path

import matplotlib.pyplot as plt

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from estimator import LagGPTEstimator


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="YAML config file.")
args = parser.parse_args()


with open(args.filename, mode="rt", encoding="utf-8") as file:
    config = yaml.safe_load(file)


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


print("Loading data...")
dataset_path = Path("../datasets")
gluonts_ds = [
    get_dataset("airpassengers", path=dataset_path).train,
    get_dataset("australian_electricity_demand", path=dataset_path).train,
    get_dataset("car_parts_without_missing", path=dataset_path).train,
    get_dataset("cif_2016", path=dataset_path).train,
    get_dataset("covid_deaths", path=dataset_path).train,
    get_dataset("electricity", path=dataset_path).train,
    get_dataset("electricity_weekly", path=dataset_path).train,
    get_dataset("exchange_rate", path=dataset_path).train,
    get_dataset("fred_md", path=dataset_path).train,
    get_dataset("hospital", path=dataset_path).train,
    get_dataset("kaggle_web_traffic_weekly", path=dataset_path).train,
    get_dataset("kdd_cup_2018_without_missing", path=dataset_path).train,
    get_dataset("london_smart_meters_without_missing", path=dataset_path).train,
    get_dataset("nn5_daily_with_missing", path=dataset_path).train,
    get_dataset("nn5_weekly", path=dataset_path).train,
    get_dataset("pedestrian_counts", path=dataset_path).train,
    get_dataset("rideshare_without_missing", path=dataset_path).train,
    get_dataset("saugeenday", path=dataset_path).train,
    get_dataset("solar-energy", path=dataset_path).train,
    get_dataset("solar_10_minutes", path=dataset_path).train,
    get_dataset("solar_weekly", path=dataset_path).train,
    get_dataset("taxi_30min", path=dataset_path).train,
    get_dataset("temperature_rain_without_missing", path=dataset_path).train,
    get_dataset("tourism_monthly", path=dataset_path).train,
    get_dataset("uber_tlc_daily", path=dataset_path).train,
    get_dataset("uber_tlc_hourly", path=dataset_path).train,
    get_dataset("vehicle_trips_without_missing", path=dataset_path).train,
    get_dataset("weather", path=dataset_path).train,
    get_dataset("wiki-rolling_nips", path=dataset_path).train,
    get_dataset("m4_daily", path=dataset_path).train,
    get_dataset("m4_hourly", path=dataset_path).train,
    get_dataset("m4_monthly", path=dataset_path).train,
    get_dataset("m4_quarterly", path=dataset_path).train,
    get_dataset("m4_yearly", path=dataset_path).train,
    get_dataset("wind_farms_without_missing", path=dataset_path).train,
]
dataset = CombinedDataset(
    gluonts_ds,
    weights=(
        [sum([len(x["target"]) for x in d]) for d in gluonts_ds]
        if config["dataset"]["weighted"]
        else None
    ),
)

experiment_name = (
    "context-window-scaling-rope"
    if config["gpt"]["rope_scaling"]
    else "context-window-scaling"
)
experiment_logger = (
    CSVLogger(save_dir="context-window-scaling-logs", name=experiment_name)
    if config["metrics"]["logger"] == "csv"
    else WandbLogger(save_dir="context-window-scaling-logs", name=experiment_name)
)
experiment_version = experiment_logger.version

print("Running " + experiment_name + " version " + str(experiment_version))

estimator = LagGPTEstimator(
    prediction_length=config["gpt"]["prediction_length"],
    context_length=config["gpt"]["context_length"],  # block_size: int = 2048
    rope_scaling=(
        dict(type="linear", factor=2.0) if gpt["gpt"]["rope_scaling"] else None
    ),
    batch_size=config["gpt"]["batch_size"],  # 4
    n_layer=config["gpt"]["n_layer"],
    # n_head=config["gpt"]["n_head"],
    # n_embd=config["gpt"]["n_embd"], # 4096
    scaling=config["gpt"]["scaling"],
    # aug_prob = config["gpt"]["aug_prob"],
    # aug_rate = config["gpt"]["aug_rate"],
    num_batches_per_epoch=config["gpt"]["batches_per_epoch"],
    trainer_kwargs=dict(
        max_epochs=config["gpt"]["max_epochs"],
        accelerator="gpu",
        precision="32",
        logger=experiment_logger,
        devices=[config["CUDA"]["device_id"]],
    ),
)


predictor = estimator.train(
    training_data=dataset, shuffle_buffer_length=1024, cache_data=True
)

test_dataset = get_dataset(config["dataset"]["test"], path=dataset_path).test
meta = get_dataset(config["dataset"]["test"], path=dataset_path).metadata
test_logger = CSVLogger(save_dir="context-window-scaling-logs", name="test")


forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor,
)
forecasts = list(forecast_it)
tss = list(ts_it)
evaluator = Evaluator()

crpss = []
crpss_rope = []
for i in range(25):
    estimator = LagGPTEstimator(
        prediction_length=meta.prediction_length,
        context_length=meta.prediction_length * (i + 1),
        scaling=config["gpt"]["scaling"],
        rope_scaling=None,
        n_layer=config["gpt"]["n_layer"],
        batch_size=32,
        num_batches_per_epoch=config["gpt"]["batches_per_epoch"],
        trainer_kwargs=dict(
            accelerator="gpu",
            precision="32",
            max_epochs=0,
            logger=test_logger,
            devices=[config["CUDA"]["device_id"]],
        ),
        ckpt_path="context-window-scaling-logs/"
        + experiment_name
        + "version_"
        + str(experiment_version)
        + "/checkpoints/epoch=49-step=5000.ckpt",
    )

    predictor = estimator.train(
        training_data=dataset.train,
        shuffle_buffer_length=1024,
        cache_data=True,
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
    )

    forecasts = list(forecast_it)
    agg_metrics, _ = evaluator(iter(tss), iter(forecasts))
    crpss.append(agg_metrics["mean_wQuantileLoss"])

    estimator = LagGPTEstimator(
        prediction_length=meta.prediction_length,
        context_length=meta.prediction_length * (i + 1),
        scaling=config["gpt"]["scaling"],
        rope_scaling=dict(type="linear", factor=2.0),
        n_layer=config["gpt"]["n_layer"],
        batch_size=32,
        num_batches_per_epoch=config["gpt"]["batches_per_epoch"],
        trainer_kwargs=dict(
            accelerator="gpu",
            precision="32",
            max_epochs=0,
            logger=test_logger,
            devices=[config["CUDA"]["device_id"]],
        ),
        ckpt_path="context-window-scaling-logs/"
        + experiment_name
        + "version_"
        + str(experiment_version)
        + "/checkpoints/epoch=49-step=5000.ckpt",
    )

    predictor = estimator.train(
        training_data=dataset.train,
        shuffle_buffer_length=1024,
        cache_data=True,
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
    )

    forecasts = list(forecast_it)
    agg_metrics, _ = evaluator(iter(tss), iter(forecasts))
    crpss_rope.append(agg_metrics["mean_wQuantileLoss"])


fig, ax = plt.subplots()
ax.plot([(i + 1) * meta.prediction_length for i in range(len(crpss))], crpss)
ax.set_xlabel("Context Length")
ax.set_ylabel("CRPS")
fig.savefig(
    "context-window-scaling-logs/"
    + experiment_name
    + "/version_"
    + str(experiment_version)
    + "/CRPS.png"
)
plt.close(fig)

pd.DataFrame(crpss, columns=["CRPS"]).to_csv(
    "context-window-scaling-logs/"
    + experiment_name
    + "/version_"
    + str(experiment_version)
    + "/CRPS.csv"
)

fig, ax = plt.subplots()
ax.plot([(i + 1) * meta.prediction_length for i in range(len(crpss))], crpss)
ax.set_xlabel("Context Length")
ax.set_ylabel("CRPS")
fig.savefig(
    "context-window-scaling-logs/"
    + experiment_name
    + "/version_"
    + str(experiment_version)
    + "/CRPS_rope.png"
)
plt.close(fig)

pd.DataFrame(crpss, columns=["CRPS"]).to_csv(
    "context-window-scaling-logs/"
    + experiment_name
    + "/version_"
    + str(experiment_version)
    + "/CRPS_rope.csv"
)
