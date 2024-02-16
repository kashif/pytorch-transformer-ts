#!/bin/bash
#SBATCH --array=12
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32000M
#SBATCH --time=08:00:00
#SBATCH --job-name=sl_ts
#SBATCH --wait

module load anaconda/3

conda activate hface

python scaling_ts.py --dataset solar-energy --seed_list 2023 2022 --layers 32 --max_epoch 30