for SEED in 1 # 2 3 4 5
do
    NAME="default_config_run"
    python lag-gpt-flows/lag-gpt-flows-scaling-data.py \
    lag-gpt-flows/configs/config.yaml --suffix "${NAME}" --seed $SEED
done
