#!/bin/bash

DATASET="llm_automated_evaluation_dataset"
DATASET_PATH="/mnt/sohn2022/Adrian/rad-llm-pmhx/dataset/${DATASET}.parquet"

OUTPUT_DIR="/mnt/sohn2022/Adrian/rad-llm-pmhx/experiments/results/outputs/${DATASET}"
NUM_SAMPLES=10
MODEL="gpt4o"

EXPERIMENTS=(
    "zero_shot_standard"
    "zero_shot_augmented"
    "one_shot_standard"
    "one_shot_augmented"
    "temp_0.0"
    "temp_0.5"
    "temp_1.0"
)

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    python run.py --model $MODEL --experiment $EXPERIMENT --output-dir $OUTPUT_DIR --dataset $DATASET_PATH --num-samples $NUM_SAMPLES
done
