#!/bin/bash

EXPERIMENTS=(
    "zero_shot_standard"
    "zero_shot_augmented"
    "one_shot_standard"
    "one_shot_augmented"
    "temp_0.0"
    "temp_0.5"
    "temp_1.0"
)

DATASET="llm_development_dataset"
DATASET_PATH="/mnt/sohn2022/Adrian/rad-llm-pmhx/dataset/${DATASET}.parquet"
NUM_SAMPLES=100
MODEL="gpt4o"

MODEL_OUTPUT_DIR="/mnt/sohn2022/Adrian/rad-llm-pmhx/experiments/results/outputs/${DATASET}"
SCORES_OUTPUT_DIR="/mnt/sohn2022/Adrian/rad-llm-pmhx/experiments/results/scores/${DATASET}/${MODEL}"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    python run.py --model $MODEL --experiment $EXPERIMENT --output-dir $MODEL_OUTPUT_DIR --dataset $DATASET_PATH --num-samples $NUM_SAMPLES
done

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    INPUT_PATH="${MODEL_OUTPUT_DIR}/${MODEL}/${EXPERIMENT}/results_${NUM_SAMPLES}.csv"
    python evaluate_llm_outputs.py --model $MODEL --input_path $INPUT_PATH --output_path $SCORES_OUTPUT_DIR
done
