#!/bin/bash

MODEL="gpt4o"
DATASET="llm_automated_evaluation_dataset"
OUTPUT_DIR="/mnt/sohn2022/Adrian/rad-llm-pmhx/experiments/results/scores/${DATASET}/${MODEL}"

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
    INPUT_PATH="/mnt/sohn2022/Adrian/rad-llm-pmhx/experiments/results/outputs/${MODEL}/${EXPERIMENT}/results_${NUM_SAMPLES}.csv"
    python evaluate_llm_outputs.py --model $MODEL --input_path $INPUT_PATH --output_path $OUTPUT_DIR
done
