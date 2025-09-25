#!/bin/bash

DATASET="llm_automated_evaluation_dataset"
DATASET_PATH="/mnt/sohn2022/Adrian/rad-llm-pmhx/dataset/${DATASET}.parquet"

OUTPUT_DIR="/mnt/sohn2022/Adrian/rad-llm-pmhx/experiments/results/outputs/${DATASET}"
NUM_SAMPLES=10
EXPERIMENT="one_shot_augmented"

MODELS=(
    # Anthropic 
    "claude3_5"
    
    # OpenAI
    "gpt4o"
    "gpt4o_mini"

    # OpenAI (Reasoning)
    "o1"
    "o1_mini"

    # HuggingFace
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "BioMistral/BioMistral-7B"
    "OpenMeditron/Meditron3-8B"
    "Qwen/Qwen2.5-7B-Instruct"

    # HuggingFace (Reasoning)
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

for MODEL in "${MODELS[@]}"; do
    python run.py --model $MODEL --experiment $EXPERIMENT --output-dir $OUTPUT_DIR --dataset $DATASET_PATH --num-samples $NUM_SAMPLES
done
