#!/bin/bash
MODELS=(
  # "mistral"
  # "llama"
  # "qwen"
  "deepseek_llama"
  "deepseek_qwen"
  # "biomistral"
  # "meditron"
  # "gpt4o"
  # "gpt4o_mini"
  # "claude3_5"
  # "o1"
  # "o1_mini"
  # "referring_physician"
)

for model in "${MODELS[@]}"; do
    printf '=%.0s' {1..20} 
	echo
	echo "$model" 
	printf '=%.0s' {1..20} 
    echo
    python evaluate_llm_outputs.py --model "$model"
done