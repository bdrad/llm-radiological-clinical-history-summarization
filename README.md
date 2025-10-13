
# Radiologically Relevant Clinical History Summarization

This repository accompanies the manuscript 
<i>Radiologically Relevant Clinical History Summarization with Large Language Models: A Multi-Reader Performance Study</i>.

## Overview
This project explores the use of large language models (LLMs) to generate radiologically relevant clinical history summaries from clinical notes. It includes code for running LLM experiments, automated and reader-based evaluation, dataset processing, and a web-based evaluation platform.

## Key Components

- **Experiments (`experiments/`)**: Unified framework for running LLM-based clinical history summarization experiments. Includes prompt templates, configuration, and provider integrations (OpenAI, Anthropic, HuggingFace, etc.).
- **Analysis (`analysis/`)**: Automated benchmark evaluation metrics (Python), reader study statistical analysis (R), and manuscript figures/tables (Jupyter).
- **Dataset (`dataset/`)**: Scripts and notebooks for dataset demographics, inclusion/exclusion criteria, and pathophysiological labeling.
- **Evaluation Platform (`evaluation_platform/`)**: Next.js web application for conducting and visualizing reader studies.

## Installation

### Python (Experiments & Analysis)
Install dependencies:
```bash
pip install -r requirements.txt
```

### Node.js (Web Platform)
Navigate to `evaluation_platform/` and install dependencies:
```bash
cd evaluation_platform
npm install
```

## Usage

### Running Experiments and Evaluations (with scripts/)

The `experiments/scripts/` directory contains shell scripts to automate running LLM experiments and evaluations:

- **run_one_model_all_expt.sh**: Run all experiment types (prompting strategies, temperature settings, etc.) for a single model on a dataset.
- **run_all_models_one_expt.sh**: Run a single experiment type across all models.
- **run_ablations_gpt4o.sh**: Run ablation experiments (various prompting strategies and temperatures) for the GPT-4o model, then evaluate outputs.
- **metrics_one_model_all_expt.sh**: Compute evaluation metrics for all experiments of a single model.

#### Example: Run all experiments for one model
```bash
cd experiments/scripts
bash run_one_model_all_expt.sh
```

#### Example: Run one experiment for all models
```bash
cd experiments/scripts
bash run_all_models_one_expt.sh
```

#### Example: Run ablation and evaluation for GPT-4o
```bash
cd experiments/scripts
bash run_ablations_gpt4o.sh
```

#### Example: Compute metrics for all experiments of a model
```bash
cd experiments/scripts
bash metrics_one_model_all_expt.sh
```