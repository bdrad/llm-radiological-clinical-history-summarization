"""
Simple unified runner for LLM indication revision experiments.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt import generate_prompt
from config import MODEL_CONFIGS, EXPERIMENT_CONFIGS
from providers.factory import ProviderFactory

def load_sample_dataset():
    """Simple test dataset."""
    return [
        {
            "patientdurablekey": "sample_001",
            "exam_type": "CT chest with contrast",
            "original_history": "Shortness of breath",
            "clinical_notes": [
                "Patient is a 65-year-old male with history of smoking presenting with dyspnea.",
                "Recent weight loss of 10 pounds over 2 months."
            ]
        },
        {
            "patientdurablekey": "sample_002", 
            "exam_type": "MRI brain without contrast",
            "original_history": "Headache",
            "clinical_notes": [
                "Patient is a 45-year-old female with persistent headaches for 3 weeks.",
                "No focal neurological deficits on examination."
            ]
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Simple LLM indication revision runner")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                       help="Model to use")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()),
                       help="Experiment configuration")
    parser.add_argument("--dataset", default=None,
                       help="Path to dataset file (CSV or Parquet)")
    parser.add_argument("--output-dir", default="/mnt/sohn2022/Adrian/rad-llm-pmhx/experiments/results",
                       help="Output directory")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Maximum samples to process")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available configurations")
    
    args = parser.parse_args()

    start_time = datetime.now()
    
    if args.list_configs:
        print("Available Models:")
        for model_key, config in MODEL_CONFIGS.items():
            print(f"  {model_key}: {config['provider']} - {config['model_name']}")
        print("\nAvailable Experiments:")
        for exp_key, config in EXPERIMENT_CONFIGS.items():
            print(f"  {exp_key}: {config['description']}")
        return
    
    # Load environment
    env_path = "/mnt/sohn2022/Adrian/Utils/Credentials/.env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
    
    print(f"Running: {args.model} + {args.experiment}")
    
    # Get configurations
    model_config = MODEL_CONFIGS[args.model]
    experiment_config = EXPERIMENT_CONFIGS[args.experiment]
    
    # Create provider
    text_generator = ProviderFactory.create_provider(
        model_config["provider"],
        model_config["model_name"], 
        model_config["max_tokens"]
    )
    
    # Load dataset
    if args.dataset and os.path.exists(args.dataset):
        try:
            if args.dataset.endswith('.csv'):
                df = pd.read_csv(args.dataset)
            elif args.dataset.endswith('.parquet'):
                df = pd.read_parquet(args.dataset)
            dataset = df.to_dict('records')
            print(f"Loaded {len(dataset)} samples from {args.dataset}")
        except Exception as e:
            print(f"Error loading {args.dataset}: {e}")
            dataset = load_sample_dataset()
            print("Using sample dataset")
    else:
        dataset = load_sample_dataset()
        print("Using sample dataset")
    
    if args.num_samples:
        dataset = dataset[:args.num_samples]
        print(f"Limited to {len(dataset)} samples")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / args.model / args.experiment
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    results = []
    for i, data_point in enumerate(tqdm(dataset, desc=f"{args.model}_{args.experiment}")):
        try:
            # Extract fields
            exam_type = data_point.get("exam_type", "")
            original_indication = data_point.get("original_history", "")
            clinical_notes = data_point.get("note_texts", [])
            
            if isinstance(clinical_notes, str):
                clinical_notes = [clinical_notes]
            
            # Generate prompt
            prompt = generate_prompt(
                exam_type=exam_type,
                original_indication=original_indication,
                clinical_notes=clinical_notes,
                prompting_strategy=experiment_config["prompting_strategy"],
                prompt_type=experiment_config["prompt_type"]
            )            
            temperature = experiment_config.get("temperature")
            
            # Generate response
            response = text_generator.generate(prompt, temperature=temperature)
            
            # Store result
            result = {
                "model": args.model,
                "experiment": args.experiment,
                "exam_type": exam_type,
                "original_history": original_indication,
                "llm_indication": response,
                "temperature": temperature,
                **{k: v for k, v in data_point.items() if k not in ["exam_type", "original_history", "note_texts", "enc_dept_names", "note_types", "auth_prov_types", "deid_service_dates"]}
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append({
                "model": args.model,
                "experiment": args.experiment,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    end_time = datetime.now()
    time_elapsed_sec = (end_time - start_time).total_seconds()
    time_elapsed_min = time_elapsed_sec / 60

    # Save results
    output_file = output_path / f"results_{args.num_samples}.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Save config
    config_file = output_path / f"config_{args.num_samples}.json"
    with open(config_file, 'w') as f:
        json.dump({
            "model_config": model_config,
            "experiment_config": experiment_config,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "time_elapsed_min": round(time_elapsed_min, 2),  
            "total_samples": len(results)
        }, f, indent=2)
    
    print(f"Completed: {len(results)} samples processed")

if __name__ == "__main__":
    main()
