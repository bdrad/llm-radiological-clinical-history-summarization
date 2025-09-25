"""
Utility functions for data loading and result analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import os

def load_dataset_from_csv(file_path: str, required_columns: List[str] = None) -> List[Dict[str, Any]]:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        List of data dictionaries
    """
    if required_columns is None:
        required_columns = ["exam_type", "original_indication", "clinical_notes"]
    
    df = pd.read_csv(file_path)
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert to list of dictionaries
    data_list = []
    for _, row in df.iterrows():
        data_point = row.to_dict()
        
        # Handle clinical notes if stored as string
        if "clinical_notes" in data_point and isinstance(data_point["clinical_notes"], str):
            try:
                # Try to parse as JSON list
                data_point["clinical_notes"] = json.loads(data_point["clinical_notes"])
            except:
                # If not JSON, split by delimiter or treat as single note
                data_point["clinical_notes"] = [data_point["clinical_notes"]]
        
        data_list.append(data_point)
    
    return data_list

def aggregate_experiment_results(results_dir: str) -> pd.DataFrame:
    """
    Aggregate results from multiple experiment runs.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Aggregated DataFrame with all results
    """
    results_path = Path(results_dir)
    all_results = []
    
    # Find all CSV result files
    for csv_file in results_path.rglob("*.csv"):
        if "config" not in csv_file.name:  # Skip config files
            try:
                df = pd.read_csv(csv_file)
                
                # Add metadata from file path
                parts = csv_file.relative_to(results_path).parts
                if len(parts) >= 2:
                    df["model_from_path"] = parts[0]
                    df["experiment_from_path"] = parts[1]
                
                df["file_path"] = str(csv_file)
                all_results.append(df)
                
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    if not all_results:
        print("No valid result files found")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df

def generate_experiment_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for experiment results.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_experiments": len(results_df),
        "unique_models": results_df["model"].nunique() if "model" in results_df.columns else 0,
        "unique_experiments": results_df["experiment"].nunique() if "experiment" in results_df.columns else 0,
        "error_rate": (results_df["error"].notna().sum() / len(results_df)) if "error" in results_df.columns else 0,
    }
    
    # Group by model and experiment
    if "model" in results_df.columns and "experiment" in results_df.columns:
        groupby_stats = results_df.groupby(["model", "experiment"]).agg({
            "revised_indication": "count",
            "error": lambda x: x.notna().sum()
        }).rename(columns={"revised_indication": "total_runs", "error": "error_count"})
        
        summary["by_model_experiment"] = groupby_stats.to_dict("index")
    
    return summary

def save_experiment_metadata(output_dir: str, metadata: Dict[str, Any]):
    """Save experiment metadata to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata_file = output_path / "experiment_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Metadata saved to: {metadata_file}")

def create_experiment_report(results_dir: str, output_file: str = None):
    """
    Create comprehensive experiment report.
    
    Args:
        results_dir: Directory containing experiment results
        output_file: Output file path (default: results_dir/experiment_report.json)
    """
    if output_file is None:
        output_file = Path(results_dir) / "experiment_report.json"
    
    # Load and aggregate results
    results_df = aggregate_experiment_results(results_dir)
    
    if results_df.empty:
        print("No results found to generate report")
        return
    
    # Generate summary
    summary = generate_experiment_summary(results_df)
    
    # Create report
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "results_directory": results_dir,
        "summary": summary,
        "sample_results": results_df.head(5).to_dict("records") if len(results_df) > 0 else []
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Experiment report saved to: {output_file}")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Error rate: {summary['error_rate']:.2%}")

class DatasetLoader:
    """Helper class for loading different dataset formats."""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path
    
    def load_polars_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset using polars (if available)."""
        try:
            import polars as pl
            df = pl.read_csv(file_path)
            return df.to_pandas().to_dict("records")
        except ImportError:
            print("Polars not available, falling back to pandas")
            return load_dataset_from_csv(file_path)
    
    def load_reader_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Load reader evaluation dataset (implement based on your structure)."""
        # Placeholder - implement based on your actual dataset structure
        raise NotImplementedError("Implement based on your reader evaluation dataset structure")
    
    def load_automated_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Load automated evaluation dataset (implement based on your structure)."""
        # Placeholder - implement based on your actual dataset structure
        raise NotImplementedError("Implement based on your automated evaluation dataset structure")
