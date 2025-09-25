import pandas as pd
import os
import evaluate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tqdm
import argparse
import subprocess
from radgraph import F1RadGraph
import torch
import torch.nn as nn
from bert_score import BERTScorer

BASEPATH = "/mnt/sohn2022/Adrian/rad-llm-pmhx/experiments/results/llm_automated_evaluation_dataset"

MODEL_DICT = {
    "mistral": "mistralai_Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama_Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen_Qwen2.5-7B-Instruct",
    "deepseek_llama": "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "deepseek_qwen": "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
    "biomistral": "BioMistral_BioMistral-7B",
    "meditron": "OpenMeditron_Meditron3-8B",
    "gpt4o": "gpt4o",
    "gpt4o_mini": "gpt4o_mini",
    "o1": "o1",
    "o1_mini": "o1_mini",
    "claude3_5": "claude3_5",
    "referring_physician": "gpt4o"
}

class BertScore(nn.Module):
    def __init__(self):
        super(BertScore, self).__init__()
        with torch.no_grad():
            self.bert_scorer = BERTScorer(
                model_type="distilbert-base-uncased",
                num_layers=4,
                batch_size=8,
                nthreads=8,
                all_layers=False,
                idf=False,
                lang="en",
                rescale_with_baseline=True,
                baseline_path=None,
            )

    def forward(self, refs, hyps):
        p, r, f = self.bert_scorer.score(
            cands=hyps,
            refs=refs,
            verbose=False,
            batch_size=8,
        )
        return f.tolist()
    
class AlignScorer(nn.Module):
    def __init__(self):
        super(AlignScorer, self).__init__()
        self.align_scorer = AlignScore(
            model='roberta-base', 
            device='cuda:0',
            batch_size=32, 
            ckpt_path='AlignScore-base.ckpt', 
            evaluation_mode='nli_sp')

    def forward(self, refs, hyps):
        f = self.align_scorer.score(
            contexts=refs,
            claims=hyps,
        )
        return f

def compute_rouge_scores(data, pred_col="llm_indication", gt_col="additional_history", variant="rougeL"):
    rouge = evaluate.load('rouge')
    predictions = data[pred_col].tolist()
    references = data[gt_col].tolist()
    individual_scores = []
    for i in tqdm.tqdm(range(len(predictions)), position=0, leave=True):
        rouge_score = rouge.compute(predictions=[predictions[i]], references=[references[i]])
        individual_scores.append(rouge_score[variant])
    return np.array(individual_scores)

def main(args):
    save_dir_basepath = '/mnt/sohn2022/Adrian/rad-llm-pmhx/analysis/llm_evaluation_scores'
    if args.model not in MODEL_DICT:
        assert "Model is not registered. Please use a registered model."
    MODEL_NAME = MODEL_DICT[args.model]

    data = pd.read_csv(f"{BASEPATH}/{MODEL_NAME}_0_1000.csv").fillna("")
    
    # ROUGE-L
    if args.model == "referring_physician":
        rouge_scores = compute_rouge_scores(data, pred_col="original_history")
    else:
        rouge_scores = compute_rouge_scores(data)
    
    # MEDCON
    medcon_csv_path = f'{save_dir_basepath}/{args.model}_medcon_temp.csv'
    medcon_output_path = f'{save_dir_basepath}/{args.model}_medcon_scores.csv'
    if args.model == "referring_physician":
        medcon_csv = data[["additional_history", "original_history"]]
        medcon_csv = medcon_csv.rename(
            columns={
                "original_history": "generated",
                "additional_history": "reference"
            }
        )
    else:
        medcon_csv = data[["additional_history", "llm_indication"]]
        medcon_csv = medcon_csv.rename(
            columns={
                "llm_indication": "generated",
                "additional_history": "reference",
            }
        )
    medcon_csv.to_csv(medcon_csv_path, index=False)
    python_path = "/home/developer/anaconda3/bin/python"
    subprocess.run([
        python_path, '/mnt/sohn2022/Adrian/Utils/Evaluation/MEDCON/main.py',
        f"--csv_path={medcon_csv_path}",
        f"--output_path={medcon_output_path}",
    ])
    medcon_scores = np.array(pd.read_csv(medcon_output_path)["MEDCON_Score"].values)
    
    # RadGraph-F1
    if args.model == "referring_physician":
        f1radgraph = F1RadGraph(reward_level="all")
        _, reward_list, _, _ = f1radgraph(
            hyps=data["original_history"].tolist(), 
            refs=data["additional_history"].tolist()
        )
    else:
        f1radgraph = F1RadGraph(reward_level="all")
        _, reward_list, _, _ = f1radgraph(
            hyps=data["llm_indication"].tolist(), 
            refs=data["additional_history"].tolist()
        )
    radgraph_scores = np.array(reward_list[-1])

    # BERTScore
    if args.model == "referring_physician":
        bert_scores = BertScore()(
            hyps=data["original_history"].tolist(),
            refs=data["additional_history"].tolist()
        )
    else:
        bert_scores = BertScore()(
            hyps=data["llm_indication"].tolist(),
            refs=data["additional_history"].tolist()
        )

    model_scores = pd.DataFrame({
        "rouge": rouge_scores,
        "medcon": medcon_scores,
        "radgraph": radgraph_scores,
        "bertscore": bert_scores,
    })
    data = pd.concat([data, model_scores], axis=1)
    data.to_csv(f"llm_evaluation_scores/{args.model}.csv", index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for model and metric selection")
    parser.add_argument("--model", type=str, required=True, help="Specify the model name or path")
    args = parser.parse_args()
    main(args)