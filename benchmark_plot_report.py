import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from utils import get_simple_name, ljust_labels


BENCHMARK_DIR = "benchmark_result"
if not os.path.exists(BENCHMARK_DIR):
    print(f"Benchmark directory {BENCHMARK_DIR} does not exist.")
    exit(1)


# compare data gen
MODELS_TO_INCLUDE_DATA = [
    "Qwen2.5-0.5B-Instruct-curriculum_12903news_4stage_A100",  # ... v1
    "Qwen2.5-0.5B-Instruct-curriculum_12903news_5stage_A100",  # ... v1

    "Qwen2.5-0.5B-Instruct-curriculum_11114news_1stage_A100_better2",  # ... v2
    "Qwen2.5-0.5B-Instruct-curriculum_11114news_4stage_A100_better2",  # ... v2
    "Qwen2.5-0.5B-Instruct-curriculum_11114news_5stage_A100_better2",  # ... v2

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3",  # ... v3
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3",  # ... v3

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3-lr_adj",  # ... v4
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3-lr_adj",  # ... v4
    "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v3-lr_adj",  # ... v4

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v4-lr_adj",  # ... v4
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v4-lr_adj",  # ... v4
    "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v4-lr_adj",  # ... v4
]


# compare training strategy
MODELS_TO_INCLUDE_STRATEGY = [
    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3",
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3",

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3-lr_adj",
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3-lr_adj",

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3-lr_adj-only_attn",
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3-lr_adj-only_attn",
    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3-lr_adj-only_mlp",
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3-lr_adj-only_mlp",

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v4-lr_adj",
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v4-lr_adj",
    "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v4-lr_adj",

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3-lr_adj_lora",
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3-lr_adj_lora",
    "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v3-lr_adj_lora",
]


# # compare overall
MODELS_TO_INCLUDE_OVERVALL = [
    "google_gemma-2-2b-it",
    "google_gemma-3-1b-it",
    "meta-llama_Llama-3.2-1B-Instruct",
    "meta-llama_Llama-3.2-3B-Instruct",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-32B-Instruct",
    "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v4-lr_adj",
]


# compare custom models
MODELS_TO_INCLUDE_CUSTOM = [
    "CustomQwen2Model_pretrained-cl_12952news_1stg_v3-lr_adj",
    "CustomQwen2Model_pretrained-cl_12952news_4stg_v3-lr_adj",
    "CustomQwen2Model-cl_5000news_1stg_v3-lr_adj",
    "CustomQwen2Model-cl_5000news_4stg_v3-lr_adj",
]


# compare stages
MODELS_TO_INCLUDE_STAGES = [
    # "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3-lr_adj",
    # "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3-lr_adj",
    # "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v3-lr_adj",

    # "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v4-lr_adj",
    # "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v4-lr_adj",
    # "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v4-lr_adj",

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3",
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3",
]


class AnalysisItem:
    def __init__(self, item: str, fields: list[str]):
        self.item = item
        self.fields = fields
        self.results: list[BenchmarkResult] = []

        match item:
            case "compare_data":
                self.models = MODELS_TO_INCLUDE_DATA
            case "compare_strategy":
                self.models = MODELS_TO_INCLUDE_STRATEGY
            case "compare_overall":
                self.models = MODELS_TO_INCLUDE_OVERVALL
            case "compare_custom_models":
                self.models = MODELS_TO_INCLUDE_CUSTOM
            case "compare_stages":
                self.models = MODELS_TO_INCLUDE_STAGES
            case _:
                raise ValueError(f"Unknown analysis item: {item}")

        self.plot_dir = os.path.join(BENCHMARK_DIR, self.item)

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)


@dataclass
class BenchmarkResult:
    modelname: str
    num_samples: int
    judge_model: str
    bert_precision: float
    bert_recall: float
    bert_f1: float
    judge_score: float

    rouge1_f: np.ndarray
    rouge1_p: np.ndarray
    rouge1_r: np.ndarray
    rougeL_f: np.ndarray
    rougeL_p: np.ndarray
    rougeL_r: np.ndarray
    rouge2_f: np.ndarray
    rouge2_p: np.ndarray
    rouge2_r: np.ndarray
    judge_scores: np.ndarray

    bert_precision_std: float
    bert_recall_std: float
    bert_f1_std: float
    rouge1_f_std: float
    rouge1_p_std: float
    rouge1_r_std: float
    rougeL_f_std: float
    rougeL_p_std: float
    rougeL_r_std: float
    rouge2_f_std: float
    rouge2_p_std: float
    rouge2_r_std: float
    judge_score_std: float

    @staticmethod
    def from_file(filename: str) -> "BenchmarkResult":
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        def stats(array: np.ndarray):
            return np.mean(array), np.std(array)

        scores = {k: np.array(v) for k, v in data["bert_scores"].items()}
        rouge = {k: np.array(v) for k, v in data["rouge_scores"].items()}
        judge_scores = np.array(data["judge_scores"]) / 20.0  # normalize
        if judge_scores.size == 0:
            judge_scores = np.array([0.0])

        return BenchmarkResult(
            modelname=data["model_name"],
            num_samples=data["num_samples"],
            judge_model=data["judge_model"],
            **{f"bert_{k}": stats(scores[k])[0] for k in scores},
            **{f"bert_{k}_std": stats(scores[k])[1] for k in scores},
            **{k: stats(rouge[k])[0] for k in rouge},
            **{f"{k}_std": stats(rouge[k])[1] for k in rouge},
            judge_score=judge_scores.mean(),
            judge_scores=judge_scores,
            judge_score_std=judge_scores.std(),
        )

    def __str__(self) -> str:
        def line(label: str, mean: float, std: float) -> str:
            return f"{label}: {mean:.4f} ± {std:.4f}"

        return "\n".join([
            f"Model: {self.modelname}",
            f"Num samples: {self.num_samples}",
            f"Judge model: {self.judge_model}",
            line("BERT precision",
                 self.bert_precision, self.bert_precision_std),
            line("BERT recall", self.bert_recall, self.bert_recall_std),
            line("BERT F1", self.bert_f1, self.bert_f1_std),
            line("ROUGE-1 F", self.rouge1_f.mean(), self.rouge1_f_std),
            line("ROUGE-1 P", self.rouge1_p.mean(), self.rouge1_p_std),
            line("ROUGE-1 R", self.rouge1_r.mean(), self.rouge1_r_std),
            line("ROUGE-L F", self.rougeL_f.mean(), self.rougeL_f_std),
            line("ROUGE-L P", self.rougeL_p.mean(), self.rougeL_p_std),
            line("ROUGE-L R", self.rougeL_r.mean(), self.rougeL_r_std),
            line("ROUGE-2 F", self.rouge2_f.mean(), self.rouge2_f_std),
            line("ROUGE-2 P", self.rouge2_p.mean(), self.rouge2_p_std),
            line("ROUGE-2 R", self.rouge2_r.mean(), self.rouge2_r_std),
            line("Judge score", self.judge_score.mean(), self.judge_score_std)
        ])


def get_field_name(field: str) -> str:
    """Get a more readable title for the field."""
    field_names = {
        "bert_precision": "BERTScore Precision",
        "bert_recall": "BERTScore Recall",
        "bert_f1": "BERTScore F1",
        "rouge1_f": "ROUGE-1 F1",
        "rouge1_p": "ROUGE-1 Precision",
        "rouge1_r": "ROUGE-1 Recall",
        "rougeL_f": "ROUGE-L F1",
        "rougeL_p": "ROUGE-L Precision",
        "rougeL_r": "ROUGE-L Recall",
        "rouge2_f": "ROUGE-2 F1",
        "rouge2_p": "ROUGE-2 Precision",
        "rouge2_r": "ROUGE-2 Recall",
        "judge_score": "Judge Score"
    }
    return field_names.get(field, field)


def get_title_name(field: str) -> str:
    """Get a more readable title for the plot."""
    return f"{get_field_name(field)} by Model"


# def plot_benchmark_results(results: list[BenchmarkResult], field):
def plot_benchmark_results(ana_item: AnalysisItem, field: str):
    results = ana_item.results
    results.sort(key=lambda x: getattr(x, field))
    values = [getattr(result, field) for result in results]
    errors = [getattr(result, f"{field}_std") for result in results]
    labels = [get_simple_name(r.modelname) for r in results]

    title_name = get_title_name(field)
    field_name = get_field_name(field)
    labels = ljust_labels(labels, width=20)

    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(20, 12))  # 2x bigger
    plt.title(title_name, fontsize=32)
    plt.barh(labels, values, xerr=errors, capsize=5)
    plt.xlabel(field_name, fontsize=28)
    plt.ylabel("Model", fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24, rotation=0, fontname="monospace")  # Use monospace font
    plt.grid(axis='x')
    plt.tight_layout()

    # if "bert" in field and "custom" not in PLOT_DIR:
    if "bert" in field and "custom" not in ana_item.plot_dir:
        plt.xlim(0.6, 0.9)

    # save the plot
    # plt.savefig(os.path.join(PLOT_DIR, f"{field}.png"))
    plt.savefig(os.path.join(ana_item.plot_dir, f"{field}.png"))
    plt.close()


def gen_results_cell(ana_item: AnalysisItem) -> None:
    """
    generate a cell of results separate by tab
            BERT-precision  BERT-recall  BERT-F1  ROUGE-1-F  ...   JUDGE-SCORE
    MODEL   0.8             0.7          0.75     0.65       ...   0.8
    ...
    """
    fields = ana_item.fields
    results = ana_item.results
    header = "MODEL\t" + "\t".join(fields) + "\n"
    header = header.replace("judge_score", "Judge")
    header = header.replace("bert_precision", "B-pre")
    header = header.replace("bert_f1", "B-F1")
    header = header.replace("rouge1_f", "R-1")
    header = header.replace("bert_recall", "B-recall")
    header = header.replace("rougeL_f", "R-L")
    header = header.replace("rouge2_f", "R-2")

    # sort results by the order of the field
    for field in fields[::-1]:
        results.sort(key=lambda x: getattr(x, field), reverse=True)

    rows = []
    for result in results:
        row = [get_simple_name(result.modelname)]
        for field in fields:
            value = getattr(result, field)
            if isinstance(value, np.ndarray):
                value = value.mean()  # take the mean of the array
            value *= 100.0
            row.append(f"{value:.1f}")
        rows.append("\t".join(row))

    with open(
        os.path.join(ana_item.plot_dir, "data.csv"), "w", encoding="utf-8"
    ) as f:
        f.write(header + "\n".join(rows) + "\n")


def load_benchmark_results(ana_item: AnalysisItem) -> list[BenchmarkResult]:
    results = []
    for name in ana_item.models:
        result_file = os.path.join(BENCHMARK_DIR, name, "results.json")
        if os.path.exists(result_file):
            try:
                # print(f"[INFO] Loading benchmark result from {result_file}")
                results.append(BenchmarkResult.from_file(result_file))
            except Exception as e:
                print(f"[ERROR] Failed to load {result_file}: {e}")
        else:
            print(f"[WARNING] Result file {result_file} does not exist.")
    ana_item.results = results


if __name__ == "__main__":

    fields_to_plot = [
        "rouge1_f",
        "bert_f1",
        "judge_score",
        # "bert_precision",
        # "bert_recall",
        "rouge2_f",
        "rougeL_f",
    ]

    ana_items = [
        AnalysisItem("compare_data", fields_to_plot),
        AnalysisItem("compare_strategy", fields_to_plot),
        AnalysisItem("compare_overall", fields_to_plot),
        AnalysisItem("compare_custom_models", fields_to_plot),
        AnalysisItem("compare_stages", fields_to_plot),
    ]

    for ana_item in ana_items:
        load_benchmark_results(ana_item)
        print(f"Loaded {len(ana_item.results)} results for {ana_item.item}.")
        for field in ana_item.fields:
            plot_benchmark_results(ana_item, field)
        gen_results_cell(ana_item)
