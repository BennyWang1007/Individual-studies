import json
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


BENCHMARK_DIR = "benchmark_result"
if not os.path.exists(BENCHMARK_DIR):
    print(f"Benchmark directory {BENCHMARK_DIR} does not exist.")
    exit(1)


@dataclass
class BenchmarkResult:
    modelname: str
    num_samples: int
    judge_model: str
    bert_precision: float
    bert_recall: float
    bert_f1: float

    rouge1_f: np.ndarray
    rouge1_p: np.ndarray
    rouge1_r: np.ndarray
    rougeL_f: np.ndarray
    rougeL_p: np.ndarray
    rougeL_r: np.ndarray
    rouge2_f: np.ndarray
    rouge2_p: np.ndarray
    rouge2_r: np.ndarray
    judge_score: np.ndarray

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
        judge = np.array(data["judge_scores"]) / 20

        return BenchmarkResult(
            modelname=data["model_name"],
            num_samples=data["num_samples"],
            judge_model=data["judge_model"],
            **{f"bert_{k}": stats(scores[k])[0] for k in scores},
            **{f"bert_{k}_std": stats(scores[k])[1] for k in scores},
            **{k: stats(rouge[k])[0] for k in rouge},
            **{f"{k}_std": stats(rouge[k])[1] for k in rouge},
            judge_score=judge.mean(),
            judge_score_std=judge.std(),
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


def plot_benchmark_results(results: list[BenchmarkResult], field):
    results.sort(key=lambda x: getattr(x, field))
    values = [getattr(result, field) for result in results]
    errors = [getattr(result, f"{field}_std") for result in results]
    labels = [get_simple_name(r.modelname) for r in results]
    # Pad labels to the same length
    max_length = max(len(label) for label in labels)
    labels = [label.ljust(max_length) for label in labels]

    plt.figure(figsize=(10, 6))
    plt.title(f"Benchmark Results: {field}")
    plt.barh(labels, values, xerr=errors, capsize=5)
    plt.xlabel(field)
    plt.ylabel("Model")
    plt.yticks(rotation=0, fontname="monospace")  # Use monospace font
    plt.grid(axis='x')
    plt.tight_layout()

    if "bert" in field:
        plt.xlim(0.6, 0.9)

    # save the plot
    plt.savefig(os.path.join(BENCHMARK_DIR, f"{field}.png"))
    plt.close()


def get_simple_name(name: str) -> str:
    name = name.replace("-lr_adj", "").replace("v2", "v3")
    name = name.replace("better2", "v2").replace("better", "v2")

    patterns = [
        # parse trained models
        (R"^(Qwen/)?Qwen([0-9\.]+)-([0-9\.]+B)-Instruct"
         R"(-curriculum|-cl)?_([0-9]+)news_([0-9])(stage|stg)"
         R"(_A100)?(.*)?$",
         lambda m: (
             f"Qw{m.group(2)}-{m.group(3)}_{m.group(5)}n"
             f"_{m.group(6)}stg{m.group(9) or ''}"
         )),
        # parse gemma models
        (R"^google/gemma-([0-9])-([0-9\.]+b)-it$",
         lambda m: f"Gemma-{m.group(1)}-{m.group(2)}"),
        # parse Qwen models
        (R"^Qwen/Qwen([0-9\.]+)-([0-9\.]+B)-Instruct$",
         lambda m: f"Qw{m.group(1)}-{m.group(2)}"),
    ]

    for pattern, fmt in patterns:
        match = re.match(pattern, name)
        if match:
            return fmt(match)

    # return name
    print(f"Unknown model name format: {name}")
    return name


def load_benchmark_results(benchmark_dir: str) -> list[BenchmarkResult]:
    results = []
    for name in os.listdir(benchmark_dir):
        path = os.path.join(benchmark_dir, name)
        if os.path.isdir(path) and name != ".ipynb_checkpoints":
            result_file = os.path.join(path, "results.json")
            if os.path.exists(result_file):
                try:
                    results.append(BenchmarkResult.from_file(result_file))
                except Exception as e:
                    print(f"[ERROR] Failed to load {result_file}: {e}")
    return results


if __name__ == "__main__":

    fields_to_plot = [
        "judge_score",
        "bert_precision",
        "bert_f1",
        "bert_recall",
        "rouge1_f",
        "rougeL_f",
        "rouge2_f",
    ]

    benchmark_results = load_benchmark_results(BENCHMARK_DIR)

    for field in fields_to_plot:
        plot_benchmark_results(benchmark_results, field)
