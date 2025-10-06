import os

import bert_score
import evaluate
import jieba
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from opencc import OpenCC
from rouge_chinese import Rouge as RougeChinese
from rouge_score import rouge_scorer
from tqdm import tqdm

from utils import ljust_labels


cc = OpenCC("s2tw")  # Simplified Chinese to Traditional Chinese

rouge_c = RougeChinese()
rouge_eval = evaluate.load("rouge")


def avg_bert_scores(bert_scores):
    avg_scores = {}
    for score in bert_scores.keys():
        avg_scores[score] = 0 if len(bert_scores) == 0 else sum(
            [bert_score for bert_score in bert_scores[score]]
        ) / len(bert_scores[score])
    return avg_scores


def avg_rouge_scores(rouge_scores):
    avg_scores = {}
    for score in rouge_scores[0].keys():
        avg_scores[score] = 0 if len(rouge_scores) == 0 else sum(
            [rouge_score[score].fmeasure for rouge_score in rouge_scores]
        ) / len(rouge_scores)
    return avg_scores


def avg_rouge_scores_chinese(rouge_scores: dict[str, list]) -> dict:
    avg_scores = {}
    for k, v in rouge_scores.items():
        avg_scores[k] = 0 if len(v) == 0 else sum(v) / len(v)
    return avg_scores


def avg_judge_scores(judge_scores):
    avg_score = 0 if len(judge_scores) == 0 else sum(
        judge_scores
    ) / len(judge_scores)

    # Normalize the score to a scale of 0 to 1
    return avg_score / 20.0


def evaluate_with_rouge(predictions, references) -> list[dict]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [
        scorer.score(pred, ref) for pred, ref in zip(predictions, references)
    ]
    scores_dict = {
        "rouge1_f": [score["rouge1"].fmeasure for score in scores],
        "rouge1_p": [score["rouge1"].precision for score in scores],
        "rouge1_r": [score["rouge1"].recall for score in scores],
        "rouge2_f": [score["rouge2"].fmeasure for score in scores],
        "rouge2_p": [score["rouge2"].precision for score in scores],
        "rouge2_r": [score["rouge2"].recall for score in scores],
        "rougeL_f": [score["rougeL"].fmeasure for score in scores],
        "rougeL_p": [score["rougeL"].precision for score in scores],
        "rougeL_r": [score["rougeL"].recall for score in scores],
    }
    return scores_dict


def evaluate_with_rouge_chinese(preds: list[str], refs: list[str]) -> dict:
    preds = [" ".join(jieba.cut(pred)) for pred in preds]
    refs = [" ".join(jieba.cut(ref)) for ref in refs]

    scores = []
    for pred, ref in tqdm(zip(preds, refs), total=len(preds),
                          desc="Evaluating with Rouge Chinese"):
        if len(pred) == 0 or len(ref) == 0 or pred.isspace() or ref.isspace():
            scores.append(
                {
                    "rouge-1": {"f": 0, "p": 0, "r": 0},
                    "rouge-l": {"f": 0, "p": 0, "r": 0},
                    "rouge-2": {"f": 0, "p": 0, "r": 0},
                }
            )
            continue
        score = rouge_c.get_scores(pred, ref)
        scores.append(score[0])

    scores_dict = {
        "rouge1_f": [score["rouge-1"]["f"] for score in scores],
        "rouge1_p": [score["rouge-1"]["p"] for score in scores],
        "rouge1_r": [score["rouge-1"]["r"] for score in scores],
        "rougeL_f": [score["rouge-l"]["f"] for score in scores],
        "rougeL_p": [score["rouge-l"]["p"] for score in scores],
        "rougeL_r": [score["rouge-l"]["r"] for score in scores],
        "rouge2_f": [score["rouge-2"]["f"] for score in scores],
        "rouge2_p": [score["rouge-2"]["p"] for score in scores],
        "rouge2_r": [score["rouge-2"]["r"] for score in scores],
    }

    return scores_dict


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


def evaluate_with_rouge_eval(preds: list[str], refs: list[str]) -> dict:
    preds = [cc.convert(pred).strip() for pred in preds]
    # preds = [" ".join(jieba.cut(pred)) for pred in preds]
    # refs = [" ".join(jieba.cut(ref)) for ref in refs]
    return rouge_eval.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True,
        rouge_types=["rouge1", "rougeL", "rougeLsum", "rouge2"]
    )


def evaluate_with_bertscore(preds, refs) -> dict:
    P, R, F1 = bert_score.score(preds, refs, lang="zh",
                                model_type="bert-base-chinese", verbose=True)
    return {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}


def plot_benchmark_results(values, errors, labels, field, plot_dir) -> None:
    """Plot benchmark results as a bar chart.
    Args:
        values: List of mean values.
        errors: List of standard deviations.
        labels: List of model names.
        field: The metric being plotted (e.g., 'bert_f1', 'rouge1_f').
        plot_dir: Directory to save the plot.
    """
    os.makedirs(plot_dir, exist_ok=True)

    assert len(values) == len(errors) == len(labels), "Values, errors, and labels must have the same length."

    values = np.array(values)
    errors = np.array(errors)
    labels = np.array(labels)

    assert values.shape == errors.shape == labels.shape, "Values, errors, and labels must have the same shape."

    # sort by values in reverse (descending) order
    sorted_indices = np.argsort(values)[::-1]
    values = values[sorted_indices]
    errors = errors[sorted_indices]
    labels = labels[sorted_indices]

    # title_name = get_title_name(field)
    field_name = get_field_name(field)
    labels = ljust_labels(labels, width=24)

    # === Dissertation-style settings ===
    plt.figure(figsize=(8, 6), dpi=300)   # smaller but publication-grade
    sns.set_style("whitegrid")
    plt.rcParams.update({
        # "font.family": "Times New Roman",   # match dissertation text
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    # === Plot ===
    sns.barplot(
        x=values,
        y=labels,
        color="gray",             # grayscale-friendly
        edgecolor="black",
        xerr=errors,
        capsize=0.2
    )

    # === Labels ===
    plt.xlabel(field_name)
    plt.ylabel("Model")

    # Remove top/right borders
    sns.despine()

    plt.tight_layout()

    # # Special case for BERT
    # if "bert" in field and "custom" not in plot_dir:
    #     plt.xlim(0.6, 0.9)

    # Save the plot
    plt.savefig(os.path.join(plot_dir, f"{field}.png"))
    plt.close()
