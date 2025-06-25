import json
import os
import statistics
from dataclasses import dataclass

from opencc import OpenCC
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_simple_name, ljust_labels


BENCHMARK_DIR = "benchmark_result"
if not os.path.exists(BENCHMARK_DIR):
    print(f"Benchmark directory {BENCHMARK_DIR} does not exist.")
    exit(1)

OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "compare_zh_tw_percentage")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cc = OpenCC('s2twp')


@dataclass
class BenchmarkZHResult:
    modelname: str
    responses: list[str]

    def __post_init__(self):
        total_char_count = 0
        zh_tw_char_count = 0
        total_result_count = len(self.responses)
        zh_tw_response_count = 0

        # For variance calculation
        self.response_binary_scores = []  # 1 if exact match, else 0
        self.response_char_match_ratios = []  # % of matching characters

        for response in self.responses:
            translated_response = cc.convert(response)
            if len(response) != len(translated_response):
                continue

            total_char_count += len(response)
            if response == translated_response:
                zh_tw_response_count += 1
                zh_tw_char_count += len(response)
                self.response_binary_scores.append(1)
                self.response_char_match_ratios.append(1.0)
            else:
                match_count = sum(
                    c1 == c2 for c1, c2 in zip(response, translated_response)
                )
                zh_tw_char_count += match_count
                self.response_binary_scores.append(0)
                self.response_char_match_ratios.append(
                    match_count / len(response)
                )

        self.zh_tw_word_percentage = (
            zh_tw_char_count / total_char_count if total_char_count > 0 else 0
        )
        self.zh_tw_result_percentage = (
            zh_tw_response_count / total_result_count
            if total_result_count > 0 else 0
        )

        if len(self.response_binary_scores) > 1:
            self.zh_tw_result_percentage_var = statistics.variance(
                self.response_binary_scores
            )
        else:
            self.zh_tw_result_percentage_var = 0.0

        if len(self.response_char_match_ratios) > 1:
            self.zh_tw_word_percentage_var = statistics.variance(
                self.response_char_match_ratios
            )
        else:
            self.zh_tw_word_percentage_var = 0.0


def get_field_name(field: str) -> str:
    field_names = {
        "zh_tw_word_percentage": "ZH-TW Word Percentage",
        "zh_tw_result_percentage": "ZH-TW Response Percentage",
    }
    return field_names.get(field, field)


def get_title_name(field: str) -> str:
    title_names = {
        "zh_tw_word_percentage": "ZH-TW Word Percentage",
        "zh_tw_result_percentage": "ZH-TW Response Percentages by Model",
    }
    return title_names.get(field, field)


def plot_benchmark_results(
    results: list[BenchmarkZHResult], field, suffix: str = ""
) -> None:
    results.sort(key=lambda x: getattr(x, field))
    values = [getattr(result, field) for result in results]
    labels = [get_simple_name(r.modelname) for r in results]

    labels = ljust_labels(labels, width=20)

    x_min = max(min(values) - 0.05, 0.0)  # Ensure x_min is not negative
    field_name = get_field_name(field)
    title_name = get_title_name(field)

    # errors = [getattr(result, f"{field}_var") ** 0.5 for result in results]
    # print(f"Error bars: {errors}")

    plt.figure(figsize=(20, 12))  # 2x bigger
    plt.title(title_name, fontsize=32)
    # plt.figure(figsize=(20, 12))  # 2x bigger
    plt.title(title_name, fontsize=32)
    plt.barh(labels, values, capsize=5)
    # plt.barh(labels, values, xerr=errors, capsize=5)
    plt.xlabel(field_name, fontsize=28)
    plt.xlim(x_min, 1.0)
    plt.ylabel("Model", fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24, fontname="monospace")
    plt.grid(axis='x')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{field}{suffix}.png"))
    plt.close()


# compare overall
INCLUDED_NAMES = [
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


# compare different stages
INCLUDED_NAMES = [
    # "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3-lr_adj",
    # "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3-lr_adj",
    # "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v3-lr_adj",

    "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v4-lr_adj",
    "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v4-lr_adj",
    "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v4-lr_adj",

    # "Qwen2.5-0.5B-Instruct-cl_24233news_1stg_v3",
    # "Qwen2.5-0.5B-Instruct-cl_24233news_4stg_v3",
]


def load_benchmark_results(benchmark_dir: str) -> list[BenchmarkZHResult]:
    results = []
    for name in tqdm(os.listdir(benchmark_dir)):
        path = os.path.join(benchmark_dir, name)
        if os.path.isdir(path) and name in INCLUDED_NAMES:
            results_file = os.path.join(path, "results.json")
            response_file = os.path.join(path, "responses.json")
            # load respones
            with open(response_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            responses: list[str] = [v["response"] for k, v in data.items()]
            # load model name
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                model_name = data["model_name"]
            results.append(BenchmarkZHResult(model_name, responses))
    return results


if __name__ == "__main__":

    fields_to_plot = [
        "zh_tw_word_percentage",
        "zh_tw_result_percentage",
    ]

    benchmark_results = load_benchmark_results(BENCHMARK_DIR)

    for field in fields_to_plot:
        # plot_benchmark_results(benchmark_results, field, "_overall")
        plot_benchmark_results(benchmark_results, field, "_stages")
