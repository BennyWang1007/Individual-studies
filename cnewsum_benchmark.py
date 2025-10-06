import os
import re
import json
import numpy as np
import pandas as pd
import ollama

from datasets import Dataset, load_from_disk
from opencc import OpenCC
from tqdm import tqdm
from transformers import AutoTokenizer

from curriculum_training.constants import MAX_INPUT_LENGTH, MAX_NEW_TOKENS
from curriculum_training.curriculum_utils import (
    PREFIX_OF_DIFFICULTY_LEVELS,
    DifficultyLevels as DF,
)
from utils import legalize_filename, get_simple_name
from utils_vllm import init_vllm_model, vllm_batch_generate, vllm_cleanup
from benchmark_utils import (
    plot_benchmark_results,
    avg_bert_scores,
    avg_rouge_scores_chinese,
    evaluate_with_rouge,
    evaluate_with_bertscore,
)

CNEWSUM_BENCHMARK_DIR = "cnewsum_benchmark"
os.makedirs(CNEWSUM_BENCHMARK_DIR, exist_ok=True)


cc = OpenCC('s2twp')  # 簡體轉繁體

ds = None
summaries = None


def preprocess_cnewsum_dataset() -> None:

    if os.path.exists("CNewsSum_test_filtered"):
        print("Filtered dataset CNewsSum_test_filtered already exists, skipping preprocessing.")
        return

    # preprocessing for cnewssum dataset
    ori_dataset_file = "CNewSum_v2/test.simple.anno.label.jsonl"

    with open(ori_dataset_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    def ensure_punctuation(summary):
        if not summary.endswith(("。", "!", "！", "?", "？", ")", "）", "」")):
            return summary + "。"
        return summary

    preprocessed_data = [
        {
            "article": ("".join(item["article"])).replace(" ", ""),
            "summary": ensure_punctuation(("".join(item["summary"])).replace(" ", "")),
        }
        for item in data
    ]

    # filtered_ds = ds["test"].filter(lambda example: len(example["text"]) < MAX_INPUT_LENGTH - 100)
    filtered_data = [
        item for item in preprocessed_data if len(item["article"]) < MAX_INPUT_LENGTH - 100
    ]
    print(f"Filtered dataset size: {len(filtered_data)}")

    def convert_example(example):
        example["text"] = cc.convert(example["article"])
        example["summary"] = cc.convert(example["summary"])
        return example

    converted_ds = Dataset.from_list([convert_example(item) for item in filtered_data])

    # print some examples to verify
    for i in range(5):
        print(f"Converted {i} text: {converted_ds[i]['text'][:50] if 'text' in converted_ds[i] else 'N/A'}...")
        print(f"Converted {i} summary: {converted_ds[i]['summary'][:50] if 'summary' in converted_ds[i] else 'N/A'}...")

    converted_ds.save_to_disk("CNewsSum_test_filtered")
    print("Filtered dataset saved to CNewsSum_test_filtered")


def generate_cnewsum_responses(model_names: list[str]) -> None:

    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DF.DIRECT_SUMMARY]

    ds = load_from_disk("CNewsSum_test_filtered")
    ds2 = {"text": [d["text"] for d in ds], "summary": [d["summary"] for d in ds]}  # input: {...}, output: {...}

    def benchmark_model_with_cnewsum(model_name):
        try:
            filename = os.path.join(
                CNEWSUM_BENCHMARK_DIR, legalize_filename(f"cnewsum_test_{model_name}_responses.json"))

            if os.path.exists(filename):
                print(f"Responses for model {model_name} already exist at {filename}, skipping...")
                return

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if model_name == "google/gemma-2-2b-it":
                prompts = [
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": sys_prompt + "\n新聞：\n" + text
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                        # enable_thinking=False,  # Disable thinking prompt for Gemma
                    )
                    for text in ds2["text"]
                ]
            else:
                prompts = [
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": "新聞：\n" + text}
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        # enable_thinking=False,  # Disable thinking prompt for Qwen3
                    )
                    for text in ds2["text"]
                ]

            llm, sampling_param = init_vllm_model(
                model_name, max_input_length=MAX_INPUT_LENGTH, max_new_tokens=MAX_NEW_TOKENS)

            responses = vllm_batch_generate(llm, prompts, sampling_param, batch_size=1000)

            responses_text = [resp.outputs[0].text for resp in responses]

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(responses_text, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error benchmarking model {model_name}: {e}")

    for model_name in model_names:
        benchmark_model_with_cnewsum(model_name)
        vllm_cleanup()


def generate_cnewsum_response_32B() -> None:
    """
    Generate CNewsSum responses using Ollama with Qwen2.5-32B-Instruct.
    Assumes Ollama is running and the model is available as 'Qwen2.5-32B-Instruct'.
    """

    ollama.pull("qwen2.5:32b-instruct")
    print("Pulled model qwen2.5:32b-instruct from Ollama.")

    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DF.DIRECT_SUMMARY]
    ds = load_from_disk("CNewsSum_test_filtered")
    texts = [d["text"] for d in ds]

    filename = os.path.join(
        CNEWSUM_BENCHMARK_DIR, legalize_filename("cnewsum_test_Qwen_Qwen2.5-32B-Instruct_responses.json")
    )

    if os.path.exists(filename):
        print(f"Responses for model Qwen2.5-32B-Instruct already exist at {filename}, skipping...")
        return

    prompts = [f"新聞：\n{text}" for text in texts]

    responses_text = []
    for prompt in tqdm(prompts, desc="Generating with Ollama qwen2.5:32b-instruct"):
        try:
            response = ollama.chat(
                model="qwen2.5:32b-instruct",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.get("message", {}).get("content", "")
            responses_text.append(content)
            if len(responses_text) % 1000 == 0:
                print(f"Generated {len(responses_text)} responses so far, saving interim results...")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(responses_text, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error generating response: {e}")
            responses_text.append("")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(responses_text, f, indent=4, ensure_ascii=False)


def clean_prefix(text: str) -> str:
    # remove prefix with order: "</think>\n\n", "摘要\n", "摘要：\n", "摘要：", "摘要如下：\n", "摘要如下："
    patterns = [
        r"</think>\n\n",
        r"摘要\n",
        r"摘要：",
        r"摘要如下：",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            text = text[match.end():]

    # remove all "**", "###"
    text = text.replace("**", "").replace("###", "")
    return text.strip("* \n#")


def benchmark_clean_cnewsum_responses(model_name: str, regen: bool = False) -> None:
    global ds, summaries

    response_file = os.path.join(
        CNEWSUM_BENCHMARK_DIR, legalize_filename(f"cnewsum_test_{model_name}_responses.json"))
    response_file_cleaned = os.path.join(
        CNEWSUM_BENCHMARK_DIR, legalize_filename(f"cnewsum_test_{model_name}_responses_cleaned.json"))

    if not os.path.exists(response_file):
        raise ValueError(f"Response file {response_file} does not exist.")

    if not regen and os.path.exists(response_file_cleaned):
        print(f"Results file {response_file_cleaned} already exists. Skipping analysis.")
        return

    with open(response_file, "r", encoding="utf-8") as f:
        responses_text = json.load(f)

    if ds is None or summaries is None:
        ds = load_from_disk("CNewsSum_test_filtered")
        summaries = [d["summary"] for d in ds]

    assert len(responses_text) == len(summaries), "Responses and summaries count mismatch"

    responses = [clean_prefix(cc.convert(resp)) for resp in tqdm(responses_text, desc="Cleaning responses")]

    with open(response_file_cleaned, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)


def benchmark_analyze_cnewsum_responses(model_name: str, regen: bool = False) -> None:
    global ds, summaries

    response_file = os.path.join(
        CNEWSUM_BENCHMARK_DIR, legalize_filename(f"cnewsum_test_{model_name}_responses_cleaned.json"))
    results_file = os.path.join(
        CNEWSUM_BENCHMARK_DIR, legalize_filename(f"cnewsum_test_{model_name}_results.json"))

    if not os.path.exists(response_file):
        raise ValueError(f"Response file {response_file} does not exist.")

    if not regen and os.path.exists(results_file):
        print(f"Results file {results_file} already exists. Skipping analysis.")
        return

    if ds is None or summaries is None:
        ds = load_from_disk("CNewsSum_test_filtered")
        summaries = [d["summary"] for d in ds]

    with open(response_file, "r", encoding="utf-8") as f:
        responses = json.load(f)

    assert len(responses) == len(summaries), "Responses and summaries count mismatch"

    # analyze the responses
    rouge_scores = evaluate_with_rouge(responses, summaries)
    bert_scores = evaluate_with_bertscore(responses, summaries)

    results = {
        "model_name": model_name,
        "num_samples": len(summaries),
        "avg_bert_scores": avg_bert_scores(bert_scores),
        "avg_rouge_scores": avg_rouge_scores_chinese(rouge_scores),
        "bert_scores": bert_scores,
        "rouge_scores": rouge_scores,
        "predictions": responses,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # show summary of results
    print("=" * 40)
    print(f"Model: {model_name}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Average BERT Scores: {results['avg_bert_scores']}")
    print(f"Average ROUGE Scores: {results['avg_rouge_scores']}")
    print(f"Results saved to {results_file}")
    print("")


FIELD_MAP = {
    "bert_f1": lambda d: d["bert_scores"]["f1"],
    "rouge1_f": lambda d: d["rouge_scores"]["rouge1_f"],
    "rouge2_f": lambda d: d["rouge_scores"]["rouge2_f"],
    "rougeL_f": lambda d: d["rouge_scores"]["rougeL_f"],
}


def benchmark_analyze_cnewsum_plot(model_names: list[str], field: str):

    data = []

    for model_name in model_names:
        results_file = os.path.join(
            CNEWSUM_BENCHMARK_DIR, legalize_filename(f"cnewsum_test_{model_name}_results.json"))
        if not os.path.exists(results_file):
            raise ValueError(f"Results file {results_file} does not exist. Please run analysis first.")
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            data.append(results)

    complete_data = [np.array(FIELD_MAP[field](d)) for d in data if field in FIELD_MAP]

    values = [complete_data.mean() for complete_data in complete_data]
    errors = [complete_data.std() for complete_data in complete_data]
    labels = [get_simple_name(d["model_name"]) for d in data]

    save_dir = os.path.join(CNEWSUM_BENCHMARK_DIR, "plots")

    plot_benchmark_results(values, errors, labels, field, save_dir)
    print(f"Benchmark plot of {field} saved to {save_dir}")


def benchmark_analyze_cnewsum_csv(model_names: list[str]) -> None:
    """Output a table of model results for each metric"""

    results_data = []

    for model_name in model_names:

        results_file = os.path.join(
            CNEWSUM_BENCHMARK_DIR, legalize_filename(f"cnewsum_test_{model_name}_results.json"))

        if not os.path.exists(results_file):
            continue

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            row = {
                "models": get_simple_name(results["model_name"]),
                "bert_f1": results["avg_bert_scores"]["f1"] * 100,
                "rouge1_f": results["avg_rouge_scores"]["rouge1_f"] * 100,
                "rouge2_f": results["avg_rouge_scores"]["rouge2_f"] * 100,
                "rougeL_f": results["avg_rouge_scores"]["rougeL_f"] * 100,
            }
            results_data.append(row)

            # create DataFrame and display, sorted by rouge1_f descending
            df = pd.DataFrame(results_data)
            df = df[["models", "rouge1_f", "rouge2_f", "rougeL_f", "bert_f1"]]
            df = df.sort_values(by="rouge1_f", ascending=False)

            # save the DataFrame to CSV in the plots directory
            os.makedirs(os.path.join(CNEWSUM_BENCHMARK_DIR, "plots"), exist_ok=True)
            csv_path = os.path.join(CNEWSUM_BENCHMARK_DIR, "plots", "data.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Results table saved to {csv_path}")

            df = df.round(2)  # round to 2 decimal places for display and CSV
            print(df.to_string(index=False))


if __name__ == "__main__":

    preprocess_cnewsum_dataset()

    model_gen_names = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen2.5-0.5B-Instruct-cl_24233news_5stg_v4-lr_adj",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "google/gemma-2-2b-it",
        "google/gemma-3-1b-it",
        # "Qwen/Qwen2.5-32B-Instruct",
    ]

    model_names = model_gen_names.copy()
    model_names.append("Qwen/Qwen2.5-32B-Instruct")

    generate_cnewsum_responses(model_gen_names)
    generate_cnewsum_response_32B()

    for model_name in model_names:
        benchmark_clean_cnewsum_responses(model_name, regen=False)
        benchmark_analyze_cnewsum_responses(model_name, regen=False)

    fields = [
        "bert_f1",
        "rouge1_f",
        "rougeL_f",
        "rouge2_f",
    ]

    for field in fields:
        benchmark_analyze_cnewsum_plot(model_names, field)

    benchmark_analyze_cnewsum_csv(model_names)
