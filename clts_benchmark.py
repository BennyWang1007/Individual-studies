import os
import re

import json
import ollama
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
from opencc import OpenCC
from tqdm import tqdm
from transformers import AutoTokenizer

from benchmark_utils import (
    avg_bert_scores,
    avg_rouge_scores_chinese,
    evaluate_with_rouge,
    evaluate_with_bertscore,
    plot_benchmark_results,
)
from curriculum_training.constants import MAX_INPUT_LENGTH, MAX_NEW_TOKENS
from curriculum_training.curriculum_utils import (
    PREFIX_OF_DIFFICULTY_LEVELS,
    DifficultyLevels as DF,
)
from utils import legalize_filename, get_simple_name
from utils_vllm import init_vllm_model, vllm_batch_generate, vllm_cleanup


CLTS_BENCHMARK_DIR = "clts_benchmark"

FIELD_MAP = {
    "bert_f1": lambda d: d["bert_scores"]["f1"],
    "rouge1_f": lambda d: d["rouge_scores"]["rouge1_f"],
    "rouge2_f": lambda d: d["rouge_scores"]["rouge2_f"],
    "rougeL_f": lambda d: d["rouge_scores"]["rougeL_f"],
}


cc = OpenCC('s2twp')  # 簡體轉繁體
sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DF.DIRECT_SUMMARY]

ds = None
ds2 = None
summaries = None


def clts_preprocess():
    clts_ds = load_dataset("Gdot/clts")

    filtered_ds = clts_ds["test"].filter(lambda example: len(example["text"]) < MAX_INPUT_LENGTH - 100)
    print(f"Filtered dataset size: {len(filtered_ds)}")

    def convert_example(example):
        example["text"] = cc.convert(example["text"])
        example["summary"] = cc.convert(example["summary"])
        return example

    converted_ds = filtered_ds.map(convert_example)
    converted_ds.save_to_disk("clts_valid_filtered")
    print("Filtered dataset saved to clts_valid_filtered")


def load_preprocessed_clts():
    global ds, ds2, summaries
    if ds is None:
        ds = load_from_disk("clts_test_filtered")
        ds2 = {"text": [d["text"] for d in ds], "summary": [d["summary"] for d in ds]}
        summaries = ds2["summary"]


def generate_clts_response(model_name):
    global ds, ds2
    try:
        load_preprocessed_clts()

        filename = os.path.join(
            "clts_benchmark", legalize_filename(f"clts_test_{model_name}_responses.json"))

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


def generate_clts_response_32B() -> None:
    """
    Generate CLTS responses using Ollama with Qwen2.5-32B-Instruct.
    Assumes Ollama is running and the model is available as 'qwen2.5:32b-instruct'.
    """
    try:
        ollama.pull("qwen2.5:32b-instruct")
        print("Pulled model qwen2.5:32b-instruct from Ollama.")
    except Exception as e:
        print(f"Error pulling model: {e}")

    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DF.DIRECT_SUMMARY]

    load_preprocessed_clts()

    texts = [d["text"] for d in ds]

    CLTS_BENCHMARK_DIR = "clts_benchmark"
    os.makedirs(CLTS_BENCHMARK_DIR, exist_ok=True)
    filename = os.path.join(
        CLTS_BENCHMARK_DIR, legalize_filename("clts_test_Qwen_Qwen2.5-32B-Instruct_responses.json")
    )

    responses_text = []

    if os.path.exists(filename):
        # print(f"Responses for model Qwen2.5-32B-Instruct already exist at {filename}, skipping...")
        responses_text = json.load(open(filename, "r", encoding="utf-8"))
        if len(responses_text) >= len(texts):
            print("All responses for model Qwen2.5-32B-Instruct already exist, skipping...")
            return
        print(f"Loaded existing responses for model Qwen2.5-32B-Instruct from {filename}")

    prompts = [f"新聞：\n{text}" for text in texts]

    prompts = prompts[len(responses_text):]  # only generate for remaining prompts
    # prompts = prompts[:1000]  # limit to 1000 samples for testing

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
            responses_text.append(content if content is not None else "")
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


def benchmark_clean_clts_responses(model_name: str, regen: bool = False) -> None:
    global summaries
    response_file = os.path.join(
        "clts_benchmark", legalize_filename(f"clts_test_{model_name}_responses.json"))
    response_file_cleaned = os.path.join(
        "clts_benchmark", legalize_filename(f"clts_test_{model_name}_responses_cleaned.json"))

    if not os.path.exists(response_file):
        raise ValueError(f"Response file {response_file} does not exist.")

    if not regen and os.path.exists(response_file_cleaned):
        print(f"Results file {response_file_cleaned} already exists. Skipping analysis.")
        return

    with open(response_file, "r", encoding="utf-8") as f:
        responses_text = json.load(f)

    if model_name != "Qwen/Qwen2.5-32B-Instruct":
        assert len(responses_text) == len(summaries), "Responses and summaries count mismatch"
    else:
        summaries = summaries[:len(responses_text)]

    responses = [clean_prefix(cc.convert(resp)) for resp in tqdm(responses_text, desc="Cleaning responses")]

    with open(response_file_cleaned, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)


def benchmark_analyze_clts_responses(model_name: str, regen: bool = False) -> None:
    response_file = os.path.join(
        "clts_benchmark", legalize_filename(f"clts_test_{model_name}_responses_cleaned.json"))
    results_file = os.path.join(
        "clts_benchmark", legalize_filename(f"clts_test_{model_name}_results.json"))

    if not os.path.exists(response_file):
        raise ValueError(f"Response file {response_file} does not exist.")

    if not regen and os.path.exists(results_file):
        print(f"Results file {results_file} already exists. Skipping analysis.")
        return

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


def benchmark_analyze_clts_plot(model_names: list[str], field: str):

    data = []

    for model_name in model_names:
        results_file = os.path.join(
            "clts_benchmark", legalize_filename(f"clts_test_{model_name}_results.json"))

        if not os.path.exists(results_file):
            raise ValueError(f"Results file {results_file} does not exist. Please run analysis first.")

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            data.append(results)

    complete_data = [np.array(FIELD_MAP[field](d)) for d in data if field in FIELD_MAP]

    values = [complete_data.mean() for complete_data in complete_data]
    errors = [complete_data.std() for complete_data in complete_data]
    labels = [get_simple_name(d["model_name"]) for d in data]

    save_dir = os.path.join("clts_benchmark", "plots")

    plot_benchmark_results(values, errors, labels, field, save_dir)
    print(f"Benchmark plot of {field} saved to {save_dir}")


def benchmark_analyze_clts_csv(model_names: list[str]) -> None:
    """ Collect results for all models and metrics """
    results_data = []
    for model_name in model_names:
        results_file = os.path.join(
            CLTS_BENCHMARK_DIR, legalize_filename(f"clts_test_{model_name}_results.json"))
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

    # create DataFrame and display, sorted by bert_f1 descending
    df = pd.DataFrame(results_data)
    df = df[["models", "rouge1_f", "rouge2_f", "rougeL_f", "bert_f1"]]
    df = df.sort_values(by="rouge1_f", ascending=False)

    # save the DataFrame to CSV in the plots directory
    os.makedirs(os.path.join(CLTS_BENCHMARK_DIR, "plots"), exist_ok=True)
    csv_path = os.path.join(CLTS_BENCHMARK_DIR, "plots", "data.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Results table saved to {csv_path}")

    df = df.round(2)
    print(df.to_string(index=False))


if __name__ == "__main__":

    clts_preprocess()
    load_preprocessed_clts()

    model_names = [
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

    for model_name in model_names:
        generate_clts_response(model_name)
        vllm_cleanup()

    generate_clts_response_32B()

    model_names.append("Qwen/Qwen2.5-32B-Instruct")

    for model_name in model_names:
        benchmark_clean_clts_responses(model_name, regen=False)
        benchmark_analyze_clts_responses(model_name, regen=False)

    fields = [
        "bert_f1",
        "rouge1_f",
        "rougeL_f",
        "rouge2_f",
    ]

    for field in fields:
        benchmark_analyze_clts_plot(model_names, field)

    benchmark_analyze_clts_csv(model_names)
