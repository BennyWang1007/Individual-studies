from typing import Literal

from utils import (
    get_response_filename,
    get_news_with_rationale_filename,
    # get_formatted_nwr_filename,
    get_zh_tw_filename,
)

ALLOW_VLLM = True
ALLOW_OLLAMA = True

USE_VLLM = True and ALLOW_VLLM

InferenceType = Literal["OLLAMA", "VLLM"]

MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_BASE_OLLAMA = "qwen2.5:0.5b-instruct"
MODEL_DISTAL_FROM = "qwen2.5:32b-instruct-q6_K"

MAX_INPUT_LENGTH = 4096
MAX_BENCHMARK_LENGTH = 2048
MAX_TRAINING_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 1024


# file names of datasets
DATASET_V1_DIR = "training_data_v1"
GENARATED_RESPONSE_FILE = get_response_filename(MODEL_DISTAL_FROM)
GENARATED_RESPONSE_FILE = f"{DATASET_V1_DIR}/{GENARATED_RESPONSE_FILE}"
GENARATED_NWR_FILE = get_news_with_rationale_filename(MODEL_DISTAL_FROM)
GENARATED_NWR_FILE = f"{DATASET_V1_DIR}/{GENARATED_NWR_FILE}"
GENARATED_ZH_TW_FILE = f"{DATASET_V1_DIR}/{get_zh_tw_filename(MODEL_BASE)}"
FORMATTED_NWR_FILE = f"{DATASET_V1_DIR}/formatted_nwr.jsonl"
FORMATTED_NWR_FILE2 = f"{DATASET_V1_DIR}/formatted_nwr2.jsonl"
NWR_TRAINING_FILE = f"{DATASET_V1_DIR}/formatted_nwr_training.jsonl"
NWR_BENCHMARK_FILE = f"{DATASET_V1_DIR}/formatted_nwr_benchmark.jsonl"

DATASET_V2_DIR = "training_data_v2"
NWR_V2 = f"{DATASET_V2_DIR}/news_with_rationale.jsonl"
FORMATTED_NWR_FILE_V2 = f"{DATASET_V2_DIR}/formatted_nwr_better.jsonl"
FORMATTED_NWR_FILE_V2_2 = f"{DATASET_V2_DIR}/formatted_nwr_better2.jsonl"
NWR_TRAINING_V2 = f"{DATASET_V2_DIR}/formatted_nwr_training.jsonl"
NWR_BENCHMARK_V2 = f"{DATASET_V2_DIR}/formatted_nwr_benchmark.jsonl"

DATASET_V3_DIR = "training_data_v3"
NWR_V3 = f"{DATASET_V3_DIR}/news_with_rationale.jsonl"
NWR_FORMATTED_V3 = f"{DATASET_V3_DIR}/news_with_rationale_formatted.jsonl"
NWR_TRAINING_V3 = f"{DATASET_V3_DIR}/formatted_nwr_training.jsonl"
NWR_BENCHMARK_V3 = f"{DATASET_V3_DIR}/formatted_nwr_benchmark.jsonl"
SUMMARY_V3 = f"{DATASET_V3_DIR}/summary.jsonl"
SUMMARY_FORMATTED_V3 = f"{DATASET_V3_DIR}/summary_formatted.jsonl"
ESSENTIALS_V3 = f"{DATASET_V3_DIR}/essential_aspects.jsonl"

BENCHMARK_PERCENTAGE = 0.2
