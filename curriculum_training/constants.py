from utils import (
    get_response_filename,
    get_news_with_rationale_filename,
    # get_formatted_nwr_filename,
    get_zh_tw_filename,
)

USE_VLLM = True

MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_BASE_OLLAMA = "qwen2.5:0.5b-instruct"
MODEL_DISTAL_FROM = "qwen2.5:32b-instruct-q6_K"

MAX_INPUT_LENGTH = 4096
MAX_NEW_TOKENS = 1024

GENARATED_RESPONSE_FILE = get_response_filename(MODEL_DISTAL_FROM)
GENARATED_NWR_FILE = get_news_with_rationale_filename(MODEL_DISTAL_FROM)
GENARATED_ZH_TW_FILE = get_zh_tw_filename(MODEL_BASE)

# FORMATTED_NWR_FILE = get_formatted_nwr_filename(MODEL_DISTAL_FROM)
FORMATTED_NWR_FILE = "formatted_nwr.jsonl"
NWR_TRAINING_FILE = "formatted_nwr_training.jsonl"
NWR_BENCHMARK_FILE = "formatted_nwr_benchmark.jsonl"

BENCHMARK_PERCENTAGE = 0.2
