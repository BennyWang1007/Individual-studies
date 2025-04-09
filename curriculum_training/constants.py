from utils import (
    get_response_filename,
    get_news_with_rationale_filename,
    get_zh_tw_filename,
)

MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_BASE_OLLAMA = "qwen2.5:0.5b-instruct"
MODEL_DISTAL_FROM = "qwen2.5:32b-instruct-q6_K"

GENARATED_RESPONSE_FILE = get_response_filename(MODEL_DISTAL_FROM)
GENARATED_NWR_FILE = get_news_with_rationale_filename(MODEL_DISTAL_FROM)
GENARATED_ZH_TW_FILE = get_zh_tw_filename(MODEL_BASE)
