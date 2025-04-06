from utils import legalize_filename

MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_BASE_OLLAMA = "qwen2.5:0.5b-instruct"
MODEL_DISTAL_FROM = "qwen2.5:32b-instruct-q6_K"

GENARATED_RESPONSE_FILE = legalize_filename(
    f"generated_responses_{MODEL_DISTAL_FROM}.jsonl"
)
GENARATED_NWR_FILE = legalize_filename(
    f"generated_news_with_rationales_{MODEL_DISTAL_FROM}.jsonl"
)
