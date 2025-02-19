
from transformers import PreTrainedTokenizer

from news_with_rationale import NewsWithRationale

MAX_LEN = 1024
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

def load_data(filepath) -> list:
    # news = NewsWithRationale.load_all()
    # data = [{"input_text": "文章：\n" + n.article + "\n摘要：\n", "output_text": n.summary} for n in news]
    news = NewsWithRationale.load_all()
    # print(news[0])
    data = [{"input_text": "文章：\n" + n.article + "\n摘要：\n", "output_text": n.summary} for n in news]
    return data

