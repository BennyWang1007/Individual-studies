import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from news_with_rationale import NewsWithRationale
from rationale import Rationale
from summarized_news import SummarizedNews

def load_generated_new_with_rationale(filepath: str = "generated_news_with_rationales.jsonl") -> list[NewsWithRationale]:
    data: list[NewsWithRationale] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            dat = json.loads(line)
            rationale: Rationale = Rationale(dat['essential_aspects'], dat['triples'], dat['summary'])
            summarized_news = SummarizedNews(dat['article'], dat['summary'], dat['id'], dat['label'])
            data.append(NewsWithRationale(summarized_news, rationale))

    return data