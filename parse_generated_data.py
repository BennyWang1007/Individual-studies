
import json

from opencc import OpenCC

from news_with_rationale import NewsWithRationale
from rationale import Rationale
from summarized_news import SummarizedNews
from utils import legalize_filename, load_udn_news

from curriculum_training.constants import (
    MODEL_DISTAL_FROM
)


MODELNAME = MODEL_DISTAL_FROM
corrupted_response_ids = []


def load_response(
    filepath: str | None = None, model_name: str = MODELNAME
) -> list[dict]:

    if filepath is None:
        if model_name == "":
            raise ValueError("Either filepath or model_name must be provided.")
        filepath = legalize_filename(
            f"generated_responses_{model_name}.jsonl"
        )
        print(f"Loading from {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = []
        for line in f:
            raw_data.append(json.loads(line))
    return raw_data


def parse_response(responses: list[dict]) -> list[NewsWithRationale]:

    global corrupted_response_ids
    cc = OpenCC('s2twp')
    data = []

    news: list[str] = load_udn_news()

    for i, d in enumerate(responses):
        if i in corrupted_response_ids:
            continue

        # remove leading \r\n in d["news"]
        d["news"] = d["news"].strip()

        # s: 核心要素：\n.....\n三元組：\n.....\n生成摘要：\n.....
        s = d["response"]

        if "核心要素：" not in s or "三元組：" not in s or "生成摘要：" not in s:
            print(f"Corrupted response at index {i}")
            corrupted_response_ids.append(i)
            continue

        idx1, idx2, idx3 = s.find("核心要素："), s.find("三元組："), s.find("生成摘要：")

        critical_elements_str = s[idx1 + 5:idx2].strip()
        triples_str = s[idx2 + 4:idx3].strip()
        summary = s[idx3 + 5:].strip()

        critical_elements_str = cc.convert(critical_elements_str)
        triples_str = cc.convert(triples_str)
        summary = cc.convert(summary)

        # print(f'\n\n核心要素：\n{critical_elements_str}\n')
        # print(f'\n\n三元組：\n{triples_str}\n')
        # print(f'\n\n生成摘要：\n{summary}\n')

        essential_aspects: list[str] = critical_elements_str.split('\n')
        triples: list[str] = triples_str.split('\n')
        # print(f'{essential_aspects=}')
        # print(f'{triples=}')

        id = news.index(d['news'])
        sum_news = SummarizedNews(d['news'], summary, id, [-1])
        rationale = Rationale(essential_aspects, triples, summary)
        data.append(NewsWithRationale(sum_news, rationale))

    return data


if __name__ == "__main__":
    data = parse_response(load_response(model_name=MODELNAME))
    print(f"Corrupted response ids: {corrupted_response_ids}")
    print(f"Total data: {len(data)} responses")

    filename = legalize_filename(
        f"generated_news_with_rationales_{MODELNAME}.jsonl"
    )

    with open(filename, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d.__dict__, ensure_ascii=False))
            f.write('\n')
