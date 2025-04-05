
import json

from opencc import OpenCC

from news_with_rationale import NewsWithRationale
from rationale import Rationale
from summarized_news import SummarizedNews


corrupted_response_ids = []


def load_data(
    filepath: str = "generated_responses.jsonl"
) -> list[NewsWithRationale]:

    raw_data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    cc = OpenCC('s2twp')

    data = []

    for i, d in enumerate(raw_data):
        if i in corrupted_response_ids:
            continue

        # remove leading \r\n in d["news"]
        d["news"] = d["news"].strip()

        s = d["response"]

        if "核心要素：" not in s or "三元組：" not in s or "生成摘要：" not in s:
            print(f"Corrupted response at index {i}")
            corrupted_response_ids.append(i)
            continue

        # s:  核心要素：\n.....\n三元組：\n.....\n生成摘要：\n.....
        critical_elements_str = s[s.find("核心要素：") + 5:s.find("三元組：")].strip()
        triples_str = s[s.find("三元組：") + 4:s.find("生成摘要：")].strip()
        summary = s[s.find("生成摘要：") + 5:].strip()

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

        sum_news = SummarizedNews(d['news'], summary, i, [-1])
        rationale = Rationale(essential_aspects, triples, summary)
        data.append(NewsWithRationale(sum_news, rationale))

    return data


if __name__ == "__main__":
    data = load_data("generated_responses_qwen2.5_32b-instruct-q6_K.jsonl")
    print(f"Corrupted response ids: {corrupted_response_ids}")
    print(f"Total data: {len(data)}")

    filename = "generated_news_with_rationales_qwen2.5_32b-instruct-q6_K.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d.__dict__, ensure_ascii=False))
            f.write('\n')
