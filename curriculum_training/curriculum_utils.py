import json
import os
import sys

from enum import Enum
from opencc import OpenCC

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from news_with_rationale import NewsWithRationale
from rationale import Rationale
from summarized_news import SummarizedNews


MODEL_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_BASE_OLLAMA = "qwen2.5:0.5b-instruct"
MODEL_DISTAL_FROM = "qwen2.5:32b-instruct-q6_K"


class DifficultyLevels(Enum):
    TO_ZHT = 0
    ESSENTIAL_ASPECTS = 1
    TRIPLES = 2
    SUMMARY = 3
    DIRECT_SUMMARY = 4


PREFIX_OF_DIFFICULTY_LEVELS = ["請為新聞生成摘要：", "請提取新聞中的核心要素：", "請根據提供的核心要素，提取新聞中的三元組：", "請根據提供的核心要素和三元組，為新聞生成摘要：", "請為新聞生成摘要："]



def load_generated_new_with_rationale(filepath: str = "generated_news_with_rationales.jsonl") -> list[NewsWithRationale]:
    data: list[NewsWithRationale] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            dat = json.loads(line)
            rationale: Rationale = Rationale(dat['essential_aspects'], dat['triples'], dat['summary'])
            summarized_news = SummarizedNews(dat['article'], dat['summary'], dat['id'], dat['label'])
            data.append(NewsWithRationale(summarized_news, rationale))
    # print(f"Loaded {len(data)} news with rationale")
    return data


def load_curriculum_datasets(dataset_name, difficulty_levels: DifficultyLevels) -> list[tuple[str, str, str]]:
    """ Load datasets with increasing difficulty. Returns a list of datasets with [system, input, output] pairs. """

    data: list[NewsWithRationale] = load_generated_new_with_rationale(dataset_name)
    ret: list[tuple[str, str, str]] = []
    
    system_str = PREFIX_OF_DIFFICULTY_LEVELS[difficulty_levels.value]

    match difficulty_levels:
        case DifficultyLevels.TO_ZHT:
            # cc = OpenCC('s2twp')
            # for d in data:
            #     ret.append((d.summary, cc.convert(d.summary)))
            # for d in data:
            #     # ret.append((d.article, ""))
            #     ret.append((system_str, d.article, ""))
            with open("curriculum_training/generated_TO_ZHT_responses_Qwen_Qwen2.5-0.5B-Instruct.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    dat = json.loads(line)
                    ret.append((system_str, dat['news'], dat['response_ZH_TW']))
                
        case DifficultyLevels.ESSENTIAL_ASPECTS:
            for d in data:
                ret.append((system_str, d.article, ", ".join(d.essential_aspects)))
        case DifficultyLevels.TRIPLES:
            for d in data:
                ret.append((system_str, d.article + "\n\n核心要素：\n\n" + ", ".join(d.essential_aspects), ", ".join(d.triples)))
        case DifficultyLevels.SUMMARY:
            for d in data:
                ret.append((system_str, d.article + "\n\n核心要素：\n\n" + ", ".join(d.essential_aspects) + "\n\n三元組：\n\n" + ", ".join(d.triples), d.summary))
        case DifficultyLevels.DIRECT_SUMMARY:
            for d in data:
                ret.append((system_str, d.article, d.summary))
        case _:
            raise Exception("Invalid difficulty level")
        

    return ret
