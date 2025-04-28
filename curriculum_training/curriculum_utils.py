import contextlib
import gc
import json
from collections.abc import Callable
from functools import lru_cache

import torch
from enum import Enum

from .constants import GENARATED_ZH_TW_FILE, USE_VLLM
from crawler.utils import Logger
from news_with_rationale import NewsWithRationale
from rationale import Rationale
from summarized_news import SummarizedNews

if USE_VLLM:
    from utils_vllm import vllm_cleanup

LoaderFunc = Callable[[NewsWithRationale], tuple[str, str, str]]

curriculum_utils_logger = Logger("curriculum_utils", verbose_level=3)


class DifficultyLevels(Enum):
    TO_ZHT = 0
    ESSENTIAL_ASPECTS = 1
    TRIPLES = 2
    SUMMARY = 3
    DIRECT_SUMMARY = 4


PREFIX_OF_DIFFICULTY_LEVELS = {
    DifficultyLevels.TO_ZHT: "請為新聞生成摘要：",
    DifficultyLevels.ESSENTIAL_ASPECTS: "請提取新聞中的核心要素：",
    DifficultyLevels.TRIPLES: "請根據提供的核心要素，提取新聞中的三元組：",
    DifficultyLevels.SUMMARY: "請根據提供的核心要素和三元組，為新聞生成摘要：",
    DifficultyLevels.DIRECT_SUMMARY: "請為新聞生成摘要：",
}

# PREFIX_OF_DIFFICULTY_LEVELS = {
#     DifficultyLevels.TO_ZHT:
#     "You are a helpful assistant. "
#     "Please generate a summary of the news article in Chinese. "
#     "The summary should be concise and informative.",
#     DifficultyLevels.ESSENTIAL_ASPECTS:
#     "You are a helpful assistant. "
#     "Please extract the essential aspects of the news article. "
#     "The essential aspects should be concise and informative.",
#     DifficultyLevels.TRIPLES:
#     "You are a helpful assistant. "
#     "Please extract the triples from the essential aspects of the news "
#     "article. "
#     "The triples should be in the format [ENTITY1 | RELATION | ENTITY2]",
#     DifficultyLevels.SUMMARY:
#     "You are a helpful assistant. "
#     "Please generate a summary of the news article based on the essential "
#     "aspects and triples. The summary should be concise and informative.",
#     DifficultyLevels.DIRECT_SUMMARY:
#     "You are a helpful assistant. "
#     "Please generate a summary of the news article. "
#     "The summary should be concise and informative."
# }


@lru_cache(maxsize=1)
def load_generated_new_with_rationale(
    filepath: str = "generated_news_with_rationales.jsonl"
) -> list[NewsWithRationale]:
    data: list[NewsWithRationale] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            dat = json.loads(line)
            data.append(NewsWithRationale(
                SummarizedNews(
                    dat['article'], dat['summary'], dat['id'], dat['label']
                ),
                Rationale(
                    dat['essential_aspects'], dat['triples'], dat['summary']
                )
            ))

    curriculum_utils_logger.info(f"Loaded {len(data)} news with rationale")
    return data


def load_curriculum_datasets(
    dataset_name: str,
    difficulty_levels: DifficultyLevels,
    finished_ids: set[int] | None = None
) -> list[tuple[str, str, str]]:
    """
    Load datasets with increasing difficulty.
    Returns a list of datasets with [system, input, output] pairs.
    """
    data: list[NewsWithRationale] = load_generated_new_with_rationale(
        dataset_name
    )
    # print(f"Loaded {len(data)} news with rationale")

    if finished_ids is not None:
        data = [d for d in data if d.id not in finished_ids]
    # print(f"Filtered {len(data)} news with rationale")

    sys_str = PREFIX_OF_DIFFICULTY_LEVELS[difficulty_levels]

    if difficulty_levels == DifficultyLevels.TO_ZHT:
        ret: list[tuple[str, str, str]] = []
        with open(GENARATED_ZH_TW_FILE, "r", encoding="utf-8") as f:
            for line in f:
                dat = json.loads(line)
                ret.append((
                    sys_str, f"新聞：\n{dat['news']}", dat['response_zh-tw']
                ))
        return ret

    # loader_fn: dict[DifficultyLevels, LoaderFunc] = {
    #     DifficultyLevels.ESSENTIAL_ASPECTS: lambda d: (
    #         sys_str, d.article_full_str(), d.essential_aspects_full_str()
    #     ),
    #     DifficultyLevels.TRIPLES: lambda d: (
    #         sys_str,
    #         f'{d.article_full_str()}\n\n{d.essential_aspects_full_str()}',
    #         d.triples_str()
    #     ),
    #     DifficultyLevels.SUMMARY: lambda d: (
    #         sys_str,
    #         f'{d.article_full_str()}\n\n{d.essential_aspects_full_str()}\n\n'
    #         f'{d.triples_full_str()}',
    #         d.summary
    #     ),
    #     DifficultyLevels.DIRECT_SUMMARY:
    #     lambda d: (sys_str, d.article_full_str(), "新聞摘要：\n" + d.summary)
    # }

    loader_fn: dict[DifficultyLevels, LoaderFunc] = {
        DifficultyLevels.ESSENTIAL_ASPECTS: lambda d: (
            sys_str, f"新聞：\n{d.article}", f"核心要素：\n{d.essential_aspects_str()}"
        ),
        DifficultyLevels.TRIPLES: lambda d: (
            sys_str,
            f"新聞：\n{d.article}\n\n核心要素：\n{d.essential_aspects_str()}",
            f"三元組：\n{d.triples_str()}",
        ),
        DifficultyLevels.SUMMARY: lambda d: (
            sys_str,
            (
                f"新聞：{d.article}\n\n"
                f"核心要素：\n{d.essential_aspects_str()}\n\n"
                f"三元組：\n{d.triples_str()}"
            ),
            f"新聞總結：{d.summary}",
        ),
        DifficultyLevels.DIRECT_SUMMARY: lambda d: (
            sys_str, f"新聞：{d.article}", f"新聞總結：{d.summary}",
        ),
    }

    return [loader_fn[difficulty_levels](d) for d in data]


def cleanup():
    if USE_VLLM:
        vllm_cleanup()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
