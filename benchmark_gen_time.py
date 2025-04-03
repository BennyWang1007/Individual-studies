import time
from tqdm import tqdm

import ollama
from ollama import ChatResponse, chat

from utils import *

# MODELNAME = "qwen:7b"         # 2.46 sec
MODELNAME = "qwen2.5:32b-instruct-q6_K" # 94.55 sec

TEST_COUNT = 15


def get_gen_time(sys_prompt: str, user_prompt: str, modelname: str) -> float:
    start_time = time.time()
    _ = chat(model=modelname, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])
    end_time = time.time()
    return end_time - start_time


def get_mean_len_idx(news_list: list[str]) -> int:
    mean_len = sum([len(news) for news in news_list]) // len(news_list)
    for i, news in enumerate(news_list):
        if len(news) == mean_len:
            return i
    raise Exception("No news found with mean length.")


# BENCHMARK_NEWS_IDX = get_mean_len_idx(load_udn_news())  # length: 824
# print(f"Benchmark news index: {BENCHMARK_NEWS_IDX}, with length: {len(load_udn_news()[BENCHMARK_NEWS_IDX])}")

BENCHMARK_NEWS_IDX = 83 # with length: 824


def test_gen_time(test_count: int, modelname: str) -> float:
    test_news_list = load_udn_news()
    sys_prompt = get_rationale_prompt_no_gt_chinese_system(test_news_list[BENCHMARK_NEWS_IDX])
    user_prompt = get_rationale_prompt_no_gt_chinese_user(test_news_list[BENCHMARK_NEWS_IDX])
    # warn up
    dt = get_gen_time(sys_prompt, user_prompt, modelname)
    print(f"Warn up time: {dt} seconds")

    start_time = time.time()
    for _ in tqdm(range(test_count)):
        _ = get_gen_time(sys_prompt, user_prompt, modelname)
    end_time = time.time()
    print(f"Response time: {end_time - start_time} seconds")
    print(f"Average response time: {(end_time - start_time) / test_count} seconds")

    return end_time - start_time


if __name__ == "__main__":
    ollama.pull(MODELNAME)

    __ = test_gen_time(TEST_COUNT, MODELNAME)
