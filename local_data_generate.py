import time
from tqdm import tqdm

import ollama
from ollama import ChatResponse, chat

from utils import *

# MODELNAME = "deepseek-r1:14b"  # 60~80 sec
# MODELNAME = "deepseek-r1:7b"  # 9.3 sec
# MODELNAME = "qwen:7b"         # 0.547 sec
# MODELNAME = "qwen:14b"        # 14.36 sec
# MODELNAME = "qwen:32b"        # 25.77 sec
# MODELNAME = "qwen2.5:32b"     # 26.7 sec
# MODELNAME = "qwen2.5:72b"     # TLE
MODELNAME = "qwen2.5:32b-instruct-q6_K" # 30.57 sec
# MODELNAME = "qwen2.5:32b-instruct-q8_0" # mem-full, 42.73 sec


def local_gen_response(news_list: list[str], modelname: str, filename: str) -> list[dict]:
    data: list[dict] = []

    for i, news in tqdm(enumerate(news_list)):
        sys_prompt = get_rationale_prompt_no_gt_chinese_system(news)
        user_prompt = get_rationale_prompt_no_gt_chinese_user(news)

        gen_response: ChatResponse = chat(model=modelname, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])
        dat = {"news": news, "response": gen_response.message.content}
        data.append(dat)

        # print(f"News {i+FINISHED_COUNT} done:")
        # print(f"News: {news}")
        # print(f"Response: {gen_response.message.content}\n")

        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    return data


def get_gen_time(sys_prompt: str, user_prompt: str, modelname: str) -> float:
    start_time = time.time()
    _ = chat(model=modelname, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])
    end_time = time.time()
    return end_time - start_time


def test_gen_time(test_count: int, modelname: str) -> float:
    prompt = get_prompt(test_news_list[0])
    sys_prompt = prompt.split("\n\n")[0]
    user_prompt = prompt.split("\n\n")[1]
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




TEST_TIME = True
TEST_TIME = False
TEST_COUNT = 10

test_news_list = load_udn_news()
FINISHED_COUNT = 230 + 313 + 2 + 676

print(f"{len(test_news_list)=}")


GENARATED_RESPONSE_FILE = legalize_filename(f"generated_responses_{MODELNAME}.jsonl")
print(f"{GENARATED_RESPONSE_FILE}")

if __name__ == "__main__":
    # ollama.pull(MODELNAME)

    #  generate response using local model
    test_news_list = test_news_list[FINISHED_COUNT:]
    _ = local_gen_response(test_news_list, MODELNAME, GENARATED_RESPONSE_FILE)

    # test response time
    if TEST_TIME:
        __ = test_gen_time(TEST_COUNT, MODELNAME)
