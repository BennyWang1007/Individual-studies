import json
from tqdm import tqdm

import ollama
from ollama import ChatResponse, chat

from curriculum_training.gen_CHT_response import gen_zh_tw_response
from curriculum_training.constants import (
    MODEL_BASE,
    MODEL_DISTAL_FROM,
)
from parse_generated_data import parse_response

from utils import (
    get_rationale_prompt_no_gt_chinese_system,
    get_rationale_prompt_no_gt_chinese_user,
    legalize_filename,
    load_udn_news
)

# MODELNAME = "deepseek-r1:14b"  # 60~80 sec
# MODELNAME = "deepseek-r1:7b"  # 9.3 sec
# MODELNAME = "qwen:7b"         # 2.46 sec
# MODELNAME = "qwen:14b"        # 14.36 sec
# MODELNAME = "qwen:32b"        # 25.77 sec
# MODELNAME = "qwen2.5:32b"     # 26.7 sec
# MODELNAME = "qwen2.5:72b"     # TLE
MODELNAME = "qwen2.5:32b-instruct-q6_K"  # 94.55 sec
# MODELNAME = "qwen2.5:32b-instruct-q8_0" # mem-full, 42.73 sec


def local_gen_response(
    news_list: list[str], modelname: str, filename: str
) -> list[dict]:

    data: list[dict] = []

    for i, news in tqdm(enumerate(news_list)):
        sys_prompt = get_rationale_prompt_no_gt_chinese_system(news)
        user_prompt = get_rationale_prompt_no_gt_chinese_user(news)

        gen_response: ChatResponse = chat(
            model=modelname,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        dat = {"news": news, "response": gen_response.message.content}
        data.append(dat)

        # print(f"News {i+FINISHED_COUNT} done:")
        # print(f"News: {news}")
        # print(f"Response: {gen_response.message.content}\n")

        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    return data


GENARATED_RESPONSE_FILE = legalize_filename(
    f"generated_responses_{MODELNAME}.jsonl"
)
# print(f"{GENARATED_RESPONSE_FILE}")

GENARATED_NWR_FILE = legalize_filename(
    f"generated_news_with_rationales_{MODELNAME}.jsonl"
)
# print(f"{GENARATED_NWR_FILE}")

# FINISHED_COUNT = 230 + 313 + 2 + 676
# FINISHED_COUNT = 4740

# number of lines in GENARATED_RESPONSE_FILE
FINISHED_NEWS_COUNT = 0
with open(GENARATED_RESPONSE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        FINISHED_NEWS_COUNT += 1
print(f"Finished count: {FINISHED_NEWS_COUNT}")

# number of lines in GENARATED_NWR_FILE
FINISHED_NWR_COUNT = 0
with open(GENARATED_NWR_FILE, "r", encoding="utf-8") as f:
    for line in f:
        FINISHED_NWR_COUNT += 1
print(f"Finished response count: {FINISHED_NWR_COUNT}")


if __name__ == "__main__":
    # make sure the model is downloaded
    ollama.pull(MODELNAME)

    # load the news
    test_news_list = load_udn_news()
    test_news_list = test_news_list[FINISHED_NEWS_COUNT:]
    print(f"{len(test_news_list)=}")

    # generate response using local model
    responses = local_gen_response(
        test_news_list, MODELNAME, GENARATED_RESPONSE_FILE
    )
    print(f"Generated {len(responses)} responses")

    # parse the response
    parsed_data: list = parse_response(responses)
    with open(GENARATED_NWR_FILE, "a", encoding="utf-8") as f:
        for d in parsed_data:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")

    print(f"Parsed {len(parsed_data)} responses")

    # generate zh-tw response MODEL_BASE and opencc
    gen_zh_tw_response(
        model_base=MODEL_BASE,
        model_distal_from=MODEL_DISTAL_FROM,
        start_index=FINISHED_NWR_COUNT,
    )
    print(f"Generated zh-tw responses from idx {FINISHED_NWR_COUNT}")
