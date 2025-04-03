import time
from tqdm import tqdm

import ollama
from ollama import ChatResponse, chat

from utils import *

# MODELNAME = "deepseek-r1:14b"  # 60~80 sec
# MODELNAME = "deepseek-r1:7b"  # 9.3 sec
# MODELNAME = "qwen:7b"         # 2.46 sec
# MODELNAME = "qwen:14b"        # 14.36 sec
# MODELNAME = "qwen:32b"        # 25.77 sec
# MODELNAME = "qwen2.5:32b"     # 26.7 sec
# MODELNAME = "qwen2.5:72b"     # TLE
MODELNAME = "qwen2.5:32b-instruct-q6_K" # 94.55 sec
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


# FINISHED_COUNT = 230 + 313 + 2 + 676
FINISHED_COUNT = 1201

test_news_list = load_udn_news()
test_news_list = test_news_list[FINISHED_COUNT:]

print(f"{len(test_news_list)=}")


GENARATED_RESPONSE_FILE = legalize_filename(f"generated_responses_{MODELNAME}.jsonl")
print(f"{GENARATED_RESPONSE_FILE}")

if __name__ == "__main__":
    # ollama.pull(MODELNAME)

    #  generate response using local model
    test_news_list = test_news_list[FINISHED_COUNT:]
    _ = local_gen_response(test_news_list, MODELNAME, GENARATED_RESPONSE_FILE)
