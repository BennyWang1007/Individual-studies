import json

import ollama
from ollama import ChatResponse, chat
from tqdm import tqdm

from crawler.utils import Logger
from curriculum_training.gen_zh_tw_response import gen_zh_tw_response
from curriculum_training.constants import MODEL_BASE, MODEL_DISTAL_FROM
from parse_generated_data import parse_response, load_response
from utils import (
    get_rationale_prompt_no_gt_chinese_system,
    get_rationale_prompt_no_gt_chinese_user,
    get_news_with_rationale_filename,
    get_zh_tw_filename,
    get_response_filename,
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

gen_logger = Logger("data_gen", verbose_level=3)

USE_VLLM = False

if USE_VLLM:
    from curriculum_training.gen_zh_tw_response_vllm import \
        gen_zh_tw_response_vllm


RESPONSE_FILE = get_response_filename(MODELNAME)
NWR_FILE = get_news_with_rationale_filename(MODELNAME)
ZH_TW_FILE = get_zh_tw_filename(MODEL_BASE)


def local_gen_response(
    news_list: list[str], modelname: str, filename: str,
    id_list: list[int]
) -> list[dict]:

    assert len(news_list) == len(id_list)
    data: list[dict] = []

    for i, news in tqdm(enumerate(news_list), total=len(news_list)):
        sys_prompt = get_rationale_prompt_no_gt_chinese_system(news)
        user_prompt = get_rationale_prompt_no_gt_chinese_user(news)

        gen_response: ChatResponse = chat(
            model=modelname,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        # dat = {"news": news, "response": gen_response.message.content}
        dat = {
            "id": id_list[i],
            "news": news,
            "response": gen_response.message.content
        }
        data.append(dat)

        # print(f"News {i+FINISHED_COUNT} done:")
        # print(f"News: {news}")
        # print(f"Response: {gen_response.message.content}\n")

        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    return data


def print_int_set(int_set) -> None:
    """
    Print the set of integers in a readable format.
    For example, if the set is {1, 2, 3, 5, 6, 7}, it will print "1-3, 5-7".
    """
    prev_id: int = -2
    continuous_count: int = 0
    id_list: list[int] = sorted(list(int_set))
    out_strs: list[str] = []
    for i in range(len(id_list)):
        if id_list[i] == prev_id + 1:
            prev_id += 1
            continuous_count += 1
            continue
        else:
            if continuous_count > 0:
                out_strs.append(f"{prev_id - continuous_count}-{prev_id}")
                continuous_count = 0
                prev_id = id_list[i]
            else:
                if prev_id != -2:
                    out_strs.append(f"{prev_id}")
                prev_id = id_list[i]

    if continuous_count > 0:
        out_strs.append(f"{prev_id - continuous_count}-{prev_id}")

    gen_logger.info(", ".join(out_strs))


def get_finished_id() -> tuple[set[int], set[int], set[int]]:
    """
    Get the finished news/NWR/zh-tw ids from the generated files.
    """
    # find finishded ids
    finished_news_ids: set[int] = set()
    with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            news = json.loads(line)
            if news["id"] in finished_news_ids:
                gen_logger.warning(f"Duplicated news id: {news['id']}")
                continue
            finished_news_ids.add(news["id"])
    gen_logger.info(f"Finished news ids count: {len(finished_news_ids)}")

    # find finished NWR ids
    finished_NWR_ids: set[int] = set()
    with open(NWR_FILE, "r", encoding="utf-8") as f:
        for line in f:
            news = json.loads(line)
            if news["id"] in finished_NWR_ids:
                gen_logger.warning(f"Duplicated NWR id: {news['id']}")
                continue
            finished_NWR_ids.add(news["id"])
    gen_logger.info(f"Finished NWR ids count: {len(finished_NWR_ids)}")

    finished_zh_tw_ids: set[int] = set()
    with open(ZH_TW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            news = json.loads(line)
            if news["id"] in finished_zh_tw_ids:
                gen_logger.warning(f"Duplicated zh-tw id: {news['id']}")
                continue
            finished_zh_tw_ids.add(news["id"])
    gen_logger.info(f"Finished zh-tw ids count: {len(finished_zh_tw_ids)}")

    finished_news_ids = finished_news_ids.union(finished_NWR_ids)

    return finished_news_ids, finished_NWR_ids, finished_zh_tw_ids


if __name__ == "__main__":
    # make sure the model is downloaded
    ollama.pull(MODELNAME)

    # get the finished ids
    finished_news_ids, finished_NWR_ids, finished_zh_tw_ids = get_finished_id()

    # load the news
    news_list: list[str] = load_udn_news()
    news_count: int = len(news_list)
    gen_logger.info(f"Loaded {news_count} news")

    # remove the finished news
    news_list = [
        news_list[i] for i in range(news_count) if i not in finished_news_ids
    ]
    ids: list[int] = [
        i for i in range(news_count) if i not in finished_news_ids
    ]
    gen_logger.info(f"Remains {len(news_list)} response to generate")
    # print_int_set(finished_news_ids)

    # generate response using local model
    responses = local_gen_response(news_list, MODELNAME, RESPONSE_FILE, ids)
    gen_logger.info(f"Generated {len(responses)} responses")

    # parse the response
    parsed_data = parse_response(load_response(model_name=MODELNAME))
    gen_logger.info(f"Parsed {len(parsed_data)} responses")
    with open(NWR_FILE, "w", encoding="utf-8") as f:
        for dat in parsed_data:
            f.write(json.dumps(dat.__dict__, ensure_ascii=False) + "\n")

    # generate zh-tw response MODEL_BASE and opencc
    if USE_VLLM:
        gen_zh_tw_response_vllm(
            model_base=MODEL_BASE,
            model_distal_from=MODEL_DISTAL_FROM,
            finished_ids=finished_zh_tw_ids
        )
    else:
        gen_zh_tw_response(
            model_base=MODEL_BASE,
            model_distal_from=MODEL_DISTAL_FROM,
            finished_ids=finished_zh_tw_ids
        )
    gen_logger.info(
        f"Generated {news_count - len(finished_zh_tw_ids)} zh-tw responses"
    )
