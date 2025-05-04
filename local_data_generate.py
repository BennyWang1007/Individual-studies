import json

import ollama
from ollama import ChatResponse, chat
from tqdm import tqdm
from transformers import AutoTokenizer

from crawler.utils import Logger
from curriculum_training.gen_zh_tw_response import gen_zh_tw_response
from curriculum_training.gen_zh_tw_response_vllm import gen_zh_tw_response_vllm
from curriculum_training.constants import (
    MODEL_BASE, MODEL_DISTAL_FROM, USE_VLLM, ALLOW_VLLM,
    MAX_INPUT_LENGTH, MAX_NEW_TOKENS,
)
from parse_generated_data import parse_response, load_response
from utils import (
    get_rationale_prompt_no_gt_chinese_system,
    get_rationale_prompt_no_gt_chinese_user,
    get_news_with_rationale_filename,
    get_zh_tw_filename,
    get_response_filename,
    load_udn_news,
    # int_set_str,
)

if ALLOW_VLLM:
    from utils_vllm import (
        init_vllm_model,
        filter_by_max_length,
        vllm_batch_generate,
    )

MODELNAME_OLLAMA = "qwen2.5:32b-instruct-q6_K"
MODELNAME_VLLM = "Qwen/Qwen2.5-32B-Instruct"

gen_logger = Logger("data_gen", verbose_level=3)

MODELNAME = MODELNAME_VLLM if USE_VLLM else MODELNAME_OLLAMA

RESPONSE_FILE = get_response_filename(MODELNAME)
NWR_FILE = get_news_with_rationale_filename(MODELNAME)
ZH_TW_FILE = get_zh_tw_filename(MODEL_BASE)


def local_gen_response(
    news_list: list[str], modelname: str, filename: str,
    id_list: list[int]
) -> list[dict]:

    assert len(news_list) == len(id_list)
    data: list[dict] = []

    if len(news_list) == 0:
        gen_logger.warning("No data need to generate.")
        return data

    if USE_VLLM:
        tokenizer = AutoTokenizer.from_pretrained(MODELNAME_VLLM)

        model, sampling_params = init_vllm_model(
            model_name=modelname,
            max_input_length=MAX_INPUT_LENGTH,
            max_new_tokens=MAX_NEW_TOKENS
        )

        prompts: list[str] = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": get_rationale_prompt_no_gt_chinese_system(
                            news
                        )
                    },
                    {
                        "role": "user",
                        "content": get_rationale_prompt_no_gt_chinese_user(
                            news
                        )
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for news in news_list
        ]

        # Filter out prompts that are too long
        prompts, id_list, news_list = filter_by_max_length(
            MAX_INPUT_LENGTH, prompts, id_list, news_list
        )

        responses = vllm_batch_generate(model, prompts, sampling_params)
        outputs = [response.outputs[0].text for response in responses]
        data = [
            {
                "id": id_list[i],
                "news": news,
                "response": response
            }
            for i, (news, response) in enumerate(zip(news_list, outputs))
        ]

    else:
        # make sure the model is downloaded
        ollama.pull(modelname)

        for i, news in tqdm(
            enumerate(news_list),
            total=len(news_list), desc="Generating response using Ollama",
        ):
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
        for dat in data:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    return data


def get_finished_id() -> tuple[set[int], set[int], set[int]]:
    """
    Get the finished news/NWR/zh-tw ids from the generated files.
    """
    # find finishded ids
    finished_news_ids: set[int] = set()
    try:
        with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                news = json.loads(line)
                if news["id"] in finished_news_ids:
                    gen_logger.warning(f"Duplicated news id: {news['id']}")
                    continue
                finished_news_ids.add(news["id"])
        gen_logger.info(f"Finished news ids count: {len(finished_news_ids)}")
    except FileNotFoundError:
        gen_logger.info(f"{RESPONSE_FILE} not found, starting from scratch.")
        finished_news_ids = set()

    # find finished NWR ids
    finished_NWR_ids: set[int] = set()
    try:
        with open(NWR_FILE, "r", encoding="utf-8") as f:
            for line in f:
                news = json.loads(line)
                if news["id"] in finished_NWR_ids:
                    gen_logger.warning(f"Duplicated NWR id: {news['id']}")
                    continue
                finished_NWR_ids.add(news["id"])
        gen_logger.info(f"Finished NWR ids count: {len(finished_NWR_ids)}")
    except FileNotFoundError:
        gen_logger.warning(f"{NWR_FILE} not found, starting from scratch.")
        finished_NWR_ids = set()

    # find finished zh-tw ids
    try:
        finished_zh_tw_ids: set[int] = set()
        with open(ZH_TW_FILE, "r", encoding="utf-8") as f:
            for line in f:
                news = json.loads(line)
                if news["id"] in finished_zh_tw_ids:
                    gen_logger.warning(f"Duplicated zh-tw id: {news['id']}")
                    continue
                finished_zh_tw_ids.add(news["id"])
        gen_logger.info(f"Finished zh-tw ids count: {len(finished_zh_tw_ids)}")
    except FileNotFoundError:
        gen_logger.warning(f"{ZH_TW_FILE} not found, starting from scratch.")
        finished_zh_tw_ids = set()

    finished_news_ids = finished_news_ids.union(finished_NWR_ids)

    return finished_news_ids, finished_NWR_ids, finished_zh_tw_ids


if __name__ == "__main__":

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
    fn = gen_zh_tw_response_vllm if USE_VLLM else gen_zh_tw_response
    fn(
        model_base=MODEL_BASE,
        model_distal_from=MODEL_DISTAL_FROM,
        finished_ids=finished_zh_tw_ids
    )
    gen_logger.info(
        f"Generated {news_count - len(finished_zh_tw_ids)} zh-tw responses"
    )
