import json
import sys
from tqdm import tqdm

import torch
from opencc import OpenCC
from transformers import AutoTokenizer, AutoModelForCausalLM

from .constants import (
    MODEL_BASE,
    MODEL_DISTAL_FROM,
)
from .curriculum_utils import (
    DifficultyLevels,
    PREFIX_OF_DIFFICULTY_LEVELS
)
from utils import get_zh_tw_filename, load_udn_news
from crawler.utils import Logger

gen_logger = Logger("data_gen", verbose_level=3)


def gen_zh_tw_response(
    model_base: str, model_distal_from: str,
    finished_ids: set[int] = set(),
) -> None:
    """
    Generate a translated response in Traditional Chinese (ZH-TW) using OpenCC.
    """
    # Initialize OpenCC for conversion
    cc = OpenCC('s2twp')

    # filename = get_news_with_rationale_filename(model_distal_from)
    save_filename = get_zh_tw_filename(model_base)

    news_list: list[str] = load_udn_news()

    id_list = [i for i in range(len(news_list)) if i not in finished_ids]
    news_list = [news_list[i] for i in id_list]

    if len(news_list) == 0:
        gen_logger.warning("No data need to generate.")
        return

    assert len(news_list) == len(id_list)

    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[
        DifficultyLevels.DIRECT_SUMMARY.value
    ]

    model = AutoModelForCausalLM.from_pretrained(model_base)
    model = model.to(torch.device("cuda"))
    tokenizer = AutoTokenizer.from_pretrained(model_base)

    for id, news in tqdm(zip(id_list, news_list), total=len(news_list)):
        # print(f"Processing id: {id}")
        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": news}
        ]
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            # temperature=
        )

        response = generated_ids[0][len(model_inputs["input_ids"][0]):]
        response_zh_CN = tokenizer.decode(response, skip_special_tokens=True)
        response_zh_TW = cc.convert(response_zh_CN)

        with open(save_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {
                    "id": id,
                    "news": news,
                    "response_ZH_CN": response_zh_CN,
                    "response_ZH_TW": response_zh_TW
                },
                ensure_ascii=False
            ) + "\n")


if __name__ == "__main__":

    start_index = 0

    # parse argument --start_index idx
    for i, arg in enumerate(sys.argv):
        gen_logger.info(f"arg {i}: {arg}")
        if arg == "--start_index":
            start_index = int(sys.argv[i + 1])
            break

    gen_logger.info(f"Start index: {start_index}")
    finished_ids = set([i for i in range(start_index)])

    # Generate the ZH-TW response
    gen_zh_tw_response(MODEL_BASE, MODEL_DISTAL_FROM, finished_ids)
