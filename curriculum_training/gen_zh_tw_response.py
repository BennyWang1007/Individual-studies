import json
import sys
from tqdm import tqdm

import torch
from opencc import OpenCC
from transformers import AutoTokenizer, AutoModelForCausalLM

from .constants import (
    MODEL_BASE,
    MODEL_DISTAL_FROM,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
)
from .curriculum_utils import (
    DifficultyLevels as DL,
    PREFIX_OF_DIFFICULTY_LEVELS,
)
from utils import get_zh_tw_filename, load_udn_news
from crawler.utils import Logger

gen_logger = Logger("data_gen", verbose_level=3)
gen_logger.info(f"CUDA allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")


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

    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DL.DIRECT_SUMMARY]

    model = AutoModelForCausalLM.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.padding_side = "left"

    batch_size = 8  # Define the batch size
    for i in tqdm(
        range(0, len(news_list), batch_size),
        total=(len(news_list) + batch_size - 1) // batch_size
    ):
        batch_news = news_list[i:i + batch_size]
        batch_ids = id_list[i:i + batch_size]

        messages = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "article:\n" + news}
            ]
            for news in batch_news
        ]
        texts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            for message in messages
        ]

        model_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
            )

        for idx, (id, news, generated_id) in enumerate(
            zip(batch_ids, batch_news, generated_ids)
        ):
            response = generated_id[len(model_inputs["input_ids"][idx]):]
            str_zh_cn = tokenizer.decode(response, skip_special_tokens=True)
            str_zh_tw = cc.convert(str_zh_cn)

            with open(save_filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {
                        "id": id,
                        "news": news,
                        "response_zh-cn": str_zh_cn,
                        "response_zh-tw": str_zh_tw
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
