import json
import sys

from opencc import OpenCC
from transformers import AutoTokenizer

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
from crawler.utils import Logger
from utils import get_zh_tw_filename, load_udn_news
from utils_vllm import (
    init_vllm_model,
    filter_by_max_length,
    vllm_batch_generate,
)

gen_logger = Logger("data_gen_vllm", verbose_level=3)


def gen_zh_tw_response_vllm(
    model_base: str, model_distal_from: str,
    finished_ids: set[int] = set(),
) -> None:
    """
    Generate a translated response in Traditional Chinese (ZH-TW) using vLLM
    and OpenCC.
    """
    cc = OpenCC('s2twp')
    save_filename = get_zh_tw_filename(model_base)
    news_list: list[str] = load_udn_news()

    id_list = [i for i in range(len(news_list)) if i not in finished_ids]
    news_list = [news_list[i] for i in id_list]

    if len(news_list) == 0:
        gen_logger.warning("No data need to generate.")
        return

    assert len(news_list) == len(id_list)

    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DL.DIRECT_SUMMARY]

    # Initialize vLLM, tokenizer and sampling parameters
    llm, sampling_params = init_vllm_model(
        model_base, MAX_INPUT_LENGTH, MAX_NEW_TOKENS
    )
    tokenizer = AutoTokenizer.from_pretrained(model_base)

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "article:\n" + news}
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
    gen_logger.info(f"Filtered prompts: {len(prompts)}")

    responses = vllm_batch_generate(llm, prompts, sampling_params)
    outputs = [response.outputs[0].text for response in responses]
    response_zh_tws = [cc.convert(output) for output in outputs]

    gen_logger.info(f"Saving {len(outputs)} responses to {save_filename}")

    with open(save_filename, "a", encoding="utf-8") as f:
        for i in range(len(id_list)):
            f.write(json.dumps(
                {
                    "id": id_list[i],
                    "news": news_list[i],
                    "response_zh-cn": outputs[i],
                    "response_zh-tw": response_zh_tws[i]
                },
                ensure_ascii=False
            ) + "\n")


if __name__ == "__main__":
    start_index = 0
    for i, arg in enumerate(sys.argv):
        gen_logger.info(f"arg {i}: {arg}")
        if arg == "--start_index":
            start_index = int(sys.argv[i + 1])
            break

    gen_logger.info(f"Start index: {start_index}")
    finished_ids = set([i for i in range(start_index)])

    gen_zh_tw_response_vllm(MODEL_BASE, MODEL_DISTAL_FROM, finished_ids)
