import json
import sys

from opencc import OpenCC
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

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

    # Initialize vLLM and tokenizer
    llm = LLM(
        model=model_base,
        dtype="bfloat16",
        max_model_len=MAX_INPUT_LENGTH + MAX_NEW_TOKENS,
        max_seq_len_to_capture=MAX_INPUT_LENGTH,
        task="generate",
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
    filtered_prompts = []
    filtered_ids = []
    filtered_news = []
    for i, prompt in enumerate(prompts):
        if len(prompt) > MAX_INPUT_LENGTH:
            continue
        filtered_prompts.append(prompt)
        filtered_ids.append(id_list[i])
        filtered_news.append(news_list[i])
    id_list = filtered_ids
    news_list = filtered_news
    prompts = filtered_prompts

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
    )
    outputs = llm.generate(prompts, sampling_params)

    for id, news, output in tqdm(
        zip(id_list, news_list, outputs),
        total=len(outputs), desc="Generating ZH-TW responses using vLLM",
    ):
        str_zh_cn = output.outputs[0].text
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
    for i, arg in enumerate(sys.argv):
        gen_logger.info(f"arg {i}: {arg}")
        if arg == "--start_index":
            start_index = int(sys.argv[i + 1])
            break

    gen_logger.info(f"Start index: {start_index}")
    finished_ids = set([i for i in range(start_index)])

    gen_zh_tw_response_vllm(MODEL_BASE, MODEL_DISTAL_FROM, finished_ids)
