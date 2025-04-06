import json
import os
import sys
from tqdm import tqdm

import torch
from opencc import OpenCC
from transformers import AutoTokenizer, AutoModelForCausalLM

from .curriculum_utils import (
    load_curriculum_datasets,
    DifficultyLevels
)
from .constants import (
    MODEL_BASE,
    MODEL_DISTAL_FROM,
)
from utils import legalize_filename


def gen_zh_tw_response(
    model_base: str, model_distal_from: str, start_index: int = 0
) -> None:
    """
    Generate a translated response in Traditional Chinese (ZH-TW) using OpenCC.
    """
    # Initialize OpenCC for conversion
    cc = OpenCC('s2twp')

    DIR = os.path.dirname(os.path.abspath(__file__))
    filename = legalize_filename(
        f"generated_news_with_rationales_{model_distal_from}.jsonl"
    )
    save_filename = os.path.join(
        DIR,
        legalize_filename(f"generated_TO_ZHT_responses_{model_base}.jsonl")
    )

    datasets = load_curriculum_datasets(
        filename, DifficultyLevels.DIRECT_SUMMARY
    )
    datasets = datasets[start_index:]
    print(datasets[0])

    model = AutoModelForCausalLM.from_pretrained(model_base)
    model = model.to(torch.device("cuda"))
    tokenizer = AutoTokenizer.from_pretrained(model_base)

    if start_index == 0:
        # Clear the file if it exists
        with open(save_filename, "w", encoding="utf-8") as f:
            pass

    for i, (sys_prompt, user_prompt, _) in tqdm(enumerate(datasets)):
        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
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
                    "news": user_prompt,
                    "response_ZH_CN": response_zh_CN,
                    "response_ZH_TW": response_zh_TW
                },
                ensure_ascii=False
            ) + "\n")


if __name__ == "__main__":

    start_index = 0

    # parse argument --start_index idx
    for i, arg in enumerate(sys.argv):
        print(f"arg {i}: {arg}")
        if arg == "--start_index":
            start_index = int(sys.argv[i + 1])
            break

    print(f"Start index: {start_index}")

    # Generate the ZH-TW response
    gen_zh_tw_response(MODEL_BASE, MODEL_DISTAL_FROM)
