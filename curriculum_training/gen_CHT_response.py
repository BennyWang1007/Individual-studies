import json
from tqdm import tqdm

import torch
from opencc import OpenCC
from transformers import AutoTokenizer, AutoModelForCausalLM

from curriculum_utils import load_curriculum_datasets, DifficultyLevels, MODEL_BASE, MODEL_BASE_OLLAMA, MODEL_DISTAL_FROM
from utils import legalize_filename, get_rationale_prompt_no_gt_chinese_system, get_rationale_prompt_no_gt_chinese_user


if __name__ == "__main__":

    cc = OpenCC('s2twp')

    filename = legalize_filename(f"generated_news_with_rationales_{MODEL_DISTAL_FROM}.jsonl")
    save_filename = legalize_filename(f"generated_TO_ZHT_responses_{MODEL_BASE}.jsonl")

    # news_without_output = load_curriculum_datasets(filename, DifficultyLevels.TO_ZHT)
    news_without_output = load_curriculum_datasets(filename, DifficultyLevels.DIRECT_SUMMARY)

    model = AutoModelForCausalLM.from_pretrained(MODEL_BASE).to(torch.device("cuda"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    with open(save_filename, "w", encoding="utf-8") as f:
        pass

    for i, (sys_prompt, user_prompt, out_str) in tqdm(enumerate(news_without_output)):

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
            max_new_tokens=512,
            # temperature=
        )
        
        response = generated_ids[0][len(model_inputs["input_ids"][0]):]
        response_in_ZH_CN = tokenizer.decode(response, skip_special_tokens=True)
        response_in_ZH_TW = cc.convert(response_in_ZH_CN)

        with open(save_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps({"news": user_prompt, "response_ZH_CN": response_in_ZH_CN, "response_ZH_TW": response_in_ZH_TW}, ensure_ascii=False) + "\n")
