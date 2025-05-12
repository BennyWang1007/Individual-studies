import json

import ollama
import re
from opencc import OpenCC
from ollama import chat
from tqdm import tqdm

from crawler.utils import Logger
from curriculum_training.constants import (
    FORMATTED_NWR_FILE,
    FORMATTED_NWR_FILE2,
    FORMATTED_NWR_FILE_V2,
    NWR_V2,
    FORMATTED_NWR_FILE_V2_2,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    USE_VLLM,
    ALLOW_VLLM,
)
from news_with_rationale import NewsWithRationale
from utils import int_set_str


if ALLOW_VLLM:
    from transformers import AutoTokenizer
    from utils_vllm import (
        init_vllm_model,
        filter_by_max_length,
        vllm_batch_generate,
    )

FORMAT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
FORMAT_MODEL_OLLAMA = "qwen2.5:32b-instruct"

FILE_TO_FORMAT = FORMATTED_NWR_FILE  # Input file
FORMATTED_FILE = FORMATTED_NWR_FILE2  # Output file

FILE_TO_FORMAT2 = FORMATTED_NWR_FILE_V2
FILE_TO_FORMAT2 = NWR_V2
FORMATTED_FILE2 = FORMATTED_NWR_FILE_V2_2

logger = Logger("data_format")
cc = OpenCC("s2twp")  # Simplified Chinese to Traditional Chinese


def get_format_system_prompt() -> str:
    """
    Get the format system prompt for the model.
    """
    return (
        "你會收到一個新聞文章以及其摘要，請評估該摘要是否為良好的摘要。\n"
        "若摘要良好（符合且文章大致內容），請輸出\"符合\"並將無關的內容移除。\n"
        "若摘要出現重複、雜亂的訊息、有未完成的句子，或是摘要本身不完整，請輸出\"不符合\"。\n"
        "範例輸入：\n"
        "新聞：\n"
        "新聞內容\n\n"
        "摘要：\n"
        "摘要內容：\n\n"
        "範例輸出：\n"
        "符合\n"
        "乾淨版摘要\n\n"
        "或者\n"
        "不符合"
    )


def get_format_user_prompt(nwr: NewsWithRationale) -> str:
    """
    Get the format user prompt for the given NewsWithRationale object.
    """
    return (
        "請評估以下內容的摘要是否是良好的新聞摘要：\n"
        "新聞：\n"
        f"{nwr.article}\n\n"
        "摘要：\n"
        f"{nwr.summary}\n\n"
    )


def process_nwr_ollama(nwr: NewsWithRationale) -> str:
    """
    Process the given NewsWithRationale object and return the formatted string.
    """
    sys_prompt = get_format_system_prompt()
    user_prompt = get_format_user_prompt(nwr)

    # Generate the response using the model from Ollama
    gen_response = chat(
        model=FORMAT_MODEL_OLLAMA,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    assert gen_response.message.content is not None

    return gen_response.message.content


def get_finished_id(format_file: str) -> set[int]:
    """
    Get the finished news/NWR/zh-tw ids from the generated files.
    """
    finished_ids = set()
    try:
        with open(format_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                finished_ids.add(data["id"])
    except FileNotFoundError:
        logger.info(f"{format_file} not found, starting from scratch.")
        pass

    return finished_ids


def read_nwr_file(filename: str, finished_ids) -> list[NewsWithRationale]:
    """
    Read the NewsWithRationale objects from the given file.
    """
    nwr_list: list[NewsWithRationale] = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            nwr = NewsWithRationale.from_dict(data)
            if nwr.id in finished_ids:
                continue
            nwr_list.append(nwr)

    logger.info(f"Loaded {len(nwr_list)} NWR from {filename}")
    return nwr_list


def extract_acceptance(response: str) -> tuple[bool, str]:
    pattern = r"^(符合)(?:\n乾淨版摘要：)?\n(.+)|^(不符合)"
    match = re.match(pattern, response)
    if match:
        if match.group(1) is not None:
            clean_summary = match.group(2)
            return True, clean_summary
        else:
            return False, ""
    else:
        logger.error(f"Invalid response format: {response}")
        return False, ""


def local_data_format_summary_main(file_to_format, file_to_save) -> None:
    """ Main function to format the data using the model. """

    # Load the finished ids from the formatted file
    finished_ids = get_finished_id(file_to_save)
    logger.info(f"Finished ids count: {len(finished_ids)}")
    logger.info(f"Finished ids: {int_set_str(finished_ids)}")

    finished_ids = set()  # For testing or reset purposes

    # Load the NewsWithRationale excluding the finished ids
    nwr_list = read_nwr_file(file_to_format, finished_ids)
    # nwr_list = nwr_list[:1000]  # For testing purposes

    logger.info(f"Total NWR to process: {len(nwr_list)}")

    ids = [nwr.id for nwr in nwr_list if nwr.id not in finished_ids]
    assert len(ids) == len(nwr_list)

    logger.info(f"Remains NWR ids: {int_set_str(set(ids))}")

    output_strs: list[str] = []
    if USE_VLLM:
        sys_prompt = get_format_system_prompt()
        model, sampling_params = init_vllm_model(
            FORMAT_MODEL, MAX_INPUT_LENGTH, MAX_NEW_TOKENS
        )
        tokenizer = AutoTokenizer.from_pretrained(FORMAT_MODEL)
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": get_format_user_prompt(nwr)}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for nwr in nwr_list
        ]

        # Filter out prompts that are too long
        prompts, nwr_list = filter_by_max_length(
            MAX_INPUT_LENGTH, prompts, nwr_list
        )

        responses = vllm_batch_generate(model, prompts, sampling_params)
        output_strs = [response.outputs[0].text for response in responses]

    else:
        # Make sure the model is downloaded
        logger.info(f"Pulling model {FORMAT_MODEL_OLLAMA}...")
        ollama.pull(FORMAT_MODEL_OLLAMA)
        desc = "Generating NWRs with Ollama"
        for nwr in tqdm(nwr_list, total=len(nwr_list), desc=desc):
            # Process each NewsWithRationale object
            formatted_response = process_nwr_ollama(nwr)
            output_strs.append(formatted_response)

    accepted_nwrs: list[NewsWithRationale] = []
    for (nwr, output) in zip(nwr_list, output_strs):
        success, summ = extract_acceptance(output)
        if success:
            nwr.essential_aspects = [
                cc.convert(ess.strip()) for ess in nwr.essential_aspects
            ]
            nwr.triples = [cc.convert(tri.strip()) for tri in nwr.triples]
            nwr.summary = cc.convert(summ.strip())
            nwr.rationale_summary = nwr.summary

            accepted_nwrs.append(nwr)

    logger.info(f"Accepted NWRs count: {len(accepted_nwrs)}")

    with open(file_to_save, "a", encoding="utf-8") as f:
        for nwr in accepted_nwrs:
            f.write(json.dumps(nwr.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # local_data_format_summary_main(FILE_TO_FORMAT, FORMATTED_FILE)
    local_data_format_summary_main(FILE_TO_FORMAT2, FORMATTED_FILE2)
