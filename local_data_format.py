import json

import ollama
from opencc import OpenCC
from ollama import chat
from tqdm import tqdm

from crawler.utils import Logger
from curriculum_training.constants import (
    MODEL_DISTAL_FROM,
)
from news_with_rationale import NewsWithRationale
from utils import get_news_with_rationale_filename


FORMAT_MODEL = "qwen2.5:14b-instruct"
FORMAT_FILENAME = "formatted_nwrs.jsonl"

NWR_FILE = get_news_with_rationale_filename(MODEL_DISTAL_FROM)


def get_format_prompt(nwr: NewsWithRationale) -> str:
    """
    Get the format prompt for the given NewsWithRationale object.
    """
    return (
        f"請完成以下任務：\n"
        f"1. 你會收到若干個關鍵要素，請將每個要素以[]與、分隔，並移除不必要的符號，如 '1.'、'；'等。"
        f"2. 你會收到若干個三元組，請將其以頓號分隔，並移除不必要的符號，如 '1.'、'；'等。"
        f"範例：\n"
        f"關鍵要素：\n"
        f"[關鍵要素1]、[關鍵要素2]、[關鍵要素3]...\n"
        f"三元組：\n"
        f"[三元組1_1 | 三元組1_2 | 三元組1_3]、[三元組2_1 | 三元組2_2 | 三元組2_3]...\n"
        f"請將以下內容轉換為上述格式：\n"
        f"關鍵要素：\n"
        f"{nwr.essential_aspects_str()}\n"
        f"三元組：\n"
        f"{nwr.triples_str()}\n"
    )


def process_nwr(nwr: NewsWithRationale) -> str:
    """
    Process the given NewsWithRationale object and return the formatted string.
    """
    format_prompt = get_format_prompt(nwr)

    # Generate the response using the model
    gen_response = chat(
        model=FORMAT_MODEL,
        messages=[
            {"role": "system", "content": format_prompt},
            {"role": "user", "content": nwr.rationale_summary}
        ]
    )

    assert gen_response.message.content is not None

    # Return the generated response
    return gen_response.message.content


def local_data_format_main() -> None:
    """
    Main function to format the data using the model.
    """
    logger = Logger("data_format")
    cc = OpenCC("s2twp")  # Simplified Chinese to Traditional Chinese

    # Load the NewsWithRationale objects
    nwr_list: list[NewsWithRationale] = []
    finished_ids: set[int] = set()

    try:
        with open(FORMAT_FILENAME, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                finished_ids.add(data["id"])
    except FileNotFoundError:
        logger.info(f"{FORMAT_FILENAME} not found, starting from scratch.")
        pass

    with open(NWR_FILE, "r", encoding="utf-8") as f:
        for line in f:
            # nwr = NewsWithRationale.from_json(line)
            data = json.loads(line)
            nwr = NewsWithRationale.from_dict(data)
            if nwr.id in finished_ids:
                continue
            nwr_list.append(nwr)

    logger.info(f"Total NWR to process: {len(nwr_list)}")
    # print(f"Sample NWR: {nwr_list[0]}")

    ess_start = len("關鍵要素：\n[")
    tri_start = len("\n[")

    # Process each NewsWithRationale object
    for i, nwr in tqdm(enumerate(nwr_list), total=len(nwr_list)):
        formatted_response = process_nwr(nwr)
        formatted_response = cc.convert(formatted_response)
        try:
            ess_str, tri_str = formatted_response.split("三元組：")
            new_essentials = ess_str.split("]、[")
            new_essentials[0] = new_essentials[0][ess_start:]
            new_essentials[-1] = new_essentials[-1][:-2]

            new_triples = tri_str.split("]、[")
            new_triples[0] = new_triples[0][tri_start:]
            new_triples[-1] = new_triples[-1][:-1]
            for j in range(len(new_triples)):
                new_triples[j] = "[" + new_triples[j] + "]"

        except Exception as e:
            logger.error(f"Error processing NWR {i + 1}: {e}")
            logger.error(f"Formatted Response:\n{formatted_response}")
            continue

        nwr.triples = new_triples
        nwr.essential_aspects = new_essentials

        # exit()

        # Save the formatted response to a file
        with open(FORMAT_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(nwr.to_dict(), ensure_ascii=False) + "\n")

        # logger.info(f"Processed NWR {i + 1}/{len(nwr_list)}")


if __name__ == "__main__":
    ollama.pull(FORMAT_MODEL)
    local_data_format_main()
