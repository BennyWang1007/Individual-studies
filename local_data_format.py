import json

import ollama
from opencc import OpenCC
from ollama import chat
from tqdm import tqdm
from vllm import LLM, SamplingParams

from crawler.utils import Logger
from curriculum_training.constants import (
    FORMATTED_NWR_FILE,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    MODEL_DISTAL_FROM,
    USE_VLLM,
)
from news_with_rationale import NewsWithRationale
from utils import get_news_with_rationale_filename, int_set_str

if USE_VLLM:
    from transformers import AutoTokenizer

FORMAT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
# FORMAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FORMAT_MODEL_OLLAMA = "qwen2.5:14b-instruct"
FORMAT_FILENAME = FORMATTED_NWR_FILE

NWR_FILE = get_news_with_rationale_filename(MODEL_DISTAL_FROM)
NWR_FILE2 = R"generated_nwr_grok.jsonl"

logger = Logger("data_format")
cc = OpenCC("s2twp")  # Simplified Chinese to Traditional Chinese

ESS_START = len("關鍵要素：\n[")
TRI_START = len("\n[")


def get_format_system_prompt() -> str:
    """
    Get the format system prompt for the model.
    """
    return (
        # "請完成以下任務：\n"
        "請將以下內容轉換為關鍵要素和三元組的格式：\n"
        "1. 你會收到若干個關鍵要素，請將每個要素以[]與、分隔，並移除不必要的符號，如 '1.'、'；'等。\n"
        "2. 你會收到若干個三元組，請將其以頓號分隔，並移除不必要的符號，如 '1.'、'；'等。\n"
        "範例：\n"
        "關鍵要素：\n"
        "[關鍵要素1]、[關鍵要素2]、[關鍵要素3]...\n"
        "三元組：\n"
        "[三元組1_1 | 三元組1_2 | 三元組1_3]、[三元組2_1 | 三元組2_2 | 三元組2_3]..."
    )


def get_format_user_prompt(nwr: NewsWithRationale) -> str:
    """
    Get the format system/user prompt for the given NewsWithRationale object.
    """
    return (
        "請將以下內容轉換為指定格式：\n"
        "關鍵要素：\n"
        f"{nwr.essential_aspects_str()}\n"
        "三元組：\n"
        f"{nwr.triples_str()}"
    )


def process_nwr(nwr: NewsWithRationale) -> str:
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


def get_finished_id() -> set[int]:
    """
    Get the finished news/NWR/zh-tw ids from the generated files.
    """
    finished_ids = set()
    try:
        with open(FORMAT_FILENAME, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                finished_ids.add(data["id"])
    except FileNotFoundError:
        logger.info(f"{FORMAT_FILENAME} not found, starting from scratch.")
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


def extract_essentials_and_triples(
    response: str
) -> tuple[list[str], list[str]]:
    # Extract the essential aspects and triples from the response
    formatted_response = cc.convert(response)

    ess_str, tri_str = formatted_response.split("三元組：")
    new_essentials = ess_str.split("]、[")
    new_essentials[0] = new_essentials[0][ESS_START:]
    new_essentials[-1] = new_essentials[-1][:-2]

    new_triples = tri_str.split("]、[")
    new_triples[0] = new_triples[0][TRI_START:]
    new_triples[-1] = new_triples[-1][:-1]
    for j in range(len(new_triples)):
        new_triples[j] = "[" + new_triples[j] + "]"

    return new_essentials, new_triples


def local_data_format_main() -> None:
    """
    Main function to format the data using the model.
    """
    # Load the finished ids from the formatted file
    finished_ids = get_finished_id()
    logger.info(f"Finished ids count: {len(finished_ids)}")
    logger.info(f"Finished ids: {int_set_str(finished_ids)}")

    # Load the NewsWithRationale excluding the finished ids
    nwr_list = read_nwr_file(NWR_FILE, finished_ids)
    nwr_list2 = read_nwr_file(NWR_FILE2, finished_ids)

    nwr_list.extend(nwr_list2)
    # nwr_list = nwr_list[:10]  # for demonstration

    logger.info(f"Total NWR to process: {len(nwr_list)}")

    ids = [nwr.id for nwr in nwr_list if nwr.id not in finished_ids]
    assert len(ids) == len(nwr_list)

    logger.info(f"Remains {len(ids)} NWR to process")
    logger.info(f"Remains NWR ids: {int_set_str(set(ids))}")

    if USE_VLLM:
        sys_prompt = get_format_system_prompt()
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
        filtered_prompts = []
        filtered_nwrs = []
        for i, prompt in enumerate(prompts):
            if len(prompt) > MAX_INPUT_LENGTH:
                continue
            filtered_prompts.append(prompt)
            filtered_nwrs.append(nwr_list[i])
        nwr_list = filtered_nwrs
        prompts = filtered_prompts

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=MAX_NEW_TOKENS,
        )
        # load the model last for better debugging
        llm = LLM(
            model=FORMAT_MODEL,
            dtype="bfloat16",
            max_model_len=MAX_INPUT_LENGTH + MAX_NEW_TOKENS,
            max_seq_len_to_capture=MAX_INPUT_LENGTH,
            task="generate",
        )
        responses = llm.generate(prompts, sampling_params)
        outputs = [responses[i].outputs[0].text for i in range(len(responses))]
    else:
        outputs = []
        for i, nwr in tqdm(
            enumerate(nwr_list),
            total=len(nwr_list), desc="Generating NWRs with Ollama"
        ):
            # Process each NewsWithRationale object
            formatted_response = process_nwr(nwr)
            outputs.append(formatted_response)

    for i, (nwr, output) in tqdm(
        enumerate(zip(nwr_list, outputs)),
        total=len(outputs), desc="Formatting NWRs"
    ):
        try:
            essentials, triples = extract_essentials_and_triples(output)
        except Exception as e:
            logger.error(f"Error processing NWR {nwr.id}: {e}")
            logger.error(f"Formatted response:\n{output}")

        nwr.essential_aspects = essentials
        nwr.triples = triples

        # Save the formatted response to a file
        with open(FORMAT_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(nwr.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if not USE_VLLM:
        # Make sure the model is downloaded
        logger.info(f"Pulling model {FORMAT_MODEL_OLLAMA}...")
        ollama.pull(FORMAT_MODEL_OLLAMA)
    local_data_format_main()
