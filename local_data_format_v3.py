import json

# import ollama
import re
from opencc import OpenCC
from ollama import chat
# from tqdm import tqdm

from crawler.utils import Logger
from curriculum_training.constants import (
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    USE_VLLM,
    ALLOW_VLLM,
    SUMMARY_V3,
    SUMMARY_FORMATTED_V3,
)
from news_with_rationale import NewsWithRationale as NWR
from utils import int_set_str

from transformers import AutoTokenizer, PreTrainedTokenizer
from utils_vllm import (
    init_vllm_model,
    filter_by_max_length,
    vllm_batch_generate,
)

assert ALLOW_VLLM

FORMAT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
FORMAT_MODEL_OLLAMA = "qwen2.5:32b-instruct"


model, sampling_params = init_vllm_model(
    FORMAT_MODEL,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(FORMAT_MODEL)


logger = Logger("data_format")
cc = OpenCC("s2twp")  # Simplified Chinese to Traditional Chinese

ESS_START = len("關鍵要素：\n[")
TRI_START = len("\n[")


def format_ess_tri_sys_prompt() -> str:
    """ Get the format system prompt for the model. """
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


def format_ess_tri_user_prompt(nwr: NWR) -> str:
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


def format_summ_sys_prompt() -> str:
    """
    Get the format system prompt for the model.
    """
    return (
        "你會收到一個新聞文章以及其摘要，請評估該摘要是否為良好的摘要。\n"
        "若摘要良好（符合且文章內容），請輸出\"符合\"並將無關的內容移除。\n"
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


def format_summ_user_prompt(data: dict) -> str:
    """
    Get the format user prompt for the given NewsWithRationale object.
    """
    return (
        "請評估以下內容的摘要是否是良好的新聞摘要：\n"
        "新聞：\n"
        f"{data['news']}\n\n"
        "摘要：\n"
        f"{data['response']}\n\n"
    )


def process_summ_ollama(data: dict) -> str:
    """
    Process the given NewsWithRationale object and return the formatted string.
    """
    sys_prompt = format_summ_sys_prompt()
    user_prompt = format_summ_user_prompt(data)

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


def get_finished_id(filename) -> set[int]:
    """ Get the finished news/NWR/zh-tw ids from the generated files. """
    finished_ids = set()
    line_bak: str
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                line_bak = line
                finished_ids.add(data["id"])
    except FileNotFoundError:
        logger.info(f"{filename} not found, starting from scratch.")
        pass
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON in line: {line_bak}.")
        exit()
    return finished_ids


def read_summ_file(filename: str, finished_ids=None) -> list[dict]:
    """ Read the summary objects from the given file. """
    if finished_ids is None:
        finished_ids = set()
    summ_data: list[dict] = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["id"] in finished_ids:
                    continue
                summ_data.append(data)
    except FileNotFoundError:
        logger.info(f"{filename} not found, starting from scratch.")
        pass
    return summ_data


def read_ess_tri_file(filename: str, finished_ids=None) -> list[dict]:
    """ Read the essentials and triples from the given file. """
    if finished_ids is None:
        finished_ids = set()
    ess_tri_data: list[dict] = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["id"] in finished_ids:
                    continue
                ess_tri_data.append(data)
    except FileNotFoundError:
        logger.info(f"{filename} not found, starting from scratch.")
        pass
    return ess_tri_data


def extract_essentials_and_triples(
    response: str
) -> tuple[list[str], list[str]]:
    # Extract the essential aspects and triples from the response
    formatted_response = cc.convert(response)

    try:
        ess_str, tri_str = formatted_response.split("三元組：")
        new_essentials = ess_str.split("]、[")
        new_essentials[0] = new_essentials[0][ESS_START:]
        new_essentials[-1] = new_essentials[-1][:-3]  # Remove \n\n

        new_triples = tri_str.split("]、[")
        new_triples[0] = new_triples[0][TRI_START:]
        new_triples[-1] = new_triples[-1][:-1]

    except ValueError:
        logger.error(f"Invalid response format: {response}")
        exit()

    return new_essentials, new_triples


def extract_summ(response: str) -> str:
    # Extract the summary from the response
    response = cc.convert(response)

    if response.startswith("生成摘要：\n"):
        summary = response[len("生成摘要：\n"):].strip()
        return summary

    logger.error(f"Invalid summ response format: {response}")
    raise ValueError(f"Invalid summ response format: {response}")


def extract_summ_acceptance(response: str) -> tuple[bool, str]:
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


def format_summ(summ_file: str, output_file: str) -> None:
    """
    Main function to format the data using the model.
    """
    # Load the finished ids from the formatted file
    finished_ids = get_finished_id(output_file)
    logger.info(f"Finished ids count: {len(finished_ids)}")
    logger.info(f"Finished ids: {int_set_str(finished_ids)}")

    # Load the NewsWithRationale excluding the finished ids
    summ_todo: list[dict] = read_summ_file(summ_file, finished_ids)
    summ_finished: list[dict] = read_summ_file(output_file, finished_ids)

    logger.info(f"Loaded {len(summ_finished)} summ from {output_file}")
    logger.info(f"Total {len(summ_todo)} summ to process")

    # summ_todo = summ_todo[:10]  # for demonstration

    ids = [summ["id"] for summ in summ_todo]
    assert len(ids) == len(summ_todo)

    logger.info(f"Remains {len(ids)} summ to process")
    logger.info(f"Remains summ ids: {int_set_str(set(ids))}")

    output_strs: list[str] = []
    if USE_VLLM:
        sys_prompt = format_summ_sys_prompt()
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": format_summ_user_prompt(summ)}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for summ in summ_todo
        ]

        assert len(prompts) == len(summ_todo)

        # Filter out prompts that are too long
        prompts, summ_todo = filter_by_max_length(
            MAX_INPUT_LENGTH, prompts, summ_todo
        )

        responses = vllm_batch_generate(model, prompts, sampling_params)
        output_strs = [response.outputs[0].text for response in responses]

    else:
        raise NotImplementedError("Ollama processing not implemented")

    for i, (summ, output) in enumerate(zip(summ_todo, output_strs)):
        try:
            success, summ_str = extract_summ_acceptance(output)
        except Exception as e:
            logger.error(f"Error processing id {summ['id']}: {e}")
            logger.error(f"Formatted response:\n{output}")
            continue

        if success:
            # Save the formatted response
            summ_finished.append(
                {
                    "id": summ["id"],
                    "news": summ["news"],
                    # "response": summ_str,
                    "response": summ,
                }
            )

    with open(output_file, "w", encoding="utf-8") as f:
        for summ in summ_finished:
            f.write(json.dumps(summ, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    format_summ(
        SUMMARY_V3,
        SUMMARY_FORMATTED_V3
    )
