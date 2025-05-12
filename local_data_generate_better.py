import json
import os
import re
from typing import Callable

from opencc import OpenCC
from transformers import AutoTokenizer

from crawler.utils import Logger
from curriculum_training.constants import (
    USE_VLLM, ALLOW_VLLM, MAX_INPUT_LENGTH, MAX_NEW_TOKENS, DATASET_V2_DIR,
)
from news_with_rationale import NewsWithRationale as NWR
from utils import load_udn_news
from utils_vllm import (
    init_vllm_model,
    filter_by_max_length,
    vllm_batch_generate,
)

assert ALLOW_VLLM
assert USE_VLLM

MODELNAME = "Qwen/Qwen2.5-32B-Instruct"

gen_logger = Logger("data_gen", verbose_level=3)

model = None
sampling_params = None
tokenizer = AutoTokenizer.from_pretrained(MODELNAME)

cc = OpenCC("s2twp")  # Simplified Chinese to Traditional Chinese

Message = list[dict[str, str]]

if not os.path.exists(DATASET_V2_DIR):
    os.makedirs(DATASET_V2_DIR)

NWR_FILE = os.path.join(DATASET_V2_DIR, "news_with_rationale.jsonl")
ESS_ASPECTS_FILE = os.path.join(DATASET_V2_DIR, "essential_aspects.jsonl")
TRIPLES_FILE = os.path.join(DATASET_V2_DIR, "triples.jsonl")
SUMMARY_FILE = os.path.join(DATASET_V2_DIR, "summary.jsonl")


def essential_aspects_prompt(nwr: NWR) -> Message:
    return [
        {
            "role": "system",
            "content": (
                "請根據以下新聞內容，提取新聞的關鍵要素，關鍵要素應為關鍵短句、名詞或事實，"
                "請用中文回答，並且不要使用任何標點符號。"
                "請將每個關鍵要素用[]與、分隔。"
                "例如："
                "關鍵要素：\n[關鍵要素1]、[關鍵要素2]、[關鍵要素3]"
            )
        },
        {"role": "user", "content": f"新聞：\n{nwr.article}"}
    ]


def triples_prompt(nwr: NWR) -> Message:
    return [
        {
            "role": "system",
            "content": (
                "請根據以下新聞內容與關鍵要素，檢索詳細的三元組，格式為 [實體1 | 關係 | 實體2]，這些三元組用於構成摘要，"
                "請用中文回答，並且不要使用任何標點符號。"
                "所有三元組用[]與、分隔，且長度必須為3。\n"
                "例如：\n"
                "三元組：\n[實體1_1 | 關係_1 | 實體1_2]、[實體2_1 | 關係_2 | 實體2_2]、..."
            )
        },
        {
            "role": "user",
            "content": (
                f"新聞：\n{nwr.article}\n\n"
                f"關鍵要素：\n{nwr.essential_aspects_str()}\n\n"
            )
        }
    ]


def summary_prompt(nwr: NWR) -> Message:
    return [
        {
            "role": "system",
            "content": (
                "請根據以下新聞內容與檢索到的關鍵要素以及三元組，為新聞生成一份摘要，"
                "請用繁體中文回答。\n"
                "例如：\n"
                "生成摘要：\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"新聞：\n{nwr.article}\n\n"
                f"關鍵要素：\n{nwr.essential_aspects_str()}\n\n"
                f"三元組：\n{nwr.triples_str()}\n\n"
            )
        }
    ]


def local_gen_response(nwrs: list[NWR], prompt_fn: Callable) -> list[dict]:
    assert model is not None
    assert sampling_params is not None

    data: list[dict] = []

    if len(nwrs) == 0:
        gen_logger.warning("No data need to generate.")
        return data

    prompts: list[str] = [
        tokenizer.apply_chat_template(
            prompt_fn(nwr),
            tokenize=False,
            add_generation_prompt=True
        )
        for nwr in nwrs
    ]

    # Filter out prompts that are too long
    prompts, _nwrs = filter_by_max_length(
        MAX_INPUT_LENGTH, prompts, nwrs
    )

    if len(prompts) == 0:
        return data

    responses = vllm_batch_generate(model, prompts, sampling_params)
    outputs = [response.outputs[0].text for response in responses]
    data = [
        {
            "id": nwr.id,
            "news": nwr.article,
            "response": response
        }
        for nwr, response in zip(_nwrs, outputs)
    ]

    return data


def parse_essential(response: dict) -> list[str]:
    """
    Parse the essential aspects from the response.
    """
    pattern = r"(?:關鍵要素：\s*)?((?:\[[^\[\]]+\]、?)+)"
    match = re.search(pattern, response["response"])
    if match:
        content = match.group(1)
        return [
            item.strip("[]") for item in content.split("、") if item
        ]
    return []


def parse_triple(response: dict) -> list[str]:
    """
    Parse the triples from the response.
    """
    pattern = r"(?:三元組：\s*)?((?:\[[^\[\]]+\]、?)+)"
    match = re.search(pattern, response["response"])
    if match:
        content = match.group(1)
        return [
            item.strip("[]") for item in content.split("、") if item
        ]
    return []


def parse_summary(response: dict) -> str:
    """
    Parse the summary from the response.
    """
    pattern = r"(?:生成摘要：\s*)?(.+)"
    match = re.search(pattern, response["response"].strip())
    if match:
        summary = match.group(1).strip()
        return summary
    return ""


def parse_essentials(nwrs: list[NWR], responses: list[dict]) -> list[NWR]:
    """
    Parse the essential aspects from the responses.
    """
    for i, response in enumerate(responses):
        nwr = nwrs[i]
        essential = parse_essential(response)
        essential = [cc.convert(ess) for ess in essential]
        if essential:
            nwr.essential_aspects = essential
    return nwrs


def parse_triples(nwrs: list[NWR], responses: list[dict]) -> list[NWR]:
    """
    Parse the triples from the responses.
    """
    for i, response in enumerate(responses):
        nwr = nwrs[i]
        triple = parse_triple(response)
        triple = [cc.convert(tri) for tri in triple]
        if triple:
            nwr.triples = triple
    return nwrs


def parse_summaries(nwrs: list[NWR], responses: list[dict]) -> list[NWR]:
    """
    Parse the summary from the responses.
    """
    for i, response in enumerate(responses):
        nwr = nwrs[i]
        summary = parse_summary(response)
        summary = cc.convert(summary)
        if summary:
            nwr.summary = summary
            nwr.rationale_summary = summary
    return nwrs


def load_data(filename: str) -> list[dict]:
    """
    Load the data from the file.
    """
    data: list[dict] = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        gen_logger.warning(f"{filename} not found, starting from scratch.")

    gen_logger.info(f"Loaded {len(data)} data from {filename}")
    return data


def load_essential_aspects() -> dict[int, list[str]]:
    data = load_data(ESS_ASPECTS_FILE)
    essential_aspects: dict[int, list[str]] = {}
    for dat in data:
        if dat["id"] in essential_aspects:
            gen_logger.warning(f"Duplicated essential id: {dat['id']}")
            continue
        essential_aspects[dat["id"]] = parse_essential(dat)
    return essential_aspects


def load_triples() -> dict[int, list[str]]:
    data = load_data(TRIPLES_FILE)
    triples: dict[int, list[str]] = {}
    for dat in data:
        if dat["id"] in triples:
            gen_logger.warning(f"Duplicated triple id: {dat['id']}")
            continue
        triples[dat["id"]] = parse_triple(dat)
    return triples


def load_summary() -> dict[int, str]:
    data = load_data(SUMMARY_FILE)
    summaries: dict[int, str] = {}
    for dat in data:
        if dat["id"] in summaries:
            gen_logger.warning(f"Duplicated summary id: {dat['id']}")
            continue
        summaries[dat["id"]] = parse_summary(dat)
    return summaries


def get_finished_id() -> tuple[set[int], set[int], set[int], set[int]]:
    """
    Get the finished essential/triple/summary/nwr ids from the generated files.
    """

    # find finished essential ids
    essential_ids: set[int] = set()
    data = load_data(ESS_ASPECTS_FILE)
    for dat in data:
        if dat["id"] in essential_ids:
            gen_logger.warning(f"Duplicated essential id: {dat['id']}")
            continue
        essential_ids.add(dat["id"])

    # find finished NWR ids
    triple_ids: set[int] = set()
    data = load_data(TRIPLES_FILE)
    for dat in data:
        if dat["id"] in triple_ids:
            gen_logger.warning(f"Duplicated NWR id: {dat['id']}")
            continue
        triple_ids.add(dat["id"])

    # find finished zh-tw ids
    summary_ids: set[int] = set()
    data = load_data(SUMMARY_FILE)
    for dat in data:
        if dat["id"] in summary_ids:
            gen_logger.warning(f"Duplicated summary id: {dat['id']}")
            continue
        summary_ids.add(dat["id"])

    nwr_ids: set[int] = essential_ids & triple_ids & summary_ids

    return essential_ids, triple_ids, summary_ids, nwr_ids


if __name__ == "__main__":

    # load the news
    news_list: list[str] = load_udn_news()
    # news_list = news_list[:10]
    gen_logger.info(f"Loaded {len(news_list)} news")

    # load finished ids
    essential_ids, triple_ids, summary_ids, nwr_ids = get_finished_id()
    gen_logger.info(f"Finished essential count: {len(essential_ids)}")
    gen_logger.info(f"Finished triple count: {len(triple_ids)}")
    gen_logger.info(f"Finished summary count: {len(summary_ids)}")
    gen_logger.info(f"Finished NWR count: {len(nwr_ids)}")

    essential_data = load_essential_aspects()
    triple_data = load_triples()
    summary_data = load_summary()

    assert len(essential_data) == len(list(essential_ids))
    assert len(triple_data) == len(list(triple_ids))
    assert len(summary_data) == len(list(summary_ids))

    nwrs = [NWR(news, id=i) for i, news in enumerate(news_list)]
    for nwr in nwrs:
        if nwr.id in nwr_ids:
            esss = [cc.convert(ess) for ess in essential_data[nwr.id]]
            tris = [cc.convert(tri) for tri in triple_data[nwr.id]]
            nwr.essential_aspects = esss
            nwr.triples = tris
            nwr.summary = cc.convert(summary_data[nwr.id])
            nwr.rationale_summary = nwr.summary

    with open(NWR_FILE, "w", encoding="utf-8") as f:
        for nwr in nwrs:
            f.write(json.dumps(nwr.to_dict(), ensure_ascii=False) + "\n")

    gen_logger.info(f"Loaded {len(nwr_ids)} NWRs")

    exit()
    model, sampling_params = init_vllm_model(
        model_name=MODELNAME,
        max_input_length=MAX_INPUT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS
    )

    # generate essential responses and parse them
    _nwrs = [nwr for nwr in nwrs if nwr.id not in essential_ids]
    responses = local_gen_response(_nwrs, essential_aspects_prompt)
    nwrs = parse_essentials(nwrs, responses)  # update the NWRs
    gen_logger.info(f"Generated {len(responses)} essential responses")

    with open(ESS_ASPECTS_FILE, "a", encoding="utf-8") as f:
        for dat in responses:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    # generate triples responses and parse them
    _nwrs = [nwr for nwr in nwrs if nwr.id not in triple_ids]
    responses = local_gen_response(_nwrs, triples_prompt)
    nwrs = parse_triples(nwrs, responses)  # update the NWRs
    gen_logger.info(f"Generated {len(responses)} triples responses")

    with open(TRIPLES_FILE, "a", encoding="utf-8") as f:
        for dat in responses:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    # generate summary responses and parse them
    _nwrs = [nwr for nwr in nwrs if nwr.id not in summary_ids]
    responses = local_gen_response(_nwrs, summary_prompt)
    nwrs = parse_summaries(nwrs, responses)  # update the NWRs
    gen_logger.info(f"Generated {len(responses)} summary responses")

    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        for dat in responses:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    # save the NWRs to file
    with open(NWR_FILE, "w", encoding="utf-8") as f:
        for nwr in nwrs:
            f.write(json.dumps(nwr.to_dict(), ensure_ascii=False) + "\n")
