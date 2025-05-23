import json
import os
import re
from typing import Callable, Optional

from opencc import OpenCC
from transformers import AutoTokenizer

from crawler.utils import Logger
from curriculum_training.constants import (
    USE_VLLM, ALLOW_VLLM, MAX_INPUT_LENGTH, MAX_NEW_TOKENS, DATASET_V3_DIR,
    SUMMARY_FORMATTED_V3, SUMMARY_V3, ESSENTIALS_V3, NWR_V3
)
from news_with_rationale import NewsWithRationale as NWR
from utils import load_udn_news
from utils_vllm import (
    LLM,
    SamplingParams,
    init_vllm_model,
    filter_by_max_length,
    vllm_batch_generate,
)

assert ALLOW_VLLM
assert USE_VLLM

MODELNAME = "Qwen/Qwen2.5-32B-Instruct"

gen_logger = Logger("data_gen", verbose_level=3)

model: Optional[LLM] = None
sampling_params: Optional[SamplingParams] = None
tokenizer = AutoTokenizer.from_pretrained(MODELNAME)

cc = OpenCC("s2twp")  # Simplified Chinese to Traditional Chinese

Message = list[dict[str, str]]

if not os.path.exists(DATASET_V3_DIR):
    os.makedirs(DATASET_V3_DIR)


def essential_aspects_prompt(nwr: NWR) -> Message:
    return [
        {
            "role": "system",
            "content": (
                "請根據以下新聞內容以及摘要，提取新聞的關鍵要素與三元組，關鍵要素應為關鍵短句、名詞或事實，"
                "三元組應為[實體1 | 關係 | 實體2]的格式，"
                "這些三元組用於構成摘要，請用繁體中文回答。"
                "請將每個關鍵要素與三元組用[]與、分隔。"
                "例如："
                "關鍵要素：\n[關鍵要素1]、[關鍵要素2]、[關鍵要素3]、...\n\n"
                "三元組：\n[實體1_1 | 關係_1 | 實體1_2]、[實體2_1 | 關係_2 | 實體2_2]、..."
                ""
            )
        },
        {
            "role": "user",
            "content": f"新聞：\n{nwr.article}\n\n摘要：\n{nwr.summary}\n\n"
        }
    ]


def summary_prompt(nwr: NWR) -> Message:
    return [
        {
            "role": "system",
            "content": (
                "請根據以下新聞內容，為新聞生成一份100字內精簡的摘要，"
                "請用繁體中文回答。\n"
                "例如：\n"
                "生成摘要：\n"
            )
        },
        {"role": "user", "content": f"新聞：\n{nwr.article}"}
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
    prompts, _nwrs = filter_by_max_length(MAX_INPUT_LENGTH, prompts, nwrs)

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


def parse_essential_and_triple(response: dict) -> tuple[list[str], list[str]]:
    """
    Parse the essential aspects from the response.
    """
    pattern = (
        r"(?:關鍵要素：\s*)?((?:\[[^\[\]]+\]、?)+)\n*"
        r"(?:三元組：\s*)?((?:\[[^\[\]]+\]、?)+)"
    )
    match = re.search(pattern, response["response"])
    if match:
        content = match.group(1)
        essentials = [
            item.strip("[]") for item in content.split("、") if item
        ]
        content = match.group(2)
        tripples = [
            item.strip("[]") for item in content.split("、") if item
        ]
        return essentials, tripples

    return [], []


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


def parse_summaries(nwrs: list[NWR], responses: list[dict]) -> list[NWR]:
    """
    Parse the summaries from the responses and update the corresponding
    NWR objects.
    """
    response_map = {response["id"]: response for response in responses}

    for nwr in nwrs:
        response = response_map.get(nwr.id)
        if not response:
            # gen_logger.warning(f"No response found for NWR with id {nwr.id}")
            continue

        summary = parse_summary(response)
        if summary:
            summary = cc.convert(summary)
            nwr.summary = summary
            nwr.rationale_summary = summary

    return nwrs


def parse_esss_and_tris(nwrs: list[NWR], responses: list[dict]) -> list[NWR]:
    """
    Parse the essential aspects and triples from the responses
    and update the corresponding NWR objects.
    """
    response_map = {response["id"]: response for response in responses}

    for nwr in nwrs:
        response = response_map.get(nwr.id)
        if not response:
            # gen_logger.warning(f"No response found for NWR with id {nwr.id}")
            continue

        essentials, triples = parse_essential_and_triple(response)
        if essentials:
            essentials = [cc.convert(ess) for ess in essentials]
            nwr.essential_aspects = essentials
        if triples:
            triples = [cc.convert(tri) for tri in triples]
            nwr.triples = triples

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


def load_essentials_and_triples() -> tuple[
    dict[int, list[str]], dict[int, list[str]]
]:
    data = load_data(ESSENTIALS_V3)
    essential_aspects: dict[int, list[str]] = {}
    triples: dict[int, list[str]] = {}
    for dat in data:
        if dat["id"] in essential_aspects:
            gen_logger.warning(f"Duplicated essential id: {dat['id']}")
            continue
        parsed_essentials, parsed_triples = parse_essential_and_triple(dat)
        essential_aspects[dat["id"]] = parsed_essentials
        triples[dat["id"]] = parsed_triples
    return essential_aspects, triples


def load_summary(filename) -> dict[int, str]:
    data = load_data(filename)
    summaries: dict[int, str] = {}
    for dat in data:
        if dat["id"] in summaries:
            gen_logger.warning(f"Duplicated summary id: {dat['id']}")
            continue
        summaries[dat["id"]] = parse_summary(dat)
    return summaries


def get_ids_from_file(filename: str) -> set[int]:
    """
    Get the ids from the file.
    """
    ids: set[int] = set()
    data = load_data(filename)
    for dat in data:
        if dat["id"] in ids:
            gen_logger.warning(f"Duplicated id: {dat['id']}")
            continue
        ids.add(dat["id"])
    return ids


def get_finished_ids() -> tuple[set[int], set[int], set[int], set[int]]:
    """
    Get the finished essential/triple/summary/nwr ids from the generated files.
    """

    # find finished essential ids
    essential_ids = get_ids_from_file(ESSENTIALS_V3)
    triple_ids = essential_ids.copy()

    # find finished zh-tw ids
    summary_ids = get_ids_from_file(SUMMARY_FORMATTED_V3)

    nwr_ids: set[int] = essential_ids & triple_ids & summary_ids

    return essential_ids, triple_ids, summary_ids, nwr_ids


if __name__ == "__main__":

    # load the news
    news_list: list[str] = load_udn_news()
    # news_list = news_list[:10]
    gen_logger.info(f"Loaded {len(news_list)} news")

    # load finished ids
    essential_ids, triple_ids, summary_ids, nwr_ids = get_finished_ids()
    gen_logger.info(f"Finished essential count: {len(essential_ids)}")
    gen_logger.info(f"Finished triple count: {len(triple_ids)}")
    gen_logger.info(f"Finished summary count: {len(summary_ids)}")
    gen_logger.info(f"Finished NWR count: {len(nwr_ids)}")

    essential_data, triple_data = load_essentials_and_triples()
    summary_data = load_summary(SUMMARY_FORMATTED_V3)

    # assert len(essential_data) == len(list(essential_ids))
    # assert len(triple_data) == len(list(triple_ids))
    # assert len(summary_data) == len(list(summary_ids))

    nwrs = [NWR(news, id=i) for i, news in enumerate(news_list)]
    for nwr in nwrs:
        if nwr.id in essential_ids:
            nwr.essential_aspects = [
                cc.convert(ess) for ess in essential_data[nwr.id]
            ]
        if nwr.id in triple_ids:
            nwr.triples = [cc.convert(tri) for tri in triple_data[nwr.id]]
        if nwr.id in summary_ids:
            nwr.summary = cc.convert(summary_data[nwr.id])
            nwr.rationale_summary = nwr.summary

    with open(NWR_V3, "w", encoding="utf-8") as f:
        for nwr in nwrs:
            if nwr.id in nwr_ids:
                f.write(json.dumps(nwr.to_dict(), ensure_ascii=False) + "\n")

    gen_logger.info(f"Loaded {len(nwr_ids)} NWRs")

    model, sampling_params = init_vllm_model(
        model_name=MODELNAME,
        max_input_length=MAX_INPUT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS
    )

    # generate summary responses and parse them
    _nwrs = [nwr for nwr in nwrs if nwr.id not in summary_ids]
    gen_logger.info(f"Generating {len(_nwrs)} summaries")

    responses = local_gen_response(_nwrs, summary_prompt)
    nwrs = parse_summaries(nwrs, responses)  # update the NWRs
    gen_logger.info(f"Generated {len(responses)} summary responses")

    with open(SUMMARY_V3, "a", encoding="utf-8") as f:
        for dat in responses:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    gen_logger.info(f"{len([nwr for nwr in nwrs if nwr.summary == ''])} "
                    f"NWRs could not generate summaries")

    # remove the NWRs that could not generate summaries
    nwrs = [nwr for nwr in nwrs if nwr.summary != ""]
    gen_logger.info(f"remaining {len(nwrs)} NWRs")


    # generate essential responses and parse them
    _nwrs = [nwr for nwr in nwrs if nwr.id not in essential_ids]
    gen_logger.info(f"Generating {len(_nwrs)} essential aspects")
    responses = local_gen_response(_nwrs, essential_aspects_prompt)
    nwrs = parse_esss_and_tris(nwrs, responses)  # update the NWRs
    gen_logger.info(f"Generated {len(responses)} essential and triples")

    with open(ESSENTIALS_V3, "a", encoding="utf-8") as f:
        for dat in responses:
            f.write(json.dumps(dat, ensure_ascii=False) + "\n")

    # remove the NWRs that could not generate essential aspects
    gen_logger.info(
        f"{len([nwr for nwr in nwrs if nwr.essential_aspects == []])} "
        f"NWRs could not generate essential aspects"
    )
    nwrs = [nwr for nwr in nwrs if nwr.essential_aspects != []]
    gen_logger.info(f"remaining {len(nwrs)} NWRs")

    # save the NWRs to file
    with open(NWR_V3, "w", encoding="utf-8") as f:
        for nwr in nwrs:
            f.write(json.dumps(nwr.to_dict(), ensure_ascii=False) + "\n")
