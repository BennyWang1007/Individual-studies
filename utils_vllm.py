from typing import Any

from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
)


def init_vllm_model(
    model_name: str, max_input_length: int, max_new_tokens: int
) -> tuple[LLM, SamplingParams]:
    model = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=max_input_length + max_new_tokens,
        # max_seq_len=max_input_length,
        max_seq_len_to_capture=max_input_length,
        # enforce_eager=True,
        task="generate",
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_new_tokens
    )
    return model, sampling_params


def filter_by_max_length(
    max_length: int, primary_list: list[str], *other_lists
) -> tuple[list[Any], ...]:
    """
    Filter out prompts that are too long, and return the filtered prompts
    """
    filtered_primary = []
    filtered_others: list[list] = [[] for _ in other_lists]

    for i in range(len(other_lists)):
        assert len(other_lists[i]) == len(primary_list)

    for i, item in enumerate(primary_list):
        if len(item) > max_length:
            continue
        filtered_primary.append(item)
        for j, other_list in enumerate(other_lists):
            filtered_others[j].append(other_list[i])

    return (filtered_primary, *filtered_others)


def vllm_batch_generate(
    llm: LLM, prompts: list[str], sampling_params: SamplingParams,
    batch_size: int = 1000,
) -> list[RequestOutput]:
    """
    Generate responses in batches using the vLLM model.
    """
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        responses = llm.generate(batch_prompts, sampling_params)
        outputs.extend(responses)
    return outputs


def vllm_cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
