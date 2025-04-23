import json

import bert_score
from rouge_score import rouge_scorer

from .curriculum_training.constants import (
    USE_VLLM,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    NWR_BENCHMARK_FILE,
)
from .curriculum_training.curriculum_utils import (
    DifficultyLevels as DL,
    PREFIX_OF_DIFFICULTY_LEVELS,
)
from crawler.utils import Logger
from news_with_rationale import NewsWithRationale
from utils import init_vllm_model, filter_by_max_length

if USE_VLLM:
    from transformers import AutoTokenizer
else:
    import ollama
    from ollama import chat

DATASET_NAME = NWR_BENCHMARK_FILE

# count the number of news in the dataset
with open(DATASET_NAME, "r", encoding="utf-8") as f:
    news_count = sum(1 for _ in f)


JUDGE_MODELNAME_OLLAMA = "qwen2.5:32b-instruct-q6_K"
JUDGE_MODELNAME = "Qwen/Qwen2.5-32B-Instruct"

TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    f"./qwen2.5-curriculum-trained_{news_count}news_4stage_A100",
    f"./qwen2.5-curriculum-trained_{news_count}news_5stage_A100",
]

eval_logger = Logger("eval score", verbose_level=3)
eval_logger.info(f"Total news count: {news_count}")


def evaluate_with_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [
        scorer.score(pred, ref) for pred, ref in zip(predictions, references)
    ]
    return scores


def evaluate_with_bertscore(predictions, references):
    P, R, F1 = bert_score.score(
        predictions, references, lang="en", verbose=False
    )
    return {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}


def judge_summary_1_to_20_prompt_sys() -> str:
    return """\
You are an expert language evaluator. Your task is to assess the quality of a \
model-generated summary based on the article and the ground-truth summary. \
Score it from 1 to 20 using the following rubric:

1: Totally irrelevant, unrelated to the article.
2: Hallucinates facts, makes no sense.
3	Severe misunderstanding, contains major errors.
4	Barely reflects the source, very incomplet.
5	Grammatical errors, lacks coherence and relevance.
6	Incomplete and partially off-topic.
7	Misses key points, contains minor hallucinations.
8	Vague summary, lacks specificity.
9	Concise and covers most major points.
10	Understandable but may miss subtle nuances.
11	Faithful with minor omissions.
12	Mostly accurate but slightly redundant.
13	Accurate, well-structured, minor stylistic issues.
14	Good coverage and clarity, tone could improve.
15	Clear, faithful, and stylistically strong.
16	Concise, elegant, and captures all key points.
17	Very close to ideal summary, only minor flaws.
18	Excellent summary, highly readable and comprehensive.
19	Near perfect, minor stylistic polish could be added.
20	Perfect — clear, faithful, complete, and elegant.

Return only an integer score (1–20), followed by a short reason \
(e.g., "Score: 17 — Very close to ideal summary, only minor flaws").
"""


def judge_summary_1_to_20_prompt_user(article, response, ground_truth) -> str:
    return (
        f"Article: {article}\n\n"
        f"Ground Truth: {ground_truth}\n\n"
        f"Model Output: {response}"
    )


def judge_summary_1_to_20(
    articles: list[str], gen_sums: list[str], ground_truths: list[str],
    judge_model: str = JUDGE_MODELNAME if USE_VLLM else JUDGE_MODELNAME_OLLAMA
) -> list[int]:
    """
    Get the responses string from the judge model.
    """
    assert len(articles) == len(gen_sums) == len(ground_truths)
    sys_prompt = judge_summary_1_to_20_prompt_sys()
    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    if USE_VLLM:
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": judge_summary_1_to_20_prompt_user(
                            article, gen_sum, ground_truth
                        )
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for article, gen_sum, ground_truth in zip(
                articles, gen_sums, ground_truths
            )
        ]
        llm, sampling_params = init_vllm_model(
            judge_model, MAX_INPUT_LENGTH, MAX_NEW_TOKENS
        )
        prompts, articles, gen_sums, ground_truths = filter_by_max_length(
            MAX_INPUT_LENGTH, prompts, articles, gen_sums, ground_truths
        )
        responses = llm.generate(prompts, sampling_params)
        outputs = [response.outputs[0].text for response in responses]
    else:
        ollama.pull(judge_model)

        sys_prompt = judge_summary_1_to_20_prompt_sys()
        outputs = []

        for i, (article, gen_sum, ground_truth) in enumerate(
            zip(articles, gen_sums, ground_truths)
        ):
            # Process each article and summary
            user_prompt = judge_summary_1_to_20_prompt_user(
                article, gen_sum, ground_truth
            )
            gen_response = chat(
                model=judge_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            assert gen_response.message.content is not None
            outputs.append(gen_response.message.content)

    scores = []
    for output in outputs:
        # Extract the score from the output
        try:
            score = int(output.split("Score:")[1].split(" —")[0].strip())
            scores.append(score)
        except (ValueError, IndexError):
            eval_logger.error(f"Invalid output format: {output}")
            scores.append(-1)

    return scores


def benchmark_model(modelname: str) -> dict:

    nwrs = []
    with open(DATASET_NAME, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            nwrs.append(NewsWithRationale.from_dict(data))
    news_list = [nwr.article for nwr in nwrs]
    summaries = [nwr.summary for nwr in nwrs]

    """ ----------------------- Generate responses ----------------------- """
    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DL.DIRECT_SUMMARY]
    if USE_VLLM:
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "article:\n" + nwr.article}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for nwr in nwrs
        ]
        llm, sampling_params = init_vllm_model(
            modelname, MAX_INPUT_LENGTH, MAX_NEW_TOKENS
        )
        responses_raw = llm.generate(prompts, sampling_params)
        responses = [response.outputs[0].text for response in responses_raw]
    else:
        ollama.pull(modelname)
        responses = []
        for nwr in nwrs:
            # Process each NewsWithRationale object
            gen_response = chat(
                model=modelname,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "article:\n" + nwr.article}
                ]
            )
            assert gen_response.message.content is not None
            responses.append(gen_response.message.content)

    """ ------------------------ Process responses ------------------------ """
    processed_responses = []
    for response in responses:
        processed_response = response.strip()
        if processed_response.startswith("新聞摘要：\n"):
            processed_response = processed_response[5:]
        elif processed_response.startswith("新聞摘要："):
            processed_response = processed_response[4:]
        elif processed_response.startswith("新聞摘要"):
            processed_response = processed_response[3:]
        processed_responses.append(processed_response)

    """ ------------------------- Evaluations ------------------------- """
    rouge_scores = evaluate_with_rouge(processed_responses, summaries)
    bert_scores = evaluate_with_bertscore(processed_responses, summaries)

    judge_scores = judge_summary_1_to_20(
        news_list,
        processed_responses,
        summaries,
        judge_model=JUDGE_MODELNAME if USE_VLLM else JUDGE_MODELNAME_OLLAMA
    )
    judge_scores = [score for score in judge_scores if score > 0]

    return {
        "predictions": processed_responses,
        "judge_scores": judge_scores,
        "rouge_scores": rouge_scores,
        "bert_scores": bert_scores,
    }


if __name__ == "__main__":
    # Example usage
    for model_name in TEST_MODELS:
        eval_logger.info(f"Evaluating model: {model_name}")
        results = benchmark_model(model_name)
        eval_logger.info(f"Results for {model_name}:")
        eval_logger.info(f"Judge scores: {results['judge_scores']}")
        eval_logger.info(f"Rouge scores: {results['rouge_scores']}")
        eval_logger.info(f"Bert scores: {results['bert_scores']}")

        with open(f"benchmark_result/{model_name}_results.json", "w") as f:
            json.dump(results, f, indent=4)
