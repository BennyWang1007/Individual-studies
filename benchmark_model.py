import json
import os

import bert_score
import re
from rouge_score import rouge_scorer
from tqdm import tqdm

from curriculum_training.constants import (
    # USE_VLLM,
    MAX_BENCHMARK_LENGTH,
    MAX_NEW_TOKENS,
    NWR_BENCHMARK_FILE,
    NWR_TRAINING_FILE,
)
from curriculum_training.curriculum_utils import (
    DifficultyLevels as DL,
    PREFIX_OF_DIFFICULTY_LEVELS,
    cleanup,
)
from crawler.utils import Logger
from news_with_rationale import NewsWithRationale
from utils import legalize_filename

USE_VLLM = True

if USE_VLLM:
    from transformers import AutoTokenizer
    from utils_vllm import (
        init_vllm_model,
        filter_by_max_length,
        vllm_batch_generate,
    )
else:
    import ollama
    from ollama import chat

DATASET_NAME = NWR_BENCHMARK_FILE

# count the number of news in the dataset
with open(NWR_TRAINING_FILE, "r", encoding="utf-8") as f:
    news_count = sum(1 for _ in f)


JUDGE_MODELNAME_OLLAMA = "qwen2.5:32b-instruct-q6_K"
JUDGE_MODELNAME_LLVM = "Qwen/Qwen2.5-32B-Instruct"

JUDGE_MODELNAME = JUDGE_MODELNAME_LLVM if USE_VLLM else JUDGE_MODELNAME_OLLAMA


TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    f"./qwen2.5-curriculum-trained_{news_count}news_4stage_A100",
    f"./qwen2.5-curriculum-trained_{news_count}news_5stage_A100",
    # R"Qwen2.5-0.5B-Instruct-curriculum_12903news_4stage_A100_old",
    # R"Qwen2.5-0.5B-Instruct-curriculum_12903news_5stage_A100_old",
    # R"Qwen2.5-0.5B-Instruct-curriculum_12903news_4stage_A100",
    # R"Qwen2.5-0.5B-Instruct-curriculum_12903news_5stage_A100",

    # R"qwen2.5-curriculum-trained_3184news_4stage_A100",
    # R"qwen2.5-curriculum-trained_3184news_5stage_A100",

    # "qwen2.5:0.5b-instruct",
    # "qwen2.5:1.5b-instruct",
    # "qwen2.5:3b-instruct",
    # "qwen2.5:7b-instruct",
    # "qwen2.5:14b-instruct",
]


""" ----------------------- Global news data -------------------------- """

nwrs: list[NewsWithRationale] = []
with open(DATASET_NAME, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        nwrs.append(NewsWithRationale.from_dict(data))

nwrs = nwrs[:100]  # for demonstration

""" ----------------------- Global news data end ----------------------- """

eval_logger = Logger("eval score", verbose_level=3)
eval_logger.info(f"Total news count: {news_count}")


def evaluate_with_rouge(predictions, references) -> list[dict]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [
        scorer.score(pred, ref) for pred, ref in zip(predictions, references)
    ]
    return scores


def evaluate_with_bertscore(predictions, references) -> dict:
    P, R, F1 = bert_score.score(
        predictions, references, lang="en", verbose=False
    )
    return {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}


def judge_summary_1_to_20_prompt_sys() -> str:
    return """\
You are an expert language evaluator. Your task is to assess the quality of a \
model-generated summary based on the article and the ground-truth summary. \
Score it from 1 to 20 using the following rubric:
0: Unreadable format.
1: Totally irrelevant, unrelated to the article.
2: Hallucinates facts, makes no sense.
3: Severe misunderstanding, contains major errors.
4: Barely reflects the source, very incomplet.
5: Grammatical errors, lacks coherence and relevance.
6: Incomplete and partially off-topic.
7: Misses key points, contains minor hallucinations.
8: Vague summary, lacks specificity.
9: Concise and covers most major points.
10: Understandable but may miss subtle nuances.
11: Faithful with minor omissions.
12: Mostly accurate but slightly redundant.
13: Accurate, well-structured, minor stylistic issues.
14: Good coverage and clarity, tone could improve.
15: Clear, faithful, and stylistically strong.
16: Concise, elegant, and captures all key points.
17: Very close to ideal summary, only minor flaws.
18: Excellent summary, highly readable and comprehensive.
19: Near perfect, minor stylistic polish could be added.
20: Perfect — clear, faithful, complete, and elegant.

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
    id_list: list[int], articles: list[str],
    gen_sums: list[str], ground_truths: list[str],
    model_name: str, judge_model: str = JUDGE_MODELNAME,
    save_result: bool = True, regenerate: bool = False
) -> list[int]:
    """
    Get the responses string from the judge model.
    """
    assert len(articles) == len(gen_sums) == len(ground_truths)
    sys_prompt = judge_summary_1_to_20_prompt_sys()

    save_name = legalize_filename(model_name)
    judge_name = legalize_filename(judge_model)
    filepath = f"benchmark_result/{save_name}_judged_by_{judge_name}.json"

    judge_history: dict[int, str] = {}

    if not regenerate and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            judge_history_raw: dict[str, str] = json.load(f)
            judge_history = {
                int(k): v for k, v in judge_history_raw.items()
            }

    id_todo = [id for id in id_list if id not in judge_history.keys()]

    articles = [news for (id, news) in zip(id_list, articles) if id in id_todo]
    gen_sums = [summ for (id, summ) in zip(id_list, gen_sums) if id in id_todo]
    gts = [gt for (id, gt) in zip(id_list, ground_truths) if id in id_todo]

    if USE_VLLM:
        tokenizer = AutoTokenizer.from_pretrained(judge_model)
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": judge_summary_1_to_20_prompt_user(
                            article, gen_sum, gt
                        )
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for article, gen_sum, gt in zip(articles, gen_sums, gts)
        ]
        llm, sampling_params = init_vllm_model(
            judge_model, MAX_BENCHMARK_LENGTH, MAX_NEW_TOKENS
        )
        prompts, articles, gen_sums, gts, id_todo = filter_by_max_length(
            MAX_BENCHMARK_LENGTH, prompts, articles, gen_sums, gts, id_todo
        )
        responses = vllm_batch_generate(llm, prompts, sampling_params)
        outputs = [response.outputs[0].text for response in responses]
        for i, id in enumerate(id_todo):
            judge_history[id] = outputs[i]
        del llm, tokenizer, sampling_params
        cleanup()
    else:
        ollama.pull(judge_model)

        sys_prompt = judge_summary_1_to_20_prompt_sys()
        outputs = []

        for i, (article, summ, gt) in tqdm(
            enumerate(zip(articles, gen_sums, gts)), total=len(articles)
        ):
            # Process each article and summary
            user_prompt = judge_summary_1_to_20_prompt_user(article, summ, gt)
            gen_response = chat(
                model=judge_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            assert gen_response.message.content is not None
            # outputs.append(gen_response.message.content)
            judge_history[id_todo[i]] = gen_response.message.content

    # Extract the scores from the output
    scores = []
    for id in id_list:
        score = -1
        output = judge_history[id]
        match = re.search(r"^\s*(Score:)?\s*(\d+)\s*—?(.*)", output)
        if match:
            try:
                score = int(match.group(2))
                judge_history[id] = output
            except ValueError:
                eval_logger.error(f"Invalid score format: {output}")
        else:
            eval_logger.error(f"Invalid output format: {output}")
        scores.append(score)

    if save_result:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(judge_history, f, indent=4, ensure_ascii=False)

    return scores


def prepare_benchmark_response(
    model_name: str, saving: bool = True, regenerate: bool = False
) -> dict[int, dict[str, str]]:

    save_name = legalize_filename(model_name)
    filepath = f"benchmark_result/{save_name}_responses.json"

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            eval_logger.info(f"Skipping generating response of {model_name}")
            histories_raw: dict[str, dict[str, str]] = json.load(f)
            histories: dict[int, dict[str, str]] = {
                int(k): v for k, v in histories_raw.items()
            }
    else:
        histories = {}

    news_list = [nwr.article for nwr in nwrs]
    summaries = [nwr.summary for nwr in nwrs]

    _nwrs: list[NewsWithRationale] = nwrs.copy()  # local nwrs to process
    finished_id = []

    if not regenerate:
        finished_id = [id for id in histories.keys()]
    # eval_logger.info(f"{finished_id:}")

    _nwrs = [nwr for nwr in _nwrs if nwr.id not in finished_id]
    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DL.DIRECT_SUMMARY]

    if USE_VLLM:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "article:\n" + nwr.article}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for nwr in nwrs if nwr.id not in finished_id
        ]
        prompts, news_list, summaries, _nwrs = filter_by_max_length(
            MAX_BENCHMARK_LENGTH, prompts, news_list, summaries, _nwrs
        )
        llm, sampling_params = init_vllm_model(
            model_name, MAX_BENCHMARK_LENGTH, MAX_NEW_TOKENS
        )
        responses_raw = vllm_batch_generate(llm, prompts, sampling_params)
        responses = [response.outputs[0].text for response in responses_raw]
        for (nwr, response) in zip(_nwrs, responses):
            histories[nwr.id] = {
                "news": nwr.article,
                "summary": nwr.summary,
                "response": response,
            }
        del llm, tokenizer, sampling_params, prompts
        cleanup()
    else:
        ollama.pull(model_name)
        responses = []
        # Process each NewsWithRationale object
        for nwr in tqdm(nwrs, desc="Generating responses", total=len(nwrs)):
            if nwr.id in finished_id:
                continue
            gen_response = chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "article:\n" + nwr.article}
                ]
            )
            assert gen_response.message.content is not None
            histories[nwr.id] = {
                "news": nwr.article,
                "summary": nwr.summary,
                "response": gen_response.message.content,
            }
            # responses.append(gen_response.message.content)

    # responses = [histories[nwr.id]["response"] for nwr in nwrs]

    if saving:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(histories, f, indent=4, ensure_ascii=False)

    return histories


def benchmark_model(model_name: str, save_results: bool = True) -> dict:

    response_file = legalize_filename(model_name)
    filepath = f"benchmark_result/{response_file}_responses.json"
    data: dict[int, dict[str, str]]

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data_raw: dict[str, dict[str, str]] = json.load(f)
            data = {int(k): v for k, v in data_raw.items()}
    else:
        data = prepare_benchmark_response(model_name, save_results)

    id_list = []
    responses = []
    summaries = []
    news_list = []

    for k, dat in data.items():
        id_list.append(k)
        responses.append(dat["response"])
        summaries.append(dat["summary"])
        news_list.append(dat["news"])

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
        id_list=id_list,
        articles=news_list,
        gen_sums=processed_responses,
        ground_truths=summaries,
        model_name=model_name,
        judge_model=JUDGE_MODELNAME,
        save_result=save_results,
        regenerate=False,
    )
    judge_scores = [score for score in judge_scores if score > 0]

    results = {
        "model_name": model_name,
        "num_samples": len(nwrs),
        "judge_model": JUDGE_MODELNAME,
        "avg_bert_scores": avg_bert_scores(bert_scores),
        "avg_rouge_scores": avg_rouge_scores(rouge_scores),
        "avg_judge_scores": avg_judge_scores(judge_scores),
        "bert_scores": bert_scores,
        "rouge_scores": rouge_scores,
        "judge_scores": judge_scores,
        "predictions": processed_responses,
    }
    eval_logger.info(f"Results for {model_name}:")
    eval_logger.info(f"Bert scores: {results['avg_bert_scores']}")
    eval_logger.info(f"Rouge scores: {results['avg_rouge_scores']}")
    eval_logger.info(f"Judge scores: {results['avg_judge_scores']}")

    if save_results:
        response_file = legalize_filename(model_name)
        filepath = f"benchmark_result/{response_file}_results.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    return results


def avg_rouge_scores(rouge_scores):
    avg_scores = {}
    for score in rouge_scores[0].keys():
        avg_scores[score] = 0 if len(rouge_scores) == 0 else sum(
            [rouge_score[score].fmeasure for rouge_score in rouge_scores]
        ) / len(rouge_scores)
    return avg_scores


def avg_bert_scores(bert_scores):
    avg_scores = {}
    for score in bert_scores.keys():
        avg_scores[score] = 0 if len(bert_scores) == 0 else sum(
            [bert_score for bert_score in bert_scores[score]]
        ) / len(bert_scores[score])
    return avg_scores


def avg_judge_scores(judge_scores):
    avg_score = 0 if len(judge_scores) == 0 else sum(
        judge_scores
    ) / len(judge_scores)

    # Normalize the score to a scale of 0 to 1
    return avg_score / 20.0


if __name__ == "__main__":
    # create a directory to save the results
    os.makedirs("benchmark_result", exist_ok=True)

    for model_name in TEST_MODELS:
        eval_logger.info(f"Generating responses: {model_name}")
        prepare_benchmark_response(model_name, saving=True)
        # prepare_benchmark_response(model_name, saving=True, regenerate=True)
        cleanup()

    cleanup()

    for model_name in TEST_MODELS:
        eval_logger.info(f"Evaluating model: {model_name}")
        benchmark_model(model_name, save_results=True)
        cleanup()
