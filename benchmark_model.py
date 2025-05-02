import json
import os
from dataclasses import dataclass

import bert_score
import jinja2
import re
from rouge_score import rouge_scorer
from tqdm import tqdm


from curriculum_training.constants import (
    USE_VLLM,
    ALLOW_VLLM,
    ALLOW_OLLAMA,
    InferenceType,
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


if ALLOW_VLLM:
    from transformers import AutoTokenizer
    from utils_vllm import (
        init_vllm_model,
        filter_by_max_length,
        vllm_batch_generate,
    )

if ALLOW_OLLAMA:
    import ollama
    from ollama import chat

DATASET_NAME = NWR_BENCHMARK_FILE

# count the number of news in the dataset
with open(NWR_TRAINING_FILE, "r", encoding="utf-8") as f:
    news_count = sum(1 for _ in f)


# JUDGE_MODELNAME_OLLAMA = "qwen2.5:32b-instruct-q6_K"
JUDGE_MODELNAME_OLLAMA = "qwen2.5:14b-instruct"
JUDGE_MODELNAME_LLVM = "Qwen/Qwen2.5-32B-Instruct"

JUDGE_MODELNAME = JUDGE_MODELNAME_LLVM if USE_VLLM else JUDGE_MODELNAME_OLLAMA


TEST_MODELS: list[str] = [
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    # f"./qwen2.5-curriculum-trained_{news_count}news_4stage_A100",
    # f"./qwen2.5-curriculum-trained_{news_count}news_5stage_A100",
    # Rf"Qwen2.5-0.5B-Instruct-curriculum_{news_count}news_4stage_A100",
    # Rf"Qwen2.5-0.5B-Instruct-curriculum_{news_count}news_5stage_A100",
    # R"Qwen2.5-0.5B-Instruct-curriculum_12903news_4stage_A100_old",
    # R"Qwen2.5-0.5B-Instruct-curriculum_12903news_5stage_A100_old",
    # R"Qwen2.5-0.5B-Instruct-curriculum_12903news_4stage_A100",
    # R"Qwen2.5-0.5B-Instruct-curriculum_12903news_5stage_A100",
    # R"Qwen2.5-0.5B-Instruct-curriculum_5000news_4stage_A100",
    # R"Qwen2.5-0.5B-Instruct-curriculum_500news_4stage_A100",
    # R"Qwen2.5-0.5B-Instruct-curriculum_50news_4stage_A100",
    # R"Qwen2.5-0.5B-Instruct-curriculum_0news_4stage_A100",

    # R"qwen2.5-curriculum-trained_3184news_4stage_A100",
    # R"qwen2.5-curriculum-trained_3184news_5stage_A100",

    # "qwen2.5:0.5b-instruct",
    # "qwen2.5:1.5b-instruct",
    # "qwen2.5:3b-instruct",
    # "qwen2.5:7b-instruct",
    # "qwen2.5:14b-instruct",

    # Gemma models
    # "google/gemma-2-2b-it",
    # "google/gemma-3-1b-it",
    # "google/gemma-3-4b-it",
]

DEFAULT_METHOD: InferenceType = "VLLM" if USE_VLLM else "OLLAMA"


@dataclass
class BenchmarkObj:
    """
    A queue for benchmarking models.
    """
    model_name: str
    inference_method: InferenceType = DEFAULT_METHOD
    use_model_judge: bool = True
    judge_method: InferenceType = DEFAULT_METHOD
    judge_model: str = "None"

    def __post_init__(self):
        """
        Post-initialization to validate and adjust attributes if necessary.
        """
        if self.use_model_judge and self.judge_model == "None":
            if self.judge_method == "VLLM":
                assert USE_VLLM
                self.judge_model = JUDGE_MODELNAME_LLVM
            else:
                assert ALLOW_OLLAMA
                self.judge_model = JUDGE_MODELNAME_OLLAMA


benchmark_queue: list[BenchmarkObj] = []

for name in TEST_MODELS:
    if name.startswith("Qwen2.5-0.5B-Instruct-curriculum"):
        assert USE_VLLM
        benchmark_queue.append(
            BenchmarkObj(name, inference_method="VLLM")
        )
    elif name.startswith("qwen2.5:"):
        benchmark_queue.append(
            BenchmarkObj(name, inference_method="OLLAMA")
        )
    else:
        benchmark_queue.append(
            BenchmarkObj(name)
        )
    print(benchmark_queue[-1])

""" ----------------------- Global data -------------------------- """

nwrs: list[NewsWithRationale] = []
with open(DATASET_NAME, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        nwrs.append(NewsWithRationale.from_dict(data))

nwrs = nwrs[:102]  # for demonstration

llm = None
sampling_params = None
tokenizer = None
prev_judge_model = None

""" ----------------------- Global data end ----------------------- """

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
        predictions, references, lang="zh-hant", verbose=False
    )
    return {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}


def judge_summary_0_to_20_prompt_sys_eng() -> str:
    return """\
You are an expert language evaluator. Your task is to assess the quality of a \
model-generated summary based on the article and the ground-truth summary. \
Score it from 0 to 20 using the following rubric:
0: Unreadable format or gibberish.
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

Return only an integer score (0–20), followed by a short reason \
(e.g., "Score: 17 — Very close to ideal summary, only minor flaws").
"""


def judge_summary_0_to_20_prompt_sys() -> str:
    return """\
你是一位語言評估專家。你的任務是根據文章與標準摘要，評估模型生成的摘要品質。
請根據以下評分標準，從 0 到 20 為其打分：
0：格式不正確或無意義的文字。
1：完全無關，與文章毫不相干。
2：虛構內容，語意不明。
3：嚴重誤解，包含重大錯誤。
4：幾乎無法反映原文，非常不完整。
5：文法錯誤，缺乏連貫性與相關性。
6：內容不完整且部分離題。
7：遺漏關鍵要點，有輕微虛構。
8：摘要過於模糊，缺乏具體性。
9：簡潔，涵蓋大部分重點。
10：可理解但可能遺漏細節。
11：忠實但略有遺漏。
12：大致正確但稍顯冗餘。
13：準確、結構良好，但有輕微風格問題。
14：涵蓋完整、清晰，語氣尚可改進。
15：清楚、忠實且具風格。
16：簡潔優雅，涵蓋所有重點。
17：非常接近理想摘要，僅有些微瑕疵。
18：優秀的摘要，易讀且內容完整。
19：幾近完美，僅可做細微風格潤飾。
20：完美——清楚、忠實、完整且優雅。
請回傳"分數："加一個整數分數（0–20），接著是一句簡短的理由（例如：「分數：17 —— 非常接近理想摘要，僅有些微瑕疵」）。
"""


def judge_summary_0_to_20_prompt_user_eng(article, response, ground_truth):
    return (
        f"Article: {article}\n\n"
        f"Ground Truth: {ground_truth}\n\n"
        f"Model Output: {response}"
    )


def judge_summary_0_to_20_prompt_user(article, response, ground_truth) -> str:
    return (
        f"文章：\n{article}\n\n"
        f"標準摘要：\n{ground_truth}\n\n"
        f"模型生成摘要：\n{response}"
    )


def extract_scores_from_responses(
    id_list: list[int], gen_sums: list[str], judge_history: dict[int, str],
    gen_responses: dict[int, str], filepath: str = "", saving: bool = True,
) -> list[int]:
    # Extract the scores from the output
    scores = []
    for (id, gen_sum) in zip(id_list, gen_sums):
        score = -1
        if id in judge_history.keys():
            output = judge_history[id]
        elif id in gen_responses.keys():
            output = gen_responses[id]
        else:
            continue

        # match = re.search(r"^\s*(Score:)?\s*(\d+)\s*—?(.*)", output)
        match = re.search(r"^\s*(分數：)?\s*(\d+)\s*—?(.*)", output)
        if match:
            try:
                score = int(match.group(2))
                judge_history[id] = output
            except ValueError:
                eval_logger.error(f"Invalid score format: {output}")
        else:
            eval_logger.error(f"Invalid output format: {output}")

        # set no output as 0
        if gen_sum == "":
            score = 0

        scores.append(score)

    if saving:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(judge_history, f, indent=4, ensure_ascii=False)

    return scores


def judge_summary_0_to_20(
    id_list: list[int], articles: list[str],
    gen_sums: list[str], ground_truths: list[str],
    model_name: str, judge_model: str = JUDGE_MODELNAME,
    judge_method: str = DEFAULT_METHOD,
    saving: bool = True, regenerate: bool = False
) -> list[int]:
    """
    Get the responses string from the judge model.
    """
    global llm, sampling_params, tokenizer, prev_judge_model

    assert len(articles) == len(gen_sums) == len(ground_truths) == len(id_list)
    sys_prompt = judge_summary_0_to_20_prompt_sys()

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

    if len(id_todo) == 0:
        eval_logger.info("No data to generate.")
        return extract_scores_from_responses(
            id_list, gen_sums, judge_history, {}, filepath, saving
        )

    articles = [news for (id, news) in zip(id_list, articles) if id in id_todo]
    gen_sums = [summ for (id, summ) in zip(id_list, gen_sums) if id in id_todo]
    gts = [gt for (id, gt) in zip(id_list, ground_truths) if id in id_todo]
    gen_responses: dict[int, str] = {}

    if judge_method == "VLLM":
        if prev_judge_model != judge_model:
            if llm is not None:
                del llm
            if sampling_params is not None:
                del sampling_params
            if tokenizer is not None:
                del tokenizer
            cleanup()
            tokenizer = AutoTokenizer.from_pretrained(judge_model)

        assert tokenizer is not None
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": judge_summary_0_to_20_prompt_user(
                            article, gen_sum, gt
                        )
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for article, gen_sum, gt in zip(articles, gen_sums, gts)
        ]
        prompts, articles, gen_sums, gts, id_todo = filter_by_max_length(
            MAX_BENCHMARK_LENGTH, prompts, articles, gen_sums, gts, id_todo
        )
        if len(prompts) == 0:
            eval_logger.info("No data to generate.")
            return extract_scores_from_responses(
                id_list, gen_sums, judge_history, {}, filepath, saving
            )
        else:
            eval_logger.info(f"Generating {len(prompts)} judge responses.")
            eval_logger.info(f"id_todo: {id_todo}")

        if prev_judge_model != judge_model:
            llm, sampling_params = init_vllm_model(
                judge_model, MAX_BENCHMARK_LENGTH, MAX_NEW_TOKENS
            )
            prev_judge_model = judge_model

        assert llm is not None and sampling_params is not None
        responses = vllm_batch_generate(llm, prompts, sampling_params)
        outputs = [response.outputs[0].text for response in responses]
        for i, id in enumerate(id_todo):
            gen_responses[id] = outputs[i]
        cleanup()
    else:
        ollama.pull(judge_model)

        sys_prompt = judge_summary_0_to_20_prompt_sys()
        outputs = []

        for (id, article, summ, gt) in tqdm(
            zip(id_todo, articles, gen_sums, gts), total=len(id_todo)
        ):
            # Process each article and summary
            user_prompt = judge_summary_0_to_20_prompt_user(article, summ, gt)
            gen_response = chat(
                model=judge_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            assert gen_response.message.content is not None
            # outputs.append(gen_response.message.content)
            # judge_history[id_todo[i]] = gen_response.message.content
            gen_responses[id_todo[i]] = gen_response.message.content

    return extract_scores_from_responses(
        id_list, gen_sums, judge_history, gen_responses, filepath, saving
    )


def prepare_benchmark_response(
    benchmark_obj: BenchmarkObj, saving: bool = True, regenerate: bool = False
) -> dict[int, dict[str, str]]:

    model_name = benchmark_obj.model_name

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

    _nwrs: list[NewsWithRationale] = nwrs.copy()  # local nwrs to process
    finished_id = []

    if not regenerate:
        finished_id = [id for id in histories.keys()]
    # eval_logger.info(f"{finished_id:}")

    _nwrs = [nwr for nwr in _nwrs if nwr.id not in finished_id]

    if len(_nwrs) == 0:
        eval_logger.info("No data to generate.")
        return histories

    news_list = [nwr.article for nwr in _nwrs]
    summaries = [nwr.summary for nwr in _nwrs]
    sys_prompt = PREFIX_OF_DIFFICULTY_LEVELS[DL.DIRECT_SUMMARY]

    if benchmark_obj.inference_method == "VLLM":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
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
        except jinja2.exceptions.TemplateError as e:
            eval_logger.error(f"Error in tokenizer: {e}")
            # retry without system prompt
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user",
                         "content": f"{sys_prompt}\n\narticle:\n{nwr.article}"}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for nwr in nwrs if nwr.id not in finished_id
            ]
        prompts, news_list, summaries, _nwrs = filter_by_max_length(
            MAX_BENCHMARK_LENGTH, prompts, news_list, summaries, _nwrs
        )
        if len(prompts) == 0:
            eval_logger.info("No data to generate.")
        else:
            eval_logger.info(f"Generating {len(prompts)} model responses.")
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
        for nwr in tqdm(_nwrs, desc="Generating responses", total=len(_nwrs)):
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


def benchmark_model(benchmark_obj: BenchmarkObj, saving: bool = True) -> dict:

    model_name = benchmark_obj.model_name

    response_file = legalize_filename(model_name)
    filepath = f"benchmark_result/{response_file}_responses.json"
    data: dict[int, dict[str, str]]

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data_raw: dict[str, dict[str, str]] = json.load(f)
            data = {int(k): v for k, v in data_raw.items()}
    else:
        data = prepare_benchmark_response(benchmark_obj, saving)

    id_list_todo = [nwr.id for nwr in nwrs]
    id_list = []
    responses = []
    summaries = []
    news_list = []

    for k, dat in data.items():
        if k not in id_list_todo:
            continue
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
    if benchmark_obj.use_model_judge:
        judge_scores = judge_summary_0_to_20(
            id_list=id_list,
            articles=news_list,
            gen_sums=processed_responses,
            ground_truths=summaries,
            model_name=model_name,
            judge_method=benchmark_obj.judge_method,
            judge_model=benchmark_obj.judge_model,
            saving=saving,
            regenerate=False,
        )
    else:
        judge_scores = []
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

    if saving:
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

    for benchmark_obj in benchmark_queue:
        eval_logger.info(f"Generating responses: {benchmark_obj.model_name}")
        prepare_benchmark_response(benchmark_obj, saving=True)
        # prepare_benchmark_response(model_name, saving=True, regenerate=True)
        cleanup()

    cleanup()

    for benchmark_obj in benchmark_queue:
        eval_logger.info(f"Evaluating model: {benchmark_obj.model_name}")
        benchmark_model(benchmark_obj, saving=True)
        cleanup()
