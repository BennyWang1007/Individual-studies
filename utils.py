import json
import re


def get_rationale_prompt(doc: str, gt_summary: str) -> str:
    return f"""\
Given a document and its ground-truth summary, do the following tasks:
(1) According to the ground-truth summary, extract essential aspects of the \
document.
(2) For each essential aspect, retrieve detailed triples in the format \
[ENTITY1 | RELATION | ENTITY2] used to compose the ground-truth summary.
(3) With the retrieved triples, compose a summary. The essential aspects, \
triples, and composed summary should be in the same response, separated by \
a new line. All triples [ENTITY1 | RELATION | ENTITY2] should be in length \
3 (separated by "|").
Example:
================Example=================
Prompt:
[Document]: [document]
[Ground-truth Summary]: [ground-truth summary]
Update:
Essential Aspects:
[aspects]
Triples:
- [ENTITY1_1 | RELATION_1 | ENTITY1_2]
- [ENTITY2_1 | RELATION_2 | ENTITY2_2]
- [ENTITY3_1 | RELATION_3 | ENTITY3_2]
- ...
Generated Summary:
[summary]
========================================
Prompt:
[Document]: {doc}
[Ground-truth Summary]: {gt_summary}
Update:
"""


def get_rationale_prompt_no_gt(doc: str) -> str:
    return f"""\
Given a document, do the following tasks:
(1) According to the document, extract the ground-truth summary.
(1) According to the ground-truth summary, extract essential aspects of the \
document.
(2) For each essential aspect, retrieve detailed triples in the format \
[ENTITY1 | RELATION | ENTITY2] used to compose the ground-truth summary.
(3) With the retrieved triples, compose a summary. The essential aspects, \
triples, and composed summary should be in the same response, separated by \
a new line. All triples [ENTITY1 | RELATION | ENTITY2] should be in length 3 \
(separated by "|").
Example:
================Example=================
Prompt:
[Document]: [document]
Update:
Essential Aspects:
[aspects]
Triples:
- [ENTITY1_1 | RELATION_1 | ENTITY1_2]
- [ENTITY2_1 | RELATION_2 | ENTITY2_2]
- [ENTITY3_1 | RELATION_3 | ENTITY3_2]
- ...
Generated Summary:
[summary]
========================================
Prompt:
[Document]: {doc}
Update:
"""


def get_rationale_prompt_chinese(doc: str, gt_summary: str) -> str:
    return f"""\
給定一份文章及其摘要，完成以下任務：
(1) 根據摘要，提取文章的核心要素。
(2) 對於每個核心要素，檢索詳細的三元組，格式為 [實體1 | 關係 | 實體2]，這些三元組用於構成真實摘要。
(3) 使用檢索到的三元組撰寫一份摘要。核心要素、三元組和撰寫的摘要應該在同一份回應中，並以換行符分隔。\
所有三元組 [實體1 | 關係 | 實體2] 的長度必須為3（以 "|" 分隔）。
範例：
================範例=================
提示：
[文件]: [文件]
[摘要]: [摘要]
更新：
核心要素：
[核心要素]
三元組：

[實體1_1 | 關係_1 | 實體1_2]
[實體2_1 | 關係_2 | 實體2_2]
[實體3_1 | 關係_3 | 實體3_2]
...
生成摘要：
[摘要]
========================================
提示：
[文件]: {doc}
[摘要]: {gt_summary}
更新：
"""


def get_rationale_prompt_chinese2(doc: str, gt_summary: str) -> str:
    return f"""\
給定一份文章及其摘要，完成以下任務：
(1) 根據摘要，提取文章的核心要素。
(2) 對於每個核心要素，檢索詳細的三元組，格式為 [實體1 | 關係 | 實體2]，這些三元組用於構成真實摘要。
(3) 使用檢索到的三元組撰寫一份摘要。核心要素、三元組和撰寫的摘要應該在同一份回應中，並以換行符分隔。\
所有三元組 [實體1 | 關係 | 實體2] 的長度必須為3（以 "|" 分隔）。
[文章]: {doc}
[摘要]: {gt_summary}
[核心要素]:
"""


def get_rationale_prompt_no_gt_chinese_system(doc: str) -> str:
    return """\
給定一份文章，完成以下任務：
(1) 提取新聞的關鍵要素，關鍵要素應為關鍵短句、名詞或事實。
(2) 對於每個關鍵要素，檢索詳細的三元組，格式為 [實體1 | 關係 | 實體2]，這些三元組用於構成摘要。
(3) 使用檢索到的三元組撰寫一份摘要。
核心要素、三元組和撰寫的摘要應該在同一份回應中，並以換行符分隔。所有三元組 [實體1 | 關係 | 實體2] 的長度必須為3（以 "|" 分隔）。
範例：
================範例=================
提示：
[新聞]: [新聞]

更新：
核心要素：
[關鍵要素1]、[關鍵要素2]、[關鍵要素3]、...

三元組：
[實體1_1 | 關係_1 | 實體1_2]
[實體2_1 | 關係_2 | 實體2_2]
[實體3_1 | 關係_3 | 實體3_2]
...

生成摘要：
[摘要]
========================================
"""


def get_rationale_prompt_no_gt_chinese_user(doc: str) -> str:
    return f"""[新聞]: {doc}"""


def get_rationale_prompt_no_gt_chinese(doc: str) -> str:
    return f"""\
給定一份文章，完成以下任務：
(1) 提取文章的核心要素。
(2) 對於每個核心要素，檢索詳細的三元組，格式為 [實體1 | 關係 | 實體2]，這些三元組用於構成摘要。
(3) 使用檢索到的三元組撰寫一份摘要。
核心要素、三元組和撰寫的摘要應該在同一份回應中，並以換行符分隔。所有三元組 [實體1 | 關係 | 實體2] 的長度必須為3（以 "|" 分隔）。
範例：
================範例=================
提示：
[文章]: [文章]

更新：
核心要素：
[核心要素]

三元組：
[實體1_1 | 關係_1 | 實體1_2]
[實體2_1 | 關係_2 | 實體2_2]
[實體3_1 | 關係_3 | 實體3_2]
...

生成摘要：
[摘要]
========================================
提示：
[文章]: {doc}

更新：
"""


def legalize_filename(filename: str) -> str:
    """ Replace illegal characters in filename to underscores """
    return re.sub(r'[\/\\:*?"<>|]', '_', filename)


def int_set_str(int_set: set[int]) -> str:
    """
    Print the set of integers in a readable format.
    For example, if the set is {1, 2, 3, 5, 6, 7}, it will print "1-3, 5-7".
    """
    prev_id: int = -2
    continuous_count: int = 0
    id_list: list[int] = sorted(list(int_set))
    out_strs: list[str] = []
    for i in range(len(id_list)):
        if id_list[i] == prev_id + 1:
            prev_id += 1
            continuous_count += 1
            continue
        else:
            if continuous_count > 0:
                out_strs.append(f"{prev_id - continuous_count}-{prev_id}")
                continuous_count = 0
                prev_id = id_list[i]
            else:
                if prev_id != -2:
                    out_strs.append(f"{prev_id}")
                prev_id = id_list[i]

    if continuous_count > 0:
        out_strs.append(f"{prev_id - continuous_count}-{prev_id}")

    return ", ".join(out_strs)


def load_udn_news() -> list[str]:
    """ Load UDN news from the saved file """
    news: list[str] = []
    with open("crawler/saved_news/udn_news.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "content" not in data or not data["content"]:
                continue
            news.append(data["content"].strip())
    return news


def get_response_filename(model_name: str) -> str:
    """ Get the response filename """
    return legalize_filename(f"generated_responses_{model_name}.jsonl")


def get_news_with_rationale_filename(model_name: str) -> str:
    """ Get the news with rationale filename """
    return legalize_filename(f"generated_nwr_{model_name}.jsonl")


def get_formatted_nwr_filename(model_name: str) -> str:
    """ Get the formatted news with rationale filename """
    return legalize_filename(f"formatted_nwr_{model_name}.jsonl")


def get_zh_tw_filename(model_name: str) -> str:
    """ Get the zh_tw filename """
    return "training_data_v1/" + legalize_filename(
        f"generated_zh-tw_responses_{model_name}.jsonl"
    )


STDOUT_HOOKED = False


def hook_stdout():
    """ Hook stdout to a file """
    import atexit
    import sys

    global STDOUT_HOOKED
    if STDOUT_HOOKED:
        return
    STDOUT_HOOKED = True

    log_file_out = open("./stdout_log.txt", "a", encoding="utf-8")
    log_file_err = open("./stderr_log.txt", "a", encoding="utf-8")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()  # immediate write
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    def cleanup_logfile():
        try:
            log_file_out.flush()
            log_file_out.close()
        except Exception:
            pass
        try:
            log_file_err.flush()
            log_file_err.close()
        except Exception:
            pass

    atexit.register(cleanup_logfile)
    sys.stdout = Tee(sys.__stdout__, log_file_out)
    sys.stderr = Tee(sys.__stderr__, log_file_err)

    print("\n\nstdout and stderr are hooked to files: "
          "stdout_log.txt and stderr_log.txt.")


def ljust_labels(labels: list[str], width: int = 20) -> list[str]:
    """
    Left-justify labels to a specified width.
    """
    # Split the longest label into two lines if needed
    max_length = max(len(label) for label in labels)

    one_line_labels = [label for label in labels if len(label) <= width]
    if one_line_labels:
        ljust_len = max(len(label) for label in labels if len(label) < width)
    else:
        ljust_len = 0
    to_adj: list[tuple[int, int]] = []

    # if max_length > width:
    for idx in range(len(labels)):
        label = labels[idx]
        if len(label) <= width:
            continue
        split_point = len(label) // 2
        # Try to split at a dash or underscore if possible
        for sep in ['-', '_']:
            idx2 = label.find(sep, split_point - 5, split_point + 5)
            if idx2 != -1:
                split_point = idx2 + 1
                break

        ljust_len = max(ljust_len, split_point)
        ljust_len = max(ljust_len, len(label) - split_point)

        to_adj.append((idx, split_point))

    # Left-justify all labels to the maximum length
    for idx in range(len(labels)):
        if len(labels[idx]) <= max_length:
            labels[idx] = labels[idx].ljust(ljust_len)

    # print(f"Longest label length: {ljust_len}")
    # print(f"Labels to adjust: {to_adj}")

    # Split the labels at the split points and ljust them
    for idx, split_point in to_adj:
        label = labels[idx]
        labels[idx] = (
            label[:split_point].ljust(ljust_len)
            + '\n'
            + label[split_point:].ljust(ljust_len)
        )

    return labels


def get_simple_name(name: str) -> str:
    name = name.replace("v2", "v3")
    name = name.replace("better2", "v2").replace("better", "v2")

    patterns = [
        # parse trained models
        (R"^(Qwen/)?Qwen([0-9\.]+)-([0-9\.]+B)-Instruct"
         R"(-curriculum|-cl)?_([0-9]+)news_([0-9])(stage|stg)"
         R"(_A100)?(.*)?$",
         lambda m: (
             f"Qwen{m.group(2)}-{m.group(3)}"
             f"_{m.group(6)}stg{m.group(9) or ''}"
         )),
        # parse gemma models
        (R"^google/gemma-([0-9])-([0-9\.]+b)-it$",
         lambda m: f"Gemma-{m.group(1)}-{m.group(2)}"),
        # parse Qwen models
        (R"^Qwen/Qwen([0-9\.]+)-([0-9\.]+B)(-Instruct$)?",
            lambda m: f"Qwen{m.group(1)}-{m.group(2)}"),
        # parse Custom models
        (
            R"^CustomQwen([0-9]+)Model(_pretrained)?-cl_([0-9]+)news_([0-9])"
            R"(stage|stg)(.*)$",
            lambda m: f"Custom{'_pre' if m.group(2) else ''}"
                      f"-{m.group(3)}n_{m.group(4)}stg{m.group(6) or ''}"
        ),
        # parse Llama models
        (R"^meta-llama/Llama-([0-9\.]+)-([0-9]+B)-Instruct$",
         lambda m: f"Llama-{m.group(1)}-{m.group(2)}"),
        # parse DeepSeek models
        (R"^deepseek-ai/DeepSeek-R1-Distill-Qwen-([0-9.]+)B$",
         lambda m: f"DeepSeek-R1-{m.group(1)}B"),
    ]

    for pattern, fmt in patterns:
        match = re.match(pattern, name)
        if match:
            return fmt(match)

    # return name
    print(f"Unknown model name format: {name}")
    return name
