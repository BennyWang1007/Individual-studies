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
    return legalize_filename(f"generated_zh-tw_responses_{model_name}.jsonl")
