
def get_rationale_prompt(doc: str, gt_summary: str) -> str:
    return f"""\
Given a document and its ground-truth summary, do the following tasks:
(1) According to the ground-truth summary, extract essential aspects of the document.
(2) For each essential aspect, retrieve detailed triples in the format [ENTITY1 | RELATION | ENTITY2] used to compose the ground-truth summary.
(3) With the retrieved triples, compose a summary. The essential aspects, triples, and composed summary should be in the same response, separated by a new line. All triples [ENTITY1 | RELATION | ENTITY2] should be in length 3 (separated by "|").
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
(1) According to the ground-truth summary, extract essential aspects of the document.
(2) For each essential aspect, retrieve detailed triples in the format [ENTITY1 | RELATION | ENTITY2] used to compose the ground-truth summary.
(3) With the retrieved triples, compose a summary. The essential aspects, triples, and composed summary should be in the same response, separated by a new line. All triples [ENTITY1 | RELATION | ENTITY2] should be in length 3 (separated by "|").
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
(3) 使用檢索到的三元組撰寫一份摘要。核心要素、三元組和撰寫的摘要應該在同一份回應中，並以換行符分隔。所有三元組 [實體1 | 關係 | 實體2] 的長度必須為3（以 "|" 分隔）。
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
(3) 使用檢索到的三元組撰寫一份摘要。核心要素、三元組和撰寫的摘要應該在同一份回應中，並以換行符分隔。所有三元組 [實體1 | 關係 | 實體2] 的長度必須為3（以 "|" 分隔）。
[文章]: {doc}
[摘要]: {gt_summary}
[核心要素]: 
"""
