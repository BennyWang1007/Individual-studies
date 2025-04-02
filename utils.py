import json
import os


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


def get_rationale_prompt_no_gt_chinese_system(doc: str) -> str:
    return f"""\
給定一份文章及其摘要，完成以下任務：
(1) 提取文章的核心要素。
(2) 對於每個核心要素，檢索詳細的三元組，格式為 [實體1 | 關係 | 實體2]，這些三元組用於構成真實摘要。
(3) 使用檢索到的三元組撰寫一份摘要。核心要素、三元組和撰寫的摘要應該在同一份回應中，並以換行符分隔。所有三元組 [實體1 | 關係 | 實體2] 的長度必須為3（以 "|" 分隔）。
(4) 生成摘要。
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
"""

def get_rationale_prompt_no_gt_chinese_user(doc: str) -> str:
    return f"""\
========================================
提示：
[文件]: {doc}
更新：
"""


def get_rationale_prompt_no_gt_chinese(doc: str) -> str:
    return f"""\
給定一份文章及其摘要，完成以下任務：
(1) 提取文章的核心要素。
(2) 對於每個核心要素，檢索詳細的三元組，格式為 [實體1 | 關係 | 實體2]，這些三元組用於構成真實摘要。
(3) 使用檢索到的三元組撰寫一份摘要。核心要素、三元組和撰寫的摘要應該在同一份回應中，並以換行符分隔。所有三元組 [實體1 | 關係 | 實體2] 的長度必須為3（以 "|" 分隔）。
(4) 生成摘要。
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
更新：
"""

def legalize_filename(filename: str) -> str:
    return filename.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")


def get_sample_news() -> str:
    """ Get a sample news for testing """
    return "華商報寶雞訊“救命啊!”3月26日下午3時許,寶雞石頭河水庫下游,兩名12歲男童不慎落水,危急時刻,候天祥、李海嘯、王凱三名青年男子相繼跳入水中,接力施救,孩子們轉危為安,但36歲的候天祥卻因體力不支,不慎溺水身亡。\n生命的最後一刻,他用盡最後一絲力氣,將抓在手裡的孩子推向岸邊。\n小女孩攔車喊救命。\n他顧不上將車熄火衝下河道石頭河水庫位於眉縣、岐山、太白三縣交界處,水庫大壩下游建有一道用於緩衝水流的小石壩,小石壩的一側匯成了一片小水域。\n據附近村民講,此處正是事發地,最深處約有5米。\n候天祥的家就位於水庫不遠處的山坡上,水庫旁邊的盤山路是他每次回家的必經之路,10年前,候天祥就曾救過一個落水兒童,但3月26日救落水兒童時自己卻不幸溺亡,十里八鄉的人都為此心痛。\n目擊者白女士說,26日當天,她就坐在候天祥駕駛的麵包車上,因此目睹了整個事發過程。\n當時,車上還坐著候天祥的愛人、兩個女兒和一位親戚程女士,一行人準備回到候天祥的家裡取野菜。\n下午3時許,途經水庫時,幾個小女孩一邊大喊“救命”,一邊攔住了他們的麵包車。\n聞聽有兩名男童落水,候天祥顧不上熄火,開啟車門便衝下河道,當時水面上只能看到了兩個小腦袋在上上下下,情況非常危急。\n伸出竹竿夠不著。\n他衝進水中游向男孩候天祥先是拾起岸邊的一個竹竿,伸向水面,但因為竹竿長度有限,距離最近的孩子仍有一尺距離。\n這時,候天祥乾脆扔掉竹竿,也顧不上脫衣服,直接衝入水中向兩個男孩游去。\n他拽住其中一個男孩,奮力向岸邊遊,此時,一名戴眼鏡的男子也跳入水中,奮力將另外一個男孩拖上了岸,但自己也累得癱坐地上。\n白女士說,此時,正在水中施救的候天祥明顯體力不支,突然下沉,而沉下去的一刻,他用盡最後一絲力氣,把孩子向岸邊推了一把。\n她趕緊請戴眼鏡的男子下水救人,但剛救完一人的男子癱坐在地上只是擺手,表示自己已沒有力氣下水了。\n還好,另外一名男子及時趕到,衝入水中,先將男童救上岸,隨後又將候天祥抱了上來。\n白女士說,候天祥被救上岸後,臉色蒼白,大家趕緊撥打120急救。\n這時,現場來了一男一女兩人,男的抬起候天祥的頭,女的不斷擠壓候天祥的胸口,併為他做人工呼吸,施救約40分鐘,候天祥嘔吐了幾次。\n120急救車到來後,一男一女離開現場。\n120急救人員現場展開施救,遺憾的是候天祥最終還是未能搶救過來。\n據候天祥的家屬介紹,候天祥有兩個女兒,一個8歲,一個2歲,候天祥是家裡的頂樑柱。\n另一名救人者說“他確實讓人感動”經過多方打聽,昨日下午,華商報記者終於聯絡到了另外兩名施救男子,戴眼鏡的男子名叫李海嘯,27歲,家住蔡家坡工業園安樂社群,陝西法士特集團公司一名車間職工;最後一位衝入水中施救的男子名叫王凱,也是27歲,眉縣齊鎮人,是一家屠宰廠的職工。\n李海嘯說,當天他也開著麵包車,因盤山路有幾個女孩喊救命,他就趕緊衝下了河道。\n成功救上一個男孩後,他已精疲力竭,胳膊都伸不直了,所以當一位女士喊他救人時,他實在無力下水了,只能無奈擺手。\n王凱表示,當時看到出事了他就衝下了河道,發現水中漂著一個男孩,許多人在圍觀,他一邊脫衣服,一邊問“誰會游泳?”結果無人應答,他便衝入水中,將男孩救上岸後,原以為沒事了,但聽說水裡還有一人,當時水很渾濁,他便按照岸邊人指的方向摸了過去,將一名中年男子抱上了岸。\n李海嘯、王凱均證實,之後確有一男一女兩人對候天祥進行施救,壓胸和人工呼吸的動作看上去非常專業。\n李海嘯及另一位目擊者程女士也都表示,他們確實看到候天祥在沉下去的一刻,用力將男孩向岸邊推去。\n李海嘯說:“我知道水裡救人很費勁,候天祥肯定是體力不支,能用最後一絲力氣把男娃推向岸邊,確實讓人感動!”兩獲救男童今年都12歲。\n系失足落水據被救小孩的父親王先生介紹,兩名被救的兒童今年都是12歲,正在讀小學6年級,一個家在岐山、一個家在眉縣,當天他們5個小夥伴相約到水庫玩耍,一個男孩踩著河道里的鵝卵石玩耍時,不慎失足絆倒另一個男孩,兩人雙雙跌入水中。\n“太感謝三位英雄了,特別是候天祥,我都不知道咋用言語表達自己的謝意……”\n昨日下午,眉縣齊鎮斜谷村三名村委會領導來到候天祥的家中,將1000元慰問金交到了候天祥家屬的手中。\n村主任尹滿成說:“被救的一個孩子是我們村的,候天祥捨身救人,斜谷人不會忘了他的恩情。”\n而候天祥愛人的公司聽聞候天祥的事蹟後,在昨日上午專門舉辦了一場追思會,恰逢公司舉辦植樹活動,36棵新栽的樹苗上綁上了36朵白花,活動負責人楊先生說:“候天祥雖然人不在了,但他見義勇為的精神,將和這些樹苗一樣茁壯成長,在大家的心裡迎來新生。”"""


def get_test_news() -> list[str]:
    """ Get test news from the test_news directory """
    TEST_NEWS_DIR = "test_news"
    test_news_list = []

    for file in os.listdir(TEST_NEWS_DIR):
        with open(os.path.join(TEST_NEWS_DIR, file), "r", encoding="utf-8") as f:
            test_news_list.append(f.read())

    return test_news_list


def get_prompt(news: str) -> str:
    """ Get the prompt for testing the model """
    return "請精簡地用台灣用語、繁體中文對以下新聞總結，不超過100字。\n\n" + news + "\n"
    return "請用台灣用語、繁體中文對以下新聞總結：\n" + news + "\n"
    return "請對以下新聞總結：\n" + news + "\n"


def load_udn_news() -> list[str]:
    """ Load UDN news from the saved file """
    news: list[str] = []
    with open("crawler/saved_news/udn_news.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            news.append(data["content"])
    return news
