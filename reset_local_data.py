
import json
from tqdm import tqdm

from curriculum_training.constants import MODEL_BASE, MODEL_DISTAL_FROM
from utils import (
    load_udn_news,
    get_news_with_rationale_filename,
    get_response_filename,
    get_zh_tw_filename,
)

REPLACE_FILE = True

RESPONSE_FILE = get_response_filename(MODEL_DISTAL_FROM)
NWR_FILE = get_news_with_rationale_filename(MODEL_DISTAL_FROM)
ZH_TW_FILE = get_zh_tw_filename(MODEL_BASE)

if REPLACE_FILE:
    RESPONSE_OUT = RESPONSE_FILE
    NWR_OUT = NWR_FILE
    ZH_TW_OUT = ZH_TW_FILE
else:
    RESPONSE_OUT = "solidated_" + RESPONSE_FILE
    NWR_OUT = "solidated_" + NWR_FILE
    ZH_TW_OUT = "solidated_" + ZH_TW_FILE


""" -------------------------- Load News Data -------------------------- """

news: list[str] = load_udn_news()


""" --------------------- Process News With Rationale --------------------- """

NWRs: list[dict] = []
NWR_ids: set[int] = set()

# load the NWRs
with open(NWR_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        NWRs.append(data)

# process the NWRs
for i, data in tqdm(enumerate(NWRs), total=len(NWRs)):
    data["id"] = news.index(data["article"])
    NWRs[i] = {**{"id": data["id"]}, **data}  # make the field id at the first
    if data["id"] in NWR_ids:
        print(f"Duplicated NWR id: {data['id']}")
        continue
    NWR_ids.add(data["id"])

# sort the NWRs by id
NWRs.sort(key=lambda x: x["id"])

# write the processed NWRs to file
with open(NWR_OUT, "w", encoding="utf-8") as f:
    for data in NWRs:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


""" -------------------------- Process Responses -------------------------- """

responses: list[dict] = []
responses_ids: set[int] = set()

# load the responses
with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        responses.append(data)

# process the responses
for i, data in tqdm(enumerate(responses), total=len(responses)):
    data["news"] = data["news"].strip()
    prev_id = data["id"] if "id" in data else None
    try:
        data["id"] = news.index(data["news"])
    except ValueError as e:
        print(f"news not found: {data}")
        raise e
    if prev_id is not None and prev_id != data["id"]:
        print(f"news id changed: {prev_id} -> {data['id']}")
    # make the field id at the first
    responses[i] = {**{"id": data["id"]}, **data}
    if data["id"] in responses_ids:
        print(f"Duplicated response id: {data['id']}")
        continue
    responses_ids.add(data["id"])

# sort the responses by id
responses.sort(key=lambda x: x["id"])

# write the processed responses to file
with open(RESPONSE_OUT, "w", encoding="utf-8") as f:
    for data in responses:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


""" ----------------------- Process zh_tw Responses ----------------------- """

zh_tw_responses: list[dict] = []
zh_tw_ids: set[int] = set()

# load the zh_tw responses
with open(ZH_TW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        zh_tw_responses.append(data)

# process the zh_tw responses
for i, data in tqdm(enumerate(zh_tw_responses), total=len(zh_tw_responses)):
    data["news"] = data["news"].strip()
    try:
        data["id"] = news.index(data["news"])
    except ValueError as e:
        print(f"news not found: {data}")
        raise e
    data["id"] = news.index(data["news"])
    # make the field id at the first
    zh_tw_responses[i] = {**{"id": data["id"]}, **data}
    if data["id"] in zh_tw_ids:
        print(f"Duplicated response id: {data['id']}")
        continue
    zh_tw_ids.add(data["id"])

# sort the responses by id
zh_tw_responses.sort(key=lambda x: x["id"])

with open(ZH_TW_OUT, "w", encoding="utf-8") as f:
    for data in zh_tw_responses:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
