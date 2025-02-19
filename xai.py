import os
import json
import requests
from crawler.utils import Logger

from utils import get_rationale_prompt_chinese
from constants import DIR_STORED_DATA, X_AI_KEY
from rationale import Rationale
from summarized_news import SummarizedNews
from news_with_rationale import NewsWithRationale

class XAI:

    _instance = None
    
    X_AI_KEY = X_AI_KEY
    RESPONSES_PATH = os.path.join(DIR_STORED_DATA, 'x_ai_responses.jsonl')
    logger = Logger("XAI")

    response: dict = {}
    responses: list[dict] = []
    rationale: Rationale = Rationale([], [], "")
    rationales: list[Rationale] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        self.logger.info("XAI instance created")
        self.responses = XAI.load_responses()

    @staticmethod
    def get_responses() -> list[dict]:
        return XAI.responses

    @staticmethod
    def get_response(prompt: str) -> dict:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {XAI.X_AI_KEY}"
        }
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a test assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response.json()
    
    @staticmethod
    # def extract_rationale(response: dict) -> list[Rationale]:
    def extract_rationale(response: dict) -> Rationale:
        rationales: list[Rationale] = []
        for choice in response['choices']:
            # only consider the completed response
            if choice['finish_reason'] != 'stop': continue
            content = choice['message']['content']

            # print(f"{content=}")
            # content is formatted as '核心要素：[aspects] 三元組：[triples] 生成摘要：[summary]'
            # separate the content by '三元组：' and '生成摘要：'
            
            essential_lines = content.split('三元組：')[0].split('核心要素：')[1].strip().split('\n')
            triples_lines = content.split('三元組：')[1].split('生成摘要：')[0].strip().split('\n')
            summary = content.split('生成摘要：')[1].strip()

            # print(essential_lines)
            # print(triples_lines)
            # print(summary)

            essential_aspects = [line.strip() for line in essential_lines if line.strip()]
            triples = [line.strip() for line in triples_lines if line.strip()]
            # remove each '- ' prefix
            essential_aspects = [line[2:] if line.startswith('- ') else line for line in essential_aspects]
            triples = [line[2:] if line.startswith('- ') else line for line in triples]
            # rationales.append(Rationale(essential_aspects, triples, summary))
            return Rationale(essential_aspects, triples, summary)


            # print(essential_aspects)
            # print(triples)
            # print(summary)

            # rationales.append(Rationale(essential_aspects, triples, summary))
            # print(rationales[-1])
        # return rationales
        raise Exception("No rationale extracted")
    
    @classmethod
    def store_responses(cls, responses: list[dict]):
        with open(cls.RESPONSES_PATH, 'w', encoding='utf-8') as f:
            for response in responses:
                f.write(json.dumps(response, ensure_ascii=False))
                f.write('\n')


    @classmethod
    def store_responses_expend(cls, response: dict):
        with open(cls.RESPONSES_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(response, ensure_ascii=False))
            f.write('\n')


    @staticmethod
    def get_rationale(news: SummarizedNews) -> Rationale:
        prompt = get_rationale_prompt_chinese(news.article, news.summary)
        XAI.response = XAI.get_response(prompt)
        XAI.responses.append(XAI.response)
        XAI.store_responses_expend(XAI.response)
        return XAI.extract_rationale(XAI.response)
    
    @classmethod
    def get_newsWithRationale(cls, news: SummarizedNews) -> NewsWithRationale:
        cls.rationale = cls.get_rationale(news)
        # return NewsWithRationale(news.article, news.summary, news.id, news.label, rationales)
        return NewsWithRationale(news, cls.rationale)


    @staticmethod
    def load_responses() -> list[dict]:
        XAI.responses = []
        if os.path.exists(XAI.RESPONSES_PATH):
            with open(XAI.RESPONSES_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    XAI.responses.append(json.loads(line))
        XAI.logger.info(f"loaded {len(XAI.responses)} responses from x_ai responses")
        return XAI.responses
    
    @staticmethod
    def load_rationales_from_responses() -> list[Rationale]:
        XAI.rationales = []
        # print(f"{len(XAI.responses)=}")
        for response in XAI.responses:
            try:
                XAI.rationale = XAI.extract_rationale(response)
                XAI.rationales.append(XAI.rationale)
            except Exception as e:
                XAI.logger.error(f"failed to extract rationale from response: {response}")
        XAI.logger.info(f"loaded {len(XAI.rationales)} rationales from x_ai responses")
        return XAI.rationales