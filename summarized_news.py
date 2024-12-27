import json
import os

from constants import DIR_STORED_DATA
from crawler.utils import Logger

class SummarizedNews:

    SAVE_PATH = os.path.join(DIR_STORED_DATA, 'processed_data.jsonl')
    logger = Logger("SummarizedNews")
    
    def __init__(self, article: str, summary: str, id: int, label: list[int]) -> None:
        self.id = id
        self.label = label
        self.article = article
        self.summary = summary

    def __str__(self) -> str:
        return f"id: {self.id}, label: {self.label}\narticle: {self.article}\nsummary: {self.summary}"""
    
    def save(self):
        with open(self.SAVE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.__dict__, ensure_ascii=False))
            f.write('\n')

    @classmethod
    def save_all(cls, ls):
        with open(cls.SAVE_PATH, 'w', encoding='utf-8') as f:
            for obj in ls:
                f.write(json.dumps(obj.__dict__, ensure_ascii=False))
                f.write('\n')
        cls.logger.info(f"Saved {len(ls)} summarized news records to {cls.SAVE_PATH}")

    @classmethod
    def load(cls):
        with open(cls.SAVE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield cls(**data)

    @classmethod
    def load_all(cls):
        with open(cls.SAVE_PATH, 'r', encoding='utf-8') as f:
            ls = [cls(**json.loads(line)) for line in f]
            cls.logger.info(f"Loaded {len(ls)} summarized news records")
            return ls