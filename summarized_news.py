import json
import os
from dataclasses import dataclass, field

from constants import DIR_STORED_DATA
from crawler.utils import Logger


@dataclass
class SummarizedNews:

    SAVE_PATH = os.path.join(DIR_STORED_DATA, 'processed_data.jsonl')
    logger = Logger("SummarizedNews")

    article: str
    summary: str = ""
    id: int = -1
    label: list[int] = field(default_factory=list[int])

    def __str__(self) -> str:
        return (
            f"id: {self.id}, label: {self.label}\n"
            f"article: {self.article}\n"
            f"summary: {self.summary}"
        )

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
        cls.logger.info(
            f"Saved {len(ls)} summarized news records to {cls.SAVE_PATH}"
        )

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

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "article": self.article,
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            article=data.get("article", ""),
            summary=data.get("summary", ""),
            id=data.get("id", -1),
            label=data.get("label", [-1]),
            **data.get("kwargs", {})
        )
