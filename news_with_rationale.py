import json
import os

from constants import DIR_STORED_DATA
from crawler.utils import Logger
from rationale import Rationale
from summarized_news import SummarizedNews


class NewsWithRationale(SummarizedNews, Rationale):

    SAVE_PATH = os.path.join(DIR_STORED_DATA, 'news_with_rationales.jsonl')
    logger = Logger("NewsWithRationale")

    def __init__(self, summarized_news: SummarizedNews, rationale: Rationale):
        SummarizedNews.__init__(
            self,
            summarized_news.article,
            summarized_news.summary,
            summarized_news.id,
            summarized_news.label
        )
        Rationale.__init__(
            self,
            rationale.essential_aspects,
            rationale.triples,
            rationale.rationale_summary
        )

    def __str__(self) -> str:
        return f'{super().__str__()}\n{Rationale.__str__(self)}'

    def save(self):
        with open(self.SAVE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.__dict__, ensure_ascii=False))
            f.write('\n')

    @classmethod
    def load(cls):
        with open(cls.SAVE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                summarized_news = SummarizedNews(**data)
                rationale = Rationale(
                    data['essential_aspects'],
                    data['triples'],
                    data['summary']
                )
                yield cls(summarized_news, rationale)

    @classmethod
    def load_all(cls):
        return list(cls.load())

    def essential_aspects_str(self):
        return ', '.join(self.essential_aspects)

    def triples_str(self):
        return ', '.join(self.triples)

    def to_dict(self):
        return {
            'id': self.id,
            'label': self.label,
            'article': self.article,
            'summary': self.summary,
            'essential_aspects': self.essential_aspects,
            'triples': self.triples,
            'rationale_summary': self.rationale_summary
        }

    @classmethod
    def from_dict(cls, data):
        summarized_news = SummarizedNews.from_dict(data)
        rationale = Rationale.from_dict(data)
        return cls(summarized_news, rationale)
