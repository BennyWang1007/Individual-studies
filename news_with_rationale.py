import json
import os
from dataclasses import dataclass

from constants import DIR_STORED_DATA
from crawler.utils import Logger
from rationale import Rationale
from summarized_news import SummarizedNews


@dataclass
class NewsWithRationale(Rationale, SummarizedNews):

    SAVE_PATH = os.path.join(DIR_STORED_DATA, 'news_with_rationales.jsonl')
    logger = Logger("NewsWithRationale")

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

    def article_full_str(self):
        """ Returns a string representation of the article with field name. """
        return f'article:\n{self.article}'

    def essential_aspects_str(self):
        """ Returns a string representation of the essential aspects. """
        return '[' + '], ['.join(self.essential_aspects) + ']'

    def essential_aspects_full_str(self):
        """
        Returns a string representation of the essential aspects with \
        field name.
        """
        return f'essential aspects:\n{self.essential_aspects_str()}'

    def triples_str(self):
        """ Returns a string representation of the triples. """
        return ', '.join(self.triples)

    def triples_full_str(self):
        """ Returns a string representation of the triples with field name. """
        return f'triples:\n{self.triples_str()}'

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
        return cls(
            article=data.get('article', ''),
            summary=data.get('summary', ''),
            id=data.get('id', -1),
            label=data.get('label', []),
            essential_aspects=data.get('essential_aspects', []),
            triples=data.get('triples', []),
            rationale_summary=data.get('rationale_summary', '')
        )
