from dataclasses import dataclass, field


@dataclass
class Rationale:
    essential_aspects: list[str] = field(default_factory=list[str])
    triples: list[str] = field(default_factory=list[str])
    rationale_summary: str = ""

    def __str__(self) -> str:
        return (
            f'Essential Aspects:\n{self.essential_aspects}\n'
            f'Triples:\n{self.triples}\n'
            f'Summary:\n{self.rationale_summary}'
        )

    def to_dict(self):
        return {
            'essential_aspects': self.essential_aspects,
            'triples': self.triples,
            'rationale_summary': self.rationale_summary
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            essential_aspects=data.get("essential_aspects", []),
            triples=data.get("triples", []),
            summary=data.get("rationale_summary", "")
        )
