class Rationale:
    def __init__(self, essential_aspects: list[str], triples: list[str], summary: str):
        self.essential_aspects = essential_aspects
        self.triples = triples
        self.rationale_summary = summary

    def __str__(self) -> str:
        return f'Essential Aspects:\n{self.essential_aspects}\nTriples:\n{self.triples}\nSummary:\n{self.rationale_summary}'
    