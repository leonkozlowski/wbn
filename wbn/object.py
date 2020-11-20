"""Reusable Objects for WBN."""
from typing import List, NamedTuple

import networkx as nx


class Attribute(NamedTuple):
    """Attribute class representing Word & Weight."""

    word: str
    weight: float
    positive: int
    negative: int

    def __repr__(self) -> str:
        return f"{self.word}:{round(self.weight, 4)}:[{self.positive}:{self.negative}]"


class Fit(NamedTuple):
    """Fit class output holding DAG and Corpus."""

    dag: nx.DiGraph
    corpus: List[str]
