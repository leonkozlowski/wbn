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
        return "<{}:{}:[{}:{}]>".format(
            self.word, round(self.weight, 4), self.positive, self.negative
        )

    @property
    def total(self) -> int:
        """Calculates total positives and negatives."""
        return self.positive + self.negative


class Fit(NamedTuple):
    """Fit class output holding DAG and Corpus."""

    dag: nx.DiGraph
    cls: str
    corpus: List[str]
