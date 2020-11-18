"""Reusable Objects for WBN."""
from typing import List, NamedTuple, Union

import networkx as nx


class Attribute(NamedTuple):
    """Attribute class representing Word & Weight."""

    word: str
    weight: Union[int, float]

    def __repr__(self) -> str:
        return f"{self.word}:{self.weight}"


class Fit(NamedTuple):
    """Fit class output holding DAG and Corpus."""

    dag: nx.DiGraph
    corpus: List[str]
