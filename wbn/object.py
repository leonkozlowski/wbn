"""Reusable Objects for WBN."""
from typing import Dict, List, NamedTuple, Union

import networkx as nx


class Attribute(NamedTuple):
    """Attribute Class representing Word & Weight."""

    word: str
    weight: Union[int, float]

    def __repr__(self) -> str:
        return f"{self.word}:{self.weight}"


class Classification(object):
    """Classification Base Class for Code Generation."""

    index: int
    name: str


class Fit(NamedTuple):
    """Fit class output holding DAG and Corpus."""

    dag: nx.DiGraph
    corpus: List[str]


class Instance(dict):
    """Facade Accessor for a Training Instance."""

    @property
    def data(self) -> List[Dict[str, int]]:
        """Access for 'data' attribute."""
        return self.get("data", [])

    @property
    def target(self) -> List[str]:
        """Access for 'target' attribute."""
        return self.get("target", [])
