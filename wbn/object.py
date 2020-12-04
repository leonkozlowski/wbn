"""Reusable Objects for WBN."""
from typing import List, NamedTuple, Tuple

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


class DocumentData(NamedTuple):
    """Document 'data' entry with paragraphs and keywords."""

    tokens: List[str]  # Paragraphs as cleaned tokens
    keywords: List[str] = None  # type: ignore


class Document(NamedTuple):
    """Document data structure for 'data' and 'target'."""

    data: DocumentData
    target: str  # Annotated classification


class Documents(list):
    """Data structure to hold an access 'Document' entries."""

    def __init__(self, documents: List[Document]):
        super(Documents, self).__init__(documents)
        self.documents = documents

    @property
    def data(self) -> List[DocumentData]:
        """Access for 'data' elements."""
        return [doc.data for doc in self.documents]

    @property
    def target(self) -> List[str]:
        """Access for 'target' elements."""
        return [doc.target for doc in self.documents]


class Classification(NamedTuple):
    """Classification output holding DAG and Corpus."""

    dag: nx.DiGraph
    cls: str
    corpus: List[str]


class ClassificationScore(NamedTuple):
    """Classification score output holding class, probability and edges."""

    cls: int
    probability: float
    edges: List[Tuple[Attribute, Attribute]]
