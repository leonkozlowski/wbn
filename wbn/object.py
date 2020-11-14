"""TODO: Module Level Docstring"""
from typing import Dict, List, NamedTuple, Union


class Attribute(NamedTuple):
    word: str
    weight: Union[int, float]

    def __repr__(self) -> str:
        return f"{self.word}:{self.weight}"


class Classification(object):
    index: int
    name: str


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
