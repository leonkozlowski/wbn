"""Weighted Bayesian Network Chain Classifier."""
import itertools
import logging
from typing import Any

import networkx as nx
import numpy as np

from wbn.object import Attribute

logging.basicConfig(level="INFO")
_LOGGER = logging.getLogger(__name__)

COMBINATION_SIZE = 2


class WBN(object):
    """TODO: Docstring"""

    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter
        self.dag = nx.DiGraph()

    def fit(self, data: np.ndarray, target: np.ndarray) -> Any:
        """TODO: Docstring"""
        for idx, entry in enumerate(data):
            # Establish universe of nodes for a given entry
            matrix = list(
                itertools.combinations(
                    [
                        Attribute(word, count / len(entry))
                        for word, count in entry.items()
                    ],
                    COMBINATION_SIZE,
                )
            )

            # Build DAG with all node combinations
            self.dag.add_edges_from(ebunch_to_add=matrix)
