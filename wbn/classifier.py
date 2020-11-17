"""Weighted Bayesian Network Chain Classifier."""
import itertools
import logging
from collections import Counter, defaultdict
from typing import List

import networkx as nx
import numpy as np

from wbn.object import Attribute, Fit

logging.basicConfig(level="INFO")
_LOGGER = logging.getLogger(__name__)


class WBN(object):
    """Weighted Bayesian Network Classifier.

    Parameters
    ----------
    size : int
        Size of node combinations

    """

    def __init__(self, size: int = 2):
        self.size = size
        self.fits = list()

    def fit(self, data: np.ndarray, target: np.ndarray) -> List[Fit]:
        """Builds directed acyclic graphs and corpora for
        class traversal and classification.

        Parameters
        ----------
        data : np.ndarray
            Array of annotated keywords

        target : np.ndarray
            Array of encoded target classifications

        Returns
        -------
        List[Fit]
            Array of dag & corpus classifications

        """
        by_class = defaultdict(list)
        for idx, entry in enumerate(data):
            # Establish universe for a target
            by_class[target[idx]] += entry

        for cls, keywords in by_class.items():
            # Create a weighted dict for weighting
            weighted = dict(Counter(keywords))

            cls_dag = nx.DiGraph()
            matrix = list(
                itertools.combinations(
                    [
                        Attribute(word, count / len(keywords))
                        for word, count in weighted.items()
                    ],
                    self.size,
                )
            )

            # Build DAG with all node combinations
            cls_dag.add_edges_from(ebunch_to_add=matrix)
            assert cls_dag.is_directed()

            # Store in instance variable for prediction
            self.fits.append(Fit(cls_dag, list(set(keywords))))

        return self.fits

    def predict(self, data: np.ndarray):
        """Predict class of for keywords in 'data'.

        Parameters
        ----------
        data : np.ndarray
            Array of cleaned words from input.

        """
        # corpus = self._make_corpus()
        # for entry in data:
        #     keywords = [word in corpus for word in entry]
        prediction = []

        return prediction

    def _make_corpus(self) -> List[str]:
        """Builds complete multi-class corpus of keywords.

        Returns
        -------
        List[str]
            List of keywords for all instances

        """
        corpus = list(
            set(
                itertools.chain.from_iterable(
                    fit_class[1] for fit_class in self.fits
                )
            )
        )

        return corpus
