"""Weighted Bayesian Network Chain Classifier."""
import itertools
import logging
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import networkx as nx
import numpy as np

from wbn.errors import InstanceCountError
from wbn.object import Attribute, Fit

logging.basicConfig(level="INFO")
_LOGGER = logging.getLogger(__name__)


class WBN(object):
    """Weighted Bayesian Network Classifier."""

    def __init__(self, size: int = 2):
        self.size = size
        self.fits = list()  # type: List[Fit]
        self.targets = dict()  # type: Dict[str, int]

    def fit(self, data: np.ndarray, target: np.ndarray) -> List[Fit]:
        """Builds directed acyclic graphs and corpora for
        class traversal and classification.

        Parameters
        ----------
        data : np.ndarray
            Array of annotated keywords

        target : np.ndarray
            Array of target classifications

        Returns
        -------
        List[Fit]
            Array of dag & corpus classifications

        """
        # Failure to validate prevents model fitting
        self._validate(data, target)

        self._encode(target=target)
        by_class = defaultdict(dict)  # type: DefaultDict
        for idx, entry in enumerate(data):
            # Establish universe for all targets
            weighted = Counter(entry)  # type: Dict[str, int]

            # Injects value for probability table
            by_word = {k: (v, 1) for k, v in weighted.items()}
            by_class[target[idx]] = self._update(
                parent=by_class[target[idx]], child=by_word
            )

        for cls, keywords in by_class.items():
            # Create a weighted dict for weighting
            cls_dag = nx.DiGraph()
            matrix = list(
                itertools.combinations(
                    [
                        Attribute(
                            word=word,
                            weight=count / len(keywords),
                            positive=positive,
                            negative=len(data) - positive,
                        )
                        for word, (count, positive) in keywords.items()
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

    def predict(self, data: np.ndarray, target: np.ndarray) -> List[int]:
        """Predict class of for keywords in 'data'.

        Parameters
        ----------
        data : np.ndarray
            Array of cleaned words from input.

        target : np.ndarray
            Array of target classifications

        """
        corpus = list(
            set(
                itertools.chain.from_iterable(
                    fit_class.corpus for fit_class in self.fits
                )
            )
        )

        instances = []  # type: List[Dict[str, int]]
        for entry in data:
            instances.append(
                dict(Counter([word for word in entry if word in corpus]))
            )

        predictions = [self.dag_traverse(instance) for instance in instances]

        return predictions

    def dag_traverse(self, instance: Dict[str, int]) -> int:
        """Iterate through and traverse class level dags
        in order to establish weighted match score.

        Parameters
        ----------
        instance : Dict[str, int]
            Instance of universe filtered words

        Returns
        -------
        int
            Predicted classification of instance

        """
        # dags = [fit_class.dag for fit_class in self.fits]

        return 1

    def _encode(self, target: List[str]) -> bool:
        """Encodes string targets to mapped integer.

        Parameters
        ----------
        target : List[str]
            Array of training classifications

        Returns
        -------
        bool
            Boolean if targets were set or not

        """
        for idx, tgt in enumerate(list(set(target))):
            self.targets[tgt] = idx

        return bool(self.targets)

    @staticmethod
    def _update(
        parent: DefaultDict, child: Dict[Any, Tuple[int, int]]
    ) -> DefaultDict:
        """Unpacks parent/child tuples and preforms addition to account
        for both instance frequency and probability.

        Parameters
        ----------
        parent : DefaultDict
            Parent 'by_class' master dictionary

        child : Dict[str, tuple]
            Attribute to be distributed into parent

        """
        for word, val in child.items():
            parent[word] = tuple(map(sum, zip(val, parent.get(word, (0, 0)))))

        return parent

    @staticmethod
    def _validate(data: np.ndarray, target: np.ndarray) -> None:
        """Validates both 'data' and 'target' for multiple rules
        including length and value existence.

        Parameters
        ----------
        data : np.ndarray
            Array of annotated keywords

        target : np.ndarray
            Array of encoded target classifications

        Raises
        ------
        InstanceCountError
            Data and target length mismatch

        """
        if len(data) != len(target):
            raise InstanceCountError(data, target)
