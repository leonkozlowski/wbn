"""Weighted Bayesian Network Text Classification Model."""
import itertools
import logging
from collections import Counter, defaultdict
from operator import itemgetter
from typing import Any, DefaultDict, Dict, List, Tuple

import networkx as nx
import numpy as np
from nltk import PorterStemmer

from wbn.config import COMBINATION_SIZE
from wbn.errors import InstanceCountError, MaxDepthExceededError
from wbn.object import (
    Attribute,
    Classification,
    ClassificationScore,
    DocumentData,
)

logging.basicConfig(level="INFO")
_LOGGER = logging.getLogger(__name__)


class WBN(object):
    """Weighted Bayesian Network Classifier."""

    def __init__(self, depth: float = 0.05):
        self.depth = depth
        self.classes = list()  # type: List[Classification]
        self.corpus = list()  # type: List[str]
        self.targets = dict()  # type: Dict[Any, int]
        self.predictions = list()  # type: List[ClassificationScore]
        self._reverse_encoded = dict()  # type: Dict[int, Any]

    def fit(
        self, data: List[DocumentData], target: List[str]
    ) -> List[Classification]:
        """Builds directed acyclic graphs and corpora for
        class traversal and classification.

        Parameters
        ----------
        data : List[DocumentData]
            Array of annotated keywords

        target : List[str]
            Array of target classifications

        Returns
        -------
        List[Classification]
            Array of dag & corpus classifications

        """
        # Failure to validate prevents model fitting
        self._validate(data, target)

        self._encode(target=target)
        stemmer = PorterStemmer()  # Instantiate stemmer
        by_class = defaultdict(dict)  # type: DefaultDict
        for idx, entry in enumerate(data):
            # Establish universe for all targets
            stemmed_entry = [stemmer.stem(word) for word in entry.keywords]
            weighted = Counter(stemmed_entry)  # type: Dict[str, int]

            # Injects value for probability table
            by_word = {k: (v, 1) for k, v in weighted.items()}
            # Create a weighted dict for weighting
            by_class[target[idx]] = self._update(
                parent=by_class[target[idx]], child=by_word
            )

        for cls, keywords in by_class.items():
            total_words = sum([kw[0] for kw in keywords.values()])
            cls_dag = nx.DiGraph()
            matrix = list(
                itertools.combinations(
                    [
                        Attribute(
                            word=word,
                            weight=count / total_words,
                            positive=positive,
                            negative=len(
                                [
                                    instance
                                    for instance in target
                                    if instance == cls
                                ]
                            )
                            - positive,  # Total minus positive values
                        )
                        for word, (count, positive) in keywords.items()
                    ],
                    COMBINATION_SIZE,
                )
            )

            # Build DAG with all node combinations
            cls_dag.add_edges_from(ebunch_to_add=matrix)
            assert cls_dag.is_directed()

            # Store in instance variable for prediction
            self.classes.append(
                Classification(
                    dag=cls_dag,
                    cls=cls,
                    corpus=list(set(keywords)),
                )
            )

        return self.classes

    def predict(self, data: List[DocumentData]) -> List[int]:
        """Predict class of for keywords in 'data'.

        Parameters
        ----------
        data : List[DocumentData]
            Array of cleaned words from input.

        Returns
        -------
        List[int]
            Array of instance class predictions

        """
        self.corpus = list(
            set(
                itertools.chain.from_iterable(
                    fit_class.corpus for fit_class in self.classes
                )
            )
        )

        instances = []  # type: List[Dict[str, int]]
        stemmer = PorterStemmer()  # Instantiate stemmer
        for entry in data:
            stemmed_entry = [stemmer.stem(word) for word in entry.tokens]
            instances.append(
                Counter(
                    [word for word in stemmed_entry if word in self.corpus]
                )
            )

        # Generate predictions for each instance
        predictions = list(map(self._evaluate, instances))

        return predictions

    def reverse_encode(self, target: List[int]) -> List[str]:
        """Reverse encodes int targets/predictions for metrics.

        Parameters
        ----------
        target : List[int]
            Array of encoded targets/predictions

        Returns
        -------
        List[str]
            Reverse encoded array of targets/predictions

        """
        return [self._reverse_encoded.get(val) for val in target]  # type: ignore

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

        self._reverse_encoded = {v: k for k, v in self.targets.items()}

        return bool(self.targets)

    def _evaluate(self, instance: Dict[str, int]) -> int:
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
        classification_probabilities = (
            list()
        )  # type: List[ClassificationScore]
        for classification in self.classes:
            edge_probabilities = list()  # type: List[Tuple[float, list]]
            for edge in classification.dag.edges:
                edge_probability = self._score_edge(
                    edge=edge, instance=instance
                )  # type: ignore
                if edge_probability:
                    edge = edge + (1 + edge_probability,)  # Assign edge probability
                    edge_probabilities.append((edge_probability, edge))

                # Sort edge scores by probability
                sorted_edge_probabilities = sorted(
                    edge_probabilities, reverse=True, key=itemgetter(0)
                )

                # Calculate depth
                depth = round(len(self.corpus) * self.depth)
                if len(sorted_edge_probabilities) >= depth:
                    # Limit probabilities to 'depth' hyper-parameter
                    depth_limited = sorted_edge_probabilities[:depth]

                    # Destructure probabilities and edges
                    probabilities, edges = list(zip(*depth_limited))

                    # Create ClassificationScore to scores
                    classification_probabilities.append(
                        ClassificationScore(
                            self.targets[classification.cls],
                            np.prod(probabilities),
                            edges,
                        )
                    )

        if not classification_probabilities:
            raise MaxDepthExceededError(self.depth)

        prediction = max(classification_probabilities, key=itemgetter(1))

        # Store verbose prediction with probability and edges
        self.predictions.append(prediction)

        return prediction.cls

    @staticmethod
    def _score_edge(
        edge: Tuple[Attribute, Attribute], instance: Dict[str, int]
    ) -> float:
        """Calculates score for edge of dag via parent/child node in order to
        identify correlation to instance.

        Using a Bayesian approach we calculate

        Parameters
        ----------
        edge : Tuple[Attribute, Attribute]
            Edge parent/child node of dag

        instance : Dict[str, int]
            Instance to be evaluated against edge

        Returns
        -------
        float
            Edge score against instance

        """
        # De-structure nodes of edge
        parent, child = edge

        words = [parent.word, child.word]
        if any(word not in instance for word in words):
            return 0  # Parent/Child edge not indicative of correlation

        # NOTE: Conditional probability calculation
        # L: Class (classification)
        # P: Parent (parent node word in edge)
        # C: Child (child node word in edge)
        # wp: Parent word weight of keywords
        # wc: Child word weight of keywords
        cls_given_parent = parent.positive / parent.total  # Pr(L | P)
        cls_given_child = child.positive / child.total  # Pr(L | C)

        weighted_joint_probability = (
            cls_given_parent * (1 + parent.weight)
        ) * (
            cls_given_child * (1 + child.weight)
        )  # Pr(L | P(wp), C(wc))

        return weighted_joint_probability

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
    def _validate(data: List[DocumentData], target: List[str]) -> None:
        """Validates both 'data' and 'target' for multiple rules
        including length and value existence.

        Parameters
        ----------
        data : List[DocumentData]
            Array of annotated keywords

        target : List[str]
            Array of encoded target classifications

        Raises
        ------
        InstanceCountError
            Data and target length mismatch

        """
        if len(data) != len(target):
            raise InstanceCountError(data, target)
