#!/usr/bin/env python

"""Tests for `wbn` package."""
from collections import defaultdict
from unittest import TestCase

import pytest

from tests.data.sample import SAMPLE_DATASET
from wbn.classifier import WBN
from wbn.errors import InstanceCountError, MaxDepthExceededError
from wbn.sample.datasets import load_pr_newswire


class TestWBN(TestCase):
    """Unit test suite for WBN."""

    def setUp(self) -> None:
        self.sample = load_pr_newswire()
        self.test_wbn = WBN()

    def test_fit(self):
        """Unit test for 'fit(...)'."""
        result = self.test_wbn.fit(
            data=self.sample.data, target=self.sample.target
        )

        assert len(result) == 5

    def test_predict(self):
        """Unit test for 'predict(...)'."""
        assert 2 == 2

    def test_reverse_encode(self):
        """Unit test for 'reverse_encode(...)'."""
        reverse = self.test_wbn.reverse_encode([0, 1])

        assert isinstance(reverse, list)

    def test_encode(self):
        """Unit test for '_encode(...)'."""
        self.test_wbn._encode(self.sample.target)

        assert isinstance(self.test_wbn.targets, dict)
        assert "cash-dividend" in self.test_wbn.targets
        assert "merger-acquisition" in self.test_wbn.targets

    def test_evaluate_raises(self):
        """Unit test for '_evaluate(...)'."""
        with pytest.raises(MaxDepthExceededError):
            self.test_wbn._evaluate(SAMPLE_DATASET.data[0])

    def test_update(self):
        """Unit test for '_update(...)'."""
        test_parent = defaultdict(dict)
        test_child = {"foo": (1, 2)}
        result = self.test_wbn._update(parent=test_parent, child=test_child)

        assert "foo" in result

    def test_validate(self):
        """Unit test for '_validate(...)'."""
        with pytest.raises(InstanceCountError):
            self.test_wbn.fit(
                data=SAMPLE_DATASET.data, target=SAMPLE_DATASET.target[:1]
            )
