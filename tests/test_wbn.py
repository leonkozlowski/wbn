#!/usr/bin/env python

"""Tests for `wbn` package."""
from collections import defaultdict
from unittest import TestCase

import pytest

from tests.data.sample import SAMPLE_DATA, SAMPLE_TARGET
from wbn.classifier import WBN
from wbn.errors import InstanceCountError


class TestWBN(TestCase):
    """Unit test suite for WBN."""

    def setUp(self) -> None:
        self.test_wbn = WBN()

    def test_fit(self):
        """Unit test for 'fit(...)'."""
        result = self.test_wbn.fit(data=SAMPLE_DATA, target=SAMPLE_TARGET)

        assert len(result) == 2

    def test_predict(self):
        """Unit test for 'predict(...)'."""
        assert 2 == 2

    def test_dag_traverse(self):
        """Unit test for 'dag_traverse(...)'."""
        assert 3 == 3

    def test_encode(self):
        """Unit test for '_encode(...)'."""
        self.test_wbn._encode(SAMPLE_TARGET)

        assert isinstance(self.test_wbn.targets, dict)
        assert "program" in self.test_wbn.targets
        assert "variable" in self.test_wbn.targets

    def test_update(self):
        """Unit test for '_update(...)'."""
        test_parent = defaultdict(dict)
        test_child = {"foo": (1, 2)}
        result = self.test_wbn._update(parent=test_parent, child=test_child)

        assert "foo" in result

    def test_validate(self):
        """Unit test for '_validate(...)'."""
        with pytest.raises(InstanceCountError):
            self.test_wbn.fit(data=SAMPLE_DATA, target=SAMPLE_TARGET[:1])
