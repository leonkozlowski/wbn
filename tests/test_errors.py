#!/usr/bin/env python

"""Tests for `wbn.errors` package."""
from unittest import TestCase

import pytest

from tests.data.sample import SAMPLE_DATA, SAMPLE_TARGET
from wbn.classifier import WBN
from wbn.errors import InstanceCountError


class TestInstanceCountError(TestCase):
    """Unit test suite for InstanceCountError."""

    def setUp(self) -> None:
        self.test_wbn = WBN()

    def test_raises(self):
        """Unit test for InstanceCountError raises."""
        with pytest.raises(InstanceCountError):
            self.test_wbn.fit(data=SAMPLE_DATA, target=SAMPLE_TARGET[:1])

    def test_str(self):
        """Unit test for '__str__()' of InstanceCountError."""
        exception = InstanceCountError(SAMPLE_DATA, SAMPLE_TARGET[:1])

        assert (
            exception.__str__()
            == "Number of instances: 2 does not match number of targets: 1"
        )
