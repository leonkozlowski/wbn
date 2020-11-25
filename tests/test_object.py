#!/usr/bin/env python

"""Tests for `wbn` package."""
from unittest import TestCase

import networkx as nx

from wbn.object import Attribute, Fit


class TestAttribute(TestCase):
    """Unit test suite for Attribute."""

    def setUp(self) -> None:
        self.test_attribute = Attribute("foo", 0.5, 1, 1)

    def test_values(self):
        """Unit test for dot notation values."""
        assert self.test_attribute.word == "foo"
        assert self.test_attribute.weight == 0.5
        assert self.test_attribute.positive == 1
        assert self.test_attribute.negative == 1

    def test_repr(self):
        """Unit test for string representation."""
        assert self.test_attribute.__repr__() == "<foo:0.5:[1:1]>"


class TestFit(TestCase):
    """Unit test suite for Fit."""

    def setUp(self) -> None:
        self.text_fit = Fit(nx.DiGraph(), "foo-bar", ["hello", "world"])

    def test_values(self):
        """Unit test for dot notation values."""
        assert isinstance(self.text_fit.dag, nx.DiGraph)
        assert self.text_fit.corpus == ["hello", "world"]
