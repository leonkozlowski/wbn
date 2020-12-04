"""Custom Exceptions for WBN."""
import numpy as np


class WBNException(Exception):
    """Base WBN Exception."""

    pass


class InstanceCountError(WBNException):
    """InstanceCountError Exception."""

    def __init__(self, data: np.ndarray, target: np.ndarray):
        self.data = data
        self.target = target

    def __str__(self) -> str:
        return "Number of instances: {} does not match number of targets: {}".format(
            len(self.data), len(self.target)
        )


class MaxDepthExceededError(WBNException):
    """MaxDepthExceededError Exception."""

    def __init__(self, depth: float):
        self.depth = depth

    def __str__(self) -> str:
        return "Max probability depth of {} exceeded for all classifications".format(
            self.depth
        )
