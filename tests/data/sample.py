"""Sample Data for Unit Tests."""
from wbn.object import Document, DocumentData, Documents

SAMPLE_DATASET = Documents(
    [
        Document(
            DocumentData(["hello", "world", "program"], ["hello", "program"]),
            "program",
        ),
        Document(
            DocumentData(["foo", "bar", "baz", "boo"], ["bar", "baz"]),
            "variable",
        ),
    ]
)
