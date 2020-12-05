"""Sample Dataset for WBN."""
import os
import pickle

from wbn.object import Document, DocumentData, Documents


def load_pr_newswire() -> Documents:
    """Loads sample PRNewswire Dataset."""
    # Load pickle of dataset
    module = os.path.dirname(__file__)
    with open(
        os.path.join(module, "data", "pr-newswire.pickle"), "rb"
    ) as infile:
        raw_data = pickle.load(infile)

    # De-structure 'data' and 'target'
    data = raw_data.get("data")
    target = raw_data.get("target")

    # Construct Document's for each data/target entry
    documents = Documents(
        [
            Document(DocumentData(paragraphs, keywords), target[idx])
            for idx, (paragraphs, keywords) in enumerate(data)
        ]
    )

    return documents
