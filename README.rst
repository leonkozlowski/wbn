===
wbn
===


.. image:: https://github.com/leonkozlowski/wbn/workflows/build/badge.svg
        :target: https://github.com/leonkozlowski/wbn

.. image:: https://readthedocs.org/projects/wbn/badge/?version=latest
        :target: https://wbn.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/leonkozlowski/wbn/shield.svg
        :target: https://pyup.io/repos/github/leonkozlowski/wbn/
        :alt: Updates

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black

.. image:: http://www.mypy-lang.org/static/mypy_badge.svg
        :target: http://mypy-lang.org/



Weighted Bayesian Network Text Classification


* Free software: MIT license
* Documentation: https://wbn.readthedocs.io.

Installation
------------

From source

.. code-block:: bash

    $ git clone https://github.com/leonkozlowski/wbn.git
    $ cd wbn

    $ python3.8 -m venv venv
    $ pip install -e .


Usage
-----

Building, training, and testing `WBN`

.. code-block:: python

    from sklearn.model_selection import train_test_split

    # Import WBN
    from wbn.classifier import WBN
    from wbn.sample.datasets import load_pr_newswire


    # Build the model
    wbn = WBN()

    # Load a sample dataset
    pr_newswire = load_pr_newswire()

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        pr_newswire.data, pr_newswire.target, test_size=0.2
    )

    # Fit the model
    wbn.fit(x_train, y_train)

    # Testing the model
    pred = wbn.predict(x_test)

    # Reverse encode the labels
    y_pred = wbn.reverse_encode(target=pred)


Constructing a new dataset:

.. code-block:: python

    import pickle

    # Import data structures for dataset creation
    from wbn.object import Document, DocumentData, Documents

    # Load your dataset from csv or pickle
    with open("dataset.pickle"), "rb") as infile:
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


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
