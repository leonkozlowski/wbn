===
wbn
===


.. image:: https://img.shields.io/pypi/v/wbn.svg
        :target: https://pypi.python.org/pypi/wbn

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

From Build

.. code-block:: bash

    $ pip install wbn

Usage
-----

Building, training, and testing `WBN`

.. code-block:: python

    from sklearn.metrics import (
        accuracy_score,
        recall_score,
        precision_score,
    )
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
    red = wbn.predict(x_test)

    # Reverse encode the labels
    y_pred = wbn.reverse_encode(target=pred)

    # Calculate key metrics
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
