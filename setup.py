#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "networkx>=2.5",
    "nltk>=3.5",
    "numpy>=1.19.4",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

extras = {
    "extras": [
        "matplotlib>=3.3.3",
        "scikit-learn>=0.23.2",
    ],
}

setup(
    author="Leon Kozlowski",
    author_email="leonkozlowski@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Weighted Bayesian Network Text Classification",
    entry_points={
        "console_scripts": [
            "wbn=wbn.cli:main",
        ],
    },
    install_requires=requirements,
    extras_requires=extras,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="wbn",
    name="wbn",
    packages=find_packages(include=["wbn", "wbn.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/leonkozlowski/wbn",
    version="0.1.0",
    zip_safe=False,
)
