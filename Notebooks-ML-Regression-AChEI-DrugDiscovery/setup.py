#!/usr/bin/env python3
"""
Setup script for ML Regression AChEI Drug Discovery project.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements = []
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        pass
    return requirements

setup(
    name="achei-regression",
    version="1.0.0",
    author="Nirajan Bhattarai, Marvin Schulte",
    author_email="bhatnira@isu.edu",
    description="Machine Learning Regression Models for Acetylcholinesterase Inhibitor Drug Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bhatnira/ML-Regression-AChEI-DrugDiscovery",
    project_urls={
        "Bug Tracker": "https://github.com/bhatnira/ML-Regression-AChEI-DrugDiscovery/issues",
        "Documentation": "https://github.com/bhatnira/ML-Regression-AChEI-DrugDiscovery/blob/main/README.md",
        "Source Code": "https://github.com/bhatnira/ML-Regression-AChEI-DrugDiscovery",
    },
    packages=find_packages(where="src", exclude=["tests*"]),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.971",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.9",
        ],
        "viz": [
            "plotly>=5.10.0",
            "bokeh>=2.4.0",
            "altair>=4.2.0",
        ],
    },
    keywords=[
        "machine learning",
        "drug discovery",
        "acetylcholinesterase",
        "molecular modeling",
        "cheminformatics",
        "regression",
        "neural networks",
        "graph neural networks",
        "molecular descriptors",
        "circular fingerprints",
    ],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            # Add command-line scripts here if needed
            # "achei-predict=achei_regression.cli:main",
        ],
    },
)
