from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="achei-classification-models",
    version="1.0.0",
    author="Nirajan Bhattarai, Marvin Schulte",
    author_email="bhatnira@isu.edu",
    description="Machine Learning models for Acetylcholinesterase Inhibitor Classification in Drug Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bhatnira/ML-AChEI-ClassificationModels-DrugDiscovery",
    project_urls={
        "Bug Tracker": "https://github.com/bhatnira/ML-AChEI-ClassificationModels-DrugDiscovery/issues",
        "Documentation": "https://github.com/bhatnira/ML-AChEI-ClassificationModels-DrugDiscovery/blob/main/README.md",
        "Source Code": "https://github.com/bhatnira/ML-AChEI-ClassificationModels-DrugDiscovery",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
        ],
        "gpu": [
            "torch-cuda>=11.7",
            "tensorflow-gpu>=2.9.0",
        ],
    },
    include_package_data=True,
    keywords=[
        "machine learning",
        "drug discovery",
        "acetylcholinesterase",
        "molecular descriptors",
        "deep learning",
        "graph neural networks",
        "cheminformatics",
        "classification",
        "bioactivity prediction",
    ],
)
