# ExplainableAI-BestModels-AChEI-DrugDiscovery

## Authors
- **Nirajan Bhattarai** - Primary Research and Development
- **Marvin Schulte** - Research Collaboration and Development

## Overview

This repository contains a comprehensive collection of Jupyter notebooks implementing and comparing various machine learning models for **Acetylcholinesterase Inhibitor (AChEI) Drug Discovery** with a focus on **Explainable AI (XAI)**. The project evaluates multiple state-of-the-art molecular classification approaches and provides detailed model interpretability analysis.

## Project Description

Acetylcholinesterase inhibitors are crucial therapeutic agents for treating neurodegenerative disorders such as Alzheimer's disease. This project applies advanced machine learning techniques to identify potential AChEI compounds from molecular datasets, with emphasis on model interpretability and explainability.

### Key Features

- **Multiple Model Architectures**: Implementation of various ML models including transformer-based, graph-based, and traditional fingerprint-based approaches
- **Explainable AI**: Comprehensive model interpretation techniques including attention visualization, SHAP analysis, and molecular substructure highlighting
- **Comparative Analysis**: Systematic comparison of model performance and interpretability
- **Drug Discovery Pipeline**: End-to-end workflow from molecular data preprocessing to compound generation

## Repository Structure

### Notebooks

1. **`FineTunedChemberta(DeepChem_ChemBERTa_10M_MLM).ipynb`**
   - Fine-tuned ChemBERTa transformer model for molecular classification
   - Utilizes DeepChem's pre-trained ChemBERTa (10M parameters) with masked language modeling
   - Attention-based visualization for model interpretability
   - Direct SMILES string processing without extensive featurization

2. **`ModelInterpretationAndCompoundGeneration_BestAggregrateModelCircularFingerprint.ipynb`**
   - Extended-Connectivity Circular Fingerprint (ECFP) based classification
   - Model interpretation and novel compound generation
   - Topological fingerprint analysis for structure-activity modeling
   - Substructure-based interpretability analysis

3. **`ModelInterpretation_GraphConvolutionalNetwork.ipynb`**
   - Graph Convolutional Network (GCN) implementation
   - Graph-based molecular representation and classification
   - Node and edge attribution analysis for model interpretation
   - Molecular graph visualization techniques

4. **`ModelInterpretability_deepNet_rdkit.ipynb`**
   - Deep neural network with RDKit molecular descriptors
   - Feature importance analysis using traditional molecular descriptors
   - SHAP (SHapley Additive exPlanations) values for model explanation
   - Comparative analysis of descriptor-based vs. structural approaches

5. **`ModelInterpretation_RdkitFeatureBasedAutoMLModel.ipynb`**
   - AutoML approach using RDKit features
   - Automated hyperparameter optimization and model selection
   - Feature importance ranking and selection
   - Ensemble model interpretation techniques

## Dataset Information

The primary molecular dataset consists of human acetylcholinesterase inhibitor compounds sourced from the **ChEMBL database** (ID: ChEMBL22). 

### Dataset Characteristics:
- **Source**: ChEMBL22 - Human Acetylcholinesterase
- **Molecular Weight Filter**: Compounds ≤ 800 Daltons
- **Standardization**: SMILES normalization and molecular standardization
- **Binary Classification**: Active vs. Inactive AChEI compounds
- **Features**: SMILES strings, molecular descriptors, circular fingerprints

## Technical Requirements

### Dependencies

```python
# Core Libraries
import numpy>=1.21.0
import pandas>=1.3.0
import matplotlib>=3.4.0
import seaborn>=0.11.0

# Machine Learning
import deepchem>=2.6.0
import tensorflow>=2.8.0
import scikit-learn>=1.0.0
import xgboost>=1.5.0

# Chemical Informatics
import rdkit>=2022.03.0
import mordred>=1.2.0

# Interpretability
import shap>=0.40.0
import lime>=0.2.0

# Transformers
import transformers>=4.15.0
import torch>=1.10.0

# Visualization
import bertviz
import plotly>=5.5.0
```

### Installation

```bash
# Install core scientific computing packages
pip install numpy pandas matplotlib seaborn scipy

# Install machine learning frameworks
pip install scikit-learn xgboost tensorflow torch

# Install chemical informatics tools
pip install rdkit-pypi deepchem mordred

# Install interpretability libraries
pip install shap lime

# Install transformer libraries
pip install transformers bertviz

# Install additional visualization tools
pip install plotly ipywidgets
```

## Usage

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhatnira/ExplainableAI-BestModels-AChEI-DrugDiscovery.git
   cd ExplainableAI-BestModels-AChEI-DrugDiscovery
   ```

2. **Set up environment:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks:**
   - Open any notebook in Jupyter Lab/Notebook
   - Execute cells sequentially
   - Modify paths for your dataset location

### Notebook Execution Order

For comprehensive analysis, we recommend running notebooks in this order:

1. Start with `ModelInterpretability_deepNet_rdkit.ipynb` for baseline understanding
2. Explore `ModelInterpretationAndCompoundGeneration_BestAggregrateModelCircularFingerprint.ipynb` for fingerprint-based approaches
3. Advance to `ModelInterpretation_GraphConvolutionalNetwork.ipynb` for graph-based methods
4. Conclude with `FineTunedChemberta(DeepChem_ChemBERTa_10M_MLM).ipynb` for transformer-based analysis
5. Compare results using `ModelInterpretation_RdkitFeatureBasedAutoMLModel.ipynb`

## Methodology

### Model Architectures

1. **ChemBERTa (Chemical BERT)**
   - Transformer architecture pre-trained on molecular SMILES
   - Attention mechanism for molecular substructure importance
   - Transfer learning from chemical language models

2. **Graph Convolutional Networks**
   - Direct molecular graph representation
   - Node embeddings for atoms, edge embeddings for bonds
   - Graph-level prediction with node attribution

3. **Circular Fingerprints (ECFP)**
   - Topological molecular fingerprints
   - Morgan algorithm for substructure enumeration
   - Traditional ML classifiers with interpretable features

4. **RDKit Descriptors**
   - Molecular descriptor calculation (>200 features)
   - Feature selection and importance ranking
   - Classical ML with interpretable features

5. **AutoML Ensemble**
   - Automated model selection and hyperparameter tuning
   - Ensemble methods for robust predictions
   - Feature importance aggregation

### Interpretability Techniques

- **Attention Visualization**: Transformer attention weights on molecular tokens
- **SHAP Values**: Feature contribution analysis
- **LIME**: Local model explanation
- **Molecular Highlighting**: Substructure importance visualization
- **Feature Importance**: Ranking and selection of molecular descriptors

## Results and Performance

The models demonstrate competitive performance in AChEI classification with the following key findings:

- **ChemBERTa**: High accuracy with interpretable attention patterns
- **Graph CNNs**: Effective molecular graph representation with node-level interpretation
- **ECFP Models**: Fast computation with interpretable substructure analysis
- **Ensemble Methods**: Robust performance across multiple feature types

Detailed performance metrics and comparison tables are available within each notebook.

## Applications

### Drug Discovery Pipeline
1. **Virtual Screening**: High-throughput compound evaluation
2. **Lead Optimization**: Structure-activity relationship analysis
3. **Mechanism Understanding**: Molecular target interaction insights
4. **Safety Assessment**: Toxicity and side effect prediction

### Research Applications
- Neurodegenerative disease research
- Alzheimer's drug development
- Computational pharmacology studies
- AI-driven drug discovery workflows

## Contributing

We welcome contributions to improve the models, add new interpretability techniques, or extend the analysis. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit your changes (`git commit -am 'Add new interpretability method'`)
4. Push to the branch (`git push origin feature/new-model`)
5. Create a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{bhattarai2025explainable,
  title={Explainable AI for Acetylcholinesterase Inhibitor Drug Discovery: A Comparative Study of Machine Learning Models},
  author={Bhattarai, Nirajan and Schulte, Marvin},
  year={2025},
  publisher={GitHub},
  url={https://github.com/bhatnira/ExplainableAI-BestModels-AChEI-DrugDiscovery}
}
```

## License

All Rights Reserved. See [LICENSE.md](LICENSE.md) for details.

## Contact

- **Nirajan Bhattarai**: [GitHub Profile](https://github.com/bhatnira)
- **Marvin Schulte**: Research Collaborator

## Acknowledgments

- **ChEMBL Database**: For providing the acetylcholinesterase inhibitor dataset
- **DeepChem Community**: For pre-trained ChemBERTa models and molecular ML tools
- **RDKit**: For chemical informatics functionality
- **Hugging Face**: For transformer model implementations

## References

1. Ahmad, W., et al. (2022). ChemBERTa-2: Towards Chemical Foundation Models. *arXiv preprint*.
2. Chithrananda, S., et al. (2020). ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. *arXiv preprint*.
3. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.
4. Gaulton, A., et al. (2017). The ChEMBL database in 2017. *Nucleic Acids Research*, 45(D1), D945-D954.

---

*This repository represents ongoing research in explainable AI for drug discovery. Models and methodologies are continuously being improved and updated.*
