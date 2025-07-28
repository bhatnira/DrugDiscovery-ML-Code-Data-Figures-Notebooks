# Changelog

All notable changes to the ExplainableAI-BestModels-AChEI-DrugDiscovery project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation and README
- Requirements.txt with all dependencies
- Contributing guidelines
- License documentation

## [1.0.0] - 2025-01-28

### Added
- **FineTunedChemberta(DeepChem_ChemBERTa_10M_MLM).ipynb**
  - Fine-tuned ChemBERTa transformer model implementation
  - Attention-based molecular interpretation
  - Transfer learning from pre-trained chemical language models
  - SMILES tokenization and processing pipeline

- **ModelInterpretationAndCompoundGeneration_BestAggregrateModelCircularFingerprint.ipynb**
  - Extended-Connectivity Circular Fingerprint (ECFP) implementation
  - Molecular substructure analysis and interpretation
  - Compound generation and optimization workflows
  - Performance comparison with other fingerprint methods

- **ModelInterpretation_GraphConvolutionalNetwork.ipynb**
  - Graph Convolutional Network (GCN) for molecular classification
  - Node and edge attribution analysis
  - Molecular graph visualization techniques
  - Graph-level prediction with interpretability

- **ModelInterpretability_deepNet_rdkit.ipynb**
  - Deep neural network with RDKit molecular descriptors
  - SHAP value analysis for feature importance
  - Traditional molecular descriptor interpretation
  - Comparative analysis of descriptor-based approaches

- **ModelInterpretation_RdkitFeatureBasedAutoMLModel.ipynb**
  - AutoML implementation with RDKit features
  - Automated hyperparameter optimization using Optuna
  - Ensemble model interpretation and analysis
  - Feature selection and importance ranking

### Features
- **Multi-Model Architecture Support**
  - Transformer-based models (ChemBERTa)
  - Graph neural networks (GCN)
  - Traditional machine learning with molecular fingerprints
  - Deep neural networks with molecular descriptors
  - AutoML ensemble methods

- **Explainable AI Techniques**
  - Attention visualization for transformer models
  - SHAP (SHapley Additive exPlanations) analysis
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Molecular substructure highlighting
  - Feature importance ranking and selection

- **Drug Discovery Pipeline**
  - Data preprocessing and standardization
  - Molecular weight filtering (≤800 Daltons)
  - SMILES normalization and validation
  - Binary classification (Active/Inactive)
  - Model performance evaluation and comparison

- **Visualization and Analysis**
  - Molecular structure visualization
  - Attention heatmaps for transformer models
  - Feature importance plots
  - Performance metrics comparison
  - ROC curves and confusion matrices

### Dataset
- **Primary Dataset**: ChEMBL22 Human Acetylcholinesterase Inhibitors
  - Source: ChEMBL database (ID: ChEMBL22)
  - Molecular weight cutoff: ≤800 Daltons
  - Standardized SMILES representation
  - Binary classification labels (Active/Inactive)

### Dependencies
- **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow, PyTorch
- **Chemical Informatics**: RDKit, DeepChem, Mordred
- **Interpretability**: SHAP, LIME
- **Transformers**: Hugging Face Transformers, BertViz
- **Optimization**: Optuna, Hyperopt

### Authors
- **Nirajan Bhattarai** - Primary Research and Development
- **Marvin Schulte** - Research Collaboration and Development

### Research Contributions
- Comprehensive comparison of molecular representation methods
- Novel application of attention mechanisms to molecular interpretation
- Integration of multiple explainable AI techniques for drug discovery
- Systematic evaluation of model interpretability in chemical space

---

## Version History Notes

### Version Numbering
- **Major Version**: Significant new models or major architectural changes
- **Minor Version**: New features, interpretability methods, or substantial improvements
- **Patch Version**: Bug fixes, documentation updates, or minor enhancements

### Categories
- **Added**: New features, models, or capabilities
- **Changed**: Modifications to existing functionality
- **Deprecated**: Features that will be removed in future versions
- **Removed**: Features that have been completely removed
- **Fixed**: Bug fixes and error corrections
- **Security**: Security-related improvements or fixes

### Future Roadmap
- Integration of additional transformer architectures
- Implementation of gradient-based attribution methods
- Extension to multi-target drug discovery
- Performance optimization for large-scale screening
- Integration with cloud computing platforms

---

*For detailed information about each release, please refer to the corresponding git tags and release notes.*
