# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-07-28

### Added

#### 🎯 Core Features
- **Data Cleaning and Preparation Pipeline**: Comprehensive notebook for processing ChEMBL acetylcholinesterase inhibitor data
- **Multiple Molecular Descriptors**: Implementation of various molecular representation methods
  - Circular Fingerprints (ECFP)
  - MACCS Keys
  - RDKit Descriptors
  - PubChem Features
  - ModRed Features
  - Mol2Vec embeddings

#### 🤖 Machine Learning Models
- **Traditional ML Pipeline**: Implementation of multiple algorithms
  - Random Forest
  - Support Vector Machines
  - Gradient Boosting (XGBoost, LightGBM)
  - Logistic Regression
  - Automated ML with TPOT

#### 🧠 Deep Learning Models
- **DeepChem Networks**: Deep neural networks for molecular property prediction
- **Graph Neural Networks**: 
  - Graph Convolutional Networks (GCN)
  - Message Passing Neural Networks (MPNN)
  - Graph Attention Transformers

#### 🔬 Transformer Models
- **Fine-tuned ChemBERTa Models**:
  - ChemBERTa 5M MLM
  - ChemBERTa 10M MLM
  - ChemBERTa 77M MLM
  - SMILES Tokenizer PubChem 1M

#### 📊 Model Interpretation
- **Graph Convolutional Network Interpretation**: Comprehensive analysis of GCN model decisions
- **Feature Importance Analysis**: SHAP values and feature importance for traditional ML models
- **Molecular Fragment Analysis**: Understanding of important molecular substructures

#### 📚 Documentation
- **Comprehensive README**: Detailed project overview, installation, and usage instructions
- **Contributing Guidelines**: Guidelines for contributing to the project
- **Requirements Management**: 
  - `requirements.txt` for pip users
  - `environment.yml` for conda users
- **Package Setup**: `setup.py` for easy installation
- **License**: MIT License for open source use

#### 🛠️ Development Infrastructure
- **Reproducible Environment**: Fixed random seeds and environment specifications
- **Comprehensive Dependencies**: All required libraries specified with versions
- **Modular Structure**: Well-organized notebook structure for different approaches

### Technical Specifications

#### 📊 Dataset Details
- **Source**: ChEMBL database (ChEMBL22)
- **Target**: Human Acetylcholinesterase (AChE)
- **Classification**: Binary (Active: IC50 ≤ 1000 nM, Inactive: IC50 > 1000 nM)
- **Molecular Weight Filter**: < 800 Daltons
- **Data Format**: Standardized SMILES strings

#### 🔧 Performance Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **ROC Analysis**: AUC-ROC curves and area under curve
- **Cross-validation**: Stratified k-fold validation
- **Statistical Analysis**: Statistical significance testing

#### 💻 System Requirements
- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended
- **GPU**: CUDA-compatible GPU recommended for deep learning models
- **Storage**: 2GB+ free space for dependencies and data

### Known Issues
- Some models require specific numpy version (1.23.3) for TPOT compatibility
- GPU memory requirements vary significantly between deep learning models
- ChemBERTa models require substantial computational resources

### Dependencies
- **Core**: pandas, numpy, scikit-learn
- **Chemistry**: RDKit, DeepChem, Mordred
- **Deep Learning**: PyTorch, TensorFlow, PyTorch Geometric
- **Visualization**: matplotlib, seaborn, plotly
- **Interpretation**: SHAP, LIME

---

## Future Releases

### Planned Features [1.1.0]
- [ ] Model ensemble methods
- [ ] Advanced hyperparameter optimization
- [ ] Additional molecular descriptors (3D descriptors, pharmacophores)
- [ ] Multi-target prediction capabilities
- [ ] Docker containerization
- [ ] Web interface for model predictions

### Long-term Goals [2.0.0]
- [ ] Integration with additional databases (PubChem, DrugBank)
- [ ] Generative model capabilities
- [ ] Active learning implementation
- [ ] Cloud deployment options
- [ ] Real-time prediction API

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{bhattarai2024achei,
  title={ML AChEI Classification Models for Drug Discovery},
  author={Bhattarai, Nirajan and Schulte, Marvin},
  year={2024},
  url={https://github.com/bhatnira/ML-AChEI-ClassificationModels-DrugDiscovery}
}
```
