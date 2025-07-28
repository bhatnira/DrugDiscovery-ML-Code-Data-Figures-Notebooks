# ML AChEI Classification Models for Drug Discovery

A comprehensive machine learning project for predicting acetylcholinesterase (AChE) inhibitor activity using various molecular descriptors and deep learning approaches.

## 🎯 Project Overview

This repository contains a collection of Jupyter notebooks implementing different machine learning and deep learning approaches for binary classification of acetylcholinesterase inhibitors. The project uses molecular data from the ChEMBL database to predict whether compounds are active (IC50 ≤ 1000 nM) or inactive (IC50 > 1000 nM) against acetylcholinesterase.

## 📊 Dataset

- **Source**: ChEMBL database (ChEMBL22 - Human Acetylcholinesterase)
- **Target**: Binary classification (Active/Inactive based on IC50 threshold of 1000 nM)
- **Molecular Weight Filter**: Compounds under 800 Daltons
- **Data Format**: Standardized SMILES strings

## 🧪 Molecular Descriptors Used

### 1. **Fingerprint-Based Features**
- **Circular Fingerprints (ECFP)**: Extended-connectivity fingerprints for structure-activity modeling
- **MACCS Keys**: 166-bit structural keys representing common chemical features

### 2. **Chemical Descriptors**
- **RDKit Descriptors**: Comprehensive set of molecular descriptors from RDKit
- **PubChem Features**: Chemical fingerprints from PubChem database
- **ModRed Features**: Molecular descriptors using reduced feature sets
- **Mol2Vec**: Molecular vector representations using unsupervised learning

### 3. **Graph-Based Representations**
- **Graph Convolutional Networks (GCN)**: Direct learning from molecular graphs
- **Message Passing Neural Networks (MPNN)**: Advanced graph neural network architecture
- **Graph Attention Transformer**: Attention-based graph neural networks

## 🤖 Machine Learning Approaches

### Traditional ML Models
Each notebook implements multiple algorithms including:
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting (XGBoost, LightGBM)
- Logistic Regression
- Neural Networks

### Deep Learning Models
- **DeepChem Networks**: Deep neural networks using DeepChem framework
- **Graph Neural Networks**: GCN, MPNN, and Graph Attention models
- **Fine-tuned ChemBERTa**: Pre-trained transformer models for chemistry
  - ChemBERTa 5M MLM
  - ChemBERTa 10M MLM
  - ChemBERTa 77M MLM
  - SMILES Tokenizer PubChem 1M

## 📁 Repository Structure

```
├── Data_Cleaning_and_Preparation.ipynb          # Data preprocessing and cleaning
├── classificationModelling_*.ipynb              # Traditional ML models
│   ├── circularFingerprint.ipynb               # ECFP-based models
│   ├── MACCSkeysFeature.ipynb                   # MACCS keys models
│   ├── RDKiTFeatures.ipynb                     # RDKit descriptor models
│   ├── PubchemFeatures.ipynb                   # PubChem feature models
│   ├── modredFeatures.ipynb                    # ModRed descriptor models
│   ├── mol2vecFeatures.ipynb                   # Mol2Vec models
│   ├── graphConvAndGroover.ipynb               # Graph convolution models
│   ├── MPNN.ipynb                              # Message passing networks
│   └── GraphAttentionTransformer.ipynb         # Graph attention models
├── deepnet_*.ipynb                             # Deep learning models
│   ├── circularfingerprint.ipynb               # Deep nets with ECFP
│   ├── MACCSkeys.ipynb                          # Deep nets with MACCS
│   ├── RDKit.ipynb                             # Deep nets with RDKit
│   ├── Pubchem.ipynb                           # Deep nets with PubChem
│   ├── modred.ipynb                            # Deep nets with ModRed
│   └── mol2Vec.ipynb                           # Deep nets with Mol2Vec
├── FineTunedChemberta*.ipynb                    # Pre-trained transformer models
├── ModelInterpretation_GraphConvolutionalNetwork.ipynb  # Model interpretability
├── requirements.txt                            # Python dependencies
├── environment.yml                             # Conda environment
├── setup.py                                    # Package setup
└── README.md                                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Conda or pip package manager
- Jupyter Notebook or JupyterLab
- CUDA-compatible GPU (recommended for deep learning models)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/bhatnira/ML-AChEI-ClassificationModels-DrugDiscovery.git
cd ML-AChEI-ClassificationModels-DrugDiscovery
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate achei-ml
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. **Install additional dependencies**
```bash
# For graph neural networks
pip install torch-geometric
pip install dgl

# For chemistry-specific libraries
pip install deepchem
pip install mordred
pip install mol2vec
```

### Usage

1. **Data Preparation**: Start with `Data_Cleaning_and_Preparation.ipynb`
2. **Traditional ML**: Run any of the `classificationModelling_*.ipynb` notebooks
3. **Deep Learning**: Explore `deepnet_*.ipynb` notebooks
4. **Transformers**: Try pre-trained models in `FineTunedChemberta*.ipynb`
5. **Interpretation**: Use `ModelInterpretation_GraphConvolutionalNetwork.ipynb`

## 📈 Model Performance

Each notebook includes comprehensive evaluation metrics:
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **ROC Analysis**: AUC-ROC curves and scores
- **Cross-validation**: Stratified k-fold validation
- **Feature Importance**: Analysis of important molecular features
- **Hyperparameter Optimization**: Grid search and random search

## 🔬 Key Features

- **Comprehensive Feature Engineering**: Multiple molecular representation approaches
- **Automated ML Pipeline**: TPOT-based automated machine learning
- **Model Interpretability**: SHAP values and feature importance analysis
- **Reproducible Results**: Fixed random seeds and detailed methodology
- **Scalable Architecture**: Efficient implementation for large datasets

## 📚 Dependencies

### Core Libraries
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- rdkit-pypi
- deepchem
- torch, tensorflow

### Specialized Libraries
- mol2vec
- mordred
- tpot
- shap
- dgl, torch-geometric

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Authors**: Nirajan Bhattarai, Marvin Schulte
- **Email**: bhatnira@isu.edu
- **GitHub**: [@bhatnira](https://github.com/bhatnira)
- **Repository**: [ML-AChEI-ClassificationModels-DrugDiscovery](https://github.com/bhatnira/ML-AChEI-ClassificationModels-DrugDiscovery)

## 🙏 Acknowledgments

- ChEMBL database for providing the acetylcholinesterase inhibitor dataset
- RDKit community for molecular descriptor calculations
- DeepChem team for deep learning frameworks
- Open source community for various machine learning libraries

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{bhattarai2024achei,
  title={ML AChEI Classification Models for Drug Discovery},
  author={Bhattarai, Nirajan and Schulte, Marvin},
  year={2024},
  url={https://github.com/bhatnira/ML-AChEI-ClassificationModels-DrugDiscovery}
}
```

---

⭐ **Star this repository if you find it helpful!**