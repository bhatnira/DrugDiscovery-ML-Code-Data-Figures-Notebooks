# ML Regression Models for AChEI Drug Discovery

A comprehensive machine learning project implementing multiple regression approaches for acetylcholinesterase inhibitor (AChEI) drug discovery using various molecular representation techniques.

## 🧬 Overview

This repository contains multiple Jupyter notebooks implementing different machine learning regression models for predicting acetylcholinesterase inhibition activity (pIC50 values) of molecular compounds. The project explores various molecular featurization techniques and modeling approaches to identify potential drug candidates.

## 📊 Project Structure

```
├── README.md
├── requirements.txt                                      # Python dependencies
├── environment.yml                                       # Conda environment file
├── data/                                                # Data directory (create as needed)
├── notebooks/
│   ├── regressionModeling_RDKitFeatures.ipynb          # RDKit descriptor-based modeling
│   ├── regressionModeling_deepNet_RDKitFeatures.ipynb  # Deep neural network with RDKit features
│   ├── RegressionModeling_GraphConvolutionalNetwork.ipynb # Graph convolutional network modeling
│   └── RegressionModelling_circularFingerprint.ipynb   # Circular fingerprint-based modeling
├── src/                                                 # Source code modules (optional)
├── results/                                             # Model outputs and results
└── docs/                                               # Additional documentation
```

## 🔬 Modeling Approaches

### 1. RDKit Features Regression (`regressionModeling_RDKitFeatures.ipynb`)
- **Description**: Traditional machine learning models using RDKit molecular descriptors
- **Features**: 2D molecular descriptors from RDKit library
- **Models**: Various regression algorithms (Random Forest, SVM, etc.)
- **Use Case**: Baseline models with interpretable molecular descriptors

### 2. Deep Neural Network with RDKit Features (`regressionModeling_deepNet_RDKitFeatures.ipynb`)
- **Description**: Deep learning approach using RDKit descriptors as input
- **Features**: RDKit molecular descriptors
- **Models**: Multi-layer neural networks
- **Use Case**: Non-linear modeling with traditional descriptors

### 3. Graph Convolutional Networks (`RegressionModeling_GraphConvolutionalNetwork.ipynb`)
- **Description**: Graph-based deep learning using molecular graph representations
- **Features**: Molecular graphs (atoms as nodes, bonds as edges)
- **Models**: Graph Convolutional Networks (GCN) using DeepChem
- **Use Case**: State-of-the-art molecular property prediction

### 4. Circular Fingerprint Models (`RegressionModelling_circularFingerprint.ipynb`)
- **Description**: Models using Extended-Connectivity Fingerprints (ECFP)
- **Features**: Circular fingerprints (ECFP4, ECFP6)
- **Models**: Various ML algorithms with fingerprint features
- **Use Case**: Substructure-based modeling with interpretability

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- CUDA-capable GPU (optional, for deep learning models)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhatnira/ML-Regression-AChEI-DrugDiscovery.git
   cd ML-Regression-AChEI-DrugDiscovery
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate achei-regression
   ```

   **Or using pip:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter lab
   ```

### Data Requirements

- **Input**: Molecular SMILES strings with corresponding IC50 values
- **Format**: Excel file with columns for SMILES and IC50 values
- **Processing**: Molecules are standardized and filtered by molecular weight (<800 Da)

## 📈 Key Features

- **Multiple Featurization Methods**: RDKit descriptors, molecular graphs, circular fingerprints
- **Comprehensive Modeling**: Traditional ML, deep learning, and graph neural networks
- **Model Evaluation**: Cross-validation, performance metrics, and visualization
- **Reproducibility**: Fixed random seeds for consistent results
- **Interpretability**: Feature importance analysis and model interpretation

## 🛠️ Dependencies

### Core Libraries
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning algorithms
- `rdkit`: Molecular informatics
- `deepchem`: Graph neural networks and molecular ML
- `tensorflow`/`keras`: Deep learning frameworks

### Visualization
- `matplotlib`, `seaborn`: Plotting and visualization
- `plotly`: Interactive plots

### Additional Tools
- `jupyter`: Notebook environment
- `google-colab`: Colab integration support

## 📊 Model Performance

Each notebook includes:
- **Data preprocessing and quality checks**
- **Feature engineering and selection**
- **Model training and hyperparameter tuning**
- **Cross-validation and performance evaluation**
- **Results visualization and interpretation**

## 🔍 Usage Examples

### Running Individual Notebooks

1. **RDKit Features Model:**
   ```bash
   jupyter notebook regressionModeling_RDKitFeatures.ipynb
   ```

2. **Graph Convolutional Network:**
   ```bash
   jupyter notebook RegressionModeling_GraphConvolutionalNetwork.ipynb
   ```

### Expected Workflow
1. Load and preprocess molecular data
2. Calculate molecular features/representations
3. Split data into train/validation/test sets
4. Train regression models
5. Evaluate model performance
6. Analyze feature importance
7. Make predictions on new compounds

## 📝 Results and Outputs

- **Model Metrics**: R², RMSE, MAE, and correlation coefficients
- **Visualizations**: Predicted vs actual plots, feature importance plots
- **Model Files**: Trained models saved for future use
- **Feature Analysis**: Identification of important molecular descriptors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -am 'Add new modeling approach'`)
4. Push to branch (`git push origin feature/new-model`)
5. Create a Pull Request

## 📚 References

- Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*
- RDKit Documentation: https://rdkit.org/docs/
- DeepChem Documentation: https://deepchem.readthedocs.io/

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Nirajan Bhattarai** - *Initial work* - [bhatnira](https://github.com/bhatnira) - bhatnira@isu.edu
- **Marvin Schulte** - *Co-author*

## 🙏 Acknowledgments

- RDKit community for molecular informatics tools
- DeepChem developers for graph neural network implementations
- Open-source community for machine learning libraries

---

**Note**: This project is designed for research and educational purposes in computational drug discovery. Results should be validated experimentally before any practical application.
