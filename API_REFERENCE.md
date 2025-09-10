# API Reference Documentation - AChE Drug Discovery Research Suite

## 📖 Overview

This API reference provides comprehensive documentation for the **AChE Drug Discovery Research Suite** containing 37 sanitized research notebooks for machine learning and AI-driven drug discovery. All notebooks have been professionally sanitized with file paths anonymized for secure sharing and collaboration.

## 🔒 Sanitization Information

**Status**: ✅ All 37 notebooks sanitized  
**Privacy**: 🛡️ All sensitive paths replaced with `"....."`  
**Functionality**: ✅ Complete research methodology preserved  
**Sharing**: 👥 Safe for public repositories and collaboration

## 📝 Notebook Collections API

### Classification Models Collection (31 Notebooks)
**Location**: `Notebooks-ML-AChEI-ClassificationModels-DrugDiscovery/`

#### Core Data Processing
##### `Data_Cleaning_and_Preparation.ipynb`
**Purpose**: Dataset preprocessing and standardization  
**Input**: Raw ChEMBL data  
**Output**: Standardized molecular datasets  
**Key Functions**:
- SMILES standardization and validation
- IC50 threshold-based classification
- Cross-species dataset preparation
- Molecular weight filtering (<800 Da)

#### Traditional ML Approaches

##### `classificationModelling_RDKiTFeatures.ipynb`
**Purpose**: RDKit molecular descriptor-based classification  
**Features**: 200+ molecular properties and topological indices  
**Models**: Random Forest, SVM, XGBoost, AdaBoost  
**Performance**: ~0.87 ROC-AUC  

##### `classificationModelling_circularFingerprint.ipynb`
**Purpose**: Morgan circular fingerprint-based classification  
**Features**: Morgan fingerprints (radius 2-3, 1024-2048 bits)  
**Models**: Ensemble methods with fingerprint optimization  
**Performance**: ~0.89 ROC-AUC  

##### `classificationModelling_MACCSkeysFeature.ipynb`
**Purpose**: MACCS structural keys analysis  
**Features**: 166-bit structural keys for pharmacophore analysis  
**Models**: Traditional ML with structural fingerprints  

##### `classificationModelling_PubchemFeatures.ipynb`
**Purpose**: PubChem database-derived fingerprints  
**Features**: Database-specific chemical representations  
**Models**: ML models with database fingerprints  

##### `classificationModelling_modredFeatures.ipynb`
**Purpose**: Reduced molecular descriptor analysis  
**Features**: Dimensionality-reduced molecular properties  
**Models**: Feature selection and traditional ML  

##### `classificationModelling_mol2vecFeatures.ipynb`
**Purpose**: Mol2Vec molecular embeddings  
**Features**: Unsupervised molecular embeddings  
**Models**: Deep learning with molecular embeddings  

#### Deep Learning Approaches

##### `deepNet_RDKit.ipynb`
**Purpose**: Deep neural networks with RDKit features  
**Architecture**: Dense layers with dropout and batch normalization  
**Features**: RDKit molecular descriptors  
**Performance**: Enhanced performance over traditional ML  

##### `deepnet_circularfingerprint.ipynb`
**Purpose**: Deep learning with circular fingerprints  
**Architecture**: Feedforward networks optimized for fingerprints  
**Features**: Morgan circular fingerprints  

##### `deepNet_Pubchem.ipynb`
**Purpose**: Neural networks with PubChem features  
**Architecture**: Deep networks for database fingerprints  

##### `deepnet_MACCSkeys.ipynb`
**Purpose**: Deep learning with MACCS keys  
**Architecture**: Networks optimized for structural keys  

##### `deepNet_modred.ipynb`
**Purpose**: Neural networks with reduced features  
**Architecture**: Optimized for dimensionality-reduced data  

##### `deepnet_mol2Vec.ipynb`
**Purpose**: Deep learning with Mol2Vec embeddings  
**Architecture**: Networks for molecular embeddings  

#### Graph Neural Networks

##### `classificationModelling_graphConvAndGroover.ipynb`
**Purpose**: Graph convolutional networks for molecular graphs  
**Architecture**: GraphConv layers with molecular graph representation  
**Features**: Node (atom) and edge (bond) feature matrices  
**Performance**: ~0.91 ROC-AUC  

##### `classificationModeling_GraphAttentionTransformer.ipynb`
**Purpose**: Graph attention mechanisms  
**Architecture**: Attention-based graph neural networks  
**Features**: Attention weights for interpretability  

##### `classificationModeling_MPNN.ipynb`
**Purpose**: Message Passing Neural Networks  
**Architecture**: MPNN with molecular graph processing  
**Performance**: ~0.93 ROC-AUC (best overall)  

#### Transformer Models

##### `FineTunedChemberta(DeepChem_ChemBERTa_5M_MLM).ipynb`
**Purpose**: ChemBERTa 5M parameter model fine-tuning  
**Architecture**: Transformer with 5M parameters  
**Features**: SMILES tokenization and attention mechanisms  

##### `FineTunedChemberta(DeepChem_ChemBERTa_10M_MLM).ipynb`
**Purpose**: ChemBERTa 10M parameter model fine-tuning  
**Architecture**: Transformer with 10M parameters  
**Performance**: ~0.91 ROC-AUC  

##### `FineTunedChemberta(DeepChem_ChemBERTa_77M_MLM).ipynb`
**Purpose**: ChemBERTa 77M parameter model fine-tuning  
**Architecture**: Large transformer with 77M parameters  
**Features**: Advanced attention mechanisms and tokenization  

##### `FineTunedChemberta(DeepChem_SmilesTokenizer_PubChem_1M).ipynb`
**Purpose**: Custom SMILES tokenizer with PubChem training  
**Architecture**: Custom tokenization strategy  
**Features**: PubChem-trained tokenizer  

### Regression Models Collection (4 Notebooks)
**Location**: `Notebooks-ML-Regression-AChEI-DrugDiscovery/`

##### `regressionModeling_RDKitFeatures.ipynb`
**Purpose**: Traditional regression with RDKit descriptors  
**Target**: Continuous IC50/pIC50 prediction  
**Models**: SVR, Random Forest Regressor, XGBoost  
**Performance**: R² ~0.69  

##### `regressionModeling_deepNet_RDKitFeatures.ipynb`
**Purpose**: Deep neural network regression  
**Architecture**: Deep networks for continuous prediction  
**Performance**: R² ~0.75  

##### `RegressionModelling_circularFingerprint.ipynb`
**Purpose**: Circular fingerprint-based regression  
**Features**: Morgan fingerprints for continuous prediction  
**Performance**: R² ~0.72  

##### `RegressionModeling_GraphConvolutionalNetwork.ipynb`
**Purpose**: Graph neural network regression  
**Architecture**: Graph convolution for continuous targets  
**Performance**: R² ~0.78 (best regression performance)  

### Explainable AI Collection (5 Notebooks)
**Location**: `Notebooks-ExplainableAI-BestModels-AChEI-DrugDiscovery/`

##### `ModelInterpretation_GraphConvolutionalNetwork.ipynb`
**Purpose**: Graph neural network interpretability  
**Techniques**: Node importance, edge weights, attention visualization  
**Features**: Atomic contribution analysis  

##### `ModelInterpretation_RdkitFeatureBasedAutoMLModel.ipynb`
**Purpose**: Feature importance analysis for AutoML models  
**Techniques**: SHAP values, feature importance ranking  
**Features**: Global and local explanations  

##### `ModelInterpretability_deepNet_rdkit.ipynb`
**Purpose**: Deep network interpretability  
**Techniques**: Layer-wise relevance propagation, gradient analysis  

##### `ModelInterpretationAndCompoundGeneration_BestAggregrateModelCircularFingerprint.ipynb`
**Purpose**: Compound generation and design using best models  
**Features**: Molecular generation, structure-activity insights  
**Applications**: Lead compound optimization  

##### `FineTunedChemberta(DeepChem_ChemBERTa_10M_MLM).ipynb`
**Purpose**: Transformer attention analysis and interpretation  
**Techniques**: Attention weight visualization, token importance  
**Features**: Sequential pattern analysis in SMILES  

## 🔧 Working with Sanitized Notebooks

### Path Replacement Requirements
All notebooks contain sanitized paths (`"....."`) that need replacement:

```python
# Common replacements needed:
df = pd.read_excel(".....")  # Replace with your data path
model.save(".....")          # Replace with your model path
results.to_csv(".....")      # Replace with your output path
```

### Environment Setup
```bash
# Essential dependencies
conda install -c conda-forge rdkit
pip install deepchem transformers torch
pip install pandas numpy scikit-learn
```

### Data Format Requirements
Notebooks expect datasets with:
- **Smiles**: SMILES notation strings
- **IC50**: Activity values in nM  
- **classLabel**: Binary classification (0/1)

## Overview

This document provides comprehensive API reference for the AChE Activity Prediction Suite. The suite consists of multiple prediction models and utilities for acetylcholinesterase inhibitor analysis.

## Core Modules

### Main Application (`main_app.py`)

#### Functions

##### `show_home_page()`
Displays the main dashboard with application overview and feature highlights.

**Returns:**
- None (renders Streamlit interface)

##### `run_chemberta_app()`
Launches the ChemBERTa transformer-based prediction application.

**Returns:**
- None (executes subprocess)

**Features:**
- Transformer-based molecular property prediction
- Attention weight visualization
- SMILES and drawing input support

##### `run_rdkit_app()`
Launches the RDKit descriptor-based prediction application.

**Returns:**
- None (executes subprocess)

**Features:**
- Traditional molecular descriptors
- Feature importance analysis
- Batch processing support

##### `run_circular_app()`
Launches the circular fingerprint-based prediction application.

**Returns:**
- None (executes subprocess)

**Features:**
- Morgan circular fingerprints
- LIME explanations
- Structural similarity analysis

##### `run_graph_app()`
Launches the graph neural network prediction application.

**Returns:**
- None (executes subprocess)

**Features:**
- Graph convolutional networks
- Atomic contribution maps
- Molecular graph analysis

---

## Graph Neural Network Module (`app_graph_combined.py`)

### Core Classes

#### `GraphPredictionPipeline`
Main pipeline class for graph-based molecular predictions.

```python
class GraphPredictionPipeline:
    """
    Pipeline for graph neural network-based molecular property prediction.
    
    Attributes:
        model_reg: Regression model for IC50 prediction
        model_class: Classification model for activity prediction
        scaler: Data scaler for preprocessing
    """
```

### Key Functions

#### `standardize_smiles(smiles, verbose=False)`
Standardizes SMILES strings using RDKit.

**Parameters:**
- `smiles` (str): Input SMILES string
- `verbose` (bool): Enable verbose logging

**Returns:**
- `str`: Standardized SMILES string

**Example:**
```python
std_smiles = standardize_smiles("CCO")
# Returns: "CCO"
```

#### `smiles_to_graph(smiles)`
Converts SMILES string to graph representation for neural network input.

**Parameters:**
- `smiles` (str): SMILES string

**Returns:**
- `ConvMol`: DeepChem ConvMol object

#### `calculate_atomic_contributions(model, mol, smiles)`
Calculates atomic contributions for model interpretability.

**Parameters:**
- `model`: Trained graph neural network model
- `mol`: RDKit molecule object
- `smiles` (str): SMILES string

**Returns:**
- `np.array`: Atomic contribution scores

#### `vis_contribs(mol, contribs)`
Visualizes atomic contributions on molecular structure.

**Parameters:**
- `mol`: RDKit molecule object
- `contribs` (np.array): Atomic contribution scores

**Returns:**
- `PIL.Image`: Contribution visualization

---

## Circular Fingerprint Module (`app_circular.py`)

### Core Functions

#### `circular_fingerprint_with_fallback(smiles, radius=2, nBits=2048)`
Generates robust circular fingerprints with multiple fallback strategies.

**Parameters:**
- `smiles` (str): Input SMILES string
- `radius` (int): Morgan fingerprint radius (default: 2)
- `nBits` (int): Fingerprint bit vector size (default: 2048)

**Returns:**
- `np.array`: Fingerprint bit vector

**Fallback Strategy:**
1. Standard Morgan fingerprint
2. Sanitized molecule fingerprint
3. Basic molecular fingerprint
4. Zero vector (if all fail)

#### `interpret_with_lime(model, input_features, X_train)`
Provides LIME-based interpretability for predictions.

**Parameters:**
- `model`: Trained prediction model
- `input_features` (np.array): Input feature vector
- `X_train` (np.array): Training data for background

**Returns:**
- `lime.explanation.Explanation`: LIME explanation object

---

## RDKit Descriptors Module (`app_rdkit.py`)

### Core Functions

#### `calculate_rdkit_descriptors(mol)`
Calculates comprehensive molecular descriptors using RDKit.

**Parameters:**
- `mol`: RDKit molecule object

**Returns:**
- `np.array`: Descriptor feature vector

**Descriptors Included:**
- Molecular weight
- LogP
- Number of rotatable bonds
- Topological polar surface area
- Number of aromatic rings
- And 200+ additional descriptors

#### `feature_importance_analysis(model, feature_names)`
Analyzes feature importance for model interpretability.

**Parameters:**
- `model`: Trained prediction model
- `feature_names` (list): Names of molecular descriptors

**Returns:**
- `pd.DataFrame`: Feature importance scores

---

## ChemBERTa Module (`app_chemberta.py`)

### Core Functions

#### `load_chemberta_model(model_path)`
Loads pre-trained ChemBERTa transformer model.

**Parameters:**
- `model_path` (str): Path to model files

**Returns:**
- `transformers.AutoModel`: Loaded ChemBERTa model

#### `tokenize_smiles(smiles, tokenizer)`
Tokenizes SMILES string for transformer input.

**Parameters:**
- `smiles` (str): Input SMILES string
- `tokenizer`: ChemBERTa tokenizer

**Returns:**
- `dict`: Tokenized input dictionary

#### `extract_attention_weights(model, inputs)`
Extracts attention weights for visualization.

**Parameters:**
- `model`: ChemBERTa model
- `inputs` (dict): Tokenized inputs

**Returns:**
- `torch.Tensor`: Attention weight tensor

---

## Utility Functions

### Data Processing

#### `validate_smiles(smiles)`
Validates SMILES string format and chemical validity.

**Parameters:**
- `smiles` (str): Input SMILES string

**Returns:**
- `bool`: True if valid, False otherwise

#### `batch_process_molecules(smiles_list, prediction_function)`
Processes multiple molecules in batch.

**Parameters:**
- `smiles_list` (list): List of SMILES strings
- `prediction_function` (callable): Prediction function to apply

**Returns:**
- `list`: List of prediction results

### Visualization

#### `create_molecular_visualization(mol, size=(300, 300))`
Creates molecular structure visualization.

**Parameters:**
- `mol`: RDKit molecule object
- `size` (tuple): Image dimensions

**Returns:**
- `PIL.Image`: Molecular structure image

#### `create_contribution_heatmap(contribs, labels)`
Creates heatmap visualization for feature contributions.

**Parameters:**
- `contribs` (np.array): Contribution scores
- `labels` (list): Feature labels

**Returns:**
- `matplotlib.Figure`: Heatmap figure

---

## Model Specifications

### Graph Neural Network Models

#### Architecture
- **Type**: Graph Convolutional Network (GraphConv)
- **Input**: Molecular graphs with atom/bond features
- **Output**: Classification (Active/Inactive) + Regression (IC50 values)
- **Framework**: DeepChem

#### Training Details
- **Dataset**: ChEMBL AChE inhibitors
- **Training samples**: ~15,000 compounds
- **Validation**: 5-fold cross-validation
- **Metrics**: ROC-AUC (classification), RMSE (regression)

### Circular Fingerprint Models

#### Architecture
- **Type**: TPOT-optimized ensemble
- **Input**: Morgan circular fingerprints (2048-bit)
- **Output**: Classification or regression
- **Framework**: TPOT + scikit-learn

#### Performance
- **Classification AUC**: 0.92±0.03
- **Regression R²**: 0.78±0.05
- **Training time**: ~10 minutes

### RDKit Descriptor Models

#### Architecture
- **Type**: TPOT-optimized ensemble
- **Input**: 208 molecular descriptors
- **Output**: Classification or regression
- **Framework**: TPOT + scikit-learn

#### Performance
- **Classification AUC**: 0.89±0.04
- **Regression R²**: 0.75±0.06
- **Training time**: ~15 minutes

### ChemBERTa Models

#### Architecture
- **Type**: Transformer (BERT-like)
- **Input**: SMILES strings (tokenized)
- **Output**: Molecular embeddings → prediction
- **Framework**: Transformers + PyTorch

#### Model Details
- **Parameters**: 24M
- **Context length**: 512 tokens
- **Attention heads**: 12
- **Hidden size**: 768

---

## Error Handling

### Common Exceptions

#### `InvalidSMILESError`
Raised when SMILES string is invalid or cannot be parsed.

```python
try:
    prediction = predict_from_smiles(smiles)
except InvalidSMILESError as e:
    print(f"Invalid SMILES: {e}")
```

#### `ModelLoadError`
Raised when model files cannot be loaded.

```python
try:
    model = load_model(model_path)
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
```

#### `PredictionError`
Raised when prediction fails due to computational issues.

```python
try:
    result = model.predict(features)
except PredictionError as e:
    print(f"Prediction failed: {e}")
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STREAMLIT_SERVER_PORT` | Application port | 10000 |
| `STREAMLIT_SERVER_ADDRESS` | Server address | 0.0.0.0 |
| `MODEL_PATH` | Path to model files | ./models |
| `CACHE_SIZE` | Prediction cache size | 1000 |

### Model Paths

```python
MODEL_PATHS = {
    'graph_regression': './GraphConv_model_files/',
    'graph_classification': './checkpoint-2000/',
    'circular_classifier': './bestPipeline_tpot_circularfingerprint_classification.pkl',
    'rdkit_classifier': './bestPipeline_tpot_rdkit_classification.pkl',
    'rdkit_regressor': './bestPipeline_tpot_rdkit_Regression.pkl'
}
```

---

## Performance Benchmarks

### Prediction Speed

| Model Type | Single Molecule | Batch (100) | Batch (1000) |
|------------|----------------|-------------|--------------|
| Graph NN | 150ms | 2.1s | 18.3s |
| Circular FP | 45ms | 0.8s | 6.2s |
| RDKit | 35ms | 0.6s | 4.8s |
| ChemBERTa | 280ms | 4.2s | 35.1s |

### Memory Usage

| Model Type | Base Memory | Peak Memory |
|------------|-------------|-------------|
| Graph NN | 1.2GB | 2.8GB |
| Circular FP | 450MB | 850MB |
| RDKit | 380MB | 720MB |
| ChemBERTa | 2.1GB | 4.5GB |

---

## Integration Examples

### REST API Integration

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:10000/api/predict',
    json={'smiles': 'CCO', 'model': 'graph'}
)
result = response.json()
```

### Python Package Integration

```python
from ache_predictor import GraphPredictor

# Initialize predictor
predictor = GraphPredictor()

# Make prediction
result = predictor.predict('CCO')
print(f"Activity: {result['activity']}")
print(f"IC50: {result['ic50']} nM")
```

### Batch Processing

```python
import pandas as pd
from ache_predictor import batch_predict

# Load data
df = pd.read_csv('compounds.csv')

# Batch prediction
results = batch_predict(
    df['smiles'].tolist(),
    model_type='circular',
    output_format='dataframe'
)
```

---

## Support and Troubleshooting

### Common Issues

1. **Model Loading Failures**: Ensure all model files are present
2. **Memory Errors**: Use smaller batch sizes or lighter models
3. **SMILES Parsing Errors**: Validate SMILES before prediction
4. **Slow Predictions**: Consider using RDKit for faster results

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

```python
from ache_predictor.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()
# ... run predictions ...
report = monitor.get_report()
```

---

## License and Attribution

This API reference is part of the AChE Activity Prediction Suite.
Licensed under Apache License 2.0.

For citation information, see the main README.md file.
