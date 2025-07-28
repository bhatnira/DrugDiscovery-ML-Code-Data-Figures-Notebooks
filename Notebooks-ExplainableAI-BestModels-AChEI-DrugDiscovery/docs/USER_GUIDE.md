# User Guide

## Getting Started

This guide will help you set up and use the ExplainableAI-BestModels-AChEI-DrugDiscovery repository for molecular classification and interpretation.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Notebook Overview](#notebook-overview)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Model Interpretation](#model-interpretation)
7. [Visualization](#visualization)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

## Installation

### Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- Git
- 8GB+ RAM (16GB recommended)
- Jupyter Lab or Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone https://github.com/bhatnira/ExplainableAI-BestModels-AChEI-DrugDiscovery.git
cd ExplainableAI-BestModels-AChEI-DrugDiscovery
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n xai-drug-discovery python=3.8
conda activate xai-drug-discovery

# Or using venv
python -m venv xai-drug-discovery
source xai-drug-discovery/bin/activate  # Linux/Mac
# xai-drug-discovery\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
# Test basic imports
python -c "
import deepchem
import rdkit
import transformers
import shap
print('All packages installed successfully!')
"
```

## Quick Start

### 1. Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

### 2. Choose Your Starting Point

**For Beginners**: Start with `ModelInterpretability_deepNet_rdkit.ipynb`
- Uses traditional molecular descriptors
- Easier to understand and interpret
- Good baseline performance

**For Advanced Users**: Try `FineTunedChemberta(DeepChem_ChemBERTa_10M_MLM).ipynb`
- State-of-the-art transformer model
- Advanced attention visualization
- Requires more computational resources

### 3. Follow the Notebook

Each notebook is self-contained with:
- Clear explanations in markdown cells
- Step-by-step code execution
- Visualization of results
- Interpretation examples

## Notebook Overview

### 1. FineTunedChemberta(DeepChem_ChemBERTa_10M_MLM).ipynb

**Purpose**: Fine-tune pre-trained ChemBERTa for AChEI classification

**What you'll learn**:
- Transformer models for chemistry
- Attention mechanism interpretation
- Transfer learning in drug discovery

**Best for**: 
- Advanced ML practitioners
- Those interested in NLP for chemistry
- GPU available (recommended)

**Estimated runtime**: 2-4 hours with GPU, 8+ hours with CPU

### 2. ModelInterpretationAndCompoundGeneration_BestAggregrateModelCircularFingerprint.ipynb

**Purpose**: Circular fingerprint-based classification with compound generation

**What you'll learn**:
- Molecular fingerprints (ECFP)
- Substructure analysis
- Novel compound generation

**Best for**:
- Traditional ML approaches
- Fast prototyping
- Substructure-based interpretation

**Estimated runtime**: 30-60 minutes

### 3. ModelInterpretation_GraphConvolutionalNetwork.ipynb

**Purpose**: Graph neural networks for molecular classification

**What you'll learn**:
- Graph representation of molecules
- Node and edge attribution
- Graph-based interpretability

**Best for**:
- Graph ML enthusiasts
- Molecular graph analysis
- Atom-level interpretations

**Estimated runtime**: 1-2 hours

### 4. ModelInterpretability_deepNet_rdkit.ipynb

**Purpose**: Deep learning with traditional molecular descriptors

**What you'll learn**:
- RDKit molecular descriptors
- SHAP value analysis
- Feature importance ranking

**Best for**:
- Beginners in cheminformatics
- Traditional ML approaches
- Interpretable features

**Estimated runtime**: 30-45 minutes

### 5. ModelInterpretation_RdkitFeatureBasedAutoMLModel.ipynb

**Purpose**: Automated machine learning with molecular features

**What you'll learn**:
- AutoML for drug discovery
- Automated hyperparameter tuning
- Ensemble methods

**Best for**:
- Quick model prototyping
- Baseline comparisons
- Automated workflows

**Estimated runtime**: 45-90 minutes

## Data Preparation

### Input Data Format

Your data should be in CSV format with the following columns:

```csv
smiles,activity,molecule_id
CCO,0,mol_001
CC(=O)OC1=CC=CC=C1C(=O)O,1,mol_002
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0,mol_003
```

**Required columns**:
- `smiles`: SMILES representation of molecules
- `activity`: Binary labels (0=inactive, 1=active)

**Optional columns**:
- `molecule_id`: Unique identifier
- `ic50`: IC50 values (if available)
- `molecular_weight`: Pre-calculated molecular weight

### Data Quality Checks

Before training, ensure your data meets these criteria:

```python
# Check SMILES validity
from rdkit import Chem

def check_smiles_validity(smiles_list):
    valid_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1
    return valid_count / len(smiles_list)

validity_ratio = check_smiles_validity(df['smiles'])
print(f"SMILES validity: {validity_ratio:.2%}")
```

### Preprocessing Steps

1. **SMILES Standardization**
   ```python
   from rdkit.Chem import rdMolStandardize
   
   def standardize_smiles(smiles):
       mol = Chem.MolFromSmiles(smiles)
       if mol is None:
           return None
       # Standardize
       mol = rdMolStandardize.Cleanup(mol)
       return Chem.MolToSmiles(mol, canonical=True)
   ```

2. **Molecular Weight Filtering**
   ```python
   from rdkit.Chem import Descriptors
   
   def filter_by_mw(smiles_list, max_mw=800):
       filtered = []
       for smiles in smiles_list:
           mol = Chem.MolFromSmiles(smiles)
           if mol and Descriptors.MolWt(mol) <= max_mw:
               filtered.append(smiles)
       return filtered
   ```

3. **Remove Duplicates**
   ```python
   # Remove duplicate SMILES
   df = df.drop_duplicates(subset=['smiles'])
   ```

## Model Training

### Configuration Options

Each model has configurable parameters:

#### ChemBERTa
```python
config = {
    'model_name': 'DeepChem/ChemBERTa-10M-MLM',
    'max_length': 128,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'epochs': 10,
    'patience': 3  # Early stopping
}
```

#### Graph CNN
```python
config = {
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64
}
```

#### Traditional ML
```python
config = {
    'model_type': 'random_forest',  # or 'xgboost', 'svm'
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}
```

### Training Process

1. **Data Splitting**
   ```python
   from sklearn.model_selection import train_test_split
   
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42
   )
   ```

2. **Model Training**
   ```python
   # Example for any model
   model.fit(X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'])
   ```

3. **Model Evaluation**
   ```python
   from sklearn.metrics import classification_report, roc_auc_score
   
   y_pred = model.predict(X_test)
   y_prob = model.predict_proba(X_test)[:, 1]
   
   print(classification_report(y_test, y_pred))
   print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
   ```

## Model Interpretation

### SHAP Analysis

SHAP (SHapley Additive exPlanations) provides feature importance:

```python
import shap

# Create explainer
explainer = shap.Explainer(model.predict, X_background)

# Calculate SHAP values
shap_values = explainer(X_test[:100])

# Visualize
shap.plots.waterfall(shap_values[0])  # Single prediction
shap.plots.beeswarm(shap_values)      # Multiple predictions
shap.plots.bar(shap_values)           # Feature importance
```

### Attention Visualization (ChemBERTa)

For transformer models, visualize attention weights:

```python
from bertviz import head_view, model_view

# Get attention weights
attention = model.get_attention_weights(input_ids)

# Visualize attention heads
head_view(attention, tokens)

# Model-level attention
model_view(attention, tokens)
```

### Molecular Highlighting

Highlight important molecular substructures:

```python
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

def highlight_atoms(mol, atom_weights, colorMap=None):
    """Highlight atoms based on importance weights"""
    if colorMap is None:
        colorMap = {i: (1, 1-w, 1-w) for i, w in enumerate(atom_weights)}
    
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), 
                       highlightAtomColors=colorMap)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
```

## Visualization

### Performance Plots

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

### Feature Importance

```python
# For tree-based models
feature_importance = model.feature_importances_
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(20), feature_importance[indices[:20]])
plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45)
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.show()
```

### Molecular Diversity

```python
from rdkit.Chem import rdMolDescriptors
from sklearn.manifold import TSNE

# Calculate molecular descriptors
descriptors = []
for smiles in df['smiles']:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        desc = rdMolDescriptors.CalcMolDescriptors(mol)
        descriptors.append(list(desc.values()))

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
coords = tsne.fit_transform(descriptors)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df['activity'], 
                     cmap='RdYlBu', alpha=0.7)
plt.colorbar(scatter)
plt.title('Molecular Diversity (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
```

## Troubleshooting

### Common Issues

#### 1. Installation Problems

**Issue**: Package conflicts or import errors
```bash
# Solution: Create fresh environment
conda create -n xai-clean python=3.8
conda activate xai-clean
pip install -r requirements.txt
```

#### 2. Memory Issues

**Issue**: Out of memory during training
```python
# Solution: Reduce batch size
batch_size = 16  # Instead of 32
```

**Issue**: GPU memory problems
```python
# Solution: Clear GPU cache
import torch
torch.cuda.empty_cache()
```

#### 3. SMILES Processing Errors

**Issue**: RDKit cannot parse SMILES
```python
# Solution: Filter invalid SMILES
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

df = df[df['smiles'].apply(is_valid_smiles)]
```

#### 4. Long Training Times

**Issue**: Training takes too long
```python
# Solutions:
# 1. Reduce dataset size for testing
df_small = df.sample(n=1000)

# 2. Use early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# 3. Reduce model complexity
hidden_dim = 64  # Instead of 128
```

### Performance Optimization

#### 1. Data Loading
```python
# Use efficient data loading
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

#### 2. GPU Utilization
```python
# Check GPU usage
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
```

#### 3. Mixed Precision Training
```python
# For faster training with less memory
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

## FAQ

### General Questions

**Q: Which notebook should I start with?**
A: For beginners, start with `ModelInterpretability_deepNet_rdkit.ipynb`. For advanced users interested in transformers, try `FineTunedChemberta(DeepChem_ChemBERTa_10M_MLM).ipynb`.

**Q: Do I need a GPU?**
A: GPU is recommended for ChemBERTa and Graph CNN models but not required. Traditional ML models work fine on CPU.

**Q: How much data do I need?**
A: Minimum 1000 molecules for meaningful results. 10,000+ molecules recommended for robust models.

**Q: Can I use my own dataset?**
A: Yes! Ensure your data has SMILES strings and binary activity labels. Follow the data preparation guidelines.

### Technical Questions

**Q: Why are some SMILES invalid?**
A: SMILES can be invalid due to:
- Incorrect syntax
- Unsupported atom types
- Invalid bond configurations
- Encoding issues

**Q: How do I improve model performance?**
A: Try:
- More training data
- Better data quality
- Hyperparameter tuning
- Ensemble methods
- Feature engineering

**Q: What if my molecules are too large?**
A: For large molecules (>100 atoms):
- Use graph CNN models
- Increase max_length for transformers
- Consider fragmenting molecules

**Q: How do I interpret SHAP values?**
A: SHAP values indicate feature contribution:
- Positive values increase prediction
- Negative values decrease prediction
- Magnitude indicates importance

### Model-Specific Questions

**Q: ChemBERTa attention doesn't make sense?**
A: Attention patterns can be complex. Consider:
- Different attention heads show different patterns
- Layer-wise attention varies
- Average attention across multiple examples

**Q: Graph CNN predictions seem random?**
A: Check:
- Graph connectivity (ensure proper bond representation)
- Node features (atomic properties)
- Model convergence during training

**Q: Traditional ML performs better than deep learning?**
A: This can happen when:
- Dataset is small
- Features are well-engineered
- Problem is relatively simple
- Deep models are overfitting

### Interpretation Questions

**Q: How do I trust the explanations?**
A: Validate explanations by:
- Comparing across different methods
- Testing on known examples
- Consulting domain experts
- Using multiple interpretation techniques

**Q: Can I use these explanations for drug design?**
A: Explanations provide insights but should be:
- Validated experimentally
- Combined with chemical knowledge
- Used as guidance, not absolute truth
- Confirmed with additional analysis

## Getting Help

### Resources
- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check technical documentation in `/docs`
- **Examples**: Look at notebook examples
- **Literature**: See references in README

### Contact
- **Nirajan Bhattarai**: [GitHub Profile](https://github.com/bhatnira)
- **Marvin Schulte**: Research Collaborator

### Community
- Join discussions in GitHub Discussions
- Share your results and insights
- Contribute improvements and extensions

---

Happy modeling! Remember that explainable AI is an iterative process - combine multiple interpretation methods for the most robust insights.
