# Technical Documentation

## Project Architecture

### Overview
This repository implements a comprehensive machine learning pipeline for acetylcholinesterase inhibitor drug discovery with emphasis on explainable AI techniques. The architecture supports multiple model types and interpretability methods.

### System Requirements

#### Minimum Hardware Requirements
- **CPU**: 4+ cores, 2.5GHz+ recommended
- **RAM**: 16GB minimum, 32GB recommended for large models
- **Storage**: 10GB free space for dependencies and models
- **GPU**: Optional but recommended for transformer models (8GB+ VRAM)

#### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Jupyter**: JupyterLab or Jupyter Notebook

### Model Architectures

#### 1. ChemBERTa (Transformer-based)
```
Input: SMILES strings
├── Tokenization (RoBERTa tokenizer)
├── Embedding Layer (768 dimensions)
├── Transformer Encoder (12 layers)
├── Attention Mechanism
├── Classification Head
└── Output: Binary prediction + attention weights
```

**Key Features:**
- Pre-trained on 10M molecular SMILES
- Attention-based interpretability
- Direct SMILES processing
- Transfer learning capabilities

#### 2. Graph Convolutional Network
```
Input: Molecular graphs
├── Node Features (atomic properties)
├── Edge Features (bond properties)
├── Graph Convolution Layers
├── Graph Pooling
├── Dense Layers
└── Output: Binary prediction + node attributions
```

**Key Features:**
- Direct molecular graph representation
- Node and edge attribution
- Chemical bond awareness
- Scalable to large molecules

#### 3. Circular Fingerprints (ECFP)
```
Input: SMILES strings
├── Molecular parsing (RDKit)
├── Circular fingerprint generation
├── Feature vector (1024/2048 bits)
├── Classical ML models
└── Output: Binary prediction + feature importance
```

**Key Features:**
- Fast computation
- Interpretable substructures
- Well-established methodology
- Good baseline performance

#### 4. RDKit Descriptors
```
Input: SMILES strings
├── Molecular descriptor calculation (200+ features)
├── Feature selection
├── Normalization/Scaling
├── Deep neural network
└── Output: Binary prediction + SHAP values
```

**Key Features:**
- Rich molecular representation
- Extensive feature set
- Traditional interpretability
- Chemical intuition

## Data Processing Pipeline

### 1. Data Acquisition
```python
# ChEMBL22 data extraction
source = "ChEMBL22"  # Human Acetylcholinesterase
target_id = "CHEMBL220"
activity_threshold = "IC50 <= 1000 nM"
```

### 2. Molecular Standardization
```python
# SMILES standardization pipeline
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Remove salts
    mol = rdMolStandardize.StandardizeSmiles(mol)
    # Canonical SMILES
    return Chem.MolToSmiles(mol, canonical=True)
```

### 3. Quality Filters
- **Molecular Weight**: ≤ 800 Daltons
- **Valid SMILES**: RDKit parseable
- **Drug-like Properties**: Lipinski's Rule of Five
- **Activity Data**: Clear IC50 values

### 4. Train/Validation/Test Split
- **Training**: 70% (stratified by activity class)
- **Validation**: 15% (hyperparameter tuning)
- **Test**: 15% (final evaluation)

## Model Training Procedures

### ChemBERTa Fine-tuning
```python
# Training configuration
config = {
    "model_name": "DeepChem/ChemBERTa-10M-MLM",
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01
}
```

### Graph CNN Training
```python
# GCN configuration
config = {
    "node_features": 75,  # Atomic features
    "edge_features": 12,  # Bond features
    "hidden_dims": [128, 128, 64],
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 64
}
```

### Hyperparameter Optimization
```python
# Optuna optimization
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    # ... model training and validation
    return validation_accuracy
```

## Interpretability Methods

### 1. Attention Visualization (ChemBERTa)
```python
# Extract attention weights
attention_weights = model.get_attention_weights(input_ids)
# Visualize on molecular structure
visualize_attention(smiles, attention_weights, tokens)
```

### 2. SHAP Analysis
```python
# SHAP explainer for any model
explainer = shap.Explainer(model.predict, X_background)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
```

### 3. Molecular Substructure Highlighting
```python
# Highlight important substructures
def highlight_substructures(mol, importances):
    highlight_atoms = []
    for atom_idx, importance in enumerate(importances):
        if importance > threshold:
            highlight_atoms.append(atom_idx)
    return Draw.MolToImage(mol, highlightAtoms=highlight_atoms)
```

### 4. Feature Importance Analysis
```python
# Calculate feature importance
importances = model.feature_importances_
feature_names = descriptor_calculator.get_feature_names()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
```

## Performance Evaluation

### Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **AUC-PR**: Area under the precision-recall curve

### Cross-Validation
```python
# Stratified k-fold cross-validation
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
```

### Statistical Significance
```python
# McNemar's test for model comparison
from statsmodels.stats.contingency_tables import mcnemar
# Compare predictions between models
table = [[n00, n01], [n10, n11]]
result = mcnemar(table, exact=True)
```

## Computational Requirements

### Memory Usage
- **ChemBERTa**: ~4GB GPU memory for batch size 32
- **Graph CNN**: ~2GB GPU memory for batch size 64
- **Traditional ML**: <1GB RAM for full dataset

### Training Time
- **ChemBERTa**: 2-4 hours on V100 GPU
- **Graph CNN**: 1-2 hours on V100 GPU
- **Traditional ML**: 10-30 minutes on CPU

### Inference Speed
- **ChemBERTa**: ~100 molecules/second
- **Graph CNN**: ~500 molecules/second
- **Traditional ML**: ~5000 molecules/second

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Use gradient accumulation
accumulation_steps = 2
```

#### 2. RDKit Parsing Errors
```python
# Handle invalid SMILES
def safe_smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except:
        return None
```

#### 3. Transformer Token Length Issues
```python
# Truncate long SMILES
max_length = 128
tokenized = tokenizer(smiles, 
                     max_length=max_length, 
                     truncation=True, 
                     padding=True)
```

### Performance Optimization

#### 1. Data Loading
```python
# Use efficient data loading
dataset = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True
)
```

#### 2. Model Optimization
```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

## API Reference

### Model Interface
```python
class MolecularClassifier:
    def __init__(self, model_type, **kwargs):
        """Initialize molecular classifier"""
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        
    def predict(self, X):
        """Make predictions"""
        
    def predict_proba(self, X):
        """Predict class probabilities"""
        
    def explain(self, X, method='shap'):
        """Generate explanations"""
```

### Visualization Functions
```python
def plot_attention(smiles, attention_weights):
    """Visualize attention on molecular structure"""
    
def plot_shap_values(shap_values, feature_names):
    """Plot SHAP value explanations"""
    
def highlight_molecule(mol, atom_scores):
    """Highlight important atoms in molecule"""
```

## Configuration Files

### Model Configuration
```yaml
# config.yaml
model:
  type: "chemberta"
  pretrained: "DeepChem/ChemBERTa-10M-MLM"
  max_length: 128
  num_classes: 2

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 10
  warmup_ratio: 0.1

data:
  train_file: "data/train.csv"
  val_file: "data/val.csv" 
  test_file: "data/test.csv"
  smiles_column: "smiles"
  target_column: "activity"
```

## Testing

### Unit Tests
```python
# test_models.py
def test_chemberta_prediction():
    model = ChemBertaClassifier()
    smiles = "CCO"  # Ethanol
    prediction = model.predict([smiles])
    assert len(prediction) == 1
    assert 0 <= prediction[0] <= 1
```

### Integration Tests
```python
# test_pipeline.py
def test_full_pipeline():
    # Test complete training pipeline
    pipeline = DrugDiscoveryPipeline()
    pipeline.load_data("test_data.csv")
    pipeline.preprocess()
    pipeline.train()
    metrics = pipeline.evaluate()
    assert metrics['auc'] > 0.7
```

## Deployment

### Model Serving
```python
# app.py - Flask API
from flask import Flask, request, jsonify

app = Flask(__name__)
model = load_model('best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.json['smiles']
    prediction = model.predict([smiles])
    return jsonify({'prediction': prediction[0]})
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.8-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["python", "app.py"]
```

## Version Control

### Git Workflow
```bash
# Feature development
git checkout -b feature/new-model
git add .
git commit -m "Add new interpretability method"
git push origin feature/new-model

# Create pull request for review
```

### Model Versioning
```python
# Use MLflow for model versioning
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_params(config)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
```
