# API Reference - AChE Inhibitor Prediction Suite

## Overview

This API reference provides programmatic access to the core prediction functions used in both applications. You can integrate these functions into your own Python scripts for automated processing and custom workflows.

## Core Modules

### 1. ChemBERTa Prediction Module

#### `chemberta_predictor.py`

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

class ChemBERTaPredictor:
    """
    ChemBERTa-based molecular activity prediction
    
    Attributes:
        model_path (str): Path to fine-tuned ChemBERTa model
        tokenizer_path (str): Path to tokenizer
        device (str): 'cuda' or 'cpu'
    """
    
    def __init__(self, model_path: str, tokenizer_path: str = None):
        """
        Initialize ChemBERTa predictor
        
        Args:
            model_path: Path to model directory or checkpoint
            tokenizer_path: Path to tokenizer (optional)
        """
        pass
    
    def predict_single(self, smiles: str) -> dict:
        """
        Predict activity for a single SMILES string
        
        Args:
            smiles (str): SMILES string of the molecule
            
        Returns:
            dict: {
                'prediction': int,  # 0 or 1
                'probability': list,  # [prob_inactive, prob_active]
                'confidence': float,  # max probability
                'attention_weights': list  # attention visualization data
            }
        """
        pass
    
    def predict_batch(self, smiles_list: list) -> list:
        """
        Predict activity for multiple SMILES
        
        Args:
            smiles_list (list): List of SMILES strings
            
        Returns:
            list: List of prediction dictionaries
        """
        pass
    
    def get_attention_weights(self, smiles: str) -> dict:
        """
        Extract attention weights for visualization
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: Attention weight data for visualization
        """
        pass
```

#### Example Usage
```python
# Initialize predictor
predictor = ChemBERTaPredictor('models/chemberta_ache/')

# Single prediction
result = predictor.predict_single('CC(=O)OC1=CC=CC=C1C(=O)O')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O']
results = predictor.predict_batch(smiles_list)
```

### 2. RDKit Descriptor Module

#### `rdkit_predictor.py`

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
import joblib

class RDKitPredictor:
    """
    RDKit molecular descriptor-based prediction
    
    Attributes:
        model_path (str): Path to trained scikit-learn model
        scaler_path (str): Path to feature scaler
        descriptor_list (list): List of descriptor names
    """
    
    def __init__(self, model_path: str, scaler_path: str = None):
        """
        Initialize RDKit predictor
        
        Args:
            model_path: Path to trained model (pickle/joblib)
            scaler_path: Path to feature scaler
        """
        pass
    
    def calculate_descriptors(self, smiles: str) -> dict:
        """
        Calculate molecular descriptors for a SMILES string
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: Dictionary of descriptor names and values
        """
        pass
    
    def predict_single(self, smiles: str) -> dict:
        """
        Predict activity using RDKit descriptors
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: {
                'prediction': int,
                'probability': list,
                'descriptors': dict,
                'feature_importance': dict
            }
        """
        pass
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance from the trained model
        
        Returns:
            dict: Feature names and importance scores
        """
        pass
    
    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate SMILES string using RDKit
        
        Args:
            smiles (str): SMILES string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        pass
```

#### Example Usage
```python
# Initialize predictor
predictor = RDKitPredictor('models/rdkit_model.pkl', 'models/scaler.pkl')

# Calculate descriptors
descriptors = predictor.calculate_descriptors('CCO')
print(f"Molecular Weight: {descriptors['MolWt']:.2f}")

# Make prediction
result = predictor.predict_single('CC(=O)OC1=CC=CC=C1C(=O)O')
print(f"Prediction: {result['prediction']}")

# Get feature importance
importance = predictor.get_feature_importance()
top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
```

### 3. Circular Fingerprint Module

#### `circular_fp_predictor.py`

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

class CircularFPPredictor:
    """
    Circular fingerprint (Morgan/ECFP) based prediction
    
    Attributes:
        model_path (str): Path to trained model
        radius (int): Fingerprint radius (default: 2)
        n_bits (int): Number of bits (default: 2048)
    """
    
    def __init__(self, model_path: str, radius: int = 2, n_bits: int = 2048):
        """
        Initialize circular fingerprint predictor
        
        Args:
            model_path: Path to trained model
            radius: Fingerprint radius
            n_bits: Number of bits in fingerprint
        """
        pass
    
    def calculate_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Calculate Morgan fingerprint for a molecule
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            np.ndarray: Binary fingerprint vector
        """
        pass
    
    def predict_single(self, smiles: str) -> dict:
        """
        Predict activity using circular fingerprints
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: {
                'prediction': int,
                'probability': list,
                'fingerprint': np.ndarray,
                'similarity_map': dict
            }
        """
        pass
    
    def calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate Tanimoto similarity between two molecules
        
        Args:
            smiles1 (str): First SMILES string
            smiles2 (str): Second SMILES string
            
        Returns:
            float: Tanimoto similarity score (0-1)
        """
        pass
    
    def get_substructure_highlights(self, smiles: str) -> dict:
        """
        Get substructures contributing to prediction
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: Substructure importance data
        """
        pass
```

### 4. Graph Neural Network Module

#### `graph_nn_predictor.py`

```python
import deepchem as dc
import numpy as np

class GraphNNPredictor:
    """
    Graph Neural Network prediction using DeepChem
    
    Attributes:
        model_path (str): Path to trained GraphConv model
        featurizer (dc.feat.ConvMolFeaturizer): Molecular featurizer
    """
    
    def __init__(self, model_path: str):
        """
        Initialize Graph NN predictor
        
        Args:
            model_path: Path to trained DeepChem model
        """
        pass
    
    def featurize_molecule(self, smiles: str) -> dc.data.Dataset:
        """
        Convert SMILES to graph features
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dc.data.Dataset: Featurized molecule dataset
        """
        pass
    
    def predict_single(self, smiles: str) -> dict:
        """
        Predict activity using graph neural network
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: {
                'prediction': int,
                'probability': list,
                'graph_features': dict,
                'node_importance': list
            }
        """
        pass
    
    def get_node_importance(self, smiles: str) -> dict:
        """
        Get importance scores for graph nodes (atoms)
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: Node importance data for visualization
        """
        pass
```

### 5. AutoML TPOT Module

#### `automl_predictor.py`

```python
from tpot import TPOTClassifier, TPOTRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

class AutoMLPredictor:
    """
    AutoML prediction using TPOT
    
    Attributes:
        model_type (str): 'classification' or 'regression'
        tpot_config (dict): TPOT configuration parameters
    """
    
    def __init__(self, model_type: str = 'classification', **tpot_kwargs):
        """
        Initialize AutoML predictor
        
        Args:
            model_type: Type of prediction task
            **tpot_kwargs: TPOT configuration parameters
        """
        pass
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train AutoML model using TPOT
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            
        Returns:
            dict: {
                'model': fitted_pipeline,
                'score': float,
                'pipeline_string': str
            }
        """
        pass
    
    def predict_single(self, features: dict) -> dict:
        """
        Make prediction for single sample
        
        Args:
            features (dict): Feature dictionary
            
        Returns:
            dict: Prediction results
        """
        pass
    
    def export_pipeline(self, filepath: str) -> None:
        """
        Export optimized pipeline to Python file
        
        Args:
            filepath: Path to save pipeline code
        """
        pass
```

## Utility Functions

### Molecular Processing

```python
def validate_smiles_batch(smiles_list: list) -> dict:
    """
    Validate a batch of SMILES strings
    
    Args:
        smiles_list (list): List of SMILES strings
        
    Returns:
        dict: {
            'valid': list,    # Valid SMILES
            'invalid': list,  # Invalid SMILES with errors
            'statistics': dict
        }
    """
    pass

def standardize_smiles(smiles: str) -> str:
    """
    Standardize SMILES string (canonical form)
    
    Args:
        smiles (str): Input SMILES
        
    Returns:
        str: Canonical SMILES
    """
    pass

def calculate_molecular_properties(smiles: str) -> dict:
    """
    Calculate basic molecular properties
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        dict: {
            'mol_weight': float,
            'logp': float,
            'hbd': int,  # H-bond donors
            'hba': int,  # H-bond acceptors
            'tpsa': float,  # Topological polar surface area
            'rotatable_bonds': int
        }
    """
    pass
```

### Data Processing

```python
def load_dataset(filepath: str, smiles_col: str = 'SMILES', 
                target_col: str = 'Activity') -> pd.DataFrame:
    """
    Load and validate dataset for prediction
    
    Args:
        filepath: Path to Excel/CSV file
        smiles_col: Name of SMILES column
        target_col: Name of target column
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    pass

def featurize_dataset(df: pd.DataFrame, method: str = 'morgan') -> np.ndarray:
    """
    Convert SMILES dataset to feature matrix
    
    Args:
        df: DataFrame with SMILES column
        method: Featurization method ('morgan', 'rdkit', 'maccs')
        
    Returns:
        np.ndarray: Feature matrix
    """
    pass

def split_dataset(X: np.ndarray, y: np.ndarray, 
                 test_size: float = 0.2) -> tuple:
    """
    Split dataset for training and testing
    
    Args:
        X: Feature matrix
        y: Target values
        test_size: Fraction for test set
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    pass
```

### Model Evaluation

```python
def evaluate_classification_model(y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                y_proba: np.ndarray = None) -> dict:
    """
    Comprehensive evaluation of classification model
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        
    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1_score': float,
            'roc_auc': float,
            'confusion_matrix': np.ndarray
        }
    """
    pass

def evaluate_regression_model(y_true: np.ndarray, 
                            y_pred: np.ndarray) -> dict:
    """
    Comprehensive evaluation of regression model
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: {
            'r2_score': float,
            'mae': float,
            'mse': float,
            'rmse': float
        }
    """
    pass
```

## Configuration Examples

### ChemBERTa Configuration
```python
chemberta_config = {
    'model_name': 'DeepChem/ChemBERTa-10M-MLM',
    'num_labels': 2,
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 100
}
```

### TPOT Configuration
```python
tpot_config = {
    'generations': 10,
    'population_size': 50,
    'cv': 5,
    'scoring': 'roc_auc',
    'verbosity': 2,
    'random_state': 42,
    'n_jobs': -1,
    'max_time_mins': 120
}
```

### Graph Model Configuration
```python
graph_config = {
    'graph_conv_layers': [64, 64],
    'dense_layer_size': 128,
    'dropout': 0.5,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}
```

## Integration Examples

### Complete Prediction Pipeline
```python
from ache_prediction import ChemBERTaPredictor, RDKitPredictor
import pandas as pd

def comprehensive_prediction(smiles_list: list) -> pd.DataFrame:
    """
    Run multiple models on a list of SMILES
    """
    # Initialize predictors
    chemberta = ChemBERTaPredictor('models/chemberta/')
    rdkit = RDKitPredictor('models/rdkit_model.pkl')
    
    results = []
    for smiles in smiles_list:
        # ChemBERTa prediction
        cb_result = chemberta.predict_single(smiles)
        
        # RDKit prediction  
        rdkit_result = rdkit.predict_single(smiles)
        
        # Combine results
        combined = {
            'SMILES': smiles,
            'ChemBERTa_Prediction': cb_result['prediction'],
            'ChemBERTa_Confidence': cb_result['confidence'],
            'RDKit_Prediction': rdkit_result['prediction'],
            'RDKit_Confidence': max(rdkit_result['probability']),
            'Consensus': 1 if (cb_result['prediction'] + rdkit_result['prediction']) >= 1 else 0
        }
        results.append(combined)
    
    return pd.DataFrame(results)

# Usage
smiles_data = ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
predictions = comprehensive_prediction(smiles_data)
print(predictions)
```

### Batch Processing with Error Handling
```python
def robust_batch_prediction(input_file: str, output_file: str) -> dict:
    """
    Process large datasets with error handling
    """
    try:
        # Load data
        df = pd.read_excel(input_file)
        
        # Validate SMILES
        validation = validate_smiles_batch(df['SMILES'].tolist())
        
        # Process valid SMILES
        predictor = ChemBERTaPredictor('models/chemberta/')
        results = []
        
        for smiles in validation['valid']:
            try:
                result = predictor.predict_single(smiles)
                results.append({
                    'SMILES': smiles,
                    'Prediction': result['prediction'],
                    'Confidence': result['confidence'],
                    'Status': 'Success'
                })
            except Exception as e:
                results.append({
                    'SMILES': smiles,
                    'Prediction': None,
                    'Confidence': None,
                    'Status': f'Error: {str(e)}'
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_file, index=False)
        
        return {
            'total_processed': len(results),
            'successful': sum(1 for r in results if r['Status'] == 'Success'),
            'failed': sum(1 for r in results if 'Error' in r['Status']),
            'output_file': output_file
        }
        
    except Exception as e:
        return {'error': str(e)}
```

## Error Handling

### Common Exceptions
```python
class AChEPredictionError(Exception):
    """Base exception for AChE prediction errors"""
    pass

class InvalidSMILESError(AChEPredictionError):
    """Raised when SMILES string is invalid"""
    pass

class ModelLoadError(AChEPredictionError):
    """Raised when model files cannot be loaded"""
    pass

class PredictionError(AChEPredictionError):
    """Raised when prediction fails"""
    pass
```

### Usage with Error Handling
```python
try:
    predictor = ChemBERTaPredictor('models/chemberta/')
    result = predictor.predict_single('invalid_smiles')
except InvalidSMILESError as e:
    print(f"Invalid SMILES: {e}")
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
except PredictionError as e:
    print(f"Prediction failed: {e}")
```

## Performance Considerations

### Memory Management
```python
import gc
import torch

def clear_gpu_memory():
    """Clear GPU memory between predictions"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def batch_predict_with_memory_management(predictor, smiles_list, batch_size=32):
    """Predict in batches to manage memory"""
    results = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        batch_results = predictor.predict_batch(batch)
        results.extend(batch_results)
        
        # Clear memory after each batch
        clear_gpu_memory()
    
    return results
```

### Parallel Processing
```python
from multiprocessing import Pool
from functools import partial

def parallel_prediction(smiles_list, model_path, n_processes=4):
    """Run predictions in parallel"""
    predictor_func = partial(single_prediction_worker, model_path=model_path)
    
    with Pool(n_processes) as pool:
        results = pool.map(predictor_func, smiles_list)
    
    return results

def single_prediction_worker(smiles, model_path):
    """Worker function for parallel processing"""
    predictor = RDKitPredictor(model_path)
    return predictor.predict_single(smiles)
```

---

## Contact and Support

For API-specific questions or integration support:
- **Email**: bhatnira@isu.edu
- **GitHub Issues**: Technical problems and feature requests
- **Documentation**: Refer to individual model documentation for detailed parameters

This API reference provides the foundation for integrating AChE prediction capabilities into your own applications and workflows.
