# User Guide - AChE Inhibitor Prediction Suite

## Table of Contents
1. [Getting Started](#getting-started)
2. [Application 1: AChE Activity Prediction Suite](#application-1-ache-activity-prediction-suite)
3. [Application 2: AI Activity Prediction - ChemML Suite](#application-2-ai-activity-prediction---chemml-suite)
4. [Model Selection Guide](#model-selection-guide)
5. [Input Formats](#input-formats)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### First Time Setup
1. **Complete Installation**: Follow the [Installation Guide](INSTALLATION_GUIDE.md)
2. **Verify Environment**: Run the verification script
3. **Choose Application**: Select based on your use case
4. **Prepare Data**: Format your molecular data appropriately

### Quick Start Workflow
```bash
# 1. Activate environment
conda activate ache-prediction

# 2. Navigate to desired application
cd Application1-AChE-Activity-Pred/  # OR Application2-AI-Activity-Prediction/

# 3. Launch application
streamlit run main_app.py

# 4. Open browser to http://localhost:8501
```

## Application 1: AChE Activity Prediction Suite

### Overview
**Best for**: Real-time predictions, model comparison, demonstration, educational use

### Features & Navigation

#### 🏠 **Home Dashboard**
- Overview of available models
- Feature comparison
- Quick navigation to prediction modules

#### 🧬 **ChemBERTa Transformer**
**Use Case**: Novel compounds, transformer-based predictions, attention analysis

**Input Methods**:
- **SMILES Text**: Direct input of SMILES strings
- **Molecular Drawing**: Interactive chemical structure editor
- **SDF Upload**: Structure data files
- **Excel Batch**: Multiple compounds in spreadsheet format

**Example SMILES**:
```
Ethanol: CCO
Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O
Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Donepezil: COc1cc2c(cc1OC)C(=O)c1ccccc1-2
```

**Outputs**:
- Binary classification (Active/Inactive)
- Confidence scores
- Attention weight visualization
- Model interpretation

#### 💊 **RDKit Descriptors**
**Use Case**: Traditional analysis, feature interpretation, fast predictions

**Features**:
- 200+ molecular descriptors
- Feature importance ranking
- Interactive visualizations
- Batch processing support

**Key Descriptors Calculated**:
- Molecular weight and LogP
- Topological indices
- Electronic properties
- 3D descriptors (when applicable)

#### 🔄 **Circular Fingerprints**
**Use Case**: Substructure analysis, similarity searching, Morgan fingerprints

**Parameters**:
- Radius: 2 (default), 1-3 range
- Bits: 2048 (default), 1024-4096 range
- Features: Presence/absence of substructures

**Applications**:
- Structural similarity analysis
- Substructure mapping
- Chemical space exploration

#### 📊 **Graph Neural Networks**
**Use Case**: Complex molecular interactions, state-of-the-art performance

**Model Types**:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Message Passing Neural Networks

**Features**:
- Node and edge feature learning
- Molecular graph visualization
- Advanced deep learning capabilities

### Navigation Tips
- Use **tabs** within each model for different input methods
- **Download results** using provided buttons
- **Compare models** by running the same compound through multiple modules

## Application 2: AI Activity Prediction - ChemML Suite

### Overview
**Best for**: AutoML workflows, batch processing, automated model selection, research

### Main Interface Navigation

#### 🏠 **Home Dashboard**
- Application overview
- Feature highlights
- Navigation to prediction modules

#### 🤖 **AutoML Activity Prediction**
**Use Case**: Binary classification with automated machine learning

**Workflow**:
1. **Data Upload**: Upload Excel file with SMILES and activity labels
2. **Feature Selection**: Choose molecular representation method:
   - Circular Fingerprints (Morgan)
   - MACCS Keys
   - Mordred Descriptors
   - RDKit Descriptors
   - PubChem Fingerprints
   - Mol2Vec Embeddings

3. **AutoML Configuration**:
   - Generations: 5-50 (default: 10)
   - Population Size: 20-100 (default: 50)
   - CV Folds: 3-10 (default: 5)
   - Scoring Metric: Accuracy, F1, ROC AUC

4. **Model Training**: TPOT automatically optimizes pipeline
5. **Results**: Download trained model and predictions

#### 📈 **AutoML Potency Prediction**
**Use Case**: Regression for IC50, Ki, or other quantitative values

**Similar workflow** to classification but with regression metrics:
- R² (coefficient of determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)

#### 🕸️ **Graph Classification/Regression**
**Use Case**: Graph neural networks with automated training

**Features**:
- DeepChem GraphConv models
- Automated hyperparameter tuning
- Graph-based molecular representations

### Data Preparation for Application 2

#### Excel File Format
```
Required columns for classification:
- SMILES: Chemical structure in SMILES format
- Activity: Binary labels (0/1 or Active/Inactive)

Required columns for regression:
- SMILES: Chemical structure in SMILES format
- Target: Numerical values (IC50, Ki, etc.)
```

#### Example Data Structure
| SMILES | Activity | IC50_nM |
|--------|----------|---------|
| CCO | 0 | 10000 |
| CC(=O)OC1=CC=CC=C1C(=O)O | 1 | 250 |
| CN1C=NC2=C1C(=O)N(C(=O)N2C)C | 0 | 5000 |

## Model Selection Guide

### When to Use Each Model

| Scenario | Recommended Model | Application | Reasoning |
|----------|------------------|-------------|-----------|
| **Novel compounds** | ChemBERTa | App 1 | Best generalization to unseen structures |
| **Large datasets** | AutoML | App 2 | Automated optimization, batch processing |
| **Fast predictions** | RDKit Descriptors | App 1 | Fastest computation, good interpretability |
| **Substructure analysis** | Circular Fingerprints | App 1 | Excellent for similarity and substructures |
| **Complex interactions** | Graph Neural Networks | App 1 or 2 | Captures spatial relationships |
| **Automated workflows** | TPOT AutoML | App 2 | Hands-off optimization |
| **Research/publication** | Multiple models | Both | Comprehensive comparison |

### Performance Expectations

| Model | Training Time | Prediction Time | Accuracy | Interpretability |
|-------|---------------|-----------------|----------|------------------|
| ChemBERTa | Medium | Fast | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| RDKit | Fast | Very Fast | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Circular FP | Fast | Fast | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Graph NN | Slow | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| AutoML | Very Slow | Fast | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Input Formats

### SMILES Strings
**Format**: Simplified Molecular Input Line Entry System
```
Valid examples:
- Simple: CCO (ethanol)
- Complex: CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)
- Aromatic: c1ccccc1 (benzene)
- Charged: [Na+].[Cl-] (salt)
```

**Common Issues**:
- Invalid characters
- Incomplete structures
- Stereochemistry ambiguity

### SDF Files
**Format**: Structure Data Format
- 3D coordinates supported
- Multiple molecules per file
- Metadata preservation

### Excel Files
**Requirements**:
- `.xlsx` format preferred
- Headers in first row
- SMILES column clearly labeled
- Target column for supervised learning

**Column naming conventions**:
- SMILES: "SMILES", "Smiles", "smiles"
- Activity: "Activity", "Label", "Class"
- Target: "IC50", "Ki", "Target", "Value"

## Interpreting Results

### Classification Results

#### Prediction Output
```
Compound: CC(=O)OC1=CC=CC=C1C(=O)O
Prediction: Active (Class 1)
Confidence: 0.85
Probability: [0.15, 0.85]  # [Inactive, Active]
```

#### Confidence Interpretation
- **> 0.9**: Very high confidence
- **0.7-0.9**: High confidence
- **0.5-0.7**: Moderate confidence
- **< 0.5**: Low confidence (classify as opposite)

### Regression Results

#### Prediction Output
```
Compound: Donepezil analog
Predicted IC50: 12.5 nM
95% Confidence Interval: [8.2, 18.9] nM
Standard Error: 2.1 nM
```

### Model Interpretability

#### LIME Explanations
- **Green highlights**: Substructures supporting the prediction
- **Red highlights**: Substructures opposing the prediction
- **Feature importance**: Numerical ranking of descriptor contributions

#### Attention Weights (ChemBERTa)
- **Darker colors**: Higher attention to specific atoms/bonds
- **Attention patterns**: Reveal which molecular regions drive predictions

#### Feature Importance (RDKit)
- **Bar charts**: Show relative importance of molecular descriptors
- **SHAP values**: Quantify individual feature contributions

## Best Practices

### Data Quality
1. **Clean SMILES**: Validate all structures before processing
2. **Balanced datasets**: Ensure reasonable class distribution
3. **Remove duplicates**: Check for identical structures
4. **Outlier detection**: Identify unusual structures or activities

### Model Selection
1. **Start simple**: Begin with RDKit descriptors for baseline
2. **Compare multiple**: Use both applications for comprehensive analysis
3. **Validate results**: Cross-check predictions across models
4. **Consider domain**: Match model to specific chemical space

### Performance Optimization
1. **Batch processing**: Use Application 2 for large datasets
2. **GPU acceleration**: Enable for ChemBERTa and Graph models
3. **Memory management**: Monitor RAM usage with large datasets
4. **Parallel processing**: Utilize multiple CPU cores when available

### Result Validation
1. **Cross-validation**: Use multiple train/test splits
2. **External validation**: Test on independent datasets
3. **Literature comparison**: Compare with published results
4. **Chemical intuition**: Validate predictions against known SAR

## Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check Python environment
python --version  # Should be 3.10+

# Verify Streamlit installation
streamlit --version

# Check for port conflicts
lsof -i :8501  # macOS/Linux
netstat -an | findstr :8501  # Windows
```

#### Invalid SMILES Error
```
Error: "Invalid SMILES string"
Solutions:
1. Validate SMILES using RDKit: Chem.MolFromSmiles(smiles)
2. Check for special characters
3. Remove salts or use canonical SMILES
4. Verify molecular structure is complete
```

#### Memory Issues
```
Error: "Out of memory"
Solutions:
1. Reduce batch size
2. Use CPU instead of GPU
3. Increase system RAM
4. Process data in smaller chunks
```

#### Model Loading Error
```
Error: "Model file not found"
Solutions:
1. Check file paths in configuration
2. Re-download model files
3. Verify model compatibility
4. Check disk space
```

### Performance Issues

#### Slow Predictions
1. **Check GPU utilization**: `nvidia-smi`
2. **Monitor CPU usage**: Task Manager/Activity Monitor
3. **Reduce model complexity**: Use simpler models for testing
4. **Optimize data preprocessing**: Cache preprocessed features

#### Inconsistent Results
1. **Set random seeds**: Ensure reproducibility
2. **Check data preprocessing**: Verify consistent feature extraction
3. **Model versioning**: Ensure same model versions across runs
4. **Validate input data**: Check for data corruption

### Getting Help

#### Documentation Resources
1. **Installation Guide**: Technical setup issues
2. **Individual README files**: Application-specific problems
3. **GitHub Issues**: Community support and bug reports
4. **Scientific literature**: Model-specific questions

#### Contact Support
- **Email**: bhatnira@isu.edu
- **GitHub**: [Issues page](https://github.com/bhatnira/AChE-Activity-Pred-1/issues)
- **Response time**: 24-48 hours for technical issues

---

## Quick Reference Commands

### Application Launch
```bash
# Application 1
cd Application1-AChE-Activity-Pred/
streamlit run main_app.py

# Application 2  
cd Application2-AI-Activity-Prediction/
streamlit run main_app.py

# Custom port
streamlit run main_app.py --server.port 8502
```

### Environment Management
```bash
# Activate environment
conda activate ache-prediction

# Update packages
pip install --upgrade -r requirements.txt

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

This guide should help you make the most of the AChE Inhibitor Prediction Suite! For additional questions, refer to the comprehensive documentation or contact support.
