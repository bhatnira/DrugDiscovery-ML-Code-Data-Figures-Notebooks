# Compound Acitivity Prediction & Drug Discovery Suite 🧬

## Overview

A comprehensive machine learning ecosystem for **Compound Acitivity prediction** and **drug discovery research**. This repository contains multiple interconnected projects implementing state-of-the-art AI/ML approaches for molecular activity prediction, featuring both production-ready applications and research notebooks with explainable AI techniques.

## 🎯 Project Portfolio

### 🚀 **Production Applications**

#### 1. **AChE Activity Prediction Suite** (`Application1-AChE-Activity-Pred/`)
**Primary deployment-ready application** with iOS-style interface featuring multiple ML models:
- **ChemBERTa Transformer**: Attention-based molecular property prediction
- **RDKit Descriptors**: Traditional molecular descriptors with ML pipelines  
- **Circular Fingerprints**: Morgan fingerprints with LIME interpretability
- **Graph Neural Networks**: Deep learning on molecular graphs

#### 2. **AI Activity Prediction - ChemML Suite** (`Application2-AI-Activity-Prediction/`)
**AutoML-powered platform** with automated machine learning capabilities:
- **TPOT AutoML**: Automated pipeline optimization for classification/regression
- **Multi-featurization**: Support for 6+ molecular representation methods
- **Batch Processing**: Excel file upload for bulk predictions
- **Model Interpretability**: LIME explanations and performance analytics

### 📚 **Research & Development**

#### 3. **Explainable AI Best Models** (`Notebooks-ExplainableAI-BestModels-AChEI-DrugDiscovery/`)
**Research notebooks** focusing on model interpretability and explainable AI:
- Fine-tuned ChemBERTa with attention visualization
- Graph Convolutional Networks with node attribution
- Circular fingerprint analysis with compound generation
- Deep neural networks with SHAP explanations

#### 4. **Classification Models Research** (`Notebooks-ML-AChEI-ClassificationModels-DrugDiscovery/`)
**Comprehensive model comparison** notebooks for binary classification tasks

#### 5. **Regression Models Research** (`Notebooks-ML-Regression-AChEI-DrugDiscovery/`)
**Quantitative prediction** models for IC50 and potency regression

---

## 🚀 Quick Start - Running the Applications

### **Application 1: AChE Activity Prediction Suite**

**Best for**: Production use, comprehensive model comparison, real-time predictions

```bash
# 1. Navigate to Application 1
cd Application1-AChE-Activity-Pred/

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch application
streamlit run main_app.py

# 5. Open browser
# → http://localhost:8501
```

**Features Available:**
- 🧬 **ChemBERTa**: State-of-the-art transformer predictions
- 💊 **RDKit**: Traditional descriptor-based models  
- 🔄 **Circular FP**: Morgan fingerprint analysis
- 📊 **Graph NN**: Neural networks on molecular graphs

---

### **Application 2: AI Activity Prediction - ChemML Suite**

**Best for**: AutoML workflows, batch processing, automated model selection

```bash
# 1. Navigate to Application 2
cd Application2-AI-Activity-Prediction/

# 2. Create virtual environment  
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch main application
streamlit run main_app.py

# 5. Open browser
# → http://localhost:8501
```

**Features Available:**
- 🤖 **AutoML Classification**: Automated binary activity prediction
- 📈 **AutoML Regression**: Automated potency prediction
- 🕸️ **Graph Models**: Graph neural networks for both tasks
- 📊 **Batch Processing**: Excel file upload and processing

---

## 🔧 System Requirements

### **Minimum Requirements**
- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB recommended for graph models)
- **Storage**: 5GB free space
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### **Key Dependencies**
- **Core**: `streamlit`, `pandas`, `numpy`, `scikit-learn`
- **Chemistry**: `rdkit`, `deepchem`, `mordred`
- **ML/DL**: `torch`, `transformers`, `tpot`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`

---


## 🧪 Usage Examples

### **Single Molecule Prediction**
```python
# Example SMILES inputs
ethanol = "CCO"
aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
```

### **Batch Processing**
- Upload Excel files with SMILES column
- Automatic feature extraction and prediction
- Download results with confidence scores

### **Model Interpretation**
- LIME explanations for prediction reasoning
- Attention weight visualization (ChemBERTa)
- Feature importance ranking (RDKit)
- Molecular substructure highlighting

---

## 🔬 Research Applications

### **Drug Discovery Pipeline**
1. **Virtual Screening**: Filter large compound libraries
2. **Lead Optimization**: Improve molecular properties
3. **ADMET Prediction**: Assess drug-like properties
4. **Mechanism Understanding**: Interpret model decisions

### **Academic Research**
- Comparative model analysis
- Explainable AI methodology development
- Molecular representation learning
- Transfer learning applications

---

## 📁 Repository Structure

```
AChE-Inhibitor-Prediction-Suite/
├── Application1-AChE-Activity-Pred/          # 🚀 Main Production App
│   ├── main_app.py                           # Streamlit interface
│   ├── app_chemberta.py                      # ChemBERTa module
│   ├── app_rdkit.py                          # RDKit module
│   ├── app_circular.py                       # Circular FP module
│   └── requirements.txt                      # Dependencies
│
├── Application2-AI-Activity-Prediction/      # 🤖 AutoML Suite
│   ├── main_app.py                           # Main interface
│   ├── app_classification.py                 # Classification models
│   ├── app_regression.py                     # Regression models
│   └── requirements.txt                      # Dependencies
│
├── Notebooks-ExplainableAI-BestModels/       # 📚 Research Notebooks
│   ├── FineTunedChemberta*.ipynb             # Transformer models
│   ├── ModelInterpretation*.ipynb            # Interpretability analysis
│   └── docs/                                 # Documentation
│
├── Notebooks-ML-*ClassificationModels/       # 🔬 Classification Research
├── Notebooks-ML-Regression*/                 # 📈 Regression Research
├── Datasets/                                  # 📊 Training Data
└── Visualization/                             # 📈 Analysis Results
```

---

## ⚡ Performance Tips

### **For Large Datasets**
- Use **Application 2** for batch processing
- Enable GPU support for ChemBERTa and Graph models
- Consider data preprocessing for very large files

### **For Real-time Predictions**
- Use **Application 1** with RDKit descriptors for fastest results
- ChemBERTa for best accuracy on novel compounds
- Graph models for complex molecular interactions

### **For Research**
- Explore research notebooks for detailed analysis
- Compare multiple models using the applications
- Use LIME/SHAP explanations for publication-quality interpretability

---

## 🤝 Contributing & Support

### **Authors**
- **Nirajan Bhattarai** 
- **Marvin Schulte** 
### **Contact**
- **Email**: bhatnira@isu.edu
- **GitHub**: [@bhatnira](https://github.com/bhatnira)

### **Citation**
If you use this work in your research, please cite:
```bibtex
@misc{bhattarai2025ache,
  title={AChE Inhibitor Prediction and Drug Discovery Suite},
  author={Bhattarai, Nirajan and Schulte, Marvin},
  year={2025},
  publisher={GitHub},
  url={https://github.com/bhatnira/AChE-Activity-Pred-1}
}
```

---

## 📄 License

This project is licensed under the MIT License - see individual project LICENSE files for details.

---

**🚀 Ready to start predicting? Choose your application and follow the setup instructions above!**

*Powered by advanced machine learning, molecular informatics, and explainable AI* 🧬✨
