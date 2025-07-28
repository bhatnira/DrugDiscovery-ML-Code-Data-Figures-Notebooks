# AI Activity Prediction - ChemML Suite

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

A comprehensive iOS-style multipage Streamlit application for chemical machine learning, featuring AutoML activity and potency prediction using TPOT, RDKit, and DeepChem. This application provides an intuitive interface for predicting molecular activity and potency using state-of-the-art machine learning models.

## 📖 Description

The AI Activity Prediction app is a powerful machine learning platform designed for chemoinformatics and drug discovery applications. It provides multiple prediction models and interfaces for various molecular prediction tasks.

### Core Capabilities:
- **Binary Activity Classification**: Predicts whether a chemical compound is active or inactive
- **Potency Regression**: Predicts the quantitative potency values of chemical compounds  
- **Graph Neural Networks**: Advanced molecular prediction using graph convolution models
- **Multiple Featurization Methods**: Support for various molecular descriptors including:
  - Circular Fingerprints (Morgan)
  - MACCS Keys
  - Mordred descriptors
  - RDKit descriptors
  - PubChem fingerprints
  - Mol2Vec embeddings

### Key Features:
- **AutoML Integration**: Automated machine learning using TPOT for optimal model selection
- **Model Interpretability**: LIME explanations for understanding predictions
- **Batch Processing**: Upload Excel files for bulk molecular predictions
- **Interactive Interface**: Modern iOS-style design with responsive layout
- **Real-time Visualization**: Interactive plots and molecular structure rendering

## 🚀 How to Run Locally

### Option 1: Direct Python Installation (Recommended for Development)

#### Prerequisites
- Python 3.10 or higher
- pip package manager

#### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/bhatnira/AI-Activity-Prediction.git
cd AI-Activity-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the main application
streamlit run main_app.py

# 6. Open your browser and navigate to:
# http://localhost:8501
```

#### Alternative: Run Individual Apps
You can also run specific components of the suite:
```bash
# Classification app
streamlit run app_classification.py

# Regression app  
streamlit run app_regression.py

# Graph classification app
streamlit run app_graph_classification.py

# Graph regression app
streamlit run app_graph_regression.py
```

### Option 2: Docker Installation (Recommended for Production)

#### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for easier management)

#### Run with Docker
```bash
# 1. Clone the repository
git clone https://github.com/bhatnira/AI-Activity-Prediction.git
cd AI-Activity-Prediction

# 2. Build and run with Docker Compose
docker-compose up -d

# 3. Or build and run manually
docker build -t chemml-suite .
docker run -p 8501:8501 chemml-suite

# 4. Access the application at:
# http://localhost:8501
```

### Option 3: Quick Setup Script
```bash
# Make the activation script executable and run
chmod +x activate_env.sh
./activate_env.sh
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom iOS-style CSS
- **Machine Learning**: TPOT (AutoML), scikit-learn
- **Chemistry**: RDKit, DeepChem
- **Graph Neural Networks**: DeepChem GraphConv models
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, Render.com

## 📱 Application Structure

### Main Interface (main_app.py)
The main application provides a unified iOS-style interface with navigation to different prediction modules:

1. **Home Dashboard**: Overview and feature highlights
2. **AutoML Activity Prediction**: Binary classification for molecular activity
3. **AutoML Potency Prediction**: Regression modeling for molecular potency
4. **Graph Classification**: Graph neural networks for activity prediction
5. **Graph Regression**: Graph neural networks for potency prediction

### Individual Applications
- `app_classification.py`: Traditional ML classification models
- `app_regression.py`: Traditional ML regression models
- `app_graph_classification.py`: Graph neural network classification
- `app_graph_regression.py`: Graph neural network regression

## 📊 Supported Data Formats

### Input Data
- **Training Data**: Excel files (.xlsx) with SMILES and target columns
- **Prediction Data**: Excel files (.xlsx) with SMILES column
- **Single Molecules**: Direct SMILES input via text box

### Output Data
- **Predictions**: CSV files with predictions and confidence scores
- **Model Performance**: Downloadable metrics and visualizations
- **LIME Explanations**: HTML files for model interpretability

## 🧪 Example Usage

### Sample SMILES Strings
- **Ethanol**: `CCO`
- **Aspirin**: `CC(=O)OC1=CC=CC=C1C(=O)O`
- **Caffeine**: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
- **Benzene**: `C1=CC=CC=C1`

### Workflow
1. **Data Preparation**: Prepare Excel file with SMILES and target values
2. **Model Training**: Upload data and configure AutoML parameters
3. **Model Evaluation**: Review performance metrics and visualizations
4. **Predictions**: Use trained model for single or batch predictions
5. **Interpretation**: Download LIME explanations for insights

## 🎯 Model Features and Capabilities

### Featurization Options
- **Circular Fingerprints**: Morgan fingerprints with customizable radius
- **MACCS Keys**: 166-bit structural keys
- **Mordred Descriptors**: Comprehensive molecular descriptors
- **RDKit Descriptors**: Standard RDKit molecular properties
- **PubChem Fingerprints**: 881-bit structural fingerprints
- **Mol2Vec**: Molecular embeddings

### AutoML Configuration
- **TPOT Integration**: Automated pipeline optimization
- **Cross-Validation**: Configurable CV folds
- **Generations**: Customizable evolution generations
- **Population Size**: Adjustable population for genetic algorithm

### Performance Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Regression**: R², MAE, MSE, RMSE
- **Visualizations**: ROC curves, confusion matrices, prediction plots

## 🔧 Configuration and Deployment

### Docker Configuration
- `Dockerfile`: Container setup with Python 3.10
- `docker-compose.yml`: Service orchestration
- `requirements.txt`: Python dependencies

### Render.com Deployment
- `render.yaml`: Service configuration
- `Procfile`: Process definition
- `runtime.txt`: Python version specification

### Environment Setup
- `activate_env.sh`: Quick environment setup script
- `build.sh`: Build automation script
- `deploy.sh`: Deployment automation

## 🔍 Model Interpretability

Each prediction includes comprehensive interpretability features:
- **Confidence Scores**: Prediction reliability indicators
- **LIME Explanations**: Local interpretable model explanations
- **Feature Importance**: Analysis of key molecular features
- **Performance Metrics**: Detailed model evaluation

## 🛡️ Security and Privacy

- **No Data Persistence**: All data processing is temporary
- **Local Processing**: Models trained and executed locally
- **No External Dependencies**: No external API calls for sensitive data
- **Secure File Handling**: Temporary file processing with cleanup

## 🤝 Contributing

We welcome contributions to improve the ChemML Suite:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support and Troubleshooting

### Common Issues
1. **Python Version**: Ensure Python 3.10+ is installed
2. **Dependencies**: Run `pip install -r requirements.txt` if modules are missing
3. **Port Conflicts**: Use `streamlit run main_app.py --server.port 8502` for alternative port
4. **Memory Issues**: Reduce TPOT generations for large datasets

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check inline documentation in the app
- **Error Messages**: Include full error traces when reporting issues

## 🔄 Updates and Maintenance

The application automatically updates when changes are pushed to the connected GitHub repository. For local development:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart the application
streamlit run main_app.py
```

---

**Built with ❤️ using Streamlit, RDKit, DeepChem, and TPOT**

For more detailed information about specific features, check the `TIME_ESTIMATION_FEATURES.md` and `DEPLOYMENT.md` files in the repository.
