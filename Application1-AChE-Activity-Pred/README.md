# AChE Activity Prediction Suite 🧬

A comprehensive machine learning application for predicting Acetylcholinesterase (AChE) inhibitory activity using multiple molecular representation methods and state-of-the-art machine learning models.

## 🌟 Features

This application prov- Check the [Issues](https://github.com/bhatnira/AChE-Activity-Pred-1/issues) section
- Review error logs in the terminal
- Ensure all dependencies are properly installed

## 📄 Licenseprediction methods through an intuitive Streamlit interface:

### Available Models
- **RDKit Descriptors**: Traditional molecular descriptors with optimized ML pipelines
- **ChemBERTa**: Transformer-based molecular property prediction
- **Circular Fingerprints**: Extended-connectivity fingerprints with LIME interpretability
- **Graph Neural Networks**: Deep learning on molecular graphs

### Key Capabilities
- 🎯 Binary classification (Active/Inactive) and regression prediction
- 🔬 Multiple molecular input methods (SMILES, molecular drawing)
- 📊 Interactive visualizations and model interpretability
- 🧪 Molecular property calculations and analysis
- 📈 Real-time prediction with confidence scores

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git
   cd AChE-Activity-Pred-1
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the main application**
   ```bash
   streamlit run main_app.py
   ```

2. **Access the application**
   - Open your web browser and navigate to `http://localhost:8501`
   - The application will automatically open in your default browser

3. **Navigate through different models**
   - Use the main launcher interface to select different prediction methods
   - Each model provides unique features and visualization capabilities

## 📱 Application Structure

### Main Interface (`main_app.py`)
The main launcher provides access to all prediction models through an iOS-style interface with the following sections:

- **🧮 RDKit Descriptors**: Classical molecular descriptors with ML pipelines
- **🤖 ChemBERTa**: Advanced transformer models for molecular property prediction
- **🔄 Circular Fingerprints**: ECFP with interpretability features
- **📊 Graph Neural Networks**: Deep learning on molecular graphs

### Individual Applications

#### RDKit Descriptors (`app_rdkit.py`)
- Traditional molecular descriptors calculation
- Multiple ML algorithms (Random Forest, XGBoost, SVM, etc.)
- TPOT automated machine learning optimization
- Comprehensive molecular property analysis

#### ChemBERTa (`app_chemberta.py`)
- Transformer-based molecular representation
- Pre-trained ChemBERTa models
- Advanced neural network architectures
- Attention visualization for model interpretability

#### Circular Fingerprints (`app_circular.py`)
- Extended-Connectivity Fingerprints (ECFP)
- LIME-based model explanations
- Interactive molecular visualization
- Feature importance analysis

#### Graph Neural Networks (`app_graphC.py`, `app_graphR.py`)
- Graph convolutional networks
- Molecular graph representation learning
- Deep learning for both classification and regression
- Advanced graph-based molecular analysis

## 🔧 Configuration

### Environment Variables

The application automatically handles various environment configurations:

```bash
# For headless environments
export DISPLAY=:99
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
```

### Model Files

The application includes pre-trained models:
- `bestPipeline_tpot_rdkit_classification.pkl`
- `bestPipeline_tpot_circularfingerprint_classification.pkl`
- `best_model_aggregrate_circular.pkl`
- Graph model checkpoints in respective directories

## 📊 Usage Examples

### 1. Predicting with SMILES Input

```python
# Example SMILES for testing
smiles_examples = [
    "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(c5ccccc5)CC4)ccc3)nc12",
    "Cc1ccc(C)c(S(=O)(=O)N2CCN(c3ccccc3)CC2)c1",
    "O=C(CSc1nnc(-c2cccnc2)n1Cc1ccccc1)N1CCN(c2ccccc2)CC1"
]
```

### 2. Using Molecular Drawing Interface

1. Navigate to any of the prediction apps
2. Select "Draw Structure" option
3. Use the integrated molecular editor
4. Submit for prediction

### 3. Batch Predictions

1. Upload CSV file with SMILES column
2. Select prediction model
3. Download results with predictions and confidence scores

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t ache-prediction .

# Run the container
docker run -p 8501:8501 ache-prediction
```

### Using Docker Compose

```bash
# Start the application
docker-compose up

# Stop the application
docker-compose down
```

## 🔬 Model Performance

### RDKit Descriptors
- **Accuracy**: ~85-90% on test set
- **Features**: 200+ molecular descriptors
- **Algorithm**: Optimized ensemble methods

### ChemBERTa
- **Accuracy**: ~88-92% on test set
- **Architecture**: Transformer-based
- **Pre-training**: Large chemical databases

### Circular Fingerprints
- **Accuracy**: ~86-91% on test set
- **Features**: ECFP with radius 2-3
- **Interpretability**: LIME explanations

### Graph Neural Networks
- **Accuracy**: ~87-93% on test set
- **Architecture**: Graph convolutional layers
- **Representation**: Molecular graphs

## 🛠️ Development

### Project Structure

```
├── main_app.py              # Main launcher application
├── app_rdkit.py             # RDKit descriptors app
├── app_chemberta.py         # ChemBERTa transformer app
├── app_circular.py          # Circular fingerprints app
├── app_graphC.py            # Graph NN classification
├── app_graphR.py            # Graph NN regression
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── style.css              # Custom CSS styling
└── models/                # Pre-trained model files
```

### Adding New Models

1. Create new app file (e.g., `app_newmodel.py`)
2. Follow the existing app structure
3. Add integration to `main_app.py`
4. Update requirements if needed

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📋 Requirements

### Core Dependencies

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.25.0
scikit-learn>=1.4.1
rdkit>=2022.9.0
torch>=2.0.0
tensorflow>=2.13.0
transformers>=4.31.0
simpletransformers>=0.70.0
```

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended for graph models)
- **CPU**: Multi-core processor recommended
- **GPU**: Optional, but recommended for ChemBERTa and Graph NN models
- **Storage**: ~2GB for all models and dependencies

## 🚨 Troubleshooting

### Common Issues

1. **RDKit Installation Issues**
   ```bash
   # Try conda installation
   conda install -c conda-forge rdkit
   ```

2. **Memory Issues with Large Datasets**
   - Process data in smaller batches
   - Increase system memory allocation
   - Use Docker with memory limits

3. **CUDA/GPU Issues**
   ```bash
   # Check PyTorch installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Streamlit Port Issues**
   ```bash
   # Use different port
   streamlit run main_app.py --server.port 8502
   ```

### Getting Help

- Check the [Issues](https://github.com/bhatnira/AChE-Activity-Pred-1/issues) section
- Review error logs in the terminal
- Ensure all dependencies are properly installed

## � Authors

- **Nirajan Bhattarai** - [@bhatnira](https://github.com/bhatnira)
  - Project Lead & Primary Developer
  - Machine Learning Model Development
  - Application Architecture & Implementation

*For a complete list of contributors, see the [Contributors](https://github.com/bhatnira/AChE-Activity-Pred-1/contributors) page.*

## �📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- RDKit community for molecular informatics tools
- Hugging Face for transformer models
- Streamlit for the web application framework
- The cheminformatics and machine learning community

## 📞 Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation for additional resources

---

**Happy Predicting! 🧬🚀**