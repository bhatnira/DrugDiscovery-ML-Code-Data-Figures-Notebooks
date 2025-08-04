# AChE Activity Prediction Suite

A comprehensive molecular prediction platform for Acetylcholinesterase (AChE) inhibition analysis using advanced machine learning and graph neural networks.

## 🚀 Features

### 🧬 **Multiple Prediction Models**
- **Graph Neural Networks**: DeepChem GraphConvModel for molecular graph analysis
- **Circular Fingerprints**: TPOT-optimized models with LIME explanations
- **RDKit Descriptors**: Traditional molecular descriptor-based predictions

### 🔬 **Analysis Capabilities**
- **Single Molecule Prediction**: Individual SMILES analysis with confidence scores
- **Batch Processing**: Excel/CSV and SDF file support for high-throughput analysis
- **LIME Explanations**: Interpretable AI with feature importance visualization
- **Similarity Maps**: Atomic contribution analysis for graph models
- **IC50 Predictions**: Quantitative inhibition concentration estimates

### 📊 **User-Friendly Interface**
- **Streamlit Web Application**: Interactive, responsive web interface
- **Progress Tracking**: Real-time processing status for batch operations
- **Export Functionality**: CSV downloads with complete prediction data
- **Molecular Visualization**: Built-in structure rendering and analysis

## 🛠️ Installation & Setup

### Prerequisites
- **Docker** (recommended) or Python 3.10+
- **Git** for repository cloning
- **4GB+ RAM** for optimal performance

### Option 1: Docker Setup (Recommended)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git
   cd AChE-Activity-Pred-1
   ```

2. **Build and Run with Docker Compose**
   ```bash
   docker-compose up -d --build
   ```

3. **Access the Application**
   - Open your web browser and navigate to: `http://localhost:10000`
   - The application will be available on ports 8501-8504 as well

4. **Stop the Application**
   ```bash
   docker-compose down
   ```

### Option 2: Local Python Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git
   cd AChE-Activity-Pred-1
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv ache_env
   source ache_env/bin/activate  # On Windows: ache_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run main_app.py --server.port 10000
   ```

5. **Access the Application**
   - Open your web browser and navigate to: `http://localhost:10000`

## 🖥️ Application Interface

### **Main Dashboard**
- **Home**: Overview and navigation hub
- **Graph Neural Networks**: Advanced molecular graph analysis
- **Circular Fingerprints**: Traditional ML with interpretability
- **RDKit Descriptors**: Descriptor-based predictions

### **Prediction Modes**

#### **Single Molecule Analysis**
1. Enter a SMILES string (e.g., `CCO` for ethanol)
2. Select prediction confidence level
3. View results with IC50 estimates and activity classification
4. Download LIME explanations (where available)

#### **Batch Processing**
1. **Excel/CSV Files**: Upload files with SMILES column
2. **SDF Files**: Upload structure data files directly
3. **Progress Monitoring**: Real-time processing status
4. **Results Export**: Download complete analysis as CSV
5. **Individual LIME**: Access explanations for specific molecules

## 📁 Project Structure

### **Root-Level Files Overview**

```
AChE-Activity-Pred-1/
├── 📱 Application Entry Points
│   ├── main_app.py                     # Primary Streamlit app (iOS-style interface)
│   ├── app_launcher.py                 # Enhanced launcher with sophisticated UI
│   └── main.py                         # Reserved for future CLI implementation
│
├── 🔬 Model-Specific Applications
│   ├── app_graph_combined.py           # Graph NN (classification + regression)
│   ├── app_graphC.py                   # Graph NN classification only
│   ├── app_graphR.py                   # Graph NN regression only
│   ├── app_graph_combined_backup.py    # Backup version
│   ├── app_circular.py                 # Circular fingerprint predictions
│   ├── app_rdkit.py                    # RDKit descriptor predictions
│   ├── app_chemberta.py                # ChemBERTa transformer predictions
│   └── app_chemberta_new.py            # Updated ChemBERTa implementation
│
├── 🤖 Pre-trained Models (Root Level)
│   ├── bestPipeline_tpot_circularfingerprint_classification.pkl
│   ├── bestPipeline_tpot_rdkit_classification.pkl
│   ├── bestPipeline_tpot_rdkit_Regression.pkl
│   ├── best_model_aggregrate_circular.pkl
│   ├── train_data.pkl                  # Training dataset
│   ├── X_train_circular.pkl            # Circular fingerprint features
│   ├── checkpoint-2000/                # ChemBERTa model directory
│   ├── GraphConv_model_files/          # Graph NN classification models
│   └── graphConv_reg_model_files 2/    # Graph NN regression models
│
├── 🐳 Deployment & Configuration
│   ├── Dockerfile                      # Production container
│   ├── Dockerfile.render               # Render.com optimized container
│   ├── docker-compose.yml              # Multi-service orchestration
│   ├── render.yaml                     # Render.com deployment config
│   ├── start.sh                        # Container startup script
│   ├── start-render.sh                 # Render-specific startup
│   └── Makefile                        # Automated build commands
│
├── 📦 Dependencies
│   ├── requirements.txt                # Full development dependencies
│   └── requirements.render.txt         # Production-optimized dependencies
│
├── 🎨 Assets & Configuration
│   ├── style.css                       # Custom UI styling
│   ├── .streamlit/                     # Streamlit configuration
│   ├── .gitignore                      # Git exclusions
│   └── .dockerignore                   # Docker build exclusions
│
├── 📚 Documentation
│   ├── README.md                       # This file
│   ├── LICENSE                         # Apache 2.0 license
│   ├── DEPLOYMENT.md                   # Deployment guide
│   ├── DEPLOYMENT_CHECKLIST.md         # Pre-deployment checklist
│   ├── DEPLOYMENT_STRATEGY.md          # Strategy overview
│   ├── DOCKER_COMPLETE.md              # Complete Docker guide
│   ├── DOCKER_SETUP.md                 # Basic Docker setup
│   ├── README_DOCKER.md                # Docker instructions
│   ├── RENDER_DEPLOYMENT.md            # Render.com guide
│   ├── RENDER_READY.md                 # Render readiness check
│   ├── FINAL_STATUS.md                 # Project status
│   └── docs/                           # Additional documentation
│       ├── API_REFERENCE.md            # Complete API documentation
│       ├── ROOT_FILES_REFERENCE.md     # Root files documentation
│       └── FILE_STRUCTURE.md           # Comprehensive file guide
│
└── 📊 Additional Folders
    ├── Datasets/                       # Training and test datasets
    ├── Notebooks-*/                    # Jupyter notebooks (multiple folders)
    ├── Final_results_data/             # Analysis results
    └── DrugDiscovery-ML-Code-Data-Figures-Notebooks/
```

### **Key Root Files Explained**

#### **🚀 Quick Start Files**
- **`main_app.py`**: Start here - modern interface with iOS styling
- **`app_launcher.py`**: Alternative launcher with enhanced UI
- **`docker-compose.yml`**: One-command deployment: `docker-compose up -d`
- **`Makefile`**: Simplified commands: `make up`, `make down`, `make logs`

#### **🔧 Configuration Files**
- **`requirements.txt`**: Local development dependencies (25+ packages)
- **`requirements.render.txt`**: Production dependencies (CPU-optimized)
- **`render.yaml`**: Cloud deployment configuration
- **`style.css`**: Custom CSS for modern UI design

#### **🐳 Docker Files**
- **`Dockerfile`**: Multi-stage production container
- **`Dockerfile.render`**: Render.com optimized container  
- **`start.sh`**: General container startup with Xvfb
- **`start-render.sh`**: Render-specific startup with logging

#### **🤖 Model Files (1.5GB total)**
```
Classification Models:
├── bestPipeline_tpot_circularfingerprint_classification.pkl  (45 MB)
└── bestPipeline_tpot_rdkit_classification.pkl               (12 MB)

Regression Models:
├── bestPipeline_tpot_rdkit_Regression.pkl                   (15 MB)
└── best_model_aggregrate_circular.pkl                       (52 MB)

Training Data:
├── train_data.pkl                                           (80 MB)
└── X_train_circular.pkl                                     (120 MB)

Deep Learning Models:
├── checkpoint-2000/                    # ChemBERTa (850 MB)
├── GraphConv_model_files/              # Graph NN classification (200 MB)
└── graphConv_reg_model_files 2/        # Graph NN regression (180 MB)
```

## 🔬 Model Information

### **Graph Neural Networks**
- **Architecture**: DeepChem GraphConvModel
- **Features**: Molecular graph representation learning
- **Output**: Classification + regression with atomic contributions
- **Interpretability**: Similarity maps and attention weights

### **Circular Fingerprints**
- **Method**: Enhanced RDKit Morgan fingerprints with fallback
- **ML Pipeline**: TPOT-optimized ensemble models
- **Features**: 3-tier robust fingerprint generation
- **Interpretability**: LIME feature importance analysis

### **RDKit Descriptors**
- **Features**: Traditional molecular descriptors
- **ML Pipeline**: TPOT-optimized classification/regression
- **Speed**: Fast predictions for large datasets
- **Reliability**: Well-established chemical descriptors

## 📊 Input Formats

### **SMILES Strings**
- Standard chemical notation (e.g., `c1ccccc1` for benzene)
- Support for complex organic molecules
- Automatic validation and preprocessing

### **SDF Files**
- Standard structure data format
- Multiple molecules per file
- Preserves 3D structural information

### **Excel/CSV Files**
- Must contain a column with SMILES strings
- Flexible column naming (auto-detection)
- Support for additional metadata columns

## 🎯 Use Cases

### **Drug Discovery**
- Screen compound libraries for AChE inhibition
- Lead optimization with structure-activity insights
- Virtual screening of chemical databases

### **Academic Research**
- Alzheimer's disease research
- Neurotransmitter system studies
- Computational chemistry education

### **Pharmaceutical Industry**
- Early-stage drug development
- Safety assessment and toxicology
- Regulatory submission support

## 🔧 Configuration

### **Environment Variables**
- `STREAMLIT_SERVER_PORT`: Application port (default: 10000)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

### **Model Configuration**
- Models are automatically loaded on application startup
- GPU acceleration available for graph neural networks
- Memory optimization for large batch processing

## 🚨 Troubleshooting

### **Common Issues**

#### **Port Already in Use**
```bash
# Check what's using the port
lsof -i :10000
# Kill the process or use a different port
docker-compose down && docker-compose up -d
```

#### **Memory Issues**
- Ensure at least 4GB RAM available
- For large batch files, process in smaller chunks
- Consider using Docker for better resource management

#### **Model Loading Errors**
- Verify all model files are present in the repository
- Check Docker container logs: `docker logs container_name`
- Restart the application: `docker-compose restart`

#### **Dependency Issues**
- Update pip: `pip install --upgrade pip`
- Clear cache: `pip cache purge`
- Use virtual environment for local setup

## 📝 Example Usage

### **Single Prediction**
```python
# Example SMILES for testing
smiles_examples = [
    "CCO",                    # Ethanol
    "c1ccccc1",              # Benzene  
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"    # Caffeine
]
```

### **Batch File Format**
```csv
SMILES,Compound_Name,MW
CCO,Ethanol,46.07
c1ccccc1,Benzene,78.11
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,Ibuprofen,206.28
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepChem**: Graph neural network implementations
- **RDKit**: Molecular informatics toolkit
- **TPOT**: Automated machine learning pipeline optimization
- **Streamlit**: Web application framework
- **LIME**: Local interpretable model-agnostic explanations

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/bhatnira/AChE-Activity-Pred-1/issues)
- **Repository**: [AChE-Activity-Pred-1](https://github.com/bhatnira/AChE-Activity-Pred-1)

---

**⚡ Quick Start**: `git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git && cd AChE-Activity-Pred-1 && docker-compose up -d --build`

**🌐 Access**: http://localhost:10000
