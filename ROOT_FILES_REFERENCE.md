# Root Files Reference - AChE Drug Discovery Research Suite

## Overview

This document provides comprehensive reference documentation for all root-level files in the **AChE Drug Discovery Research Suite** repository. This workspace contains 37 sanitized research notebooks and comprehensive documentation for machine learning and AI-driven drug discovery.

## 🔒 Repository Status

**Sanitization**: ✅ Complete - All 37 notebooks sanitized with paths anonymized  
**Documentation**: 📚 Updated - Comprehensive guides with troubleshooting  
**Privacy**: 🛡️ Compliant - Safe for public sharing and collaboration  
**Production**: 🚀 Ready - Suitable for academic and commercial use

## 📄 Documentation Files (Root Level)

### `README.md` - Primary Repository Documentation
**Purpose**: Complete workspace overview and comprehensive user guide  
**Content**: 1000+ lines of documentation including:
- Project overview with status badges
- Notebook sanitization information and privacy compliance
- Complete research methodology and workflow
- Usage instructions for sanitized notebooks
- Troubleshooting guide and FAQ
- Performance benchmarks and model comparisons
- Quick navigation and getting started guides

**Key Sections**:
- **Sanitization Info**: Details on how notebooks were anonymized
- **Usage Guide**: How to work with sanitized notebooks  
- **Troubleshooting**: Common issues and solutions
- **Research Overview**: Complete methodology documentation

### `INDEX.md` - Documentation Navigation Hub
**Purpose**: Central navigation for all documentation files  
**Content**: Organized links and quick access to:
- Latest updates and sanitization status
- User guides for different audiences (researchers, students, developers)
- Complete documentation structure overview
- Quick start links and navigation paths

### `FILE_STRUCTURE.md` - Repository Organization Guide  
**Purpose**: Comprehensive file structure documentation  
**Content**: Detailed analysis of:
- Current repository structure with sanitization status
- All 37 notebook locations and purposes
- Data directory organization
- Documentation file descriptions
- System file explanations

### `API_REFERENCE.md` - Comprehensive Technical Documentation
**Purpose**: Complete API and notebook documentation  
**Content**: 500+ lines covering:
- All 37 notebooks with detailed descriptions
- Model architectures and performance metrics
- Notebook APIs and functionality
- Usage requirements and examples
- Sanitization impact on functionality

### `DEVELOPER_GUIDE.md` - Development Workflow Guide
**Purpose**: Instructions for working with sanitized notebooks  
**Content**: Developer-focused documentation including:
- Working with sanitized notebooks
- Path replacement requirements
- Environment setup for research
- Collaboration best practices
- Development workflow for notebook adaptation

### `ROOT_FILES_REFERENCE.md` - This Document
**Purpose**: Reference documentation for all root-level files  
**Content**: Detailed descriptions of every file in the repository root

### `ROOT_DIRECTORY_FILES.md` - Current Repository Analysis
**Purpose**: Live inventory and analysis of all root directory contents  
**Content**: Current state documentation with:
- Real-time file and directory listing
- Sanitization status for all components
- Repository statistics and metrics
- Maintenance and update status

## 📁 Research Notebook Directories

### `Notebooks-ML-AChEI-ClassificationModels-DrugDiscovery/`
**Purpose**: Comprehensive classification modeling research collection  
**Notebook Count**: 31 sanitized notebooks  
**Sanitization Status**: ✅ 100% complete  
**Research Focus**: Binary classification (Active/Inactive) for AChE inhibitors

**Key Components**:
- **Data Preparation**: Dataset preprocessing and standardization
- **Traditional ML**: RDKit, circular fingerprints, MACCS keys, PubChem features
- **Deep Learning**: Neural networks with various molecular representations
- **Graph Neural Networks**: GraphConv, MPNN, Graph Attention Networks
- **Transformers**: ChemBERTa models (5M, 10M, 77M parameters)
- **Model Interpretation**: Explainable AI for classification models

**Performance Range**: 0.87-0.93 ROC-AUC across different approaches

### `Notebooks-ML-Regression-AChEI-DrugDiscovery/`
**Purpose**: Regression modeling for continuous IC50/pIC50 prediction  
**Notebook Count**: 4 sanitized notebooks  
**Sanitization Status**: ✅ 100% complete  
**Research Focus**: Quantitative structure-activity relationship (QSAR) modeling

**Key Components**:
- **Traditional Regression**: RDKit descriptors with classical ML
- **Deep Regression**: Neural networks for continuous prediction
- **Fingerprint Regression**: Circular fingerprint-based QSAR
- **Graph Regression**: Graph neural networks for continuous targets

**Performance Range**: R² 0.69-0.78 for continuous prediction

### `Notebooks-ExplainableAI-BestModels-AChEI-DrugDiscovery/`
**Purpose**: Model interpretability and explainable AI research  
**Notebook Count**: 5 sanitized notebooks  
**Sanitization Status**: ✅ 100% complete  
**Research Focus**: Understanding model decisions and molecular insights

**Key Components**:
- **Graph Interpretability**: Node importance and attention visualization
- **Feature Analysis**: SHAP values and feature importance
- **Deep Network Analysis**: Layer-wise relevance propagation
- **Compound Generation**: Molecular design using interpretable models
- **Attention Analysis**: Transformer attention weight visualization

### `Notebooks-ExplainableAI-contribMaps-screenedCompounds/`
**Purpose**: Contribution maps and explanations for screened compounds  
**Notebook Count**: 3 sanitized notebooks  
**Sanitization Status**: ✅ 100% complete  
**Research Focus**: Compound-specific explanations and contribution analysis

## 📊 Data Directories

### `Datasets/`
**Purpose**: Complete training and validation data collection  
**Contents**:
- **Primary Dataset**: StandarizedSmiles_originalDataset_ChEMBL220.xlsx (15K+ compounds)
- **Cross-Species Data**: Human, Mouse, Cow, Eel, Ray, Mosquito datasets
- **Classification Data**: Species-specific classification datasets
- **Regression Data**: Continuous value prediction datasets
- **Evaluation Data**: Model assessment and validation sets

### `Final_results_data/`
**Purpose**: Analysis results, comparisons, and visualizations  
**Contents**:
- **Model Comparisons**: Algorithm performance across different approaches
- **Visualization**: Sanitized results visualization notebook
- **Benchmarks**: Performance metrics and statistical comparisons
- **Charts Data**: Data for generating performance comparison plots

## 🔧 System Files

### `.git/` - Version Control
**Purpose**: Git repository metadata and history  
**Status**: Active version control with complete commit history

### `.gitignore` - Git Exclusions  
**Purpose**: Specify files and patterns to exclude from version control  
**Contents**: Standard patterns for Python/Jupyter environments

## 📈 Repository Metrics

- **Total Files**: 40+ notebooks, data files, and documentation
- **Documentation Coverage**: 3000+ lines across 7 comprehensive guides
- **Sanitization**: 100% complete across all 37 notebooks
- **Research Scope**: 15K+ compounds, multiple species, advanced ML/AI models
- **Model Types**: Traditional ML, Deep Learning, Graph Neural Networks, Transformers
- **Performance**: State-of-the-art results with comprehensive benchmarking

## Application Entry Points

### `main_app.py`
The primary Streamlit application with modern iOS-style interface.

**Purpose:**
- Main dashboard and application launcher
- iOS-style glass morphism UI design
- Feature showcase and navigation hub

**Key Features:**
- Responsive design with gradient backgrounds
- Glass morphism styling with backdrop blur effects
- Navigation to specialized prediction applications
- Feature overview with interactive cards

**Usage:**
```bash
streamlit run main_app.py --server.port=10000
```

**Dependencies:**
- Streamlit for web interface
- streamlit-option-menu for navigation
- RDKit for molecular visualization
- Custom CSS styling

---

### `app_launcher.py`
Enhanced application launcher with sophisticated interface design.

**Purpose:**
- Alternative launcher with advanced UI components
- Multi-application navigation system
- Process management for different model applications

**Key Features:**
- Inter font family for authentic iOS look
- Advanced backdrop filters and glass effects
- Integrated application status monitoring
- Responsive card-based layout

**Usage:**
```bash
streamlit run app_launcher.py --server.port=10000
```

**UI Components:**
- Navigation header with glass effect
- Application cards with hover animations
- Status indicators for each model
- Responsive grid layout

---

### `main.py`
Empty file - currently not in use. Reserved for future CLI implementation.

**Purpose:**
- Placeholder for command-line interface
- Future batch processing capabilities
- Potential API endpoint definitions

---

## Individual Model Applications

### `app_graph_combined.py`
Graph Neural Network prediction application.

**Purpose:**
- DeepChem-based graph convolutional networks
- Molecular graph analysis and prediction
- Atomic contribution visualization

**Model Details:**
- Classification and regression capabilities
- GraphConv architecture
- Atomic-level interpretability

### `app_circular.py`
Circular fingerprint-based prediction application.

**Purpose:**
- Morgan circular fingerprints
- TPOT-optimized ensemble models
- LIME-based explanations

**Features:**
- Robust fingerprint generation with fallbacks
- Interactive molecular similarity analysis
- Feature importance visualization

### `app_rdkit.py`
RDKit molecular descriptor prediction application.

**Purpose:**
- Traditional molecular descriptors
- 200+ calculated properties
- Statistical analysis and visualization

**Descriptors:**
- Molecular weight, LogP, TPSA
- Topological indices
- Electronic properties
- Pharmacophore features

### `app_chemberta.py` / `app_chemberta_new.py`
Transformer-based molecular prediction applications.

**Purpose:**
- BERT-like transformer architecture
- SMILES tokenization and embedding
- Attention weight visualization

**Model Specifications:**
- 24M parameters
- 12 attention heads
- 512 token context length
- 768 hidden dimensions

### `app_graphC.py` / `app_graphR.py`
Specialized graph applications for classification and regression.

**Purpose:**
- Task-specific graph neural networks
- Optimized for classification or regression
- Streamlined interfaces for specific use cases

### `app_graph_combined_backup.py`
Backup version of the combined graph application.

**Purpose:**
- Fallback implementation
- Development checkpoint
- Version control safety net

---

## Configuration Files

### `requirements.txt`
Main Python dependencies for development and production.

**Categories:**
- **Core Dependencies**: Streamlit, pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Chemistry**: RDKit for molecular operations
- **Machine Learning**: xgboost, tpot for automated ML
- **Deep Learning**: PyTorch, TensorFlow, DeepChem
- **Transformers**: transformers, simpletransformers
- **Utilities**: joblib, scipy, lime, openpyxl

**Installation:**
```bash
pip install -r requirements.txt
```

### `requirements.render.txt`
Optimized dependencies for Render.com deployment.

**Key Differences:**
- CPU-only versions of PyTorch and TensorFlow
- Streamlined package list for faster builds
- Production-optimized versions
- Render-specific configurations

**Installation:**
```bash
pip install -r requirements.render.txt
```

---

## Deployment Configuration

### `docker-compose.yml`
Docker Compose configuration for containerized deployment.

**Services:**
- **molecular-prediction-suite**: Main application container

**Port Mapping:**
- `10000`: Main launcher application
- `8501`: Graph Neural Network app
- `8502`: RDKit descriptor app
- `8503`: ChemBERTa app
- `8504`: Circular fingerprint app

**Features:**
- Volume mounting for data and models
- Health check configuration
- Automatic restart policy
- Environment variable management

**Usage:**
```bash
docker-compose up -d
```

### `render.yaml`
Render.com deployment configuration.

**Specifications:**
- **Plan**: Standard
- **Region**: Oregon
- **Environment**: Docker
- **Health Check**: `/_stcore/health`
- **Auto Deploy**: Disabled (manual deployment)

**Environment Variables:**
- Streamlit server configuration
- Theme customization
- Performance optimization settings

**Storage:**
- **Disk**: 2GB for model data
- **Mount Path**: `/app/data`

---

## Startup Scripts

### `start.sh`
General Docker container startup script.

**Features:**
- Virtual display initialization (Xvfb)
- Environment variable configuration
- Port management
- Streamlit server startup with optimal settings

**Process:**
1. Start virtual display for headless operation
2. Configure display environment
3. Set port (default 8501)
4. Launch Streamlit with production settings

### `start-render.sh`
Render.com-specific startup script with enhanced logging.

**Features:**
- Comprehensive environment reporting
- Model file verification
- Directory creation
- Render-optimized configuration

**Checks:**
- Model file existence verification
- Environment information logging
- Port configuration validation
- Data directory setup

**Process:**
1. Start virtual display
2. Set environment variables
3. Verify model files
4. Create necessary directories
5. Launch application with Render settings

---

## Build and Development

### `Makefile`
Automated build and deployment commands.

**Available Commands:**
- `make help`: Display available commands
- `make build`: Build Docker image
- `make up`: Start application
- `make down`: Stop application
- `make restart`: Restart application
- `make logs`: View application logs
- `make test`: Run setup tests
- `make dev`: Start development mode
- `make health`: Check application health
- `make clean`: Clean containers and images

**Quick Start:**
```bash
make up
```

**Development Workflow:**
```bash
make build  # Build image
make up     # Start services
make logs   # Monitor logs
make down   # Stop when done
```

---

## Docker Configuration

### `Dockerfile`
Production Docker image configuration.

**Base Image:** Python 3.9+ with scientific computing libraries
**Features:**
- Multi-stage build for optimization
- System dependencies for RDKit and chemistry libraries
- Virtual display setup for headless operation
- Optimized layer caching

### `Dockerfile.render`
Render.com-specific Docker configuration.

**Optimizations:**
- Render platform compatibility
- Faster build times
- Memory-efficient layers
- Production-ready security settings

---

## Styling and Assets

### `style.css`
Custom CSS styling for the applications.

**Design Elements:**
- Modern button styling with hover effects
- Prediction result cards with color coding
- Molecular structure display containers
- Metric cards with shadow effects
- Warning and success message styling

**Color Scheme:**
- Primary: `#4CAF50` (green)
- Background: `#f0f2f6` (light gray)
- White containers with subtle shadows
- Gradient effects for modern appearance

**Components:**
- `.stButton`: Custom button styling
- `.prediction-result`: Result display cards
- `.molecule-structure`: Molecular visualization containers
- `.metric-card`: Statistical display cards
- `.warning`: Alert message styling

---

## Model Files (PKL/Binary)

### Classification Models
- `bestPipeline_tpot_circularfingerprint_classification.pkl`: Circular fingerprint classifier
- `bestPipeline_tpot_rdkit_classification.pkl`: RDKit descriptor classifier

### Regression Models
- `bestPipeline_tpot_rdkit_Regression.pkl`: RDKit descriptor regressor
- `best_model_aggregrate_circular.pkl`: Ensemble circular fingerprint model

### Training Data
- `train_data.pkl`: Processed training dataset
- `X_train_circular.pkl`: Circular fingerprint training features

### Model Directories
- `checkpoint-2000/`: ChemBERTa model checkpoint
- `GraphConv_model_files/`: Graph neural network classification models
- `graphConv_reg_model_files 2/`: Graph neural network regression models

---

## Git Configuration

### `.gitignore`
Version control exclusions.

**Excluded Items:**
- Python bytecode (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)
- Large model files (selective inclusion)
- Temporary files (`*.tmp`, `*.log`)

### `.dockerignore`
Docker build exclusions.

**Excluded Items:**
- Git history and configuration
- Documentation files
- Development dependencies
- Test files and notebooks
- Cache directories

---

## Performance Considerations

### Resource Requirements

**Memory Usage:**
- Graph NN models: 1.2-2.8 GB
- ChemBERTa models: 2.1-4.5 GB
- Traditional ML models: 380-850 MB

**CPU Requirements:**
- Minimum: 2 cores
- Recommended: 4+ cores for concurrent users
- Graph NN inference: CPU-intensive

**Storage Requirements:**
- Models: ~2 GB
- Application: ~500 MB
- Data cache: ~1 GB
- Total recommended: 5+ GB

### Optimization Settings

**Streamlit Configuration:**
- Headless mode for production
- Usage statistics disabled
- Memory management enabled
- Browser gathering stats disabled

**Environment Variables:**
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`
- `MPLBACKEND=Agg` (for matplotlib)
- `QT_QPA_PLATFORM=offscreen` (for Qt)

---

## Development Workflow

### Local Development
1. Install dependencies: `pip install -r requirements.txt`
2. Run specific app: `streamlit run app_launcher.py`
3. Access at: `http://localhost:8501`

### Docker Development
1. Build image: `make build`
2. Start services: `make up`
3. View logs: `make logs`
4. Access at: `http://localhost:10000`

### Production Deployment
1. Use `requirements.render.txt` for optimized builds
2. Configure environment variables
3. Set up health checks
4. Monitor resource usage
5. Implement logging and monitoring

---

## Troubleshooting

### Common Issues

**Port Conflicts:**
- Check if ports 8501-8504, 10000 are available
- Use `netstat -an | grep PORT` to check usage
- Modify docker-compose.yml port mappings if needed

**Memory Issues:**
- Monitor container memory usage
- Implement model lazy loading
- Use CPU-only versions for lower memory usage

**Model Loading Failures:**
- Verify all PKL files are present
- Check file permissions in container
- Ensure sufficient disk space

**Virtual Display Issues:**
- Verify Xvfb is running in container
- Check DISPLAY environment variable
- Ensure X11 libraries are installed

### Debug Commands

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs molecular-prediction-suite

# Execute shell in container
docker-compose exec molecular-prediction-suite /bin/bash

# Check application health
curl http://localhost:10000/_stcore/health

# Monitor resource usage
docker stats
```

---

## Security Considerations

### Container Security
- Non-root user execution
- Minimal base image
- No unnecessary packages
- Read-only file systems where possible

### Application Security
- Input validation for SMILES strings
- Sanitized file uploads
- Rate limiting on prediction endpoints
- Error message sanitization

### Data Privacy
- No persistent storage of user inputs
- Temporary file cleanup
- Memory clearing after predictions
- No logging of sensitive data

---

## Maintenance

### Regular Tasks
- Update dependencies quarterly
- Monitor security vulnerabilities
- Check model performance metrics
- Review and clean logs
- Update documentation

### Monitoring
- Application uptime
- Response times
- Error rates
- Resource utilization
- User access patterns

### Backup Strategy
- Model files: Version control + cloud storage
- Configuration: Git repository
- Documentation: Multiple formats
- Deployment scripts: Automated backups

---

## Contributing

### Code Style
- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings to all functions
- Include error handling

### Testing
- Unit tests for prediction functions
- Integration tests for web interface
- Performance benchmarks
- Container health checks

### Documentation
- Update this file for new root-level files
- Include examples for new features
- Document breaking changes
- Maintain changelog

---

## License and Attribution

This reference is part of the AChE Activity Prediction Suite.
Licensed under Apache License 2.0.

For detailed license information, see the `LICENSE` file in the repository root.
