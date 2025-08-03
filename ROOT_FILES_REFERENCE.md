# Root Files Reference - AChE Activity Prediction Suite

## Overview

This document provides comprehensive documentation for all root-level files in the AChE Activity Prediction Suite repository. These files handle application startup, configuration, deployment, and core functionality.

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
