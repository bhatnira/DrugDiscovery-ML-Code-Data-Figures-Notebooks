# Developer Guide - AChE Activity Prediction Suite

## Overview

This guide provides comprehensive instructions for developers working with the AChE Activity Prediction Suite. It covers the entire development workflow, from local setup to production deployment.

## 🏗️ Architecture Overview

### Application Structure
```
Frontend (Streamlit) → Model Applications → ML Models → Predictions
                    ↓
                 Web Interface → User Interactions → Results Display
```

### Component Hierarchy
```
main_app.py / app_launcher.py (Entry Points)
    ├── app_graph_combined.py (Graph Neural Networks)
    ├── app_circular.py (Circular Fingerprints)  
    ├── app_rdkit.py (Molecular Descriptors)
    └── app_chemberta.py (Transformer Models)
```

## 🛠️ Development Environment Setup

### Prerequisites
- **Python**: 3.9+ (3.10 recommended)
- **Git**: Latest version
- **Docker**: 20.10+ (optional but recommended)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ free space

### Local Development Setup

#### 1. Repository Setup
```bash
# Clone the repository
git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git
cd AChE-Activity-Pred-1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Environment Configuration
```bash
# Create .env file for local development
cat > .env << EOF
STREAMLIT_SERVER_PORT=10000
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_SERVER_HEADLESS=false
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
DEBUG=true
EOF
```

#### 3. Verify Installation
```bash
# Test import of key dependencies
python -c "import streamlit; import rdkit; import sklearn; print('Setup successful!')"

# Check model files
ls -la *.pkl
ls -la checkpoint-2000/
ls -la GraphConv_model_files/
```

### Docker Development Setup

#### 1. Build Development Container
```bash
# Build the image
docker build -t ache-pred-dev .

# Alternative: Use docker-compose
docker-compose build
```

#### 2. Run Development Container
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access shell in container
docker-compose exec molecular-prediction-suite /bin/bash
```

## 📝 Working with Root-Level Files

### Application Entry Points

#### `main_app.py` - Primary Application
**Purpose**: Modern iOS-style interface with comprehensive navigation

**Development Notes:**
- Uses glass morphism CSS effects
- Implements responsive design patterns
- Handles subprocess launching for model apps

**Key Functions:**
```python
def show_home_page():
    """Displays main dashboard with feature highlights"""
    
def run_chemberta_app():
    """Launches ChemBERTa application in subprocess"""
    
def run_graph_app():
    """Launches graph neural network application"""
```

**Customization Points:**
- CSS styling in embedded `<style>` tags
- Navigation menu items and icons
- Feature descriptions and layouts
- Color schemes and animations

#### `app_launcher.py` - Enhanced Launcher
**Purpose**: Alternative launcher with sophisticated UI components

**Development Notes:**
- Uses Inter font for iOS authenticity
- Advanced backdrop filters and effects
- Application status monitoring
- Card-based navigation system

**Key Features:**
- Real-time application status checking
- Hover animations and transitions
- Responsive grid layouts
- Process management utilities

### Model-Specific Applications

#### Development Pattern
All model applications follow a similar structure:

```python
# 1. Imports and configuration
import streamlit as st
import relevant_ml_libraries

# 2. Page configuration
st.set_page_config(
    page_title="Model Name",
    page_icon="🔬",
    layout="wide"
)

# 3. Model loading
@st.cache_resource
def load_model():
    """Load and cache the ML model"""
    return model

# 4. Prediction functions
def predict_single(smiles):
    """Single molecule prediction"""
    
def predict_batch(data):
    """Batch prediction processing"""

# 5. UI components
def main():
    """Main application interface"""
    # Sidebar controls
    # Input methods
    # Results display
    # Visualizations
```

#### Extending Model Applications

**Adding New Features:**
1. Create new function in appropriate app file
2. Add UI components in `main()` function
3. Update navigation if needed
4. Test thoroughly with various inputs

**Example: Adding New Visualization**
```python
def create_custom_plot(prediction_data):
    """Create custom visualization for predictions"""
    import plotly.express as px
    
    fig = px.scatter(
        prediction_data, 
        x='feature1', 
        y='prediction',
        title='Custom Analysis'
    )
    return fig

# In main() function:
if st.button("Show Custom Plot"):
    fig = create_custom_plot(results)
    st.plotly_chart(fig, use_container_width=True)
```

### Configuration Files

#### `requirements.txt` - Development Dependencies
**Management:**
```bash
# Add new package
pip install new-package
pip freeze > requirements.txt

# Update specific package
pip install --upgrade package-name
pip freeze > requirements.txt

# Security updates
pip-audit  # Check for vulnerabilities
pip install --upgrade package-name
```

**Testing Dependencies:**
```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
# Run application tests
```

#### `requirements.render.txt` - Production Dependencies
**Purpose**: Optimized for cloud deployment

**Key Differences:**
- CPU-only versions of PyTorch/TensorFlow
- Minimal package set for faster builds
- Fixed versions for stability

**Updating:**
1. Test changes in `requirements.txt` locally
2. Update `requirements.render.txt` with CPU versions
3. Test in Render.com or similar environment
4. Deploy and monitor performance

### Docker Configuration

#### `Dockerfile` - Production Container
**Multi-stage Build Process:**
```dockerfile
# Stage 1: Base dependencies
FROM python:3.10-slim as base
# Install system dependencies

# Stage 2: Python dependencies  
FROM base as python-deps
# Install Python packages

# Stage 3: Application
FROM python-deps as app
# Copy application code
# Set up runtime environment
```

**Customization:**
- Modify base image for different Python versions
- Add system dependencies in first stage
- Optimize layer caching for faster builds
- Add health checks and monitoring

#### `docker-compose.yml` - Service Orchestration
**Service Configuration:**
```yaml
services:
  molecular-prediction-suite:
    build: .
    ports: ["10000:10000", "8501:8501", ...]
    volumes: ["./data:/app/data", ...]
    environment: [...]
    healthcheck: [...]
```

**Development Overrides:**
Create `docker-compose.override.yml`:
```yaml
services:
  molecular-prediction-suite:
    volumes:
      - .:/app  # Mount source for live reload
    environment:
      - DEBUG=true
    command: streamlit run main_app.py --server.runOnSave=true
```

### Startup Scripts

#### `start.sh` - General Container Startup
**Process Flow:**
1. Start virtual display (Xvfb) for RDKit
2. Set environment variables
3. Configure Streamlit settings
4. Launch application

**Customization:**
```bash
# Add custom initialization
echo "Custom setup commands..."

# Set additional environment variables
export CUSTOM_VAR=value

# Add health checks
curl -f http://localhost:$PORT/_stcore/health || exit 1
```

#### `start-render.sh` - Render.com Startup
**Enhanced Features:**
- Environment information logging
- Model file verification
- Directory setup
- Render-specific optimizations

**Development Mode:**
```bash
# Add debug mode
if [ "$DEBUG" = "true" ]; then
    echo "Starting in debug mode..."
    streamlit run main_app.py --server.runOnSave=true
else
    streamlit run main_app.py
fi
```

## 🎨 Frontend Development

### Styling with `style.css`

#### Current Design System
```css
/* Color Palette */
:root {
    --primary-color: #4CAF50;
    --secondary-color: #45a049;
    --background-color: #f0f2f6;
    --text-color: #333333;
    --border-radius: 8px;
    --shadow: 0 2px 4px rgba(0,0,0,0.1);
}
```

#### Component Styling
```css
/* Custom button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    border: none;
    border-radius: var(--border-radius);
    transition: all 0.3s ease;
}

/* Result cards */
.prediction-result {
    background: white;
    padding: 1rem;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    border-left: 4px solid var(--primary-color);
}
```

#### Adding New Styles
1. Define new CSS classes in `style.css`
2. Use `st.markdown()` to inject styles
3. Apply classes using `unsafe_allow_html=True`

**Example:**
```python
# In application file
st.markdown("""
<style>
.custom-metric {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Use the style
st.markdown('<div class="custom-metric">Custom Content</div>', 
           unsafe_allow_html=True)
```

### Streamlit Best Practices

#### State Management
```python
# Use session state for persistence
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Cache expensive operations
@st.cache_data
def load_training_data():
    return pd.read_pickle('train_data.pkl')

@st.cache_resource  
def load_model():
    return joblib.load('model.pkl')
```

#### Layout Optimization
```python
# Use columns for responsive design
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("Left panel")
    
with col2:
    st.write("Main content")
    
with col3:
    st.write("Right panel")

# Use containers for organization
with st.container():
    st.header("Section Title")
    # Section content
```

## 🧪 Testing and Quality Assurance

### Unit Testing

#### Test Structure
```
tests/
├── test_models.py          # Model prediction tests
├── test_utils.py           # Utility function tests  
├── test_apps.py            # Application logic tests
└── conftest.py             # Pytest configuration
```

#### Example Test Cases
```python
# test_models.py
import pytest
from app_graph_combined import standardize_smiles, predict_molecule

def test_smiles_standardization():
    """Test SMILES standardization function"""
    assert standardize_smiles("CCO") == "CCO"
    assert standardize_smiles("c1ccccc1") == "c1ccccc1"
    
def test_invalid_smiles():
    """Test handling of invalid SMILES"""
    with pytest.raises(ValueError):
        standardize_smiles("invalid_smiles")

def test_prediction_output_format():
    """Test prediction output format"""
    result = predict_molecule("CCO")
    assert 'prediction' in result
    assert 'confidence' in result
    assert isinstance(result['prediction'], (int, float))
```

#### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Integration Testing

#### Application Testing
```python
# test_integration.py
import subprocess
import time
import requests

def test_app_startup():
    """Test that applications start successfully"""
    # Start app in background
    process = subprocess.Popen([
        'streamlit', 'run', 'main_app.py', 
        '--server.port=8888', '--server.headless=true'
    ])
    
    # Wait for startup
    time.sleep(10)
    
    # Test health check
    response = requests.get('http://localhost:8888/_stcore/health')
    assert response.status_code == 200
    
    # Cleanup
    process.terminate()
```

#### Docker Testing
```bash
# Test Docker build
docker build -t ache-test .

# Test container health
docker run -d --name test-container -p 8888:10000 ache-test
sleep 30
curl -f http://localhost:8888/_stcore/health
docker stop test-container && docker rm test-container
```

### Performance Testing

#### Load Testing
```python
# load_test.py
import concurrent.futures
import time
import requests

def make_prediction_request():
    """Make a single prediction request"""
    data = {'smiles': 'CCO'}
    response = requests.post('http://localhost:10000/predict', json=data)
    return response.status_code == 200

def test_concurrent_requests():
    """Test handling of concurrent requests"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_prediction_request) for _ in range(50)]
        results = [future.result() for future in futures]
    
    success_rate = sum(results) / len(results)
    assert success_rate > 0.95  # 95% success rate
```

#### Memory Profiling
```python
# profile_memory.py
import tracemalloc
from app_graph_combined import predict_molecule

tracemalloc.start()

# Run prediction
result = predict_molecule("CCO")

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

## 🚀 Deployment Guide

### Local Development Deployment

#### Quick Start
```bash
# Clone and setup
git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git
cd AChE-Activity-Pred-1

# Docker deployment (recommended)
make up  # or docker-compose up -d

# Local Python deployment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run main_app.py --server.port=10000
```

#### Development Configuration
```bash
# Set development environment variables
export DEBUG=true
export STREAMLIT_SERVER_HEADLESS=false
export STREAMLIT_SERVER_RUN_ON_SAVE=true

# Run with auto-reload
streamlit run main_app.py --server.runOnSave=true
```

### Production Deployment

#### Pre-deployment Checklist
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] Health checks implemented
- [ ] Monitoring setup

#### Docker Production Deployment
```bash
# Build production image
docker build -f Dockerfile -t ache-prod .

# Run with production settings
docker run -d \
  --name ache-production \
  -p 80:10000 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  -e STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
  ache-prod
```

#### Cloud Deployment (Render.com)
```bash
# Use render.yaml configuration
# Deploy via Render dashboard or CLI
render deploy

# Monitor deployment
render logs --service=molecular-prediction-suite
```

### Environment-Specific Configuration

#### Development
```yaml
# docker-compose.override.yml
services:
  molecular-prediction-suite:
    environment:
      - DEBUG=true
      - STREAMLIT_SERVER_HEADLESS=false
    volumes:
      - .:/app
    command: streamlit run main_app.py --server.runOnSave=true
```

#### Staging
```yaml
# docker-compose.staging.yml
services:
  molecular-prediction-suite:
    environment:
      - DEBUG=false
      - STREAMLIT_SERVER_HEADLESS=true
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

#### Production
```yaml
# docker-compose.prod.yml
services:
  molecular-prediction-suite:
    environment:
      - DEBUG=false
      - STREAMLIT_SERVER_HEADLESS=true
      - LOG_LEVEL=ERROR
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:10000/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 📊 Monitoring and Logging

### Application Monitoring

#### Health Checks
```python
# health_check.py
import requests
import sys

def check_app_health():
    """Check application health"""
    try:
        response = requests.get('http://localhost:10000/_stcore/health', timeout=5)
        if response.status_code == 200:
            print("✅ Application is healthy")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    healthy = check_app_health()
    sys.exit(0 if healthy else 1)
```

#### Performance Monitoring
```python
# monitor.py
import time
import psutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_resources():
    """Monitor system resources"""
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        logger.info(f"CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
        
        if memory.percent > 90:
            logger.warning("High memory usage detected!")
        
        time.sleep(60)  # Monitor every minute

if __name__ == "__main__":
    monitor_resources()
```

### Logging Configuration

#### Application Logging
```python
# logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Setup application logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Use in applications
logger = setup_logging()
logger.info("Application started")
```

## 🔧 Troubleshooting Common Issues

### Development Issues

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Clear cache
pip cache purge
```

#### Model Loading Errors
```python
# Debug model loading
import pickle
import traceback

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    traceback.print_exc()
```

#### Memory Issues
```bash
# Monitor memory usage
top -p $(pgrep -f streamlit)

# Reduce memory usage
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
```

### Deployment Issues

#### Port Conflicts
```bash
# Find process using port
lsof -i :10000
sudo kill -9 <PID>

# Use alternative port
streamlit run main_app.py --server.port=8080
```

#### Container Issues
```bash
# Debug container
docker run -it --entrypoint=/bin/bash ache-pred

# Check logs
docker logs container-name

# Inspect container
docker inspect container-name
```

#### Permission Issues
```bash
# Fix file permissions
chmod +x start.sh start-render.sh

# Fix directory permissions
sudo chown -R $(id -u):$(id -g) .
```

## 🤝 Contributing Guidelines

### Code Style

#### Python Code Standards
- Follow PEP 8
- Use type hints where possible
- Add comprehensive docstrings
- Maximum line length: 88 characters (Black formatter)

#### Example Function
```python
def predict_molecule(smiles: str, model_type: str = 'graph') -> Dict[str, Any]:
    """
    Predict molecular activity for a given SMILES string.
    
    Args:
        smiles: Valid SMILES string representation of molecule
        model_type: Type of model to use ('graph', 'circular', 'rdkit')
        
    Returns:
        Dictionary containing prediction results with keys:
        - 'prediction': Predicted activity value
        - 'confidence': Confidence score (0-1)
        - 'model_used': Model type used for prediction
        
    Raises:
        ValueError: If SMILES string is invalid
        ModelError: If model loading fails
        
    Example:
        >>> result = predict_molecule('CCO', 'graph')
        >>> print(result['prediction'])
        0.85
    """
    # Implementation here
    pass
```

### Git Workflow

#### Branch Naming
- `feature/new-feature-name`
- `bugfix/issue-description`
- `hotfix/critical-fix`
- `docs/documentation-update`

#### Commit Messages
```
type(scope): short description

Detailed description if needed

- List specific changes
- Reference issues: Fixes #123
```

#### Pull Request Process
1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Submit PR with description
5. Address review comments
6. Merge after approval

### Testing Requirements

#### New Features
- Unit tests for all new functions
- Integration tests for UI components
- Performance tests for model changes
- Documentation updates

#### Example Test Addition
```python
# tests/test_new_feature.py
import pytest
from app_graph_combined import new_function

class TestNewFunction:
    def test_valid_input(self):
        """Test with valid input"""
        result = new_function("valid_input")
        assert result is not None
        assert isinstance(result, dict)
        
    def test_invalid_input(self):
        """Test with invalid input"""
        with pytest.raises(ValueError):
            new_function("invalid_input")
            
    def test_edge_cases(self):
        """Test edge cases"""
        assert new_function("") == {}
        assert new_function(None) is None
```

## 📚 Additional Resources

### Documentation
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Root Files Reference](docs/ROOT_FILES_REFERENCE.md) - Root files documentation  
- [File Structure Guide](docs/FILE_STRUCTURE.md) - Comprehensive file guide

### External Libraries
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [DeepChem Documentation](https://deepchem.readthedocs.io/)
- [TPOT Documentation](https://epistasislab.github.io/tpot/)

### Development Tools
- [Black Code Formatter](https://black.readthedocs.io/)
- [pytest Testing Framework](https://docs.pytest.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Git Best Practices](https://git-scm.com/doc)

---

This developer guide serves as a comprehensive reference for working with the AChE Activity Prediction Suite. For questions or suggestions, please open an issue in the GitHub repository.
