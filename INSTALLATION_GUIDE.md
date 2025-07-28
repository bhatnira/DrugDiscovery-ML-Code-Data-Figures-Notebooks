# Installation Guide - AChE Inhibitor Prediction Suite

## System Requirements

### Operating System Support
- **Windows**: 10 or later
- **macOS**: 10.15 (Catalina) or later  
- **Linux**: Ubuntu 18.04 LTS or later, CentOS 7+

### Hardware Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (minimum), Intel i7 or AMD Ryzen 7 (recommended)
- **RAM**: 8GB minimum, 16GB recommended for graph models
- **Storage**: 5GB free space minimum, 10GB recommended
- **GPU**: Optional but recommended for ChemBERTa and Graph Neural Networks

## Python Environment Setup

### Step 1: Python Installation

#### Option A: Using Anaconda (Recommended)
```bash
# Download and install Anaconda from https://www.anaconda.com/
# Create new environment
conda create -n ache-prediction python=3.10
conda activate ache-prediction
```

#### Option B: Using System Python
```bash
# Ensure Python 3.10+ is installed
python --version

# Create virtual environment
python -m venv ache-prediction-env

# Activate environment
# On macOS/Linux:
source ache-prediction-env/bin/activate
# On Windows:
ache-prediction-env\Scripts\activate
```

### Step 2: Core Dependencies Installation

#### For Application 1 (AChE Activity Prediction Suite)
```bash
cd Application1-AChE-Activity-Pred/

# Install base requirements
pip install streamlit==1.28.0
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0

# Chemistry libraries
pip install rdkit==2023.3.2
pip install deepchem==2.7.1

# Deep learning
pip install torch==2.0.1
pip install transformers==4.33.0
pip install simpletransformers==0.63.9

# Visualization
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install plotly==5.15.0

# UI components
pip install streamlit-option-menu==0.3.6
pip install streamlit-ketcher==0.1.3

# Install all at once
pip install -r requirements.txt
```

#### For Application 2 (ChemML Suite)
```bash
cd Application2-AI-Activity-Prediction/

# AutoML and additional libraries
pip install tpot==0.12.0
pip install mordred==1.2.0
pip install lime==0.2.0.1

# Install all requirements
pip install -r requirements.txt
```

## Troubleshooting Common Installation Issues

### Issue 1: RDKit Installation Problems
```bash
# If pip installation fails, use conda
conda install -c conda-forge rdkit

# Alternative: Use mamba for faster installation
mamba install -c conda-forge rdkit
```

### Issue 2: PyTorch GPU Support
```bash
# For CUDA 11.8 (check your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 3: DeepChem Dependencies
```bash
# If DeepChem installation fails
conda install -c conda-forge deepchem

# Or install with specific dependencies
pip install deepchem[tensorflow]
```

### Issue 4: Streamlit Port Conflicts
```bash
# Run on different port if 8501 is occupied
streamlit run main_app.py --server.port 8502
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU Setup
1. **Install NVIDIA Drivers**: Latest version from NVIDIA website
2. **Install CUDA Toolkit**: Version 11.8 or 12.1
3. **Install cuDNN**: Compatible version with CUDA
4. **Verify Installation**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Environment Variables

### Set Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc (macOS/Linux)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"

# Windows (Command Prompt)
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%PYTHONPATH%;C:\path\to\your\project
```

## Verification Steps

### Test Application 1
```bash
cd Application1-AChE-Activity-Pred/
python -c "import streamlit, rdkit, torch; print('Application 1 dependencies OK')"
streamlit run main_app.py --server.headless true --server.port 8501
```

### Test Application 2
```bash
cd Application2-AI-Activity-Prediction/
python -c "import tpot, deepchem, mordred; print('Application 2 dependencies OK')"
streamlit run main_app.py --server.headless true --server.port 8502
```

## Docker Installation (Alternative)

### Using Docker Compose
```bash
# Clone repository
git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git
cd AChE-Activity-Pred-1

# Build and run with Docker
docker-compose up -d

# Access applications
# App 1: http://localhost:8501
# App 2: http://localhost:8502
```

### Individual Docker Images
```bash
# Application 1
cd Application1-AChE-Activity-Pred/
docker build -t ache-pred-app1 .
docker run -p 8501:8501 ache-pred-app1

# Application 2
cd Application2-AI-Activity-Prediction/
docker build -t ache-pred-app2 .
docker run -p 8502:8501 ache-pred-app2
```

## Performance Optimization

### Memory Optimization
```bash
# Increase Java heap size for RDKit
export JAVA_OPTS="-Xmx8g"

# Set OMP threads for CPU optimization
export OMP_NUM_THREADS=4
```

### GPU Memory Management
```python
# Add to Python scripts for GPU memory management
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)
```

## Quick Verification Script

Save as `verify_installation.py`:
```python
#!/usr/bin/env python3
"""Installation verification script for AChE Prediction Suite"""

def check_imports():
    required_packages = {
        'streamlit': 'Streamlit web framework',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'rdkit': 'Chemical informatics',
        'torch': 'PyTorch deep learning',
        'transformers': 'Transformer models',
        'deepchem': 'Chemical machine learning',
        'tpot': 'AutoML framework',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization'
    }
    
    success = []
    failures = []
    
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'rdkit':
                from rdkit import Chem
            else:
                __import__(package)
            success.append(f"✅ {package}: {description}")
        except ImportError:
            failures.append(f"❌ {package}: {description}")
    
    print("Installation Verification Results:")
    print("=" * 50)
    for item in success:
        print(item)
    
    if failures:
        print("\nMissing packages:")
        for item in failures:
            print(item)
        return False
    else:
        print("\n🎉 All packages installed successfully!")
        return True

if __name__ == "__main__":
    check_imports()
```

Run verification:
```bash
python verify_installation.py
```

## Support and Next Steps

1. **Successful Installation**: Proceed to the main applications
2. **Issues**: Check troubleshooting section or contact support
3. **Performance Issues**: Review optimization settings
4. **Updates**: Regularly update packages for latest features

## Contact Support

- **Email**: bhatnira@isu.edu
- **GitHub Issues**: [Report problems](https://github.com/bhatnira/AChE-Activity-Pred-1/issues)
- **Documentation**: Refer to individual project READMEs for specific issues
