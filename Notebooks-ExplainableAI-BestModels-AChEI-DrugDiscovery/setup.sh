#!/bin/bash

# ExplainableAI-BestModels-AChEI-DrugDiscovery Setup Script
# Authors: Nirajan Bhattarai and Marvin Schulte

echo "=========================================="
echo "ExplainableAI Drug Discovery Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python is installed
check_python() {
    print_header "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        print_status "Python $PYTHON_VERSION found"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible (3.8+)"
            return 0
        else
            print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        return 1
    fi
}

# Check if pip is installed
check_pip() {
    print_header "Checking pip installation..."
    if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
        print_status "pip found"
        return 0
    else
        print_error "pip is not installed. Please install pip."
        return 1
    fi
}

# Check if conda is available
check_conda() {
    print_header "Checking conda availability..."
    if command -v conda &> /dev/null; then
        print_status "Conda found - recommended for this project"
        return 0
    else
        print_warning "Conda not found - will use pip for installation"
        return 1
    fi
}

# Create virtual environment
create_environment() {
    print_header "Setting up virtual environment..."
    
    if check_conda; then
        print_status "Creating conda environment 'xai-drug-discovery'..."
        conda create -n xai-drug-discovery python=3.8 -y
        if [ $? -eq 0 ]; then
            print_status "Conda environment created successfully"
            print_status "To activate: conda activate xai-drug-discovery"
            return 0
        else
            print_error "Failed to create conda environment"
            return 1
        fi
    else
        print_status "Creating virtual environment with venv..."
        python3 -m venv xai-drug-discovery
        if [ $? -eq 0 ]; then
            print_status "Virtual environment created successfully"
            print_status "To activate: source xai-drug-discovery/bin/activate"
            return 0
        else
            print_error "Failed to create virtual environment"
            return 1
        fi
    fi
}

# Install requirements
install_requirements() {
    print_header "Installing Python packages..."
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing packages from requirements.txt..."
        
        # Try to install with pip
        if command -v pip3 &> /dev/null; then
            pip3 install -r requirements.txt
        else
            pip install -r requirements.txt
        fi
        
        if [ $? -eq 0 ]; then
            print_status "All packages installed successfully"
            return 0
        else
            print_error "Some packages failed to install"
            print_warning "You may need to install them manually"
            return 1
        fi
    else
        print_error "requirements.txt not found"
        return 1
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying installation..."
    
    python3 -c "
import sys
packages = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy',
    'scikit-learn', 'tensorflow', 'torch', 
    'deepchem', 'rdkit', 'shap', 'transformers'
]

failed = []
for package in packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        print(f'✗ {package}')
        failed.append(package)

if failed:
    print(f'\\nFailed to import: {failed}')
    print('You may need to install these packages manually.')
    sys.exit(1)
else:
    print('\\n✓ All core packages imported successfully!')
    sys.exit(0)
"
    
    return $?
}

# Check system requirements
check_system_requirements() {
    print_header "Checking system requirements..."
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -ge 16 ]; then
            print_status "Memory: ${MEMORY_GB}GB (Recommended: 16GB+)"
        elif [ "$MEMORY_GB" -ge 8 ]; then
            print_warning "Memory: ${MEMORY_GB}GB (Minimum met, 16GB+ recommended)"
        else
            print_warning "Memory: ${MEMORY_GB}GB (Below recommended 8GB minimum)"
        fi
    fi
    
    # Check available disk space
    if command -v df &> /dev/null; then
        DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
        if [ "$DISK_GB" -ge 10 ]; then
            print_status "Disk space: ${DISK_GB}GB available (10GB+ required)"
        else
            print_warning "Disk space: ${DISK_GB}GB available (10GB+ recommended)"
        fi
    fi
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected (recommended for large models)"
    else
        print_warning "No NVIDIA GPU detected (models will run on CPU)"
    fi
}

# Main setup function
main() {
    print_header "Starting setup process..."
    echo ""
    
    # Check prerequisites
    if ! check_python; then
        exit 1
    fi
    
    if ! check_pip; then
        exit 1
    fi
    
    # Check system requirements
    check_system_requirements
    echo ""
    
    # Ask user if they want to proceed
    read -p "Do you want to create a virtual environment? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if ! create_environment; then
            print_error "Environment creation failed"
            exit 1
        fi
        echo ""
    fi
    
    # Ask about package installation
    read -p "Do you want to install required packages? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if ! install_requirements; then
            print_warning "Some packages may need manual installation"
        fi
        echo ""
        
        # Verify installation
        if verify_installation; then
            print_status "Installation verification successful!"
        else
            print_warning "Some packages may need attention"
        fi
    fi
    
    echo ""
    print_header "Setup Complete!"
    echo ""
    print_status "Next steps:"
    echo "  1. Activate your environment:"
    if command -v conda &> /dev/null; then
        echo "     conda activate xai-drug-discovery"
    else
        echo "     source xai-drug-discovery/bin/activate"
    fi
    echo "  2. Launch Jupyter:"
    echo "     jupyter lab"
    echo "  3. Open a notebook to get started!"
    echo ""
    print_status "For help, see docs/USER_GUIDE.md"
    echo ""
}

# Run main function
main "$@"
