#!/usr/bin/env python3
"""
ExplainableAI-BestModels-AChEI-DrugDiscovery Setup Script
Cross-platform setup script for the project

Authors: Nirajan Bhattarai and Marvin Schulte
"""

import sys
import subprocess
import importlib
import platform
import shutil
from pathlib import Path

# ANSI color codes for cross-platform colored output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(message, color=Colors.WHITE):
    """Print colored message"""
    print(f"{color}{message}{Colors.END}")

def print_header(message):
    """Print header message"""
    print_colored(f"\n{'='*50}", Colors.BLUE)
    print_colored(message, Colors.BOLD + Colors.BLUE)
    print_colored('='*50, Colors.BLUE)

def print_success(message):
    """Print success message"""
    print_colored(f"✓ {message}", Colors.GREEN)

def print_warning(message):
    """Print warning message"""
    print_colored(f"⚠ {message}", Colors.YELLOW)

def print_error(message):
    """Print error message"""
    print_colored(f"✗ {message}", Colors.RED)

def print_info(message):
    """Print info message"""
    print_colored(f"ℹ {message}", Colors.CYAN)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print_info(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible (3.8+)")
        return True
    else:
        print_error("Python 3.8 or higher is required")
        return False

def check_package_managers():
    """Check available package managers"""
    print_header("Checking Package Managers")
    
    managers = {}
    
    # Check for pip
    if shutil.which('pip'):
        managers['pip'] = True
        print_success("pip found")
    else:
        managers['pip'] = False
        print_error("pip not found")
    
    # Check for conda
    if shutil.which('conda'):
        managers['conda'] = True
        print_success("conda found (recommended)")
    else:
        managers['conda'] = False
        print_warning("conda not found")
    
    return managers

def check_system_info():
    """Display system information"""
    print_header("System Information")
    
    print_info(f"Operating System: {platform.system()} {platform.release()}")
    print_info(f"Architecture: {platform.machine()}")
    print_info(f"Processor: {platform.processor()}")
    
    # Check memory (cross-platform)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 16:
            print_success(f"Memory: {memory_gb:.1f}GB (Recommended)")
        elif memory_gb >= 8:
            print_warning(f"Memory: {memory_gb:.1f}GB (Minimum met)")
        else:
            print_warning(f"Memory: {memory_gb:.1f}GB (Below recommended)")
    except ImportError:
        print_warning("psutil not available - cannot check memory")

def create_environment():
    """Create virtual environment"""
    print_header("Creating Virtual Environment")
    
    env_name = "xai-drug-discovery"
    
    try:
        # Try conda first if available
        if shutil.which('conda'):
            print_info("Creating conda environment...")
            result = subprocess.run([
                'conda', 'create', '-n', env_name, 'python=3.8', '-y'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print_success(f"Conda environment '{env_name}' created")
                print_info(f"To activate: conda activate {env_name}")
                return True
            else:
                print_error("Failed to create conda environment")
                print_error(result.stderr)
        
        # Fallback to venv
        print_info("Creating virtual environment with venv...")
        result = subprocess.run([
            sys.executable, '-m', 'venv', env_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success(f"Virtual environment '{env_name}' created")
            if platform.system() == "Windows":
                print_info(f"To activate: {env_name}\\Scripts\\activate")
            else:
                print_info(f"To activate: source {env_name}/bin/activate")
            return True
        else:
            print_error("Failed to create virtual environment")
            print_error(result.stderr)
            return False
            
    except Exception as e:
        print_error(f"Error creating environment: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print_header("Installing Required Packages")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False
    
    try:
        print_info("Installing packages from requirements.txt...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("All packages installed successfully")
            return True
        else:
            print_error("Some packages failed to install")
            print_error(result.stderr)
            return False
            
    except Exception as e:
        print_error(f"Error installing packages: {e}")
        return False

def verify_installation():
    """Verify package installation"""
    print_header("Verifying Installation")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy',
        'sklearn', 'tensorflow', 'torch', 'deepchem', 
        'rdkit', 'shap', 'transformers'
    ]
    
    failed_packages = []
    
    for package in required_packages:
        try:
            # Handle special cases
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'rdkit':
                importlib.import_module('rdkit.Chem')
            else:
                importlib.import_module(package)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package}")
            failed_packages.append(package)
    
    if failed_packages:
        print_warning(f"\nFailed to import: {', '.join(failed_packages)}")
        print_warning("You may need to install these packages manually")
        return False
    else:
        print_success("\nAll core packages imported successfully!")
        return True

def check_gpu():
    """Check for GPU availability"""
    print_header("Checking GPU Availability")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"CUDA GPU detected: {gpu_name}")
            print_info(f"GPU count: {gpu_count}")
        else:
            print_warning("No CUDA GPU detected - models will run on CPU")
    except ImportError:
        print_warning("PyTorch not available - cannot check GPU")

def display_next_steps():
    """Display next steps for the user"""
    print_header("Setup Complete!")
    
    print_colored("\nNext Steps:", Colors.BOLD)
    print_info("1. Activate your virtual environment:")
    
    if shutil.which('conda'):
        print_colored("   conda activate xai-drug-discovery", Colors.WHITE)
    else:
        if platform.system() == "Windows":
            print_colored("   xai-drug-discovery\\Scripts\\activate", Colors.WHITE)
        else:
            print_colored("   source xai-drug-discovery/bin/activate", Colors.WHITE)
    
    print_info("2. Launch Jupyter:")
    print_colored("   jupyter lab", Colors.WHITE)
    
    print_info("3. Open a notebook to get started!")
    print_colored("   Start with: ModelInterpretability_deepNet_rdkit.ipynb", Colors.WHITE)
    
    print_info("4. For help, see:")
    print_colored("   docs/USER_GUIDE.md", Colors.WHITE)
    
    print_colored(f"\n{'='*50}", Colors.GREEN)
    print_colored("Happy modeling!", Colors.BOLD + Colors.GREEN)
    print_colored('='*50, Colors.GREEN)

def main():
    """Main setup function"""
    print_colored("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           ExplainableAI Drug Discovery Setup                 ║
    ║                                                              ║
    ║  Authors: Nirajan Bhattarai and Marvin Schulte             ║
    ╚══════════════════════════════════════════════════════════════╝
    """, Colors.BOLD + Colors.MAGENTA)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    managers = check_package_managers()
    if not managers['pip']:
        print_error("pip is required but not found")
        sys.exit(1)
    
    check_system_info()
    
    # Interactive setup
    try:
        # Ask about environment creation
        response = input(f"\n{Colors.CYAN}Create virtual environment? (y/n): {Colors.END}")
        if response.lower() in ['y', 'yes']:
            create_environment()
        
        # Ask about package installation
        response = input(f"\n{Colors.CYAN}Install required packages? (y/n): {Colors.END}")
        if response.lower() in ['y', 'yes']:
            if install_requirements():
                verify_installation()
                check_gpu()
        
        display_next_steps()
        
    except KeyboardInterrupt:
        print_colored("\n\nSetup interrupted by user", Colors.YELLOW)
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
