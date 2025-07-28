# Contributing to ML Regression AChEI Drug Discovery

Thank you for your interest in contributing to this machine learning project for acetylcholinesterase inhibitor drug discovery! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

1. **Check existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Environment details (Python version, OS, etc.)
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Error messages and stack traces

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards
4. **Test your changes** thoroughly
5. **Commit with descriptive messages**:
   ```bash
   git commit -m "Add: new molecular fingerprint method"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** with a clear description

## 📝 Coding Standards

### Python Code Style
- Follow **PEP 8** style guidelines
- Use **Black** for code formatting: `black .`
- Use **isort** for import sorting: `isort .`
- Use **flake8** for linting: `flake8 .`

### Jupyter Notebooks
- **Clear markdown documentation** for each section
- **Descriptive variable names** and comments
- **Remove output cells** before committing (except for example outputs)
- **Use consistent cell structure**:
  1. Import statements
  2. Data loading
  3. Preprocessing
  4. Model training
  5. Evaluation
  6. Visualization

### Documentation
- **Docstrings** for all functions and classes
- **Type hints** where appropriate
- **Clear commit messages** following conventional commits
- **Update README.md** for new features

## 🧪 Testing Guidelines

### Before Submitting
- [ ] Code runs without errors
- [ ] All dependencies are listed in requirements.txt
- [ ] Notebooks execute from top to bottom
- [ ] Results are reproducible (fixed random seeds)
- [ ] No hardcoded file paths (use relative paths)

### Testing New Models
- [ ] Compare against baseline models
- [ ] Include cross-validation results
- [ ] Document hyperparameter choices
- [ ] Provide performance metrics
- [ ] Include visualization of results

## 🔬 Types of Contributions

### New Modeling Approaches
- Novel featurization methods
- Different regression algorithms
- Ensemble methods
- Transfer learning approaches

### Data Processing Improvements
- Better data cleaning methods
- Feature engineering techniques
- Data augmentation strategies
- Outlier detection methods

### Evaluation and Metrics
- Additional evaluation metrics
- Better visualization methods
- Statistical significance tests
- Model interpretation techniques

### Documentation and Examples
- Tutorial notebooks
- Code documentation
- Usage examples
- Best practices guides

## 📊 Data and Model Guidelines

### Data Handling
- **No raw data in repository** (use data loading scripts)
- **Document data sources** and preprocessing steps
- **Include data validation** checks
- **Handle missing values** appropriately

### Model Development
- **Reproducible results** with fixed random seeds
- **Cross-validation** for model evaluation
- **Hyperparameter tuning** documentation
- **Model serialization** for reuse

### Performance Reporting
- **Multiple metrics**: R², RMSE, MAE, correlation
- **Statistical significance** testing
- **Confidence intervals** where appropriate
- **Comparison with baselines**

## 🚀 Development Environment

### Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ML-Regression-AChEI-DrugDiscovery.git
cd ML-Regression-AChEI-DrugDiscovery

# Create conda environment
conda env create -f environment.yml
conda activate achei-regression

# Install development dependencies
pip install -e .
```

### Code Quality Tools
```bash
# Format code
black .
isort .

# Check code quality
flake8 .

# Run tests (if available)
pytest tests/
```

## 📋 Pull Request Checklist

- [ ] **Descriptive title** and clear description
- [ ] **Link to related issues** (if applicable)
- [ ] **Code follows style guidelines**
- [ ] **All tests pass**
- [ ] **Documentation updated** (if needed)
- [ ] **No merge conflicts** with main branch
- [ ] **Reasonable commit history** (squash if needed)

## 🎯 Contribution Ideas

### Beginner-Friendly
- Fix typos in documentation
- Add more comments to existing code
- Create simple visualization improvements
- Add error handling to existing functions

### Intermediate
- Implement new molecular descriptors
- Add cross-validation strategies
- Create model comparison notebooks
- Improve data preprocessing pipelines

### Advanced
- Implement new deep learning architectures
- Add multi-task learning capabilities
- Create automated hyperparameter tuning
- Develop model interpretation tools

## 📞 Getting Help

- **Open an issue** for questions about contributing
- **Start a discussion** for ideas and suggestions
- **Check existing documentation** before asking questions
- **Be respectful** and constructive in all interactions

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Academic publications (if applicable)

## 📚 Resources

- [RDKit Documentation](https://rdkit.org/docs/)
- [DeepChem Tutorials](https://deepchem.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

Thank you for contributing to the advancement of computational drug discovery! 🧬
