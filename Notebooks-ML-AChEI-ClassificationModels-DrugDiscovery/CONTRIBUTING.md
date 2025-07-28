# Contributing to ML AChEI Classification Models

Thank you for your interest in contributing to this project! This guide will help you get started with contributing to the acetylcholinesterase inhibitor classification models.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Conda or pip
- Basic knowledge of machine learning and cheminformatics

### Setting up Development Environment

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ML-AChEI-ClassificationModels-DrugDiscovery.git
cd ML-AChEI-ClassificationModels-DrugDiscovery
```

2. **Create development environment**
```bash
conda env create -f environment.yml
conda activate achei-ml
```

3. **Install development dependencies**
```bash
pip install -e ".[dev]"
```

## 📝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Found an issue? Please report it!
2. **Feature Requests**: Have an idea for improvement? Let us know!
3. **Code Contributions**: Fix bugs, add features, or improve existing code
4. **Documentation**: Improve README, add comments, or create tutorials
5. **New Models**: Add new machine learning models or molecular descriptors
6. **Performance Improvements**: Optimize existing code for better performance

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates.

**Bug Report Template:**
- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: Python version, OS, library versions
- **Additional Context**: Screenshots, error logs, etc.

### Suggesting Features

Feature requests are welcome! Please provide:

- **Clear description** of the feature
- **Use case** explaining why this feature would be useful
- **Possible implementation** if you have ideas
- **Examples** of how the feature would be used

### Code Contributions

#### Pull Request Process

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
```bash
pytest tests/
```

4. **Commit your changes**
```bash
git add .
git commit -m "Add: descriptive commit message"
```

5. **Push and create pull request**
```bash
git push origin feature/your-feature-name
```

#### Coding Standards

- **Python Style**: Follow PEP 8 guidelines
- **Code Formatting**: Use `black` for code formatting
- **Import Sorting**: Use `isort` for import organization
- **Linting**: Use `flake8` for code quality checks
- **Documentation**: Add docstrings for all functions and classes

**Format your code:**
```bash
black .
isort .
flake8 .
```

#### Notebook Guidelines

- **Clear Structure**: Use markdown cells to explain each section
- **Reproducible**: Set random seeds for reproducibility
- **Clean Output**: Clear outputs before committing
- **Documentation**: Add comments explaining complex logic

### Adding New Models

When adding new machine learning models:

1. **Create a new notebook** following the naming convention
2. **Include comprehensive evaluation** (accuracy, precision, recall, F1, AUC)
3. **Add model interpretation** where possible (SHAP, feature importance)
4. **Document the methodology** in markdown cells
5. **Compare with existing models** if applicable

### Adding New Molecular Descriptors

For new molecular descriptor implementations:

1. **Research validation**: Ensure the descriptor is scientifically sound
2. **Performance comparison**: Compare with existing descriptors
3. **Documentation**: Explain the descriptor calculation method
4. **Code efficiency**: Optimize for computational performance

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_specific.py
```

### Writing Tests

- Write tests for all new functions
- Include edge cases and error conditions
- Use descriptive test names
- Mock external dependencies when necessary

## 📚 Documentation

### Code Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Include parameter types and return values
- Add usage examples where helpful

### Notebook Documentation

- Use markdown cells to explain methodology
- Include literature references
- Add interpretation of results
- Explain parameter choices

## 🔄 Review Process

### What We Look For

1. **Code Quality**: Clean, readable, and well-documented code
2. **Testing**: Adequate test coverage for new functionality
3. **Performance**: Efficient implementation
4. **Documentation**: Clear documentation and comments
5. **Scientific Validity**: Sound methodology and interpretation

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] No sensitive information is included

## 🎯 Project Priorities

Current areas where contributions are especially welcome:

1. **Model Optimization**: Hyperparameter tuning and architecture improvements
2. **New Descriptors**: Novel molecular representations
3. **Interpretability**: Enhanced model interpretation methods
4. **Performance**: Code optimization and parallelization
5. **Documentation**: Improved tutorials and examples

## 📞 Getting Help

If you need help or have questions:

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact the maintainer directly for sensitive issues

## 🏆 Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **Release Notes**: Major contributions mentioned
- **Citations**: Academic publications when applicable

## 📋 Code of Conduct

Please note that this project follows a Code of Conduct. By participating, you are expected to uphold this code:

- **Be respectful** and inclusive
- **Welcome newcomers** and help them learn
- **Focus on constructive feedback**
- **Respect different viewpoints** and experiences

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the advancement of computational drug discovery! 🧬💊
