# Contributing to ExplainableAI-BestModels-AChEI-DrugDiscovery

We appreciate your interest in contributing to this project! This document provides guidelines for contributing to the ExplainableAI-BestModels-AChEI-DrugDiscovery repository.

## Authors and Maintainers

- **Nirajan Bhattarai** - Primary Research and Development
- **Marvin Schulte** - Research Collaboration and Development

## Types of Contributions

We welcome the following types of contributions:

### 🔬 Research Contributions
- New model implementations for molecular classification
- Novel interpretability techniques
- Performance improvements and optimizations
- Comparative analysis of additional methods

### 📚 Documentation
- Improvements to existing documentation
- Additional examples and tutorials
- Code comments and docstrings
- Performance benchmarking results

### 🐛 Bug Reports and Fixes
- Issue identification and reporting
- Bug fixes with proper testing
- Performance issue resolution

### 💡 Feature Requests
- New interpretability methods
- Additional molecular descriptors
- Visualization improvements
- Dataset preprocessing enhancements

## Before Contributing

### Prerequisites
1. **Domain Knowledge**: Basic understanding of machine learning and chemical informatics
2. **Technical Skills**: Proficiency in Python, Jupyter notebooks, and relevant ML libraries
3. **Research Background**: Familiarity with drug discovery and molecular modeling concepts

### Development Environment
Ensure you have the following installed:
- Python 3.8 or higher
- Jupyter Lab/Notebook
- Git for version control
- Required dependencies (see `requirements.txt`)

## How to Contribute

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ExplainableAI-BestModels-AChEI-DrugDiscovery.git
cd ExplainableAI-BestModels-AChEI-DrugDiscovery
```

### 2. Create a Branch
```bash
# Create a descriptive branch name
git checkout -b feature/new-interpretability-method
# or
git checkout -b bugfix/model-loading-issue
# or
git checkout -b docs/improve-readme
```

### 3. Set Up Development Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Install additional development tools (optional)
pip install black flake8 pytest
```

### 4. Make Your Changes

#### For Code Contributions:
- Follow existing code style and conventions
- Add comprehensive comments and docstrings
- Include appropriate error handling
- Ensure reproducibility with random seeds

#### For Notebook Contributions:
- Clear all outputs before committing (unless they demonstrate important results)
- Include markdown cells with explanations
- Follow the existing notebook structure
- Test all cells execute successfully

#### For Documentation:
- Use clear, concise language
- Include code examples where appropriate
- Maintain consistent formatting
- Verify all links work correctly

### 5. Test Your Changes
```bash
# Test notebook execution
jupyter nbconvert --to notebook --execute your_notebook.ipynb

# Run any unit tests (if applicable)
pytest tests/

# Check code style (optional)
black --check .
flake8 .
```

### 6. Commit Your Changes
```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "Add SHAP analysis for graph convolutional networks

- Implement node-level SHAP value calculation
- Add molecular substructure highlighting
- Include performance comparison with baseline
- Update documentation with usage examples"
```

### 7. Push and Create Pull Request
```bash
# Push to your fork
git push origin feature/new-interpretability-method

# Create a pull request on GitHub with detailed description
```

## Contribution Guidelines

### Code Style
- **Python Style**: Follow PEP 8 guidelines
- **Naming Conventions**: Use descriptive variable and function names
- **Comments**: Include inline comments for complex logic
- **Docstrings**: Use NumPy-style docstrings for functions and classes

### Notebook Guidelines
- **Structure**: Follow the existing notebook organization pattern
- **Markdown**: Use appropriate headers and explanatory text
- **Outputs**: Clear outputs before committing (unless demonstrating results)
- **Reproducibility**: Set random seeds and document environment requirements

### Documentation Standards
- **Clarity**: Write for users with varying expertise levels
- **Examples**: Include practical usage examples
- **Links**: Verify all external links are functional
- **Citations**: Properly cite academic papers and external resources

### Research Ethics
- **Data**: Ensure compliance with data usage policies
- **Citations**: Properly attribute all sources and prior work
- **Reproducibility**: Provide sufficient detail for result reproduction
- **Validation**: Include appropriate validation and testing

## Pull Request Process

### PR Description Template
```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Research contribution

## Changes Made
- Detailed list of changes
- Files modified
- New features added

## Testing
- [ ] Tested locally
- [ ] All notebooks execute successfully
- [ ] Documentation updated
- [ ] Examples provided

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] New and existing tests pass
```

### Review Process
1. **Initial Review**: Authors will review for alignment with project goals
2. **Technical Review**: Code quality, methodology, and implementation
3. **Testing**: Verification that changes work as expected
4. **Documentation**: Ensure adequate documentation is provided
5. **Final Approval**: Both authors must approve significant changes

## Research Collaboration

### Academic Partnerships
- Open to collaborations with academic institutions
- Joint research projects welcome
- Publication opportunities for significant contributions

### Conference Presentations
- Contributors may be invited to co-present work
- Acknowledgment in academic publications
- Collaboration on workshop submissions

## Recognition

### Contributor Acknowledgment
- Significant contributors will be acknowledged in the README
- Academic contributions may lead to co-authorship opportunities
- Technical contributions will be credited appropriately

### Hall of Fame
We maintain a contributor hall of fame recognizing:
- Major feature additions
- Significant bug fixes
- Documentation improvements
- Research contributions

## Getting Help

### Questions and Support
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and research discussions
- **Direct Contact**: Reach out to authors for collaboration inquiries

### Resources
- **ChEMBL Documentation**: https://chembl.gitbook.io/chembl-interface-documentation/
- **DeepChem Documentation**: https://deepchem.readthedocs.io/
- **RDKit Documentation**: https://www.rdkit.org/docs/
- **Scikit-learn Documentation**: https://scikit-learn.org/stable/

## Code of Conduct

### Research Integrity
- Maintain high standards of scientific rigor
- Properly cite all sources and prior work
- Report results honestly and transparently
- Respect intellectual property rights

### Collaboration
- Be respectful and constructive in all interactions
- Provide helpful feedback and suggestions
- Welcome contributors of all backgrounds and experience levels
- Foster an inclusive research environment

### Quality Standards
- Ensure code is well-tested and documented
- Follow reproducible research practices
- Maintain backward compatibility when possible
- Consider computational efficiency and scalability

## License and Copyright

All contributions become part of the project under the terms specified in the [LICENSE.md](LICENSE.md) file. By contributing, you agree that:

- Your contributions will be subject to the same licensing terms
- You have the right to submit the contributions
- You understand the all-rights-reserved nature of this project

## Contact Information

For questions about contributing, please contact:

- **Nirajan Bhattarai**: [GitHub Profile](https://github.com/bhatnira)
- **Marvin Schulte**: Research Collaborator

---

Thank you for your interest in contributing to this explainable AI research project! Your contributions help advance the field of AI-driven drug discovery and molecular modeling.
