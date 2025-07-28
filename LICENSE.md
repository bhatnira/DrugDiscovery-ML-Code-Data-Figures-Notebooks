# License

MIT License

Copyright (c) 2025 Nirajan Bhattarai and Marvin Schulte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Third-Party Licenses

This project includes or depends on several third-party libraries and tools. Below is a summary of their licenses:

### Core Dependencies

#### RDKit
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2006-2015, Rational Discovery LLC, Greg Landrum, and Julie Penzotti
- **Usage**: Molecular informatics and cheminformatics toolkit
- **License URL**: https://github.com/rdkit/rdkit/blob/master/license.txt

#### Streamlit
- **License**: Apache License 2.0
- **Copyright**: Copyright 2018-2023 Streamlit Inc.
- **Usage**: Web application framework for machine learning applications
- **License URL**: https://github.com/streamlit/streamlit/blob/develop/LICENSE

#### PyTorch
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2016-2023 Facebook, Inc. (Meta Platforms, Inc.)
- **Usage**: Deep learning framework for neural networks
- **License URL**: https://github.com/pytorch/pytorch/blob/master/LICENSE

#### Transformers (Hugging Face)
- **License**: Apache License 2.0
- **Copyright**: Copyright 2018 The HuggingFace Inc. team
- **Usage**: Transformer models including ChemBERTa
- **License URL**: https://github.com/huggingface/transformers/blob/main/LICENSE

#### scikit-learn
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2007-2023 The scikit-learn developers
- **Usage**: Machine learning library
- **License URL**: https://github.com/scikit-learn/scikit-learn/blob/main/COPYING

#### Pandas
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
- **Usage**: Data manipulation and analysis
- **License URL**: https://github.com/pandas-dev/pandas/blob/main/LICENSE

#### NumPy
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2005-2023, NumPy Developers
- **Usage**: Numerical computing library
- **License URL**: https://github.com/numpy/numpy/blob/main/LICENSE.txt

### Machine Learning Dependencies

#### TPOT
- **License**: GNU Lesser General Public License v3.0
- **Copyright**: Copyright (c) 2015-2021 University of Pennsylvania
- **Usage**: Automated machine learning pipeline optimization
- **License URL**: https://github.com/EpistasisLab/tpot/blob/master/LICENSE

#### DeepChem
- **License**: MIT License
- **Copyright**: Copyright (c) 2016 Stanford University and the Authors
- **Usage**: Deep learning for chemistry and materials science
- **License URL**: https://github.com/deepchem/deepchem/blob/master/LICENSE

#### LIME
- **License**: BSD 2-Clause License
- **Copyright**: Copyright (c) 2016, Marco Tulio Ribeiro
- **Usage**: Local interpretable model-agnostic explanations
- **License URL**: https://github.com/marcotcr/lime/blob/master/LICENSE

#### SHAP
- **License**: MIT License
- **Copyright**: Copyright (c) 2018 Scott Lundberg
- **Usage**: SHapley Additive exPlanations for model interpretability
- **License URL**: https://github.com/slundberg/shap/blob/master/LICENSE

### Visualization Dependencies

#### Matplotlib
- **License**: PSF-based License (BSD-compatible)
- **Copyright**: Copyright (c) 2012-2023 Matplotlib Development Team
- **Usage**: Plotting and visualization library
- **License URL**: https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE

#### Plotly
- **License**: MIT License
- **Copyright**: Copyright (c) 2016-2023 Plotly, Inc.
- **Usage**: Interactive plotting and visualization
- **License URL**: https://github.com/plotly/plotly.py/blob/master/LICENSE.txt

#### Seaborn
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2012-2023, Michael Waskom
- **Usage**: Statistical data visualization
- **License URL**: https://github.com/mwaskom/seaborn/blob/master/LICENSE.md

### Chemistry-Specific Dependencies

#### Mordred
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2018 Hirotomo Moriwaki
- **Usage**: Molecular descriptor calculator
- **License URL**: https://github.com/mordred-descriptor/mordred/blob/master/LICENSE

#### PubChemPy
- **License**: MIT License
- **Copyright**: Copyright (c) 2014 Matt Swain
- **Usage**: PubChem database interface
- **License URL**: https://github.com/mcs07/PubChemPy/blob/master/LICENSE

### Web Framework Dependencies

#### Streamlit Option Menu
- **License**: MIT License
- **Copyright**: Copyright (c) 2021 Victor Yan
- **Usage**: Navigation component for Streamlit
- **License URL**: https://github.com/victoryhb/streamlit-option-menu/blob/main/LICENSE

#### Streamlit Ketcher
- **License**: Apache License 2.0
- **Copyright**: Copyright (c) 2022 Streamlit Ketcher Contributors
- **Usage**: Molecular drawing component for Streamlit
- **License URL**: https://github.com/AndrejJurkin/streamlit-ketcher/blob/main/LICENSE

---

## Model Licenses and Attributions

### Pre-trained Models

#### ChemBERTa Models
- **Source**: DeepChem/Hugging Face Model Hub
- **License**: Apache License 2.0
- **Citation**: 
  ```
  @article{chithrananda2020chemberta,
    title={ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction},
    author={Chithrananda, Seyone and Grand, Gabriel and Ramsundar, Bharath},
    journal={arXiv preprint arXiv:2010.09885},
    year={2020}
  }
  ```

#### RDKit Descriptors
- **Source**: RDKit Open-Source Cheminformatics Software
- **License**: BSD 3-Clause License
- **Citation**:
  ```
  @misc{rdkit,
    author = {Greg Landrum and others},
    title = {RDKit: Open-source cheminformatics},
    url = {http://www.rdkit.org},
    note = {Version 2023.03.1}
  }
  ```

---

## Dataset Licenses

### Training Data
- **Source**: ChEMBL Database
- **License**: Creative Commons Attribution-ShareAlike 3.0 Unported License
- **Attribution**: European Bioinformatics Institute (EMBL-EBI)
- **URL**: https://www.ebi.ac.uk/chembl/
- **Citation**:
  ```
  @article{chembl2019,
    title={ChEMBL: towards direct deposition of bioassay data},
    author={Mendez, David and others},
    journal={Nucleic acids research},
    volume={47},
    number={D1},
    pages={D930--D940},
    year={2019},
    publisher={Oxford University Press}
  }
  ```

---

## Compliance Notes

### Commercial Use
This software is released under the MIT License, which permits commercial use. However, users should be aware of the following:

1. **Third-party Dependencies**: Some dependencies may have different license terms
2. **Model Usage**: Pre-trained models may have specific usage restrictions
3. **Data Sources**: Training data from ChEMBL requires attribution
4. **Patent Considerations**: Users should conduct their own patent clearance

### Academic Use
For academic and research use:

1. **Citation Required**: Please cite this work and its dependencies appropriately
2. **Data Attribution**: Acknowledge ChEMBL and other data sources
3. **Model Credits**: Credit pre-trained model authors
4. **Reproducibility**: Consider sharing derived datasets and model configurations

### Distribution
When distributing this software or derivatives:

1. **Include License**: Include this license file and all third-party licenses
2. **Attribution**: Maintain copyright notices and attributions
3. **Disclaimer**: Include appropriate disclaimers for warranty and liability
4. **Dependencies**: Ensure compliance with all dependency licenses

---

## Disclaimer

THE SOFTWARE AND MODELS PROVIDED ARE FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. THE AUTHORS MAKE NO REPRESENTATIONS OR WARRANTIES REGARDING THE ACCURACY, RELIABILITY, OR SUITABILITY OF THE SOFTWARE FOR ANY PARTICULAR PURPOSE, INCLUDING BUT NOT LIMITED TO DRUG DISCOVERY, MEDICAL DIAGNOSIS, OR THERAPEUTIC APPLICATIONS.

USERS ASSUME ALL RESPONSIBILITY FOR THE USE OF THIS SOFTWARE AND ANY DECISIONS MADE BASED ON ITS OUTPUT. THE AUTHORS SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF THIS SOFTWARE.

FOR COMMERCIAL OR CLINICAL APPLICATIONS, USERS SHOULD CONDUCT APPROPRIATE VALIDATION, VERIFICATION, AND REGULATORY COMPLIANCE PROCEDURES.

---

## Contact

For licensing questions or permissions beyond the scope of this license:

- **Primary Contact**: Nirajan Bhattarai - bhatnira@isu.edu
- **Institution**: Idaho State University
- **Repository**: https://github.com/bhatnira/AChE-Activity-Pred-1

---

*Last updated: July 28, 2025*
