# Quantum Phase Classification Project

## Project Overview
This project implements machine learning techniques to classify quantum phases in spin chain systems, based on correlation functions as feature vectors.

## Models Implemented
The project includes three quantum spin chain models:
- XXZ chains with uniaxial single-ion anisotropy (H1)
- Bond alternating XXZ chain (H2)
- Bilinear biquadratic chain (H3)

## Directory Structure
```
/
├── correlators.py      # Implements correlation function calculations
├── data_generation.py  # Generates data from quantum models
├── hamiltonians.py     # Defines quantum Hamiltonians
├── main.ipynb         # Main analysis notebook
└── utils.py           # Helper functions and constants
```

## Key Features
- Implementation of quantum spin chain Hamiltonians
- Correlation function calculations
- Data generation for different model parameters
- Machine learning phase classification using KNN
- PCA visualization of quantum phases

## Dependencies
- NumPy
- SciPy
- Pandas
- Scikit-learn
- Matplotlib
- tqdm

## Author
Edgard
