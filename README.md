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
├── data/              # Generated correlation data CSVs
├── images/            # Generated figures from notebook analyses
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

## Running data generation

The main script to compute correlation CSVs is `data_generation.py`. It uses multiprocessing to parallelize Hamiltonian diagonalizations and correlator evaluations.

Basic usage:

```bash
python data_generation.py [N] [n_cores] [H1 H2 H3]
```

- `N` — number of spins (default 8)
- `n_cores` — number of worker processes to spawn (default 2)
- Optional trailing arguments: list which Hamiltonians to generate (`H1`, `H2`, `H3`). Example: `python data_generation.py 12 4 H1 H2`

Outputs are written to `data/H1/N=<N>.csv`, `data/H2/N=<N>.csv`, `data/H3/N=<N>.csv`.

Notes about performance:
- Exact diagonalization scales badly with system size (matrix dimension grows as 3^N for spin-1). Keep `N` small (commonly 4, 8, 12 in this repo's data).
- When `n_cores` is larger than your available CPU count, the script reduces it automatically.


## Author
Edgard Macena Cabral

