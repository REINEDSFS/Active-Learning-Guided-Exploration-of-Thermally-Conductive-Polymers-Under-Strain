# [Active Learning-Guided Exploration of Thermally Conductive Polymers Under Strain](https://an-xu.github.io/AL-TCpolymer.github.io/)

This repository contains the code and data for the paper "*Active Learning-Guided Exploration of Thermally Conductive Polymers Under Strain*". The goal is to enable researchers to reproduce the results presented in the work and utilize the tools for further exploration.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Files in the Repository](#files-in-the-repository)
4. [How to Run the Jupyter Notebook](#how-to-run-the-jupyter-notebook)
5. [Instructions for Running the MD Simulations](#instructions-for-running-the-md-simulations)

---

## Introduction

This repository provides the code for:
- Active learning workflows implemented using Gaussian Process Regression (GPR) for discovering polymers with high thermal conductivity (TC) under strain.
- Molecular Dynamics (MD) simulations performed using LAMMPS to compute polymer thermal conductivity.

---

## Prerequisites
### Required Software
- **Python**: Version 3.10 or above.
- **Jupyter Notebook**: For running the interactive notebook. Install via pip:
  ```bash
  pip install notebook
  ```
### Required Python Packages
Ensure the following packages are installed in your environment:
- `mol2vec`
- `rdkit`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`
- `scipy`
- `csv`

You can install them using:
```bash
pip install mol2vec rdkit scikit-learn numpy pandas matplotlib seaborn tqdm scipy
```

## Additional Software

- **LAMMPS**: A molecular dynamics simulator. Refer to the [LAMMPS installation guide](https://www.lammps.org/doc/Install.html) for setup instructions.

---

## Files in the Repository

### Jupyter Notebook
- **`Active_Learning_Workflow.ipynb`**:
  - This notebook implements the active learning workflow described in the paper.
  - It includes:
    - GPR model training.
    - Bayesian Optimization.
    - Data visualization.

### LAMMPS Files
- **`amorphous_polymer_P522013.lmps`**: LAMMPS data file for the amorphous polymer used in simulations.
- **`lammps-2.in`**: LAMMPS input script for equilibrating the amorphous polymer.
- **`lammpsdeform-2.in`**: LAMMPS input script for applying strain and computing thermal conductivity via Non-Equilibrium Molecular Dynamics (NEMD).

---

## How to Run the Jupyter Notebook

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/REINEDSFS/Active-Learning-Guided-Exploration-of-Thermally-Conductive-Polymers-Under-Strain.git
   cd Active-Learning-Guided-Exploration-of-Thermally-Conductive-Polymers-Under-Strain
   ```
2. **Install the Required Packages**:
Use the commands provided in the [Prerequisites](#prerequisites) section to set up the required environment.
3. **Run the Notebook**:
   Open the notebook using the following command:
   ```bash
   jupyter notebook Active_Learning_Workflow.ipynb
   ```
   
## Instructions for Running the MD Simulations

### Prepare the Polymer Data File
- The file **`amorphous_polymer_P522013.lmps`** contains the data for the amorphous polymer generated using PySimm.

### Run the Equilibrium Simulation
- Use **`lammps-2.in`** to equilibrate the amorphous polymer. Run the following command:
  ```bash
  lmp_mpi -in lammps-2.in
  ```
### Run the Deformation Simulation
- Use lammpsdeform-2.in to apply strain and calculate thermal conductivity. Run the following command:
  ```bash
  lmp_mpi -in lammpsdeform-2.in
  ```

