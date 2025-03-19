This package processes BH mergers in the cosmological simulation FABLE (link to data repository of FABLE ideally) and generates macrophysical time delays according to the method described in Buttigieg et al. (2025) (link to paper). 

# Installation Guide

This project requires a Python environment with specific dependencies, including MPI for parallel computing. Follow the steps below to set up your environment.

## Prerequisites
Before installing the Python environment, ensure that an MPI implementation is installed on your system. Supported options include:

- **MPICH** (Recommended)
  ```bash
  sudo apt update
  sudo apt install mpich  # For Ubuntu/Debian
  ```

- **Open MPI**
  ```bash
  sudo apt update
  sudo apt install openmpi-bin  # For Ubuntu/Debian
  ```

- **Microsoft MPI (Windows only)**
  [Download and install MS-MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)

On HPC systems, you may need to load an MPI module:
```bash
module load openmpi  # Example command for HPC environments
```

## Creating the Python Environment
1. **Create a virtual environment:**
   ```bash
   python -m venv my_env
   source my_env/bin/activate  # On Linux/Mac
   .\my_env\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

If you encounter issues, please refer to the Troubleshooting section below or consult the project documentation.

