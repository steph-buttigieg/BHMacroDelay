This package processes BH mergers in the cosmological simulation FABLE \([Henden et al. 2018,](http://dx.doi.org/10.1093/mnras/sty1780) [2019,](http://dx.doi.org/10.1093/mnras/stz2301) [2020](http://dx.doi.org/10.1093/mnras/staa2235) \) which employs the Arepo code \([Springel 2010;](http://dx.doi.org/10.1111/j.1365-2966.2009.15715.x) [Pakmor et al. 2015](http://dx.doi.org/10.1093/mnras/stv2380)\).
It generates macrophysical time delays according to the method described in [Buttigieg et al. (2025)](https://arxiv.org/abs/2504.17549). Note that the package is still in development and does not currently implement the full functionality intended.

Note that the FABLE data is currently not publicly available through a web server, but is available upon request. However, the BH merger data can be found in the fable_merger_files directory. With slight modifications, these scripts can also be used to analyse BH mergers in the Illustris simulation. 

 Author: [Stephanie Buttigieg](mailto:sb2583@cam.ac.uk)

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

Note that one of the package requirements is the illustris_python package which is included in the git repository of this module but for which further documentation can be obtained on the original git repository [here](https://github.com/illustristng/illustris_python). 

