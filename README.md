![build](https://github.com/LiborKudela/heat-battery/actions/workflows/python-app.yml/badge.svg)

## Battery included heat equation solver for solid heat storages

### What is it?
A python package for thermal storage simulation and optimisation based on finite element method.

## Table of Contents
- [Installation](#installation)
- [Main Features](#main-features)
- [Examples](#examples)

# Installation
### You will need a linux environment!
If you have Ubuntu/Debian system you can skip to step 2. If you use Windows, read the folowing paragraph.

It is recomended to use Debian/Linux. If you wish to install under Windows you
will need to install Windows subsystem for Linux (WSL) first. Select a Ubuntu
distribution. You can find the details
[here](#https://learn.microsoft.com/en-us/windows/wsl/install). The open new
terminal with the WSL distrubution and proceed with the following steps.

### 1. Install and test a MPI environment such as OpenMPI
```bash
sudo apt-get update 
sudo apt install build-essential 
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
mpirun -n 5 python3 -c "print('This message should repeat five times.')"
```
You can read more information on MPI instalation [here](#https://webpages.charlotte.edu/abw/coit-grid01.uncc.edu/ParallelProgSoftware/Software/OpenMPIInstall.pdf).

### 2. Install gmsh for mesh generation
```bash
sudo apt update
sudo apt install gmsh
```

### 3. Install latest release version of FEniCSx
```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx
```
You can read detailed instructions [here](#https://fenicsproject.org/download/#:~:text=The%20easiest%20way%20to%20install%20FEniCSx%20on%20Debian%20or%20Ubuntu%20Linux%20is%20via%20apt%3A).

### 4. Install latest release versio of HeatBattery:
You can use pypi repo (:warning: not functional yet, I did not decide on LICENCSE yet)
```bash
pip install heat_battery
```

You can clone the repository and install directly.
```bash
git clone https://github.com/LiborKudela/heat-battery
cd heat_battery
pip install .
```

### 5. Test the package
```bash
python3 -c "import heat_battery; heat_battery.test_package()"
```

### 6. Install Paraview (optional!! for postprocessing)
Paraview can be installed in Linux and Windows. Please see the details options [here](#https://www.paraview.org/download/).

## Main Features
- Complex geometry import via STL files import
- Sensitivity analysis (forward and adjoint derivative)
- Mathematical optimisation, algorithms included
- Least square fitting to data
- Simulation of in loop controller such as PID
- Live visualization via web aplication ploting

## Examples
- No examples yet