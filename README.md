[![build-ubuntu20.04](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-20.04.yml/badge.svg)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-20.04.yml)  
[![build-ubuntu22.04](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml/badge.svg)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)  
[![build-ubuntu24.04](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-24.04.yml/badge.svg)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-24.04.yml)  


## Battery included heat equation solver for solid heat storages

### What is it?
A python package for thermal storage simulation and optimisation based on finite element method.

## Table of Contents
- [Installation](#installation)
- [Main Features](#main-features)
- [Examples](#examples)

# Installation
### Ubuntu installation
First clone this repository:
```bash
git clone https://github.com/LiborKudela/heat-battery.git
```
The most convenient way to install the package is to use the installation script.  
Run the script with --help option to see the installation options.
```bash
bash install_scripts/install_ubuntu.sh --help
```

The two most common combinations are:
#### Worker installation:
```bash
bash install_scripts/install_ubuntu.sh -y
```
#### Worker + database installation:
```bash
bash install_scripts/install_ubuntu.sh -y -p --ppass db_password, --ppassc db_password
```

### Windows installation
If you wish to install under Windows you will need to install Windows subsystem 
for Linux (WSL) first.  
Select a Ubuntu distribution*(e.g. 22.04 LTS). You can find the details for this step
[here](#https://learn.microsoft.com/en-us/windows/wsl/install).  
Then open new terminal with the WSL distribution and proceed with the steps for [Ubuntu installation](#ubuntu-installation).


### Test the package
After installation you can test the package with the following command:
```bash
python3 -c "import heat_battery; heat_battery.test_package()"
```
# Other compatible software
### Paraview (for postprocessing)
Paraview can be installed in Linux and Windows. 
Please see the details options [here](#https://www.paraview.org/download/).

## Main Features
Some of this features are in development  
- <text style="color: green">Simulation of heat storages with complex geometry,</text>
- <text style="color: green"> Co-simulation of attached controllers (e.g. PID, python code etc.),</text>
- <text style="color: green"> Co-simulation of attached systems (e.g. buildings, heat pumps, etc.),</text>
- <text style="color: red"> Co-simulation of OpenModelica models (e.g. buildings, heat pumps, etc.),</text>
- <text style="color: green"> Distributed parallel simulation (via MPI),</text>
- <text style="color: green"> Distributed large parameter studies (via MPI, workers send data to master),</text>
- <text style="color: green"> Web based UI for management of the distributed parameter studies (requires database),</text>
- <text style="color: green"> Complex geometry import via STL files import,</text>
- <text style="color: green"> Sensitivity analysis (adjoint derivative),</text>
- <text style="color: red"> Second order sensitivity analysis (Hessian) (in development),</text>
- <text style="color: orange"> Mathematical optimisation (Gradient descent, Newton's method, etc.),</text>
- <text style="color: orange"> Least square fitting to data with SGD</text>

## Examples/Tutorials
- [Example_01: Simple cylindrical heat storage](examples/Example_01)
- [Example_02: Same but STEP file imported geometry](examples/Example_02)
- [Example_03: Transient hot wire method simulation](examples/Example_03)
- [Example_04: Simulation of passive heat storage](examples/Example_04)
- [Example_05: Large parameter study of passive heat storage](examples/Example_05)
- [Example_06: Simulation of active heat storage with controller](examples/Example_06)
