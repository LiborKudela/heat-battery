![osname-ubuntu-22.04](https://img.shields.io/badge/Ubuntu--22.04-8A2BE2?logo=ubuntu)
[![build-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/success-build-ubuntu-22.04.json&logo=gnubash)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)
[![unittest-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/success-unittest-ubuntu-22.04.json&logo=python)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)
[![diskusage-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/disk-usage-ubuntu-22.04.json)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)  
![osname-ubuntu-24.04](https://img.shields.io/badge/Ubuntu--24.04-8A2BE2?logo=ubuntu)
[![build-ubuntu-24.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/success-build-ubuntu-24.04.json&logo=gnubash)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-24.04.yml) 
[![unittest-ubuntu-24.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/success-unittest-ubuntu-24.04.json&logo=python)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-24.04.yml)
[![diskusage-ubuntu-24.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/disk-usage-ubuntu-24.04.json)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-24.04.yml)  
[![cloc-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/cloc-ubuntu-22.04.json)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-cloc.yml)
[![coverage-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/coverage-ubuntu-22.04.json)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)

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
bash install_scripts/install_ubuntu.sh -y -p --ppass db_password --ppassc db_password
```

### Windows installation
If you wish to install under Windows you will need to install Windows subsystem 
for Linux (WSL) first.  
Select a Ubuntu distribution (e.g. 22.04 LTS). You can find the details for this step
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
- <span text="color: green">Simulation of heat storages with complex geometry,</span>
- <span style="color: green">Co-simulation of attached controllers (e.g. PID, python code etc.),</span>
- <span style="color: green">Co-simulation of attached systems (e.g. PV panels, heat pumps, etc.),</span>
- <span style="color: red">Co-simulation of OpenModelica models (e.g. pipes, heat pumps, etc.),</span>
- <span style="color: green">Distributed parallel simulation (via MPI),</span>
- <span style="color: green">Distributed large parameter studies (via MPI, workers send data to master),</span>
- <span style="color: green">Web based UI for management of the distributed parameter studies (requires database),</span>
- <span style="color: green">Complex geometry import via STL files import,</span>
- <span style="color: green">Sensitivity analysis (adjoint derivative),</span>
- <span style="color: red">Second order sensitivity analysis (Hessian) (in development),</span>
- <span style="color: orange">Mathematical optimisation (Gradient descent, Newton's method, etc.),</span>
- <span style="color: orange">Least square fitting to data with SGD</span>

## Examples/Tutorials
- [Example_01: Simple cylindrical heat storage](examples/Example_01)
- [Example_02: Same but STEP file imported geometry](examples/Example_02)
- [Example_03: Transient hot wire method simulation](examples/Example_03)
- [Example_04: Simulation of passive heat storage](examples/Example_04)
- [Example_05: Large parameter study of passive heat storage](examples/Example_05)
- [Example_06: Simulation of active heat storage with controller](examples/Example_06)
