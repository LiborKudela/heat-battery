# HiTepOptim

Battery-included heat-equation simulation and optimization for solid thermal
storages.

![osname-ubuntu-22.04](https://img.shields.io/badge/Ubuntu--22.04-8A2BE2?logo=ubuntu)
[![build-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/success-build-ubuntu-22.04.json&logo=gnubash)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)
[![unittest-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/success-unittest-ubuntu-22.04.json&logo=python)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)
[![diskusage-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/disk-usage-ubuntu-22.04.json)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)
![osname-ubuntu-24.04](https://img.shields.io/badge/Ubuntu--24.04-8A2BE2?logo=ubuntu)
[![build-ubuntu-24.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/success-build-ubuntu-24.04.json&logo=gnubash)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-24.04.yml)
[![unittest-ubuntu-24.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/success-unittest-ubuntu-24.04.json&logo=python)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-24.04.yml)
[![diskusage-ubuntu-24.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/disk-usage-ubuntu-24.04.json)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-24.04.yml)
[![cloc-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/cloc-ubuntu-22.04.json)](assets/readme_cloc_history.md)
[![coverage-ubuntu-22.04](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/LiborKudela/6f402d08683e16acdfc545b806f60259/raw/coverage-ubuntu-22.04.json)](https://github.com/LiborKudela/heat-battery/actions/workflows/ubuntu-22.04.yml)

## Overview

HiTepOptim is a Python package, imported as `heat_battery`, for simulating and
optimizing solid thermal energy storages with the finite element method. It uses
DOLFINx/FEniCSx for the numerical model, Gmsh for geometry and mesh workflows,
MPI for parallel runs, and optional PostgreSQL support for distributed parameter
studies.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Testing](#testing)
- [Features](#features)
- [Examples](#examples)
- [Additional software](#additional-software)

## Requirements

- Ubuntu 22.04 or 24.04 is recommended and covered by CI.
- Python 3.8 or newer is required by the package metadata.
- The installer sets up system dependencies including FEniCSx/DOLFINx, Gmsh,
  ADIOS2, PostgreSQL client libraries, and optional PostgreSQL server support.
- Windows users should install Ubuntu through
  [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install)
  and then follow the Ubuntu instructions below.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/LiborKudela/heat-battery.git
cd heat-battery
```

The recommended installation path is the Ubuntu installer:

```bash
bash install_scripts/install_ubuntu.sh --help
```

For a worker/local simulation installation without a PostgreSQL server:

```bash
bash install_scripts/install_ubuntu.sh -y --hbdir "$HOME/heat_battery_data"
```

For a worker plus local PostgreSQL installation, provide and confirm the
PostgreSQL `postgres` user password:

```bash
bash install_scripts/install_ubuntu.sh -y -p --ppass db_password --ppassc db_password --hbdir "$HOME/heat_battery_data"
```

If you want to install into a virtual environment, create it with access to
system site packages before running the installer. This is useful because
FEniCSx is commonly installed from Ubuntu packages:

```bash
python3 -m venv --system-site-packages hb_venv
source hb_venv/bin/activate
bash install_scripts/install_ubuntu.sh -y --hbdir "$HOME/heat_battery_data"
```

## Testing

After installation, check that the package imports:

```bash
python3 -c "import heat_battery; heat_battery.test_package()"
```

Run the unit tests from the repository root:

```bash
python3 -m unittest discover -v -s ./test -p 'test_*.py'
```

## Features

- Finite element simulation of heat storage systems with complex geometry.
- Co-simulation with controllers, for example PID or custom Python logic.
- Co-simulation with attached systems such as PV panels and heat pumps.
- Parallel simulation and distributed parameter studies via MPI.
- PostgreSQL-backed project storage and web UI for distributed studies.
- Geometry import and mesh generation workflows using STL/STEP/Gmsh files.
- Sensitivity analysis with adjoint derivatives.
- Optimization tools including gradient-based methods and least-squares fitting.
- Experimental OpenModelica co-simulation and second-order sensitivity analysis.

## Examples

- [Example_01: Simple cylindrical heat storage](examples/Example_01)
- [Example_02: STEP-file imported geometry](examples/Example_02)
- [Example_03: Transient hot wire method simulation](examples/Example_03)
- [Example_04: Passive heat storage simulation](examples/Example_04)
- [Example_05: Large parameter study of passive heat storage](examples/Example_05)
- [Example_06: Active heat storage simulation with controller](examples/Example_06)

## Additional software

[ParaView](https://www.paraview.org/download/) is recommended for inspecting
spatial simulation outputs such as XDMF files.

For a detailed Czech guide, see [README_NAVOD.md](README_NAVOD.md).
