# What is this example
This is the introductory example of the HeatBattery framework. It simulates a
small-scale cylindrical sand heat storage with a single electrical cartridge
heater, matching the geometry of a real laboratory experiment. It demonstrates
how to build a complete transient thermal simulation — from mesh generation to
probe output — in just three short files.

### Physical setup
The storage is a double-walled stainless-steel cylinder filled with sand. A
single cartridge heater sits in the centre along the vertical axis. The
construction consists of (from centre outward):

1. **Heated cartridge** — an electrically heated rod (diameter 14 mm, heated
   length 191 mm) with a threaded top section and bolt head.
2. **Sand fill** — the thermal storage medium surrounding the cartridge.
3. **Inner steel wall** — thin stainless-steel liner (0.6 mm).
4. **Insulation layer** — mineral insulation between inner and outer walls
   (29 mm on the side, 5 mm on top).
5. **Outer steel wall** — outer shell (0.5 mm).
6. **Contact layer** — a thin resistive layer around the cartridge surface
   modelling imperfect thermal contact with the sand.

A steel bottom plate and top plate close the vessel. The outer surface of the
entire assembly is exposed to ambient air via convective cooling.

### Geometry
The geometry is generated programmatically with Gmsh in `geometry.py`. It
supports both 2D axisymmetric (default) and full 3D modes, with optional
half- or quarter-symmetry for the 3D case. The Jacobian factor is set
accordingly (`2*pi*r` for 2D axisymmetric, `1` for full 3D).

The mesh is refined near the cartridge surface using a distance-based size
field to resolve the steep temperature gradients there.

### Simulation model
The model (`model.py`) derives from the base `Simulation` class and defines:

- **Source term** — a `UniformHeatSource` applied to the `"heated cartridge"`
  subdomain. Its power is driven by a time-dependent function `Qc(t)`.
- **Boundary term** — an `AmbientCooling` term on the `"outer_surface"`,
  parameterised by ambient temperature `T_amb(t)` and heat transfer
  coefficient `alpha(t)`.

The model supports both steady-state and transient solvers. In the default
run, the cartridge power follows a sinusoidal profile
`Qc(t) = 100 + 10·sin(2π·0.01·t)` W, and the ambient temperature is fixed
at 10 °C.

### Tracked quantities
Temperature probes are placed at 15 points inside the storage arranged in a
3×3 radial-axial grid plus cartridge surface and outer cover locations:

| Probe | Description |
|-------|-------------|
| `T[0]`–`T[2]` | Top row — three radial distances from cartridge |
| `T[3]`–`T[5]` | Middle row |
| `T[6]`–`T[8]` | Bottom row |
| `T[9]`–`T[11]` | Cartridge surface (A, B, C) |
| `T[12]`–`T[14]` | Outer cover at three heights |
| `heat` | Total thermal energy stored in the domain [J] |

### Running the example
The `run_example.py` script performs the following steps:

1. Generates a 2D axisymmetric mesh via Gmsh.
2. Builds the FEM simulation and both solver types.
3. Runs a 100-second transient simulation with adaptive time stepping.
4. Writes probe data to a CSV file in the `results/` folder.

### Files

| File | Purpose |
|------|---------|
| `geometry.py` | Gmsh geometry builder (2D axi or 3D, with symmetry options) |
| `model.py` | FEM simulation class with heat source and ambient cooling terms |
| `run_example.py` | Mesh generation, simulation setup, and transient solve |
