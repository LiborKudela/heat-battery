# What is this example
This example simulates a sand-based thermal energy storage (  ) system that
is charged by PV-powered electrical cartridge heaters and provides heating to
an industrial building (Hall C3, Brno FME). It runs a large parametric study
over the storage geometry and operating conditions to evaluate different
design variants.

### Main idea
A cylindrical sand-filled storage core is surrounded by an insulation layer.
Electrical cartridge heaters with beam-fin spreaders are embedded inside the
sand to charge the storage using PV electricity. Heat is extracted from the 
storage through two paths:

1. **Through-hole pipes (THPs)** — pipes that pass through the storage body,
   equipped with beam-fin spreaders to increase heat transfer area. Air from
   the building is circulated through these pipes.
2. **Outer surface** — passive heat loss from the insulation surface into the
   room where the storage is located.

The heat transfer coefficient on the THP surfaces is dynamically adjusted so
that the storage delivers just enough heat to maintain the room at the
reference temperature (equithermal control). When the storage alone cannot
cover the full heating demand, a bivalent backup source fills the gap.

### Co-simulation with building model
The FEM heat conduction solver is coupled at each time step with a lumped
building thermal model (`HallC3`). The building model:

- Tracks the well-mixed room air temperature based on heat gains from the
  storage, heat losses to the ambient, and bivalent source contribution.
- Uses real meteorological data (ambient temperature, solar irradiance) from
  the NSRDB database for the selected location.
- Implements heating season logic — heating is only active during the heating
  season and respects heating pause schedules acording to Czech rules.

### Parametric study
The `run_example.py` script defines a `ParameterGrid` that sweeps over:

- **Storage size** (diameter = height): 1–5 m
- **Insulation thickness**: 0.6–1.0 m
- **Number of cartridge heaters**: 6 or 10
- **THP diameter and count**: controlled by `tht_d`, `tht_n_ratio`
- **Spreader fin dimensions**: for both cartridge and THP spreaders
- **PV peak power**: 30 kW

Each combination is meshed with Gmsh, solved transiently over 2 years, and
the results (temperatures, heat flows, energies, efficiencies) are stored in
a project database.

### Tracked quantities
The simulation records a rich set of probes at each output time step:

| Probe | Description |
|-------|-------------|
| `T[0]` | Temperature at the cartridge centre |
| `T_avg_sand` | Volume-averaged sand temperature |
| `T_avg_m` | Surface-averaged THP temperature |
| `T_avg_room` | Well-mixed room air temperature |
| `Q_pv` | Instantaneous PV power available |
| `Q_c` | Power actually delivered to cartridges |
| `Q_s2r_total` | Total heat flow from storage to room |
| `Q_bivalent` | Bivalent backup heating power |
| `H_storage` | Total thermal energy stored |
| `Eff_demand_covered` | Fraction of heating demand covered by storage |
| `Eff_pv_converted` | Fraction of PV energy converted to heat |

### Dashboard
The `view_project.py` script launches a Dash-based dashboard with
authentication and role-based access control for monitoring running jobs and
inspecting results interactively (time series, comparator charts, data
transforms).

### Files

| File | Purpose |
|------|---------|
| `geometry_v7.py` | Gmsh geometry builder (storage + cartridges + THPs + spreaders) |
| `model.py` | FEM simulation class and building thermal model (`HallC3`) |
| `run_example.py` | Parametric study definition and job management |
| `view_project.py` | Dash dashboard for result visualisation |
