"""Run Example 05 directly (no Project/job queue) for a single size=5 case.

Picks one concrete parameter set out of the parametric study defined in
``run_example.py`` -- specifically the ``size=5`` storage variant -- builds the
geometry locally (with mesh caching) and runs the transient simulation in the
current process.

Run as a module from the project root:

    python -m examples.Example_05.run_example_size5
"""


def run():

    import os

    from .model import C3_passive
    from .geometry_v7 import build_geometry # geometry generator
    from heat_battery.data import meteodata # meteorological data fetcher
    from heat_battery.geometry import CachedGeometryBuilder # mesh caching helper

    example_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_cache_dir = os.path.join(example_dir, "meshes", "C3_passive")
    results_dir = os.path.join(example_dir, "results_size5_direct")
    checkpoint_dir = os.path.join(results_dir, "checkpoint")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # If a previous run left a checkpoint behind, resume from it; otherwise
    # start from t=0. The presence of metadata.json is what
    # Simulation.load_metadata_checkpoint expects.
    load_initial_checkpoint = (
        checkpoint_dir
        if os.path.isfile(os.path.join(checkpoint_dir, "metadata.json"))
        else None
    )

    # Mesh (geometry) parameters: a single concrete case with size=5
    mesh_p = dict(
        name='mesh',
        dir=mesh_cache_dir,
        verbosity=4,
        fltk=False,
        size=5,
        t_insulation=0.5,
        mesh_size_max=0.1,
        mesh_size_from_curvature=18,
        cartridge_n=4,
        cartridge_d_ratio=0.5,
        cartridge_diameter=0.014,
        cartridge_h_ratio=0.9,
        cartridge_spreader_lb=0.02,
        cartridge_spreader_nb=3,
        cartridge_spreader_tb=0.005,
        cartridge_spreader_mesh_size_min=0.0025,
        cartridge_spreader_mesh_grow_factor=0.8,
        tht_in_sand=True,
        tht_d=0.04,
        tht_d_ratio=0.1,
        tht_n_ratio=0.2,
        thp_mesh_size_min=0.0025,
        thp_mesh_grow_factor=0.8,
        tht_spreader_h_ratio=0.9,
        tht_spreader_lb=0.02,
        tht_spreader_nb=3,
        tht_spreader_tb=0.005,
        thp_spreader_mesh_size_min=0.0025,
        thp_spreader_mesh_grow_factor=0.8,
        thp_surface_segments=10,
        sand_material="SandTheory",
        insulation_material="Standard_insulation",
        thp_spreader_material="Steel04",
        cartridge_material="Steel04",
        cartridge_spreader_material="Steel04",
    )

    # Simulation parameters: same numerical settings as run_example.py.
    sim_p = dict(
        verbose=False,
        t_max=2 * 365 * 24 * 3600,  # 2 years
        dt_start=0.01,
        dt_min=0.000001,
        dt_max=1200,
        dt_xdmf=24*3600, # save temperature field every day
        xdmf_file='unsteady.xdmf',
        result_dir=results_dir,
        force_explicit_terms=False,
        dt_ctrl_interval=(1.0, 2.0),
        T0=18,
        h0_T_ref=18,
        atol=1e-6,
        rtol=1e-7,

        # Example-specific (consumed by C3_passive.solve_unsteady)
        T_room_ctrl_interval=(0.1, 0.2),
        converge_tol_T_room=0.1,
        converge_tol_Q_amb=10,
        alpha_s=5.0,
        alpha_m_lims=(0.1, 20.0),
        location=meteodata.locations['Brno-FME'],
        pv_peak=30000,
        Tc_limit=500.0,
        max_bivalent_power=30000,
        max_mem_power=30000,
        datetime_start='2007-6-1 00:00:00.0',  # y-m-d h:m:s

        # Direct CSV probe output (no PostgreSQL project involved). The CSV
        # destination participates in checkpointing -- on resume it truncates
        # back to the byte offset stored in the checkpoint.
        probe_destinations=[
            {
                'type': 'csv',
                'result_dir': results_dir,
                'file_name': 'unsteady.csv',
            }
        ],

        # Checkpointing: dump full simulation state every `checkpoint_dt`
        # seconds (1 week). If `checkpoint_dir`
        # already contains a previous checkpoint, the run resumes from it
        # automatically.
        checkpoint_dt=7 * 24 * 3600,
        checkpoint_dir=checkpoint_dir,
        load_initial_checkpoint=load_initial_checkpoint,
    )

    # Resolve / build the mesh files (cached on disk, keyed by mesh_p hash).
    mesh_builder = CachedGeometryBuilder(build_geometry, mesh_cache_dir)
    geometry_dir, model_name = mesh_builder.get_single(**mesh_p)

    # Instantiate the FEM model and run the transient simulation.
    sim = C3_passive(geometry_dir=geometry_dir, model_name=model_name)
    sim.solve_unsteady(**sim_p)


if __name__ == '__main__':
    run()
