def run():

    # local imports
    from .geometry import build_geometry
    from .model import PassiveStorage
    import os

    # what folder is this file in?
    example_dir = os.path.dirname(__file__)

    # build geometry for simulation
    g_dir = os.path.join(example_dir, 'meshes')
    if not os.path.exists(os.path.join(g_dir, 'passive_storage.msh')):
        build_geometry(g_dir, example_dir)

    # create simulation
    r_dir = os.path.join(example_dir, 'results')
    sim = PassiveStorage(
        geometry_dir=g_dir,
        model_name='passive_storage',
        build_solvers=['unsteady'],
    )

    # simulate 1 month of PID-controlled charging
    sim.solve_unsteady(
        dt_min=0.01,
        dt_max=3600.0,
        dt_start=1.0,
        dt_ctrl_interval=(1.0, 2.0),
        h0_T_ref=18,
        atol=1e-8,
        rtol=1e-10,
        dt_xdmf=3600,
        T0=18,
        verbose=True,
        t_max=3600*24*31,
        T_pid_input_control=lambda t: 400.0,
        pid=(100.0, 0.1, 10.0),
        pid_power_lims=[0, 20000],
        T_amb_t=lambda t: 18.0,
        alpha_t=lambda t: 2.5,
        alpha_mem_t=lambda t: 1.0,
        probe_destinations=[{
            'type': 'csv',
            'file_name': 'unsteady.csv',
            'result_dir': r_dir,
            'flush': True,
        }],
    )

if __name__ == "__main__":
    run()

