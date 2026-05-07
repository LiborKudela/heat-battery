
def run():

    # local imports
    from .geometry import build_geometry
    from .model import Experiment_v1
    import math
    import os

    # what folder is this file in?
    example_dir = os.path.dirname(__file__)

    # builde geometry for simulation
    g_dir = os.path.join(example_dir, 'meshes')
    build_geometry(g_dir, example_dir)

    # build simulation (only for unsteady mode)
    r_dir = os.path.join(example_dir, 'results')
    sim = Experiment_v1(
        geometry_dir=g_dir,
        model_name='inventor_v1',
        build_solvers=['unsteady'],
    )

    # simulate 1000 seconds of sin heat source and save resutls from probes
    sim.solve_unsteady(
        verbose=True,
        t_max = 100.0,
        dt_max=1.0,
        Qc_t=lambda t: 100+100*math.sin(2*math.pi*0.001*t),
        T_amb_t=lambda t: 20,
        probe_destinations=[{
            'type': 'csv',
            'file_name': 'unsteady.csv',
            'result_dir': r_dir,
            'flush': True,
        }],
        )

if __name__ == "__main__":
    run()