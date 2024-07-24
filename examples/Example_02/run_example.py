
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
        result_dir=r_dir,
        build_solvers=['unsteady'],
        atol=1e-8,
        rtol=1e-10,
    )

    # simulate 1000 seconds of sin heat source and save resutls from probes
    sim.solve_unsteady(
        verbose=False,
        t_max = 1000.0,
        Qc_t=lambda t: 100+100*math.sin(2*math.pi*0.001*t),
        T_amb_t=lambda t: 20,
        )

if __name__ == "__main__":
    run()