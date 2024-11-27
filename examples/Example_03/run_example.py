def run():

    from .geometry import build_geometry
    from .model import THW_twowire
    import os

    # what folder is this file in?
    example_dir = os.path.dirname(__file__)

    # builde geometry for simulation
    g_dir = os.path.join(example_dir, 'meshes')
    #build_geometry(g_dir, example_dir)

    # create simulation
    r_dir = os.path.join(example_dir, 'results')
    sim = THW_twowire(
        geometry_dir=g_dir, 
        model_name='two_wire',
        
    )

    # run 5 second of THW simulation and save results
    sim.run_experiment(
        t_max=5,
        P=0.12,
        R_std=1.0,
        verbose=False,
        result_dir=r_dir,
        probes_file='unsteady.csv',
        dt_min=0.00000001,
        dt_start=0.0000001,
        dt_max=10.0,
        dt_ctrl_interval=(0.0025, 0.005),
        atol=1e-10,
        rtol=1e-12,
        )
    
if __name__ == "__main__":
    run()