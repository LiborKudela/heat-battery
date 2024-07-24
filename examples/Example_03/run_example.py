def run():

    from .model import THW_twowire
    import os

    # what folder is this file in?
    example_dir = os.path.dirname(__file__)

    # create simulation
    sim = THW_twowire(
        geometry_dir=os.path.join(example_dir, 'meshes/two_wire'), 
        model_name='mesh',
        result_dir=os.path.join(example_dir, 'results'),
        dt_min=0.00000001,
        dt_start=0.0000001,
        dt_max=10.0,
        dt_ctrl_interval=(0.0025, 0.005),
    )

    # run 5 second of THW simulation and save results
    sim.run_experiment(
        t_max=5,
        P=0.12,
        R_std=1.0,
        T0=20, 
        T_guess=20, 
        verbose=False,
        )
    
if __name__ == "__main__":
    run()