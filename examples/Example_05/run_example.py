def run():

    # local imports
    from .model import C3_passive
    from .geometry_v7 import build_geometry
    from heat_battery.data import meteodata
    from heat_battery.simulations import (
        Project, new_jobs_generator,
        ParameterGrid, ParameterList, NoNumericalEffect, ParameterEvaluation
    )
    import os
    import time
    import pandas as pd
    p_grid = ParameterGrid(dict(
        mesh_p=ParameterGrid(dict(
            name='mesh',
            dir=os.path.join("meshes/C3_passive"),
            verbosity=NoNumericalEffect(4), # this will not affect the simulation results
            fltk=NoNumericalEffect(False),
            size=ParameterList([1, 3, 5]),  # three different sizes of the geometry
            t_insulation=ParameterList([0.5, 1.0]),
            mesh_size_max=0.1,
            mesh_size_from_curvature=18,
            cartridge_n=ParameterList([4, 10]), 
            cartridge_d_ratio=0.5,
            cartridge_diameter=0.014,
            cartridge_h_ratio=0.9,
            cartridge_spreader_lb=ParameterList([0.02, 0.06]),
            cartridge_spreader_nb=3,
            cartridge_spreader_tb=0.005,
            cartridge_spreader_mesh_size_min=0.0025,
            cartridge_spreader_mesh_grow_factor=0.8,
            tht_in_sand=True,
            tht_d=ParameterList([0.04, 0.08]),
            tht_d_ratio=ParameterList([0.1, 0.15]),
            tht_n_ratio=ParameterList([0.2, 0.4, 0.6]),
            thp_mesh_size_min=0.0025,
            thp_mesh_grow_factor=0.8,
            tht_spreader_h_ratio=0.9,
            tht_spreader_lb=ParameterList([0.02, 0.06]),
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
        ),
        sim_p=ParameterGrid(dict(
            
            #general simulation class parameters

            verbose=NoNumericalEffect(False),
            t_max=2*365*24*3600, #2 years test
            dt_start=0.01,
            dt_min=0.000001,    
            dt_max=1200,
            dt_xdmf=3600,
            xdmf_file=None,
            force_explicit_terms=False,
            dt_ctrl_interval=(1.0, 2.0),
            T0=18,
            h0_T_ref=18,
            atol=1e-6,
            rtol=1e-7,
            checkpoint_dt=NoNumericalEffect(7*24*3600),
            #checkpoint_dt=NoNumericalEffect(3600),

            #example specific parameters
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
            datetime_start='2007-6-1 00:00:00.0', # y-m-d h:m:s
            )       
        ),
        )
    )

    project = Project(
        'project_example_05',
        #if_exists='override',
    )
    
    # jobs_gen = new_jobs_generator(
    #     sim_class=C3_passive, 
    #     mesh_builder=build_geometry,
    #     group_name='Example_5',
    #     group_priority=0,
    #     runner='solve_unsteady', # to whicn method to pass the simulation parameters
    #     p_grid=p_grid
    # )

    #project.add_jobs(jobs_gen)

    while True:
        df = project.get_jobs(as_dataframe=True)
        print(df.head())
        running_jobs = df[df['status'].str.contains('RUNNING')]
        if len(running_jobs) > 0:
            print('Running jobs:')
            print(running_jobs)
            project.mark_interrupted_jobs()
        time.sleep(10)

if __name__ == '__main__':
    run()