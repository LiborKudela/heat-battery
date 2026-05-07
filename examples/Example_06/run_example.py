def run():

    # local imports
    from .model import C3_passive
    from .geometry_v4 import build_geometry
    import os

    dt_max = 1200
    verbose = False
    dt_ctrl_interval = (0.05, 0.1)

    example_dir = os.path.dirname(__file__)
    g_dir = os.path.join(example_dir, 'meshes')
    # build_geometry(
    #     name='mesh',
    #     dir=g_dir,
    #     verbosity=1,
    #     mesh_size_max=0.05,
    #     cartridge_mesh_size_min=0.0025,
    #     cartridge_mesh_grow_factor=0.8,
    #     mem_mesh_size_min = 0.01,
    #     mem_mesh_grow_factor = 0.8,
    #     mesh_size_from_curvature=18,
    #     fltk=False,
    #     symmetry=True,
    #     size=1,
    #     t_insulation=0.3,
    #     n_c=10,
    #     c_position=0.5,
    #     d_c=0.017,
    #     h_c_ratio=0.9,
    #     m_position=0.3,
    #     mem_in_sand=True,
    #     d_m=0.08,
    #     n_m_ratio=0.3,
    #     spreader_db=0.15,
    #     spreader_nb=3,
    #     spreader_tb=0.005,
    # )
    
    r_dir = os.path.join(example_dir, 'results')

    sim = C3_passive(
        geometry_dir=g_dir,
        model_name='mesh',
        build_solvers=['unsteady'],
    )
    sim.solve_unsteady(
        verbose=verbose,
        t_max=2000.0,
        dt_start=0.01,
        dt_min=0.000001,    
        dt_max=dt_max,
        dt_xdmf=3600,
        xdmf_file=None,
        result_dir=r_dir,
        #probes_file='unsteady_true.csv',
        probe_destinations=[
            {
                'type': 'csv',
                'result_dir': r_dir,
                'file_name': 'unsteady_true.csv',
                'flush': True
            }
        ],
        force_explicit_terms=False,
        dt_ctrl_interval=dt_ctrl_interval,
        T0=18,
        h0_T_ref=18,
        atol=1e-7,
        rtol=1e-8,

        #example specific parameters
        alpha_s=5.0,
        pv_peak=30000,
        Tc_limit=500.0,
        max_bivalent_power=30000,
        max_mem_power=30000,
        alpha_m_lims=(0.1, 30.0),
        #checkpoint_dt=100.0,
        #checkpoint_path=os.path.join(r_dir, 'checkpoint_test'),
    )


    sim1 = C3_passive(
        geometry_dir=g_dir,
        model_name='mesh',
        build_solvers=['unsteady'],
    )


    sim1.solve_unsteady(
        verbose=verbose,
        t_max=1000.0,
        dt_start=0.01,
        dt_min=0.000001,    
        dt_max=dt_max,
        dt_xdmf=1000.0,
        #xdmf_file='sim1_domain.xdmf',
        result_dir=r_dir,
        #probes_file='unsteady_1.csv',
        probe_destinations=[
            {
                'type': 'csv',
                'result_dir': r_dir,
                'file_name': 'unsteady_1.csv',
                'flush': True
            }
        ],
        force_explicit_terms=False,
        dt_ctrl_interval=dt_ctrl_interval,
        T0=18,
        h0_T_ref=18,
        atol=1e-7,
        rtol=1e-8,

        #example specific parameters
        alpha_s=5.0,
        pv_peak=30000,
        Tc_limit=500.0,
        max_bivalent_power=30000,
        max_mem_power=30000,
        alpha_m_lims=(0.1, 30.0),
        checkpoint_dt=100.0,
        checkpoint_dir=os.path.join(r_dir, 'checkpoint_test'),
    )

    sim2 = C3_passive(
        geometry_dir=g_dir,
        model_name='mesh',
        build_solvers=['unsteady'],
    )

    sim2.solve_unsteady(
        verbose=verbose,
        t_max=2000.0,
        dt_start=0.01,
        dt_min=0.000001,    
        dt_max=dt_max,
        dt_xdmf=1000.0,
        #xdmf_file='sim2_domain.xdmf',
        result_dir=r_dir,
        #probes_file='unsteady_2.csv',
        probe_destinations=[
            {
                'type': 'csv',
                'result_dir': r_dir,
                'file_name': 'unsteady_2.csv',
                'flush': True
            }
        ],
        force_explicit_terms=False,
        dt_ctrl_interval=dt_ctrl_interval,
        T0=18,
        h0_T_ref=18,
        atol=1e-7,
        rtol=1e-8,

        #example specific parameters
        alpha_s=5.0,
        pv_peak=30000,
        Tc_limit=500.0,
        max_bivalent_power=30000,
        max_mem_power=30000,
        alpha_m_lims=(0.1, 30.0),
        checkpoint_dt=None,
        load_initial_checkpoint=os.path.join(r_dir, 'checkpoint_test'),
    )
    sim.print_r0(sim.unsteady_probes.get_value('Q_amb'))
    sim2.print_r0(sim2.unsteady_probes.get_value('Q_amb'))

if __name__ == '__main__':
    run()