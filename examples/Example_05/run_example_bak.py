def run():

    # local imports
    from .model import C3_passive
    from .geometry_v4 import build_geometry
    from heat_battery.simulations.sweep import CachedSweep, ParameterGrid, ListParameter
    import os

    example_dir = os.path.dirname(__file__)

    p_gen = ParameterGrid(dict(
        mesh_p=ParameterGrid(dict(
            name='mesh',
            dir=os.path.join(example_dir, "meshes/C3_passive"),
            verbosity=0,
            mesh_size_max=0.2,
            cartridge_mesh_size_min=0.0025,
            cartridge_mesh_grow_factor=0.8,
            mem_mesh_size_min = 0.01,
            mem_mesh_grow_factor = 0.8,
            mesh_size_from_curvature=18,
            fltk=False,
            symmetry=True,
            size=ListParameter([5]),
            t_insulation=0.6,
            n_c=10,
            c_position=0.5,
            d_c=0.017,
            h_c_ratio=0.9,
            m_position=ListParameter([0.1, 0.2, 0.3, 0.4, 0.5]),
            mem_in_sand=True,
            d_m=0.08,
            n_m_ratio=ListParameter([0.2, 0.3, 0.4]),
            spreader_db=0.15,
            spreader_nb=3,
            spreader_tb=0.005,
            )
        ),
        sim_p=ParameterGrid(dict(
            verbose=False,
            t_max=3600*24*365*4, #4years
            xdmf_file=None,
            force_explicit_terms=False,
            pv_peak=30000,
            alpha_m_lims=(0.1, 30.0),
            datetime_start='2007-6-1 00:00:00.0',
            )
        ),
        )
    )

    cs = CachedSweep(C3_passive, build_geometry, p_gen, example_dir)
    cs.override_sim_defaults(
        dt_min=0.000001,
        dt_max=1200,
        dt_start=0.01,
        dt_xdmf=3600,
        dt_ctrl_interval=(1.0, 2.0),
        h0_T_ref=18,
        T0=18,
        atol=1e-7,
        rtol=1e-8,
        )
    #cs.prebuild_meshes(parallel=True, error_on_fail=False)
    cs.loop()
    
if __name__ == '__main__':
    run()
