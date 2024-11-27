from heat_battery.visualization import pages


def run(skip_geo_build=False):

    # local imports
    from .geometry import build_geometry
    from .model import PassiveStorage
    from heat_battery.data import Visualizer
    import os

    # what folder is this file in?
    example_dir = os.path.dirname(__file__)

    # builde geometry for simulation
    
    g_dir = os.path.join(example_dir, 'meshes')
    if not os.path.exists(os.path.join(g_dir, 'passive_storage.msh')):
        build_geometry(g_dir, example_dir)

    # create simulation
    r_dir = os.path.join(example_dir, 'results')
    sim = PassiveStorage(
        result_dir=r_dir,
        model_name='passive_storage',
    )

    # define pages in Live post procesing server
    V = Visualizer()
    V.register_page(pages.FigurePage("Unsteady", sim.probes_time_plot, 
        x='t_sim', 
        y=["heat_loss", "heat_loss_mem", "heat", "Ts_avg", "Tc_avg", "Tm_avg", "power"]
        )
    )
    V.register_page(pages.FigurePage("lmbd", sim.material_plot, property='k'))
    V.register_page(pages.FigurePage("cp", sim.material_plot, property='cp'))
    V.register_page(pages.FigurePage("rho", sim.material_plot, property='rho'))
    V.build_app()
    V.start_app()

    # simulate 1 week of charging with the webserver above attatched
    sim.solve_unsteady(
        dt_min=0.01,
        dt_max=3600.0,
        dt_start=1.0,
        dt_ctrl_interval=(1.0, 2.0),
        geometry_dir=g_dir, 
        model_name='passive_storage',
        result_dir=r_dir,
        h0_T_ref=18,
        atol=1e-8,
        rtol=1e-10,
        dt_xdmf=3600,
        T0=18,

        verbose=True,
        t_max=3600*24*31,
        T_pid_input_control=lambda t: 400.0,
        pid=(100.0, 0.1, 10.0),
        pid_power_lims = [0, 20000],
        T_amb_t=lambda t: 18.0,
        alpha_t=lambda t: 2.5,
        alpha_mem_t=lambda t: 1.0,
        call_back=V.update_data,
        call_back_each_step=10,
        )

if __name__ == "__main__":
    run()

