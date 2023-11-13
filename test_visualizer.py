from heat_battery.data import Visualizer, FigurePage, ResampingFigurePage, Experiment_data, PseudoExperimentalData
from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
from mpi4py import MPI

sim = Experiment(
    dim = 2,
    geometry_dir='meshes/experiment', 
    result_dir='results/experiment_test')

# fake experimental data
Qc = 100
T_amb = 20
res = sim.solve_steady(Qc=Qc, T_amb=T_amb,save_xdmf=False)
exp_fake = PseudoExperimentalData()
exp_fake.feed_steady_state(res, Qc=Qc, T_amb=T_amb)

# create 
fitter = SteadyStateComparer(sim, [exp_fake])
true_k = fitter.get_k(4)
exp_real = Experiment_data('data/experiments/20231009_third/Test_TF24_Third_measurement_054411.csv')

def two_big_figs():
    fig = exp_real.plot_data_series()
    if MPI.COMM_WORLD.rank == 0:
        return [fig, fig]

V = Visualizer()
V.register_page(ResampingFigurePage("Conductivity", sim.material_plot, property="k", include_density=False))
V.register_page(ResampingFigurePage("TwoBigFigs", two_big_figs))
#V.register_page(ResampingFigurePage("StaticBigData", exp_real.plot_data_series))
V.build_app()
V.start_app()
V.update_data()

k0 = true_k.copy()
k0 *= 1.1
loss = fitter.generate_loss_for_material(4)
opt = optimizers.ADAM(loss=loss, k0=k0, alpha=1e-3)

for i in range(300):
    opt.step()
    opt.alpha *= 0.995
    if i % 10 == 0:
        fitter.set_k(opt.get_k(), m=4)
        fitter.update()
        V.update_data()
    #opt.print_state()
print(opt.get_k())

sim.close_results() 