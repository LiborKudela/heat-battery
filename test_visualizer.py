from heat_battery.data import Visualizer, pages, Experiment_data, PseudoExperimentalData
from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
from heat_battery.optimization.derivatives import taylor_test
from mpi4py import MPI
import numpy as np

sim = Experiment(
    dim = 2,
    T_guess=200,
    geometry_dir='meshes/experiment', 
    result_dir='results/experiment_test')
m = None

# fake experimental data to test against
Qc = 50
T_amb = 20
res = sim.solve_steady(Qc=Qc, T_amb=T_amb,save_xdmf=False)
exp_fake_1 = PseudoExperimentalData()
exp_fake_1.feed_steady_state(res, Qc=Qc, T_amb=T_amb)

Qc = 100
T_amb = 20
res = sim.solve_steady(Qc=Qc, T_amb=T_amb,save_xdmf=False)
exp_fake_2 = PseudoExperimentalData()
exp_fake_2 .feed_steady_state(res, Qc=Qc, T_amb=T_amb)

# create 
fitter = SteadyStateComparer(sim, [exp_fake_1, exp_fake_2])
true_k = fitter.get_k(m)
#exp_real = Experiment_data('data/experiments/20231009_third/Test_TF24_Third_measurement_054411.csv')

V = Visualizer()
V.register_page(pages.FigurePage("Conductivity", fitter.material_plot, property="k", include_density=True))
V.register_page(pages.FigurePage("Compare", fitter.compare_plot))
V.register_page(pages.VtkMeshPage("VTK_TEST", fitter.domain_state_plot))
#V.register_page(pages.ResampingFigurePage("StaticBigData", exp_fake.data_series_plot))

V.build_app()
V.start_app()
V.stay_alive(1)

k0 = true_k.copy()
k0 *= 0.5+np.random.rand(*k0.shape)
k0 = MPI.COMM_WORLD.bcast(k0) # each process needs the same guess
grad = fitter.generate_gradient_for_material(m, batch_size=1)


opt = optimizers.ADAM(loss="none", grad=grad, grad_returns_loss=True, k0=k0, alpha=1e-2, k_min=0.01)

for i in range(1500):
    opt.step()
    opt.alpha *= 0.999
    if opt.loss_value < 50 and opt.alpha > 2e-4:
        opt.alpha = 2e-4
    if i % 50 == 0:
        fitter.set_k(opt.get_k(), m=m)
        fitter.update()
        V.update_data()
    opt.print_state()

#V.stay_alive(100)
sim.close_results() 