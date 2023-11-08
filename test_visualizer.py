from heat_battery.data import Visualizer, SubPlotsPage, FigurePage
from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers

sim = Experiment(
    dim = 2,
    geometry_dir='meshes/experiment', 
    result_dir='results/experiment_test')
exp1 = sim.pseudoexperimental_data_steady(Qc=100, save_xdmf=False)
fitter = SteadyStateComparer(sim, [exp1])
true_k = fitter.get_k(4)
        
V = Visualizer()
V.register_page(SubPlotsPage("Conductivity", sim.material_plot, property="k", include_density=False))
V.register_page(SubPlotsPage("Capacity", sim.material_plot, property="cp", include_density=False))
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
    if i % 2 == 0:
        fitter.set_k(opt.get_k(), m=4)
        fitter.update()
        V.update_data()
    opt.print_state()
print(opt.get_k())

sim.close_results()