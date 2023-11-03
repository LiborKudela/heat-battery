from heat_battery.simulations import Experiment
from heat_battery.data import Experiment_data
from heat_battery.optimization import SteadyStateComparer, optimizers

import numpy as np
from mpi4py import MPI

sim = Experiment(dim = 2)
exp = sim.pseudoexperimental_data_steady()
fitter = SteadyStateComparer(sim, [exp])
true_k = fitter.get_k()
k0 = true_k[-3:].copy()
k0 += 0.1

loss = fitter.generate_loss_for_material(4)
opt = optimizers.ADAM(loss=loss, k0=k0, alpha=5e-4)

for i in range(1000):
    opt.step()
    opt.print_state()

if MPI.COMM_WORLD.rank == 0:
    print(true_k[-3:])  
    print(k0)
    print(opt.get_k())

