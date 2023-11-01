from heat_battery.simulations import Experiment
from heat_battery.data import Experiment_data
from heat_battery.optimization import DataFitter, SteadyStateComparer
from scipy import optimize

import numpy as np
from mpi4py import MPI

sim = Experiment(dim = 2)
exp = sim.pseudoexperimental_data_steady()
fitter = DataFitter(sim, exp)
true_k = fitter.get_k()
k0 = [[20.0], [0.04, 0.04], [2.0], [2.0], [0.4157130394942282, 0.7002761686885639, 0.7]]
k0 = [[20.0], [0.04, 0.04], [2.0], [2.0], [0.5257129703536888, 0.5302761370038594, 0.5899998258155854]]
k0 = [[20.0], [0.04, 0.05], [2.0], [2.0], [0.29073105551270717, 0.49288810919649617, 0.48602103357527493]]
alpha = 1e-4
fitter.optimise_steady_state_v2(max_iter=500, alpha=alpha, k0=k0, beta_1=0.0, beta_2=0.0)



#print(fitter.steady_state.get_k(1))
#loss = fitter.steady_state.generate_loss_for_material(1)
#print(loss([0.04, 0.042]))

