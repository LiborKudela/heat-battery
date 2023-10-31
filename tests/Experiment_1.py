from heat_battery.simulations import Experiment
from heat_battery.experiments import Experiment_data
from heat_battery.optimization import DataFitter, SteadyStateComparer
from scipy import optimize

import numpy as np
from mpi4py import MPI

sim = Experiment(dim = 2)
exp = sim.pseudoexperimental_data_steady()
fitter = DataFitter(sim, exp)
true_k = fitter.get_k()
true_k = [[20, 20], [0.04, 0.04], [2.0, 2.0], [2.0, 2.0], [0.42, 0.7]]
k0 = [[20.0, 20.0], [0.04, 0.04], [2.0, 2.0], [2.0, 2.0], [1.0, 2.0]]
k0 = [[20.0, 20.0], [0.04, 0.04], [2.0, 2.0], [2.0, 2.0], [0.4157130394942282, 0.7002761686885639]]
alpha = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0.0001, 0.0001]]).flatten()
fitter.optimise_steady_state_v3(max_iter=500, alpha=alpha, k0=k0)



#print(fitter.steady_state.get_k(1))
#loss = fitter.steady_state.generate_loss_for_material(1)
#print(loss([0.04, 0.042]))

