from heat_battery.simulations import Experiment
from heat_battery.data import Experiment_data
from heat_battery.optimization import DataFitter, SteadyStateComparer
from scipy import optimize

import numpy as np
from mpi4py import MPI

sim = Experiment(dim = 2)
exp = sim.pseudoexperimental_data_steady()
exp_real = Experiment_data('data/experiments/20231009_third/Test_TF24_Third_measurement_054411.csv')
fitter = DataFitter(sim, exp_real)
fitter.steady_state.print_data()
#true_k = fitter.get_k()
#true_k = [[20, 20], [0.04, 0.04], [2.0, 2.0], [2.0, 2.0], [0.42, 0.7]]
#k0 = [[20.0, 20.0], [0.04, 0.04], [2.0, 2.0], [2.0, 2.0], [1.0, 2.0]]
k0 = [[16.36238981024193, 16.536933101685726], [0.01, 0.3795482227790676], [2.3836363473746762, 2.8151312724201634], [0.5149101878143033, 0.6917468917453583], [0.053775803389092845, 0.5536663349442154, 0.5084728138518837]]
k0 = [[17.098470353156618], [0.01, 0.5835987827827844], [5.962876729291803], [0.1464233119387374], [0.17187061834192716, 0.5500053091546767, 0.01]]
alpha = 0.001
fitter.optimise_steady_state_v3(max_iter=1500, alpha=alpha, k0=k0, beta_1=0.8, beta_2=0.9)
fitter.steady_state.print_data()

sim.mats[0].plot()

#print(fitter.steady_state.get_k(1))
#loss = fitter.steady_state.generate_loss_for_material(1)
#print(loss([0.04, 0.042]))

