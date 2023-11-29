from typing import List
import pandas as pd
import plotly.graph_objects as go
from mpi4py import MPI
import os
import numpy as np
from ..simulations import Experiment
from ..data import Experiment_data
from .derivatives import AdjointDerivative, Point_wise_lsq_objective
from scipy.linalg import block_diag

class SteadyStateComparer:
    def __init__(self, sim: Experiment, exp_data: List[Experiment_data]):
        self.sim = sim
        if isinstance(exp_data, Experiment_data):
            exp_data = [exp_data]

        self.exp_data = exp_data
        self.n = len(self.exp_data)
        self.data = [pd.Series()]*self.n
        self.domain_data = [None]*self.n 
        self.total_abs_error = [0.0]*self.n 
        self.total_square_error = [0.0]*self.n 
        self.total_max_error = [0.0]*self.n
        self.total_error = 0.0

    def get_k(self, m=None):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)
        k = []
        for i in m:
            k.append(self.sim.mats[i].k.get_values())
        return np.concatenate(k)
    
    def set_k(self, k, m=None):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)
        start_idx = 0
        for i in m:
            end_idx = start_idx + self.sim.mats[i].k.n_values
            self.sim.mats[i].k.set_values(k[start_idx:end_idx])
            start_idx = end_idx

    def generate_gradient_for_material(self, m=None):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)

        controls = [self.sim.mats[i].k.fem_const for i in m]
        transform_jac = block_diag(*[self.sim.mats[i].k.transform_jac for i in m])
        points = self.sim.T_probes_coords
        true_vals = self.exp_data[0].steady_state_mean[self.sim.T_probes_names].to_numpy()
       
        J = Point_wise_lsq_objective(points, self.sim.T, controls, true_vals)
        self.adjoint_derivative = AdjointDerivative(J, controls, self.sim.Fss, self.sim.steady_solver, self.sim.T)

        def objective(k):
            original_k = self.get_k(m=m)
            original_T = self.sim.T.x.array.copy()

            self.set_k(k, m=m)
            g, l = self.adjoint_derivative.compute_gradient()
            g = g.dot(transform_jac)

            self.set_k(original_k, m=m)
            self.sim.T.x.array[:] = original_T
            return g, l

        return objective

    def loss_function(self, k, m=None):
        # save original state
        original_k = self.get_k(m=m)
        original_T = self.sim.T.x.array.copy()

        # calculate new state
        self.set_k(k, m=m)
        self.update()
        err = self.total_error

        # return to the original state
        self.set_k(original_k, m=m)
        self.sim.T.x.array[:] = original_T
        return err.copy() 

    def generate_loss_for_material(self, m):
        def restricted_loss(k):
            return self.loss_function(k, m=m)    
        return restricted_loss
    
    def generate_getter_for_material(self, m):
        def restricted_getter():
            return self.get_k(m=m)
        return restricted_getter
    
    def generate_setter_for_material(self, m):
        def restricted_setter(k):
            self.set_k(k, m=m)
        return restricted_setter

    def update(self):
        for i, exp_data in enumerate(self.exp_data):
            self.data[i], self.domain_data[i] = self.compare_steady_state(exp_data)
            self.total_abs_error[i] = self.data[i]["Abs Error"].sum()
            self.total_square_error[i] = self.data[i]["Square Error"].sum()
            self.total_max_error[i] = self.data[i]["Abs Error"].max()
        self.total_error = np.sum(self.total_square_error)

    def compare_steady_state(self, exp_data):
        T_amb = exp_data.steady_state_mean['16 - Ambient [°C]']
        Qc = exp_data.steady_state_mean['Power [W]']
        s_sim = self.sim.solve_steady(Qc=Qc, T_amb=T_amb, save_xdmf=False) # this must run on all ranks
        domain_data = self.sim.T.copy()
        exp_mean = exp_data.steady_state_mean
        exp_std = exp_data.steady_state_std
        data = pd.concat([exp_mean, exp_std, s_sim], axis=1)
        data["Difference"] = data["Experiment Mean"] - data["Simulation"]
        data["Square Error"] = data["Difference"].pow(2)
        data["Abs Error"] = data["Difference"].abs()
        data = data.dropna()
        return data, domain_data
        
    def compare_plot(self):
        if MPI.COMM_WORLD.rank == 0:
            figs = []
            for data in self.data:
                fig = go.Figure()
                fig.add_bar(x=data.index, y=data["Experiment Mean"], name='Experiment')
                fig.add_bar(x=data.index, y=data["Simulation"], name='Simulation')
                figs.append(fig)
                #TODO: add titles to compare charts
            return figs
        
    def material_plot(self, m=None, property='k', include_density=False):
        return self.sim.material_plot(m=m, property=property, include_density=include_density, T=self.domain_data)
    
    def domain_state_plot(self):
        return self.sim.domain_state_plot(T=self.domain_data)
        
    def print_data(self):
        if MPI.COMM_WORLD.rank == 0:
            print(self.data)

    def save_data(self):
        if MPI.COMM_WORLD.rank == 0:
            self.data.to_csv(os.path.join(self.sim.result_dir, "comprarer.csv"))