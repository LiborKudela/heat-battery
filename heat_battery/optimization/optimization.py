from typing import List
import pandas as pd
import plotly.graph_objects as go
from mpi4py import MPI
import os
import numpy as np
from ..simulations import Experiment_v1
from ..data import Experiment_data
from .derivatives import AdjointDerivative, Point_wise_lsq_objective, ForwardDerivative_dudk
from scipy.linalg import block_diag

class SteadyStateComparer:
    def __init__(self, sim: Experiment_v1, exp_data: List[Experiment_data]):
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
        self.sim.mats.set_property_values(k, 'k', m)

    def generate_loss_gradient_for_material(self, m=None, batch_size=None):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)

        controls = [self.sim.mats[i].k.fem_const for i in m]
        transform_jac = block_diag(*[self.sim.mats[i].k.transform_jac for i in m])
        points = self.sim.T_probes_coords
        true_vals = None
       
        J = Point_wise_lsq_objective(points, self.sim.T, controls, true_vals)
        adjoint_derivative = AdjointDerivative(J, controls, self.sim.Fss, self.sim.solve_steady, self.sim.T)

        def gradient(k):
            original_k = self.get_k(m=m)
            original_T = self.sim.T.x.array.copy()

            self.set_k(k, m=m)
            g = np.zeros_like(original_k)
            l = 0.0

            # select batch
            if batch_size is not None:
                idx_batch = np.random.choice(len(self.exp_data), size=batch_size, replace=False)
                idx_batch = MPI.COMM_WORLD.bcast(idx_batch)
            else:
                idx_batch = np.arange(len(self.exp_data))

            for i in idx_batch:
                exp_data = self.exp_data[i]
                T_amb = exp_data.steady_state_mean['16 - Ambient [°C]']
                Qc = exp_data.steady_state_mean['Power [W]']
                J.true_values = exp_data.steady_state_mean[self.sim.T_probes_names].to_numpy()
                adjoint_derivative.forward(Qc=Qc, T_amb=T_amb, save_xdmf=False)
                _g = adjoint_derivative.compute_gradient()
                g += _g.dot(transform_jac)
                l += adjoint_derivative.compute_loss()

            self.set_k(original_k, m=m)
            self.sim.T.x.array[:] = original_T
            return g, l

        return gradient
    
    def generate_solution_jacobian(self, m=None, full_domain=False):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)

        controls = [self.sim.mats[i].k.fem_const for i in m]
        transform_jac = block_diag(*[self.sim.mats[i].k.transform_jac for i in m])
        points = self.sim.T_probes_coords if not full_domain else None

        forward_jacobian = ForwardDerivative_dudk(controls, self.sim.Fss, self.sim.solve_steady, self.sim.T, p_coords=points)
        
        def jacobian(k):
            original_k = self.get_k(m=m)
            original_T = self.sim.T.x.array.copy()

            self.set_k(k, m=m)
            j = np.zeros((len(k), len(points)*len(self.exp_data)))

            for i in range(len(self.exp_data)):
                exp_data = self.exp_data[i]
                T_amb = exp_data.steady_state_mean['16 - Ambient [°C]']
                Qc = exp_data.steady_state_mean['Power [W]']
                forward_jacobian.forward(Qc=Qc, T_amb=T_amb, save_xdmf=False)
                _j = forward_jacobian.compute_jacobian()
                z = i*_j.shape[1]
                j[:, z:z+_j.shape[1]] = transform_jac.T.dot(_j)
                
            self.set_k(original_k, m=m)
            self.sim.T.x.array[:] = original_T

            return j
        return jacobian
    
    def generate_loss_component_jacobian(self, m=None):
        return self.generate_solution_jacobian(m=m, full_domain=False)

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
        self.total_error = np.sum(self.total_square_error)/2

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
        data["Rel Error"] = data["Difference"]/(data["Experiment Mean"]-T_amb)
        data = data.dropna()
        return data, domain_data
        
    def compare_plot(self):
        if MPI.COMM_WORLD.rank == 0 and not self.data[0].empty:
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
        if self.domain_data[0] is not None:
            return self.sim.domain_state_plot(T=self.domain_data)
        
    def print_data(self):
        if MPI.COMM_WORLD.rank == 0:
            print(self.data)

    def save_data(self):
        if MPI.COMM_WORLD.rank == 0:
            self.data.to_csv(os.path.join(self.sim.result_dir, "comprarer.csv"))

class MultiSimSteadyStateComparer:
    "Allows to optimise agains multiple geometries."
    def __init__(self, comparers: List[SteadyStateComparer]):
        self.comparers = comparers

