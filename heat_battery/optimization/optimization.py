from typing import List
import pandas as pd
import plotly.graph_objects as go
from mpi4py import MPI
import os
import numpy as np
from ..simulations import Simulation
from .derivatives import AdjointDerivative, Point_wise_lsq_objective, ForwardDerivative_dudk
from scipy.linalg import block_diag

class SteadyStateComparer:
    def __init__(self, sim: Simulation, inputs, outputs):
        self.sim = sim
        assert len(inputs) == len(outputs), "not the same length"
        self.n = len(inputs)
        self.inputs = inputs
        self.outputs = outputs
        
        self.data = [pd.Series()]*self.n
        self.domain_data = [None]*self.n 

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
                idx_batch = np.random.choice(self.n, size=batch_size, replace=False)
                idx_batch = MPI.COMM_WORLD.bcast(idx_batch)
            else:
                idx_batch = np.arange(self.n)

            try:
                for i in idx_batch:
                    input_data = self.inputs[i]
                    output_data = self.outputs[i]
                    J.true_values = output_data
                    adjoint_derivative.forward(**input_data)
                    _g = adjoint_derivative.compute_gradient()
                    g += _g.dot(transform_jac)
                    l += adjoint_derivative.compute_loss()
            except Exception as e:
                raise e
            finally:
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
        n_points = len(points) if not full_domain else self.sim.T.x.array.shape[0]

        forward_jacobian = ForwardDerivative_dudk(controls, self.sim.Fss, self.sim.solve_steady, self.sim.T, p_coords=points)
        
        def jacobian(k):
            original_k = self.get_k(m=m)
            original_T = self.sim.T.x.array.copy()

            self.set_k(k, m=m)
            j = np.zeros((len(k), n_points*self.n))

            for i in range(self.n):
                input_data = self.inputs[i]
                forward_jacobian.forward(**input_data)
                _j = forward_jacobian.compute_jacobian()
                z = i*_j.shape[1]
                j[:, z:z+_j.shape[1]] = transform_jac.T.dot(_j)
                
            self.set_k(original_k, m=m)
            self.sim.T.x.array[:] = original_T

            return j
        return jacobian
    
    def generate_loss_component_jacobian(self, m=None):
        return self.generate_solution_jacobian(m=m, full_domain=False)
    
    def generate_loss_for_material(self, m):
        if m is None:
            m = np.arange(len(self.sim.mats))
        m = np.atleast_1d(m)

        points = self.sim.T_probes_coords
        controls = [self.sim.mats[i].k.fem_const for i in m]
        true_vals = None
        J = Point_wise_lsq_objective(points, self.sim.T, controls, true_vals)

        def loss_function(k):
            original_k = self.get_k(m=m)
            original_T = self.sim.T.x.array.copy()
            self.set_k(k, m=m)

            l = 0.0
            for i in range(self.n):
                input_data = self.inputs[i]
                output_data = self.outputs[i]
                J.true_values = output_data
                self.sim.solve_steady(**input_data)
                l += J.evaluate()

            self.set_k(original_k, m=m)
            self.sim.T.x.array[:] = original_T
            return l
        return loss_function
    
    def generate_getter_for_material(self, m):
        def restricted_getter():
            return self.get_k(m=m)
        return restricted_getter
    
    def generate_setter_for_material(self, m):
        def restricted_setter(k):
            self.set_k(k, m=m)
        return restricted_setter

    def update(self):
        for i, input_data in enumerate(self.inputs):
            probes = self.sim.solve_steady(**input_data)
            self.data[i] = probes.get_value('T')
            self.domain_data[i] = self.sim.T.copy()
        
    def compare_plot(self):
        if MPI.COMM_WORLD.rank == 0:
            figs = []
            for i in range(self.n):
                fig = go.Figure()
                fig.add_bar(y=self.data[i], name='Simulation')
                fig.add_bar(y=self.outputs[i], name='Experiment')
                figs.append(fig)
                #TODO: add titles to compare charts
                #TODO: add x index so there are names
            return figs
        
    def material_plot(self, m=None, property='k', include_density=False):
        return self.sim.material_plot(m=m, property=property, include_density=include_density, T=self.domain_data)
    
    def domain_state_plot(self):
        if self.domain_data[0] is not None:
            return self.sim.domain_state_plot(T=self.domain_data)
        
    def print_data(self):
        if MPI.COMM_WORLD.rank == 0:
            print(self.data)