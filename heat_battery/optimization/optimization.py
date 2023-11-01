import pandas as pd
import plotly.graph_objects as go
from mpi4py import MPI
import os
from ..simulations import Experiment
from ..data import Experiment_data
import random
import numpy as np


class DataFitter:
    def __init__(self, sim : Experiment, exp_data: Experiment_data):
        self.exp_data = exp_data
        self.sim = sim
        self.steady_state = SteadyStateComparer(sim, exp_data)

        self.idx_couples = []
        for m in range(len(self.sim.mats)):
            for k in range(self.sim.mats[m].k.order+1):
                self.idx_couples.append((m, k))

    def get_k(self):
        return [mat.k.get_values() for mat in self.sim.mats]

    def gradient_aproximation(self, perturbation=1e-4):
        
        g = []
        self.steady_state.update()
        org_err = self.steady_state.total_square_error.copy()
        
        for i, ic in enumerate(self.idx_couples):
            m = ic[0]
            k = ic[1]
            original_value = self.sim.mats[m].k.get_value(k)
            self.sim.mats[m].k.set_value(k, original_value+perturbation)
            self.steady_state.update()
            pert_err = self.steady_state.total_square_error.copy()
            g_err = (pert_err-org_err)/perturbation
            self.sim.mats[m].k.set_value(k, original_value)
            g.append(g_err)

        self.steady_state.update()

        g = np.array(g)
        return org_err, g
    
    def optimise_steady_state_v2(self, k0=None, max_iter=100, alpha=1e-1, beta_1=0.8, beta_2=0.8):
        if k0 is not None:
            for i, y_values in enumerate(k0):
                self.sim.mats[i].k.set_values(y_values)

        self.steady_state.update()
        if MPI.COMM_WORLD.rank == 0:
            abs_e = self.steady_state.total_abs_error.copy()
            sqr_e = self.steady_state.total_square_error.copy()
            print(f"initial state:", f"abs_err: {abs_e}" , f"sqr_err: {sqr_e}")

        g_w = 0
        g_norm_w = 1e5
        g_s = 1
        for j in range(max_iter):
            org_err, g = self.gradient_aproximation()
            prev_k = np.array([self.sim.mats[m].k.get_value(k) for m, k in self.idx_couples])

            # update values
            g = g*np.concatenate([[0], [0, 0], [0], [0], [1.0, 1.0, 1.0]])
            g_norm = np.linalg.norm(g)
            g_w = beta_1*g_w + (1-beta_1)*g
            g_norm_w = beta_2*g_norm_w + (1-beta_2)*g_norm
            g_s = beta_2*g_s + (1-beta_2)*g**2

            update = alpha * g_w/(g_norm_w+1e-3)
            for i, ic in enumerate(self.idx_couples):
                self.sim.mats[ic[0]].k.set_value(ic[1], max(2e-2, prev_k[i]-update[i]))
            self.steady_state.update()

            res = [mat.k.get_values() for mat in self.sim.mats]
            if MPI.COMM_WORLD.rank == 0:
                abs_e = self.steady_state.total_abs_error.copy()
                sqr_e = self.steady_state.total_square_error.copy()
                print(f"iter {j}", f"abs_err: {abs_e}" , f"sqr_err: {sqr_e}", f"g_norm: {g_norm}", f"g_norm_w: {g_norm_w}")
                if j % 10 == 0:
                    print("k = ", res)
        
        return res
    
    def optimise_steady_state_v3(self, k0=None, max_iter=100, alpha=1e-1, beta_1=0.8, beta_2=0.8):
        if k0 is not None:
            for i, y_values in enumerate(k0):
                self.sim.mats[i].k.set_values(y_values)

        self.steady_state.update()
        if MPI.COMM_WORLD.rank == 0:
            abs_e = self.steady_state.total_abs_error.copy()
            sqr_e = self.steady_state.total_square_error.copy()
            print(f"initial state:", f"abs_err: {abs_e}" , f"sqr_err: {sqr_e}")

        g_w = 0
        g_norm_w = 1e5
        g_s = 1
        for j in range(max_iter):
            org_err, g = self.gradient_aproximation()
            prev_k = np.array([self.sim.mats[m].k.get_value(k) for m, k in self.idx_couples])[-3:]

            # update values
            #g = g*np.concatenate([[0], [0, 0], [0], [0], [1.0, 1.0, 1.0]])
            g_norm = np.linalg.norm(g)
            g_w = beta_1*g_w + (1-beta_1)*g
            g_norm_w = beta_2*g_norm_w + (1-beta_2)*g_norm
            g_s = beta_2*g_s + (1-beta_2)*g**2

            update = alpha[-3:] * g_w/(g_norm_w+1e-3)
            for i, ic in enumerate(self.idx_couples[-3:]):
                self.sim.mats[ic[0]].k.set_value(ic[1], max(2e-2, prev_k[i]-update[i]))
            self.steady_state.update()

            res = [mat.k.get_values() for mat in self.sim.mats]
            if MPI.COMM_WORLD.rank == 0:
                abs_e = self.steady_state.total_abs_error.copy()
                sqr_e = self.steady_state.total_square_error.copy()
                print(f"iter {j}", f"abs_err: {abs_e}" , f"sqr_err: {sqr_e}", f"g_norm: {g_norm}", f"g_norm_w: {g_norm_w}")
                if j % 10 == 0:
                    print("k = ", res)
        
        return res

class SteadyStateComparer:
    def __init__(self, sim: Experiment, exp_data: Experiment_data):
        self.exp_data = exp_data
        self.sim = sim
        self.data = 0
        self.total_error = 0

        self.idx_couples = []
        for m in range(len(self.sim.mats)):
            for k in range(self.sim.mats[m].k.order+1):
                self.idx_couples.append((m, k))

    def get_k(self, m=None):
        if m is None:
            array = []
            for i in range(len(self.sim.mats)):
                array.append(self.sim.mats[i].k.get_values())
            return np.array(array)
        elif type(m) == int:
            return self.sim.mats[m].k.get_values()
    
    def set_k(self, k, m=None):
        if m is None:
            for i in range(len(self.sim.mats)):
                self.sim.mats[i].k.set_values(k[i])
        elif type(m) == int:
            self.sim.mats[m].k.set_values(k)

    def loss_function(self, k, m=None):
        original_k = self.get_k(m=m)
        self.set_k(k, m=m)
        data = self.compare_steady_state()
        self.set_k(original_k, m=m)
        return data["Square Error"].sum()    

    def generate_loss_for_material(self, m):
        def restricted_loss(k):
            return self.loss_function(k, m=m)    
        return restricted_loss

    def update(self):
        self.data = self.compare_steady_state()
        self.total_abs_error = self.data["Abs Error"].sum()
        self.total_square_error = self.data["Square Error"].sum()
        self.total_max_error = self.data["Abs Error"].max()
        return self.total_abs_error.copy(), self.total_square_error.copy()

    def compare_steady_state(self):
        T_amb = self.exp_data.steady_state_mean['16 - Ambient [°C]']
        Qc = self.exp_data.steady_state_mean['Power [W]']
        s_sim = self.sim.solve_steady(Qc=Qc, T_amb=T_amb, save_xdmf=False) # this must run on all ranks
        exp_mean = self.exp_data.steady_state_mean
        exp_std = self.exp_data.steady_state_std
        data = pd.concat([exp_mean, exp_std, s_sim], axis=1)
        data["Difference"] = data["Experiment Mean"] - data["Simulation"]
        data["Square Error"] = data["Difference"].pow(2)
        data["Abs Error"] = data["Difference"].abs()
        data = data.dropna()
        return data
        
    def create_figure(self, save_name=None, show=False):
        if MPI.COMM_WORLD.rank == 0:
            fig = go.Figure()
            fig.add_bar(x=self.data.index, y=self.data["Experiment Mean"], name='Experiment')
            fig.add_bar(x=self.data.index, y=self.data["Simulation"], name='Simulation')
            if show:
                fig.show()
            if save_name is not None:
                fig.write_html(f'{save_name}.html')
                fig.write_image(f'{save_name}.jpg', scale=2)
    
    def print_data(self):
        if MPI.COMM_WORLD.rank == 0:
            print(self.data)

    def save_data(self):
        if MPI.COMM_WORLD.rank == 0:
            self.data.to_csv(os.path.join(self.sim.result_dir, "comprarer.csv"))