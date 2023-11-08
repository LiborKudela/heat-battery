from typing import List
import pandas as pd
import plotly.graph_objects as go
from mpi4py import MPI
import os
import numpy as np
from ..simulations import Experiment
from ..data import Experiment_data

class SteadyStateComparer:
    def __init__(self, sim: Experiment, exp_data: List[Experiment_data]):
        self.sim = sim
        if isinstance(exp_data, Experiment_data):
            exp_data = [exp_data]

        self.exp_data = exp_data
        self.n = len(self.exp_data)
        self.data = [pd.Series()]*self.n 
        self.total_abs_error = [0.0]*self.n 
        self.total_square_error = [0.0]*self.n 
        self.total_max_error = [0.0]*self.n
        self.total_error = 0.0

    def get_k(self, m=None):
        if m is None:
            array = []
            for i in range(len(self.sim.mats)):
                array.append(self.sim.mats[i].k.get_values())
            return np.concatenate(array)
        elif type(m) == int:
            return self.sim.mats[m].k.get_values()
    
    def set_k(self, k, m=None):
        if m is None:
            start_idx = 0
            for i in range(len(self.sim.mats)):
                end_idx = start_idx + self.sim.mats[i].k.n_values
                self.sim.mats[i].k.set_values(k[start_idx:end_idx])
                start_idx = end_idx 
        elif type(m) == int:
            self.sim.mats[m].k.set_values(k)

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

    def update(self):
        for i, exp_data in enumerate(self.exp_data):
            self.data[i] = self.compare_steady_state(exp_data)
            self.total_abs_error[i] = self.data[i]["Abs Error"].sum()
            self.total_square_error[i] = self.data[i]["Square Error"].sum()
            self.total_max_error[i] = self.data[i]["Abs Error"].max()
        self.total_error = np.sum(self.total_square_error)

    def compare_steady_state(self, exp_data):
        T_amb = exp_data.steady_state_mean['16 - Ambient [°C]']
        Qc = exp_data.steady_state_mean['Power [W]']
        s_sim = self.sim.solve_steady(Qc=Qc, T_amb=T_amb, save_xdmf=False) # this must run on all ranks
        exp_mean = exp_data.steady_state_mean
        exp_std = exp_data.steady_state_std
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