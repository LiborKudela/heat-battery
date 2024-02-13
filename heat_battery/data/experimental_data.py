from mpi4py import MPI
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import time
import functools
import json

class Experiment_data():
    def __init__(self, csv_path, decimal=',', delimiter=';', cache=False) -> None:
        self.full_path = csv_path
        self.dir_path, self.file_name = os.path.split(self.full_path)
        self.cache_dir = os.path.join(self.dir_path + "/.cache")
        self.cache_full_path = os.path.join(self.cache_dir, self.file_name + ".feather")
        self.metadata_path = os.path.join(self.dir_path, self.file_name + '.json')


        with open(self.metadata_path) as f:
            metadata = json.load(f)
        meta_steady_state_start = metadata['steady_state']['start']
        meta_steady_state_end = metadata['steady_state']['end']

        self.io_stats = {'file': self.file_name}

        # load csv and measur load time
        if os.path.exists(self.cache_full_path):
            _start = time.time()
            self.df = pd.read_feather(self.cache_full_path)
            self.io_stats["Series load/processing"] = time.time() - _start
            self.io_stats["Series load method"] = "Cached feather"
        else:
            _start = time.time()
            self.df = pd.read_csv(
                self.full_path, 
                delimiter=delimiter, 
                encoding='unicode_escape', 
                decimal=decimal)
            self.df['Time']=pd.to_datetime(self.df['Time'], format='%d.%m.%y %H:%M:%S,%f')
            self.df.set_index('Time', inplace=True)
            self.df.sort_index(inplace=True)
            self.df['total_seconds'] = (self.df.index - self.df.index[0]).total_seconds()
            self.df['dt'] = self.df['total_seconds'].diff()
            self.df['Power [W]'] = self.df['Power [W]'].clip(0, np.inf)
            self.io_stats["Series load/processing"] = time.time() - _start
            self.io_stats["Series load method"] = "csv processing"
            if MPI.COMM_WORLD.rank == 0:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.df.to_feather(self.cache_full_path)

        # detect steady states in the data series
        _start = time.time()
        self.detect_steady_state(meta_steady_state_start, meta_steady_state_end)
        self.io_stats["Steady state processing"] = time.time() - _start

        self.T_names = [
            '1 - Top [°C]', '2 - Top [°C]', '3 - Top [°C]', '4 - Middle [°C]',
            '5 - Middle [°C]', '6 - Middle [°C]', '7 - Bottom [°C]',
            '8 - Bottom [°C]', '9 - Bottom [°C]', '10 - A - Surface [°C]',
            '11 - B - Surface [°C]', '12 - C - Surface [°C]', '13 - I. Cover [°C]',
            '14 - II. Cover [°C]', '15 - III. Cover [°C]', '16 - Ambient [°C]',
        ]

    def print_io_stats(self):
        for item in self.io_stats.items():
            print(item)

    def detect_steady_state(self, start, end):
        #TODO: make this automatic
        self.steady_state_start = start
        self.steady_state_end = end
        self.steady_state_data = self.df[(self.df.index > start) & (self.df.index < end)]
        self.steady_state_mean = self.steady_state_data.mean().rename("Experiment Mean")
        self.steady_state_std = self.steady_state_data.std().rename("Experiment Std")

    def auto_dectect_steady_state(self):
        data = self.df - self.df.iloc[0]
        return data

    @functools.cache
    def data_series_plot(self):
        return px.line(self.df, y=self.T_names)

    @functools.cache
    def plot_steady_data(self):
        fig = go.Figure()
        fig.add_bar(x=self.steady_state_mean.index, y=self.steady_state_mean.values, name='Experiment')
        return fig
    
    @functools.cache
    def plot_steady_series(self):
        fig = go.Figure()
        fig.add_trace(px.scatter(self.steady_state_data))
        return fig

class PseudoExperimentalData():
    def __init__(self) -> None:
        pass

    def feed_steady_state(self, sim_res=None, Qc=0.0, T_amb=0.0):
        self.steady_state_mean = sim_res
        self.steady_state_mean['Power [W]'] = Qc
        self.steady_state_mean['16 - Ambient [°C]'] = T_amb
        self.steady_state_std = self.steady_state_mean.copy()
        self.steady_state_std.values[:] = 0.0
        self.steady_state_mean.rename("Experiment Mean", inplace=True)
        self.steady_state_std.rename("Experiment Std", inplace=True)

    def feed_unsteady(self, sim_res=None):
        pass

    def plot_data_series(self):
        return self.figure

    def plot_steady_data(self):
        fig = go.Figure()
        fig.add_bar(x=self.steady_state_mean.index, y=self.steady_state_mean.values, name='Experiment')
        return fig
    

    