import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly_resampler import FigureResampler, FigureWidgetResampler

#experiment_file = 'experiment_data/20231004_first/Test_TF46_First_measurement_135651.csv'
#experiment_file = 'experiment_data/20231005_top_insulated/Test_TF46_Second_measurement_065323_prvnich_6h.csv'
experiment_file = 'experiment_data/20231009_third/Test_TF24_Third_measurement_054411.csv'
#experiment_file = 'experiment_data/20231009_third/Test_TF24_Third_measurement_163955.csv'

class Experiment_data():
    def __init__(self, csv_path=None, decimal=',', delimiter=';') -> None:
        self.path = csv_path
        if csv_path is not None:
            self.df = pd.read_csv(self.path, 
                                delimiter=delimiter, 
                                encoding='unicode_escape', 
                                decimal=decimal)
        
            self.df['Time']=pd.to_datetime(self.df['Time'], format='%d.%m.%y %H:%M:%S,%f')
            self.df.set_index('Time', inplace=True)
            self.df.sort_index(inplace=True)
            self.df['total_seconds'] = (self.df.index - self.df.index[0]).total_seconds()
            self.df['dt'] = self.df['total_seconds'].diff()
            self.df['Power [W]'] = self.df['Power [W]'].clip(0, np.inf)
            self.detect_steady_state()

        self.T_names = [
            '1 - Top [°C]', '2 - Top [°C]', '3 - Top [°C]', '4 - Middle [°C]',
            '5 - Middle [°C]', '6 - Middle [°C]', '7 - Bottom [°C]',
            '8 - Bottom [°C]', '9 - Bottom [°C]', '10 - A - Surface [°C]',
            '11 - B - Surface [°C]', '12 - C - Surface [°C]', '13 - I. Cover [°C]',
            '14 - II. Cover [°C]', '15 - III. Cover [°C]', '16 - Ambient [°C]',
        ]
    
    def detect_steady_state(self, start='2023-10-13 15:00', end='2023-10-13 17:00'):
        #TODO: make this automatic
        self.steady_state_start = start
        self.steady_state_end = end
        self.steady_state_data = self.df[(self.df.index > start) & (self.df.index < end)]
        self.steady_state_mean = self.steady_state_data.mean().rename("Experiment Mean")
        self.steady_state_std = self.steady_state_data.std().rename("Experiment Std")

    def plot_data_series(self):
        f1 = px.line(self.df, y=self.T_names)
        fig = FigureResampler(f1)
        return fig

    def plot_steady_data(self):
        fig = go.Figure()
        fig.add_bar(x=self.steady_state_mean.index, y=self.steady_state_mean.values, name='Experiment')
        return fig
    