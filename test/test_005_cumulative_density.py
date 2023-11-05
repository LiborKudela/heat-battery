from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest
import plotly.graph_objects as go

class TestPlotCumulativeDensity(unittest.TestCase):
    def test_contact_self_identification(self):
        sim = Experiment(
            dim = 2,
            geometry_dir='meshes/experiment_contact', 
            result_dir='results/experiment_contanct_test')
        sim.solve_steady()
        res = sim.get_current_temperature_density(cell_tag=5, sampling=1e-1, smoothness=1, cumulative=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res[0], y=res[1]))
        #fig.show()

        sim.close_results()

if __name__ == '__main__':
    unittest.main()