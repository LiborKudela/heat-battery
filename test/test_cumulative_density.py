from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest
import plotly.graph_objects as go

class TestPlotCumulativeDensity(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment(
            dim = 2,
            geometry_dir='meshes/experiment_contact', 
            result_dir='results/experiment_contact_test',
            )
    
    def test_contact_self_identification(self):
        self.sim.solve_steady()
        res = self.sim.get_temperature_density(cell_tag=5, sampling=1e-1, smoothness=1, cumulative=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res[0], y=res[1]))
        #fig.show()

    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()