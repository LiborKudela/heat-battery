from heat_battery.simulations import Experiment_v1
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest
import plotly.graph_objects as go

class TestTemperatureSpectrum(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment_v1(
            model_name="mesh_2d",
            geometry_dir='meshes/experiment_contact', 
            result_dir='results/experiment_contact_test',
            )
    
    def test_temperature_spectrum(self):
        self.sim.solve_steady()
        res = self.sim.get_temperature_spectrum(cell_tag=5, sampling=1e-1, smoothness=1, cumulative=False)

    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()