from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest

class TestPlotProperty(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment(
                dim = 2,
                geometry_dir='meshes/experiment', 
                result_dir='results/experiment_test')
    
    def test_material_property_plots(self):
       fig = self.sim.mats.plot_property(m=1)
       #fig.show()

    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()