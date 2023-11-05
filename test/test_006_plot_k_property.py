from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest

class TestPlotProperty(unittest.TestCase):
    def test_contact_self_identification(self):
        sim = Experiment(
            dim = 2,
            geometry_dir='meshes/experiment_contact', 
            result_dir='results/experiment_contact_test')
        
        for mat in sim.mats:
            mat.k.plot(show=False)

if __name__ == '__main__':
    unittest.main()