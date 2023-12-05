from heat_battery.simulations import Experiment
from heat_battery.data import PseudoExperimentalData
from heat_battery.optimization import SteadyStateComparer
import numpy as np
import unittest

class TestMaterialAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment(dim = 2)

    def tearDown(self) -> None:
        self.sim.close_results()

    def test_material_get_set_methods(self):
    
        # get original values
        k0 = self.sim.mats[0].k.get_values().copy()
        k0_femconst = self.sim.mats[0].k.fem_const.value.copy()

        #change value
        self.sim.mats[0].k.set_values([k0+1.0])

        # read if it changed
        k = self.sim.mats[0].k.get_values().copy()
        k_femconst = self.sim.mats[0].k.fem_const.value.copy()

        #assert it did change correctly
        self.assertTrue(np.allclose(k0, k - 1.0, atol=1e-6))
        self.assertTrue(np.allclose(k0_femconst, k_femconst - 1.0, atol=1e-6))

    #TODO add test for material conversion lagrange to polynomial
    #TODO add test for material conversion polynomial to lagrange