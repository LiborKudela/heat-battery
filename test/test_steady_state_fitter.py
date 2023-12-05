from heat_battery.simulations import Experiment
from heat_battery.data import PseudoExperimentalData
from heat_battery.optimization import SteadyStateComparer
import numpy as np
import unittest

class TestSteadyStateComparerAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment(dim = 2)
        self.exp = PseudoExperimentalData()
        Qc = 100
        T_amb = 20
        res = self.sim.solve_steady(Qc=Qc, T_amb=T_amb,save_xdmf=False)
        self.exp.feed_steady_state(res, Qc=Qc, T_amb=T_amb)
        self.fitter = SteadyStateComparer(self.sim , [self.exp])

    def tearDown(self) -> None:
        self.sim.close_results()

    def test_fitter_get_set_methods_single_material(self):
        k0 = self.fitter.get_k(0).copy()
        self.fitter.set_k(k0 + 1.0, 0)
        k = self.fitter.get_k(0).copy()
        self.assertTrue(np.allclose(k0, k - 1.0, atol=1e-6))

    def test_fitter_get_set_methods_all_materials(self):

        k0 = self.fitter.get_k().copy()
        self.fitter.set_k(k0 + 1.0)
        k = self.fitter.get_k().copy()
        self.assertTrue(np.allclose(k0, k - 1.0, atol=1e-6))

    def test_fitter_loss_fucntion(self):
        
        k0 = self.fitter.get_k().copy()
        k0_err = self.fitter.loss_function(k0)
        new_err = self.fitter.loss_function(k0 + 1.0)
        self.assertNotEqual(k0_err, new_err)

    def test_fitter_generated_loss_function(self):

        k0 = self.fitter.get_k(0).copy()
        gen_loss = self.fitter.generate_loss_for_material(0)
        k0_err = gen_loss(k0)
        new_err = gen_loss(k0 + 1.0)
        self.assertNotEqual(k0_err, new_err)

    def test_fitter_generated_gradient(self):

        k0 = self.fitter.get_k(0).copy()
        gen_loss = self.fitter.generate_loss_for_material(0)
        k0_err = gen_loss(k0)
        new_err = gen_loss(k0 + 1.0)
        self.assertNotEqual(k0_err, new_err)

if __name__ == '__main__':
    unittest.main()