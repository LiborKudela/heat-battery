from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.sim = Experiment(dim = 2)
        self.exp = self.sim.pseudoexperimental_data_steady()
        self.fitter = SteadyStateComparer(self.sim, [self.exp])
        self.true_k = self.fitter.get_k(4)

    def test_material_identification(self):

        k0 = self.true_k.copy()
        k0 *= 1.1
        loss = self.fitter.generate_loss_for_material(4)
        opt = optimizers.ADAM(loss=loss, k0=k0, alpha=1e-3)

        for i in range(300):
            opt.step()
            opt.alpha *= 0.995
            opt.print_state()
            
        self.assertTrue(np.allclose(self.true_k, opt.get_k(), atol=1e-03), "Values do not agree")

    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()