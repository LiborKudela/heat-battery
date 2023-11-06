from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest

class TestOptimization(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment(
                dim = 2,
                geometry_dir='meshes/experiment_contact', 
                result_dir='results/experiment_contact_test')
        self.exp = self.sim.pseudoexperimental_data_steady()
        self.fitter = SteadyStateComparer(self.sim, [self.exp])
        self.true_k = self.fitter.get_k(m=5)

    def test_contact_self_identification(self):
          
        k0 = self.true_k.copy()
        k0 *= 1.1
        loss = self.fitter.generate_loss_for_material(5)
        opt = optimizers.ADAM(loss=loss, k0=k0, alpha=1e-3)

        for i in range(200):
            opt.step()
            opt.alpha *= 0.995
            opt.print_state()

        self.assertTrue(np.isclose(self.true_k, opt.get_k(), atol=1e-4).all(), "Values do not agree")
        
    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()