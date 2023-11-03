from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest

class TestOptimization(unittest.TestCase):
    def test_self_identification(self):
        sim = Experiment(dim = 2)
        exp = sim.pseudoexperimental_data_steady()
        fitter = SteadyStateComparer(sim, [exp])
        true_k = fitter.get_k()
        k0 = true_k[-3:].copy()
        k0 += 0.01
        loss = fitter.generate_loss_for_material(4)
        opt = optimizers.ADAM(loss=loss, k0=k0, alpha=3e-4)

        for i in range(200):
            opt.step()
            opt.print_state()

        self.assertTrue(np.isclose(true_k[-3:], opt.get_k(), atol=3e-03).all(), "Values do not agree")

if __name__ == '__main__':
    unittest.main()