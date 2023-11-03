from heat_battery.simulations import Experiment
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest

class TestOptimization(unittest.TestCase):
    def test_contact_self_identification(self):
        sim = Experiment(
            dim = 2,
            geometry_dir='meshes/experiment_contact', 
            result_dir='results/experiment_contact_test')
        exp = sim.pseudoexperimental_data_steady()
        fitter = SteadyStateComparer(sim, [exp])
        true_k = fitter.get_k()
        
        k0 = true_k[-2:].copy()
        k0 += 100
        loss = fitter.generate_loss_for_material(5)
        opt = optimizers.ADAM(loss=loss, k0=k0, alpha=10)

        for i in range(200):
            opt.step()
            opt.print_state()

        print(opt.get_k())
        self.assertTrue(np.isclose(true_k[-2:], opt.get_k(), atol=1).all(), "Values do not agree")

if __name__ == '__main__':
    unittest.main()