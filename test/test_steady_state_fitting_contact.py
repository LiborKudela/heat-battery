from heat_battery.simulations import Experiment
from heat_battery.data import PseudoExperimentalData
from heat_battery.optimization import SteadyStateComparer, optimizers
from heat_battery.optimization.derivatives import finite_diferences
import numpy as np
import unittest

class TestOptimization(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment(
                dim = 2,
                geometry_dir='meshes/experiment_contact', 
                result_dir='results/experiment_contact_test')
        self.exp = PseudoExperimentalData()
        self.m = 6
        self.adam_max_iter = 300
        Qc = 100
        T_amb = 20
        res = self.sim.solve_steady(Qc=Qc, T_amb=T_amb,save_xdmf=False)
        self.exp.feed_steady_state(res, Qc=Qc, T_amb=T_amb)
        self.fitter = SteadyStateComparer(self.sim, [self.exp])
        self.true_k = self.fitter.get_k(m=self.m)

    def test_contact_self_identification_fd(self):
        
        k = self.true_k.copy() # initial guess
        k *= 1.1
        loss = self.fitter.generate_loss_for_material(self.m)
        grad = finite_diferences(loss)
        opt = optimizers.ADAM(alpha=1e-4)

        for i in range(self.adam_max_iter):
            g, l = grad(k)
            update = opt.step(g)
            k += update
            opt.alpha *= 0.999
            print(f'loss: {l}')

        self.assertTrue(np.isclose(self.true_k, k, atol=1e-4).all(), "FD - Values do not agree")

    def test_contact_self_identification_adjoint(self):
          
        k = self.true_k.copy()
        k *= 1.1
        grad = self.fitter.generate_gradient_for_material(m=self.m)
        opt = optimizers.ADAM(alpha=1e-4)

        for i in range(self.adam_max_iter):
            g, l = grad(k)
            update = opt.step(g)
            k += update
            opt.alpha *= 0.999
            print(f'loss: {l}')

        self.assertTrue(np.isclose(self.true_k, k, atol=1e-4).all(), "ADJOINT - Values do not agree")
        
    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()