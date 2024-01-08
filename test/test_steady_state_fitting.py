from heat_battery.simulations import Experiment
from heat_battery.data import PseudoExperimentalData
from heat_battery.optimization import SteadyStateComparer, optimizers
import numpy as np
import unittest

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.sim = Experiment(dim = 2)
        self.exp = PseudoExperimentalData()
        Qc = 80
        T_amb = 20
        res = self.sim.solve_steady(Qc=Qc, T_amb=T_amb,save_xdmf=False)
        self.exp.feed_steady_state(res, Qc=Qc, T_amb=T_amb)
        self.fitter = SteadyStateComparer(self.sim, [self.exp])
        self.m = 5
        self.true_k = self.fitter.get_k(self.m)
        self.grad = self.fitter.generate_gradient_for_material(m=self.m)

    def test_material_identification_fd(self):

        k0 = self.true_k.copy()
        k0 *= 1.5
        loss = self.fitter.generate_loss_for_material(self.m)
        opt = optimizers.ADAM(loss=loss, k0=k0, alpha=1e-2)

        for i in range(300):
            opt.step()
            opt.alpha *= 0.99
            opt.print_state()

        self.assertTrue(np.allclose(self.true_k, opt.get_k(), atol=1e-02), "Values do not agree")

    def test_material_identification_adjoint(self):

        k0 = self.true_k.copy()
        k0 *= 1.5
        loss = self.fitter.generate_loss_for_material(self.m)
        opt = optimizers.ADAM(loss=loss, grad=self.grad, k0=k0, alpha=1e-2)

        for i in range(300):
            opt.step()
            opt.alpha *= 0.99
            opt.print_state()
        
        self.assertTrue(np.allclose(self.true_k, opt.get_k(), atol=1e-02), "Values do not agree")

    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()