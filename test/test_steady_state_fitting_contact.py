from heat_battery.simulations import Experiment
from heat_battery.data import PseudoExperimentalData
from heat_battery.optimization import SteadyStateComparer, optimizers
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
        self.grad = self.fitter.generate_gradient_for_material(m=self.m)

    def test_contact_self_identification_fd(self):
          
        k0 = self.true_k.copy()
        k0 *= 1.1
        loss = self.fitter.generate_loss_for_material(self.m)
        opt = optimizers.ADAM(loss=loss, k0=k0, alpha=1e-4)

        for i in range(self.adam_max_iter):
            opt.step()
            opt.alpha *= 0.999
            opt.print_state()

        self.assertTrue(np.isclose(self.true_k, opt.get_k(), atol=1e-4).all(), "FD - Values do not agree")

    def test_contact_self_identification_adjoint(self):
          
        k0 = self.true_k.copy()
        k0 *= 1.1
        loss = self.fitter.generate_loss_for_material(self.m)
        opt = optimizers.ADAM(loss=loss, grad=self.grad, k0=k0, alpha=1e-4)

        for i in range(self.adam_max_iter):
            opt.step()
            opt.alpha *= 0.999
            opt.print_state()

        self.assertTrue(np.isclose(self.true_k, opt.get_k(), atol=1e-4).all(), "ADJOINT - Values do not agree")
        
    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()