from examples.Example_01.model import Experiment_v1
from heat_battery.optimization import SteadyStateComparer, optimizers
from heat_battery.optimization.derivatives import finite_diferences
import numpy as np
import unittest

class TestOptimization(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment_v1(
            geometry_dir='examples/Example_01/meshes/',
            model_name="mesh_2d",
            )
        inputs = [{'Qc': 100, 'T_amb': 20}]
        probes = self.sim.solve_steady(**inputs[0])
        outputs = [probes.get_value('T')]
        self.fitter = SteadyStateComparer(self.sim, inputs, outputs)
        self.m_sand = 5
        self.m_contact = 6
        self.adam_max_iter = 1000

    def minimisation_test_recipe(self, m):
        loss = self.fitter.generate_loss_for_material(m)
        grad = self.fitter.generate_loss_gradient_for_material(m)
        jac = self.fitter.generate_loss_component_jacobian(m)
        true_k = self.fitter.get_k(m=m).copy()
        k = true_k.copy() * 1.1 # initial guess -> make it 10% wrong
        opt = optimizers.GaussNewtonDescent(loss, grad, jac, alpha=0.1)
        sol_k = opt.optimise(k, max_iter=10)
        self.assertTrue(np.isclose(true_k, sol_k, atol=5e-4).all(), 
                        f"Values do not agree {true_k} vs {sol_k}")

    def test_sand_identification(self):
        self.minimisation_test_recipe(self.m_sand)

    def test_contact_identification(self):
        self.minimisation_test_recipe(self.m_contact)

if __name__ == '__main__':
    unittest.main()