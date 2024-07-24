from examples.Example_01.model import Experiment_v1
from heat_battery.optimization import SteadyStateComparer
import numpy as np
import unittest

class TestSteadyStateComparerAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment_v1(
            geometry_dir='examples/Example_01/meshes/',
            model_name="mesh_2d",
            )
        inputs = [{'Qc': 100, 'T_amb': 20}]
        probes = self.sim.solve_steady(**inputs[0])
        outputs = [probes.get_value('T')]
        self.fitter = SteadyStateComparer(self.sim , inputs, outputs)

    def tearDown(self) -> None:
        pass

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