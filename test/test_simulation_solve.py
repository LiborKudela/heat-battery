from examples.Example_01.model import Experiment_v1
import unittest
import numpy as np

class TestSimulation(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment_v1(
            geometry_dir='examples/Example_01/meshes/',
            model_name="mesh_2d",
            )
    
    def test_steady_solve(self):
        T_amb = 20.0
        Qc = 100.0
        alpha = 5.0
        probes = self.sim.solve_steady(
            T_amb=T_amb,
            Qc=Qc,
            alpha=alpha)
        stored_heat = probes.get_value('heat')
        print(stored_heat)

        # ignoring little bit of heat loss but t_max is small so it should be ok
        self.assertTrue(np.isclose(stored_heat, 6949441, atol=1), "Stored heat is wrong")

    def test_unsteady_solve(self):
        T_amb = 20.0
        Qc = 100.0
        alpha = 5.0
        t_max = 100.0 # terminate at this time
        probes = self.sim.solve_unsteady(
            T_amb_t=lambda t: T_amb,
            Qc_t=lambda t: Qc,
            alpha_t=lambda t: alpha,
            t_max=t_max)
        stored_heat = probes.get_value('heat')

        # ignoring little bit of heat loss but t_max is small so it should be ok
        self.assertTrue(np.isclose(stored_heat, Qc*t_max, atol=1), "Stored heat is wrong")

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()