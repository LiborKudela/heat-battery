from heat_battery.simulations import Experiment
import unittest
import numpy as np

class TestOptimization(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment(dim = 2)
    
    def test_self_fitter(self):
        T_amb = 20.0
        Qc = 100.0
        t_max = 100.0
        T_amb_t = lambda t: T_amb
        Qc_t = lambda t: Qc
        res = self.sim.solve_unsteady(T_amb_t=T_amb_t, Qc_t=Qc_t, t_max=t_max)
        stored_heat = self.sim.probes.get_value('heat')

        # ignoring little bit of heat loss but t_max is small so it should be ok
        self.assertTrue(np.isclose(stored_heat, Qc*t_max, atol=1e-5), "Stored heat is wrong")

    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()