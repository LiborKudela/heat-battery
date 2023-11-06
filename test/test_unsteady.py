from heat_battery.simulations import Experiment
import unittest

class TestOptimization(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment(dim = 2)
    
    def test_self_fitter(self):

        T_amb_t = lambda t: 20.0
        Qc_t = lambda t: 100.0
        res = self.sim.solve_unsteady(T_amb_t=T_amb_t, Qc_t=Qc_t, t_max=100)
        self.assertTrue(True, "Test did not fail")

    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()