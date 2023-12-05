from mpi4py import MPI
import unittest
import time

from heat_battery.data import Experiment_data

class TestExperimentalDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        path = 'data/experiments/20231009_third/Test_TF24_Third_measurement_054411.csv'
        self.exp_real = Experiment_data(path)
        self.exp_real.print_io_stats()
       

    def test_steady_state_detection(self):
        print(self.exp_real.steady_state_mean)
    

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()