import unittest
from examples.Example_01.model import Experiment_v1

class TestTemperatureSpectrum(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment_v1(
            geometry_dir='examples/Example_01/meshes/',
            model_name="mesh_2d",
            )
    
    def test_temperature_spectrum(self):
        self.sim.solve_steady()
        res = self.sim.get_temperature_spectrum(cell_tag=5, sampling=1e-1, smoothness=1, cumulative=False)

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()