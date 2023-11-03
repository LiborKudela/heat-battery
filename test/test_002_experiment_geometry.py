import unittest
from heat_battery.geometry.experiment import build_geometry

class TestExperimentGeometryGeneration(unittest.TestCase):
    def test_2d_experiment(self):
        _max=0.01
        _min=0.005
        build_geometry(mesh_size_max=_max, mesh_size_min=_min, dim=2)
        self.assertTrue(True, "2D Geometry Experiment 1 Failed to generate")

    def test_3d_experiment(self):
        _max=0.01
        _min=0.005
        build_geometry(mesh_size_max=_max, mesh_size_min=_min, dim=3, symetry_3d="quarter")
        self.assertTrue(True, "3D Geometry Experiment 1 Failed to generate")

if __name__ == '__main__':
    unittest.main()
