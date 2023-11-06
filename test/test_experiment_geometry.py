import unittest
from heat_battery.geometry.dummy import build_geometry as build_dummy
from heat_battery.geometry.experiment import build_geometry as build_without_contact
from heat_battery.geometry.experiment_contact import build_geometry as build_with_contact

class TestGeometryBuilders(unittest.TestCase):

    def test_gmsh_dummy(self):
        build_dummy()
        self.assertTrue(True, "Dummy failed")

    def test_2d_experiment(self):
        _max=0.005
        _min=0.001
        build_without_contact(mesh_size_max=_max, mesh_size_min=_min, dim=2)
        self.assertTrue(True, "2D Geometry Experiment 1 Failed to generate")

    def test_2d_experiment_contact(self):
        _max=0.005
        _min=0.001
        build_with_contact(mesh_size_max=_max, mesh_size_min=_min, dim=2)
        self.assertTrue(True, "2D Geometry Experiment 1 Failed to generate")

    def test_3d_experiment(self):
        _max=0.005
        _min=0.001
        build_without_contact(mesh_size_max=_max, mesh_size_min=_min, dim=3, symetry_3d="quarter")
        self.assertTrue(True, "3D Geometry Experiment 1 Failed to generate")

    def test_3d_experiment_contact(self):
        _max=0.005
        _min=0.001
        build_with_contact(mesh_size_max=_max, mesh_size_min=_min, dim=3, symetry_3d="quarter")
        self.assertTrue(True, "3D Geometry Experiment 1 Failed to generate")

if __name__ == '__main__':
    unittest.main()
