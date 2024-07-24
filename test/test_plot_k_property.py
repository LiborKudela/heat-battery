from examples.Example_01.model import Experiment_v1
import unittest

class TestPlotProperty(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment_v1(
            geometry_dir='examples/Example_01/meshes/',
            model_name="mesh_2d",
            )
    
    def test_material_property_plots(self):
       fig = self.sim.mats.plot_property(m=1, property='k')
       #fig.show()

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()