from heat_battery.simulations import Experiment_v1
import unittest

class TestPlotProperty(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment_v1(
                model_name="mesh_2d",
                geometry_dir='meshes/experiment', 
                result_dir='results/experiment_test')
    
    def test_material_property_plots(self):
       fig = self.sim.mats.plot_property(m=1)
       #fig.show()

    def tearDown(self) -> None:
        self.sim.close_results()

if __name__ == '__main__':
    unittest.main()