from heat_battery.simulations import Experiment_v1
import numpy as np
import unittest

class TestMaterialAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Experiment_v1(model_name="mesh_2d")
        self.m_int = 0
        self.m_ints = [0,1]
        self.m_str = 'steel'
        self.m_strs = ['steel', 'insulation']
        self.m_mixed = ['steel', 1]

    def tearDown(self) -> None:
        self.sim.close_results()

    def material_tree_test_recipe(self, index):
        # get property from deep inside tree
        prop = self.sim.mats.get_material(index).get_property('k')
        k0 = prop.get_values()
        k0_femconst = prop.fem_const.value.copy()
        #change value
        prop.set_values(k0+1.0)
        # read if it changed
        k = prop.get_values().copy()
        k_femconst = prop.fem_const.value.copy()
        #assert it did change correctly (for Lagrange polynomial property)
        self.assertTrue(np.allclose(k0, k - 1.0, atol=1e-6))
        self.assertTrue(np.allclose(k0_femconst[0], k_femconst[0] - 1.0, atol=1e-6))

    def single_index_test_recipe(self, index):
        # get original values
        k0 = self.sim.mats.get_property_values('k', index)
        k0_femconst = self.sim.mats.get_material(index).get_property('k').fem_const.value.copy()
        #change value
        self.sim.mats.set_property_values(k0+1.0, 'k', index)
        # read if it changed
        k = self.sim.mats.get_property_values('k', index)
        k_femconst = self.sim.mats.get_material(index).get_property('k').fem_const.value.copy()
        #assert it did change correctly
        self.assertTrue(np.allclose(k0, k - 1.0, atol=1e-6))
        self.assertTrue(np.allclose(k0_femconst[0], k_femconst[0] - 1.0, atol=1e-6))

    def multiple_index_test_recipe(self, indexes):
        # get original values
        k0 = self.sim.mats.get_property_values('k', indexes)
        k0_femconst = self.sim.mats.get_material(indexes[0]).get_property('k').fem_const.value.copy()
        # change value
        self.sim.mats.set_property_values(k0+1.0, 'k', indexes)
        # read if it changed
        k = self.sim.mats.get_property_values('k', indexes)
        k_femconst = self.sim.mats.get_material(indexes[0]).get_property('k').fem_const.value.copy()
        # assert it did change correctly
        self.assertTrue(np.allclose(k0, k - 1.0, atol=1e-6))
        self.assertTrue(np.allclose(k0_femconst[0], k_femconst[0] - 1.0, atol=1e-6))

    def test_transformation(self):  
        prop = self.sim.mats.get_material(self.m_int).get_property('k')
        k0 = prop.get_values()
        fem_const = prop.fem_const.value.copy()
        k1 = prop.transform_form_coefficients_to_values(fem_const)
        self.assertTrue(np.allclose(k0, k1, atol=1e-6))

    # all ttest are below
    def test_material_tree_by_int(self):
        self.material_tree_test_recipe(self.m_int)

    def test_material_tree_by_str(self):
        self.material_tree_test_recipe(self.m_str)

    def test_direct_by_single_int(self):
        self.single_index_test_recipe(self.m_int)

    def test_direct_by_single_str(self):
        self.single_index_test_recipe(self.m_int)

    def test_direct_by_multiple_ints(self):
        self.multiple_index_test_recipe(self.m_ints)

    def test_direct_by_multiple_strs(self):
        self.multiple_index_test_recipe(self.m_strs)

    def test_direct_by_multiple_mixed(self):
        self.multiple_index_test_recipe(self.m_mixed)

    #TODO add test for material conversion lagrange to polynomial
    #TODO add test for material conversion polynomial to lagrange