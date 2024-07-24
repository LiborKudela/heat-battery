import unittest
from heat_battery.geometry.dummy import build_geometry as build_dummy
from heat_battery.geometry.step_loader import build_geometry_from_stepfile
from heat_battery.materials import materials

class TestGeometryBuilders(unittest.TestCase):

    def test_gmsh_dummy(self):
        build_dummy()
        self.assertTrue(True, "Dummy failed")
        
    def test_mesh_form_step(self):
        path = "./test/Experiment_v1.stp"
        mats = {
            'walls':(materials.Steel04, [1]), 
            'insulation bottom':(materials.Standard_insulation, [2]),    
            'insulation top':(materials.Standard_insulation, [3]),
            'sand':(materials.SandTheory, [4]), 
            'unheated cartridge':(materials.Cartridge_unheated, [5,8]),
            'cartridge contact':(materials.new_contact_class(0.0001), [6]), 
            'heated cartridge':(materials.Cartridge_heated, [7]), 
            'top plate':(materials.Steel04, [9]),    
            'insulation':(materials.Standard_insulation, [10]), 
        }
        h_ref = 0.235
        r_c = 0.007
        r_outer_t = 0.3112/2
        r_0 = r_c + 0.005
        r_1 = r_c + 0.027
        r_2 = r_1 + 0.029
        r_3 = r_2 + 0.029
        r_4 = r_3 + 0.020

        points = {
            'T':{
                '0 - Top [°C]': [r_0, 0.0, h_ref-0.06],
                '1 - Top [°C]': [r_1, 0.0, h_ref-0.06], 
                '2 - Top [°C]': [r_2, 0.0, h_ref-0.06],
                '3 - Top [°C]': [r_3, 0.0, h_ref-0.06],
                '4 - Top [°C]': [r_4, 0.0, h_ref-0.06],

                '0 - Middle [°C]':[r_0, 0.0, h_ref-2*0.06], 
                '1 - Middle [°C]':[r_1, 0.0, h_ref-2*0.06], 
                '2 - Middle [°C]':[r_2, 0.0, h_ref-2*0.06], 
                '3 - Middle [°C]':[r_3, 0.0, h_ref-2*0.06],
                '4 - Middle [°C]':[r_4, 0.0, h_ref-2*0.06],

                '0 - Bottom [°C]':[r_0, 0.0, h_ref-3*0.06],
                '1 - Bottom [°C]':[r_1, 0.0, h_ref-3*0.06],
                '2 - Bottom [°C]':[r_2, 0.0, h_ref-3*0.06],
                '3 - Bottom [°C]':[r_3, 0.0, h_ref-3*0.06],
                '4 - Bottom [°C]':[r_4, 0.0, h_ref-3*0.06],

                '3 - A - Surface [°C]':[r_c, 0.0, h_ref-0.064],
                '11 - B - Surface [°C]':[r_c, 0.0, h_ref-2*0.064],
                '12 - C - Surface [°C]':[r_c, 0.0, h_ref-3*0.064],

                '13 - I. Cover [°C]':[r_outer_t, 0.0, h_ref-0.06],
                '14 - II. Cover [°C]':[r_outer_t, 0.0, h_ref-2*0.06],
                '15 - III. Cover [°C]':[r_outer_t, 0.0, h_ref-3*0.06],

                '0 - top plate [°C]':[0.0, 0.0, h_ref],
                '1 - top plate [°C]':[0.03, 0.0, h_ref],
            },
        }

        bcs = {
            'outer_surface': [12, 20, 21, 22, 23, 24, 25, 26, 28, 29, 64],
            }
        build_geometry_from_stepfile(
            path = path,
            dir = 'meshes/experiment_v1_inventor',
            mesh_size_max=0.001,
            points=points,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            )

if __name__ == '__main__':
    unittest.main()
