import unittest
from heat_battery.geometry.dummy import build_geometry as build_dummy
from heat_battery.geometry.experiment import build_geometry as build_without_contact
from heat_battery.geometry.experiment_contact import build_geometry as build_with_contact
from heat_battery.geometry.experiment_v2 import build_geometry as build_experiment_v2
from heat_battery.geometry.tantalum_wire import build_geometry as build_tantalum_wire
from heat_battery.geometry.experiment_cartridge import build_geometry as build_cartridge
from heat_battery.geometry.step_loader import build_geometry_from_stepfile
from heat_battery.materials import materials

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

    def test_2d_experiment_v2(self):
        _max=0.0001
        _min=0.0001
        build_experiment_v2(mesh_size_max=_max, mesh_size_min=_min, dim=2)
        self.assertTrue(True, "2D Geometry Experiment 2 Failed to generate")

    def test_tantalum_wire(self):
        _max=0.00001
        _min=0.00001
        build_tantalum_wire(mesh_size_max=_max, mesh_size_min=_min, dim=2)
        self.assertTrue(True, "2D Geometry Experiment 2 Failed to generate")

    def test_cartridge(self):
        _max=0.001
        _min=0.001
        build_cartridge(mesh_size_max=_max, mesh_size_min=_min, dim=2)
        self.assertTrue(True, "2D Geometry Cartridge Failed to generate")
        
    def test_experiment_v1_inventor_axisymetry(self):
        _max=0.00001
        _min=0.00001
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/Experiment_v1.stp"
        groups = {
            'walls':[1], 
            'insulation bottom':[2],    
            'insulation top':[3],
            'sand':[4], 
            'unheated cartridge':[5,8],
            'cartridge contact':[6], 
            'heated cartridge':[7], 
            'top plate':[9],    
            'insulation':[10], 
        }
        h_ref = 0.235
        r_c = 0.007
        r_outer_t = 0.3112/2
        r_1 = r_c + 0.027
        r_2 = r_1 + 0.029
        r_3 = r_2 + 0.029
        probes_coords = [
            [r_1, 0.0, h_ref-0.06],[r_2, 0.0, h_ref-0.06],[r_3, 0.0, h_ref-0.06], # radial top sensors
            [r_1, 0.0, h_ref-2*0.06],[r_2, 0.0, h_ref-2*0.06],[r_3, 0.0, h_ref-2*0.06], # radial mid sensors
            [r_1, 0.0, h_ref-3*0.06],[r_2, 0.0, h_ref-3*0.06],[r_3, 0.0, h_ref-3*0.06], # radial bottom sensors
            [r_c, 0.0, h_ref-0.064],[r_c, 0.0, h_ref-2*0.064],[r_c, 0.0, h_ref-3*0.064], # cartridge surface
            [r_outer_t, 0.0, h_ref-0.06],[r_outer_t, 0.0, h_ref-2*0.06],[r_outer_t, 0.0, h_ref-3*0.06], # outer surface sensors
        ]

        probes_names = [
            '1 - Top [°C]', '2 - Top [°C]', '3 - Top [°C]',
            '4 - Middle [°C]', '5 - Middle [°C]', '6 - Middle [°C]',
            '7 - Bottom [°C]', '8 - Bottom [°C]', '9 - Bottom [°C]',
            '10 - A - Surface [°C]', '11 - B - Surface [°C]', '12 - C - Surface [°C]',
            '13 - I. Cover [°C]', '14 - II. Cover [°C]', '15 - III. Cover [°C]'
        ]

        mats = [
            (materials.Steel04, 'Steel parts'), 
            (materials.Standard_insulation, 'Insulation bottom'),
            (materials.Standard_insulation, 'Insulation top'),
            (materials.SandTheory, 'Sand'),
            (materials.Cartridge_unheated, 'Unheated part of cartridge'),
            (materials.new_contact_class(0.0001), 'Cartridge contact'), 
            (materials.Cartridge_heated, 'Heated part of cartridge'),
            (materials.Steel04, 'Top plate'), 
            (materials.Standard_insulation, 'Insulation'),
        ]

        bcs = {
            'outer_surface': [12, 20, 21, 22, 23, 24, 25, 26, 28, 29, 64],
            }
        build_geometry_from_stepfile(
            path = path,
            dir = 'meshes/experiment_v1_inventor',
            mesh_size_max=0.001,
            groups=groups, 
            probes_coords=probes_coords,
            probes_names=probes_names,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            )

    def test_experiment_v21_inventor_axisymetry(self):
        _max=0.00001
        _min=0.00001
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/Experiment_v2.1.stp"
        groups = {'wire':[2], 'electrodes':[1, 3], 'sand':[4], 'case':[7], 'th':[5, 6], 'lid':[8]}
        probes_coords = [[0.0, 0.0, 0.0], [0.0001-1e-6, 0.0, 0.0], [0.001, 0.0, 0.0], [0.042/2, 0.0, 0.0]]
        probes_names = ["T - wire mid", "T - wire surf", "T - sand", "T - surf can"]
        mats = [
            (materials.TantalumWire, 'wire'),
            (materials.Steel04, 'electrodes'),
            (materials.Constant_sand, 'sand'),
            (materials.Steel04, 'case'),
            (materials.Standard_insulation, 'th'),
            (materials.Steel04, 'lid'),
        ]
        bcs = {'outer_surface': [7,8,9,14,15,16,30,31,32,33,34,35,36,37,41,42,43,44,45]}
        build_geometry_from_stepfile(
            path = path,
            dir='meshes/experiment_inventor',  
            mesh_size_max=0.0001, 
            groups=groups, 
            probes_coords=probes_coords,
            probes_names=probes_names,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            custom_data = dict(d_wire=0.0002, l_wire=0.15),
            )
        self.assertTrue(True, "2D Geometry Experiment 2 Failed to generate")

    def test_experiment_v22_inventor_axisymetry(self):
        _max=0.00001
        _min=0.00001
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/Experiment_v2.2.stp"
        groups = {'wire':[2], 'electrodes':[1, 3], 'contact':[4], 'sand':[5], 'th':[6, 7], 'case':[8], 'lid':[9]}
        probes_coords = [[0.0, 0.0, 0.0], [0.0001-1e-6, 0.0, 0.0], [0.001, 0.0, 0.0], [0.032/2, 0.0, 0.0]]
        probes_names = ["T - wire mid", "T - wire surf", "T - sand", "T - surf can"]
        mats = [
            (materials.TantalumWire, 'wire'),
            (materials.Steel04, 'electrodes'),
            (materials.new_contact_class(0.000001), 'contact'),
            (materials.Constant_sand, 'sand'),
            (materials.Standard_insulation, 'th'),
            (materials.Steel04, 'case'),
            (materials.Steel04, 'lid'),
        ]
        bcs = {'outer_surface': [8,9,10,15,16,17,33,34,35,36,37,38,39,40,44,45,46,47,48]}
        build_geometry_from_stepfile(
            path = path,
            name='mesh_contact',
            dir='meshes/experiment_inventor',  
            mesh_size_max=0.0001, 
            groups=groups, 
            probes_coords=probes_coords,
            probes_names=probes_names,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            custom_data = dict(d_wire=0.0002, l_wire=0.15),
            )
        self.assertTrue(True, "2D Geometry Experiment 2 Failed to generate")

    def test_TF46(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/TF46.stp"
        groups = {'heated':[1], 'unheated':[2, 3]}
        probes_coords = []
        probes_names = []
        mats = [
            (materials.Steel04, 'Heated part of cartridge'),
            (materials.Steel04, 'Unheated part of cartridge'),
        ]
        bcs = {'outer_surface': [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]}  
        build_geometry_from_stepfile(
            path = path,
            dir = 'meshes/experiment_TF46',
            mesh_size_max=0.001,
            groups=groups, 
            probes_coords=probes_coords,
            probes_names=probes_names,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=False,
            )
        self.assertTrue(True, "geometry TF46 inventor Failed to generate")

    def test_TF46_axisymetric(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/TF46.stp"
        groups = {'heated':[2], 'unheated':[1, 3]}
        probes_coords = []
        probes_names = []
        mats = [
            (materials.Steel04, 'Heated part of cartridge'),
            (materials.Steel04, 'Unheated part of cartridge'),
        ]
        bcs = {'outer_surface': [3,4,5,6,7,10,12,13,14,15,16,17,18,19,20,21]}  
        
        build_geometry_from_stepfile(
            path = path,
            dir = 'meshes/experiment_TF46_axisymetry',
            mesh_size_max=0.001,
            groups=groups, 
            probes_coords=probes_coords,
            probes_names=probes_names,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            )
        self.assertTrue(True, "geometry TF46 inventor Failed to generate")

if __name__ == '__main__':
    unittest.main()
