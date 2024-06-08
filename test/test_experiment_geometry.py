import unittest
from heat_battery.geometry.dummy import build_geometry as build_dummy
from heat_battery.geometry.experiment import build_geometry as build_without_contact
from heat_battery.geometry.experiment_contact import build_geometry as build_with_contact
from heat_battery.geometry.experiment_v2 import build_geometry as build_experiment_v2
from heat_battery.geometry.tantalum_wire import build_geometry as build_tantalum_wire
from heat_battery.geometry.experiment_cartridge import build_geometry as build_cartridge
from heat_battery.geometry.step_loader import build_geometry_from_stepfile
from heat_battery.materials import materials
from math import sqrt

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
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/Experiment_v1.stp"
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

        probes = {
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
            probes=probes,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            )

    def test_experiment_v21_inventor_axisymetry(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/Experiment_v2.1.stp"
     
        probes = {
            'T':{
                'T - wire mid': [0.0, 0.0, 0.0], 
                'T - wire surf': [0.0001-1e-6, 0.0, 0.0],
                'T - sand': [0.001, 0.0, 0.0],
                'T - surf can':[0.042/2, 0.0, 0.0], 
            },
        }

        mats = {
            'wire':(materials.TantalumWire, [2]), 
            'electrodes':(materials.Steel04, [1, 3]), 
            'sand':(materials.Linear_sand, [4]), 
            'case':(materials.Steel04, [7]), 
            'th':(materials.Standard_insulation, [5, 6]), 
            'lid':(materials.Steel04, [8])
            }
        custom_data = {
            'wire':{'length':0.15, 'diameter':0.0002},
        }
        bcs = {'outer_surface': [7,8,9,14,15,16,30,31,32,33,34,35,36,37,41,42,43,44,45]}
        build_geometry_from_stepfile(
            path = path,
            dir='meshes/experiment_inventor',  
            mesh_size_max=0.0001, 
            mats=mats,
            probes=probes,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            custom_data=custom_data,
            )
        self.assertTrue(True, "2D Geometry Experiment 2 Failed to generate")

    def test_experiment_v22_inventor_axisymetry(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/Experiment_v2.2.stp"
       
        probes = {
            'T':{
                'T - wire mid': [0.0, 0.0, 0.0], 
                'T - wire surf': [0.0001-1e-6, 0.0, 0.0],
                'T - sand': [0.001, 0.0, 0.0],
                'T - surf can':[0.032/2, 0.0, 0.0], 
            },
        }

        mats = {
            'wire':(materials.TantalumWire, [2]), 
            'electrodes':(materials.Steel04, [1, 3]), 
            'contact':(materials.new_contact_class(0.000001), [4]), 
            'sand':(materials.Linear_sand, [5]), 
            'th':(materials.Standard_insulation, [6, 7]), 
            'case':(materials.Steel04, [8]), 
            'lid':(materials.Steel04, [9])
            }
        custom_data = {
            'wire':{'length':0.15, 'diameter':0.0002},
        }
        bcs = {'outer_surface': [8,9,10,15,16,17,33,34,35,36,37,38,39,40,44,45,46,47,48]}
        build_geometry_from_stepfile(
            path = path,
            name='mesh_contact',
            dir='meshes/experiment_inventor',  
            mesh_size_max=0.0001, 
            probes=probes,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            custom_data=custom_data,
            )
        self.assertTrue(True, "2D Geometry Experiment 2 Failed to generate")

    def test_twowire_stp(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/TwoWire.stp"
 
        probes = {
            'T':{
                'T - wire mid': [0.0, 0.0, 0.0], 
                'T - wire surf': [0.0001-1e-6, 0.0, 0.0],
                'T - sand': [0.001, 0.0, 0.0],
                'T - surf can':[0.032/2, 0.0, 0.0], 
            },
        }

        mats = {
            'short_wire': (materials.TantalumWire, [2]),
            'long_wire': (materials.TantalumWire, [7]),
            'electrodes': (materials.Steel04, [1, 3, 5, 8]),
            'sand': (materials.Linear_sand, [4, 9]),
            'th': (materials.Standard_insulation, [6, 10, 11, 13]),
            'case': (materials.Steel04, [12, 16]),
            'lid': (materials.Steel04, [14, 15]),
        }
        custom_data = {
            'short_wire':{'length':0.07, 'diameter':0.0002},
            'long_wire':{'length': 0.15, 'diameter':0.0002},
        }
        bcs = {'outer_surface': [7,8,9,14,15,16,28,29,30,39,40,41,42,51,52,53,61,62,63,64,67,68,69,70,74,75,76,77,80,81,82,83,84,85,87,88,89,90]}
        build_geometry_from_stepfile(
            path = path,
            name='mesh',
            dir='meshes/two_wire',
            mesh_size_max=0.0001,  
            probes=probes,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            custom_data=custom_data,
            )
        self.assertTrue(True, "Two wire method experiment")

    def test_TF46(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/TF46.stp"
        mats = {
            'heated cartridge':(materials.Steel04, [1]), 
            'unheated':(materials.Steel04, [2, 3])
            }
        probes={}
        bcs = {'outer_surface': [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]}  
        build_geometry_from_stepfile(
            path = path,
            dir = 'meshes/experiment_TF46',
            mesh_size_max=0.001, 
            probes=probes,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=False,
            )
        self.assertTrue(True, "geometry TF46 inventor Failed to generate")

    def test_TF46_axisymetric(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/TF46.stp"
        mats = {
            'heated':(materials.Steel04, [2]), 
            'unheated':(materials.Steel04, [1, 3])}
        probes={}
        bcs = {'outer_surface': [3,4,5,6,7,10,12,13,14,15,16,17,18,19,20,21]}  
        build_geometry_from_stepfile(
            path = path,
            dir = 'meshes/experiment_TF46_axisymetry',
            mesh_size_max=0.001,
            probes=probes,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=True,
            )
        self.assertTrue(True, "geometry TF46 inventor Failed to generate")

    def test_urbanek(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/Urbanek_v2.stp"
        mats = {
            'sand':(materials.Sand_urbanek, [1]), 
            'insulation':(materials.Insulation_urbanek, [4]),    
            'heated cartridge':(materials.Cartridge_heated, [2, 3]), 
        }

        probes = {
            'T':{
                'inner_cartridge': [0, -0.16/2,-1.7/2],
                'outer_cartridge': [0, -0.36/2,-1.7/2], 
            },
        }

        bcs = {
            'outer_surface': [14,15,18],
            'cartridge_surface': [1,2,3,4,5,6],
            }
        
        build_geometry_from_stepfile(
            path = path,
            dir = 'meshes/urbanek',
            mesh_size_max=0.04,
            probes=probes,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=False,
            override_jac=lambda x: 8,
            mesh_size_from_curvature=18,
            )
        
    def test_urbanek_mem(self):
        path = "/home/numlab/Projects/CurrentProjects/HeatBattery/test/Urbanek_membrane.stp"
        mats = {
            'sand':(materials.Sand_urbanek, [1]), 
            'insulation':(materials.Insulation_urbanek, [4]),    
            'heated cartridge':(materials.Cartridge_heated, [2, 3]), 
        }

        probes = {
            'T':{
                'inner_cartridge': [0, -0.16/2,-1.7/2],
                'outer_cartridge': [0, -0.36/2,-1.7/2], 
            },
        }

        bcs = {
            'outer_surface': [21,22,25],
            'mebrane_surface': [14,15,16,17,18,19],
            'cartridge_surface': [1,2,3,4,5,6],
            }
        
        build_geometry_from_stepfile(
            path = path,
            dir = 'meshes/urbanek_mem',
            mesh_size_max=0.04,
            probes=probes,
            mats=mats,
            bcs=bcs,
            fltk=False,
            extract_axisymetry=False,
            override_jac=lambda x: 8,
            mesh_size_from_curvature=18,
            )

if __name__ == '__main__':
    unittest.main()
