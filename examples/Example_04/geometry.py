from heat_battery.geometry.step_loader import build_geometry_from_stepfile
from heat_battery.materials import materials

path = "Passive_storage.stp"
mats = {
    'sand':(materials.Sand_urbanek, [1]), 
    'insulation':(materials.Insulation_urbanek, [4]),    
    'heated cartridge':(materials.Cartridge_heated, [2, 3]), 
}

points = {
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

if __name__ == '__main__':
    build_geometry_from_stepfile(
        path = path,
        dir = 'meshes',
        mesh_size_max=0.04,
        points=points,
        mats=mats,
        bcs=bcs,
        fltk=False,
        extract_axisymetry=False,
        override_jac=lambda x: 8,
        mesh_size_from_curvature=18,
        )