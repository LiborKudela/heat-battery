from heat_battery.geometry.step_loader import build_geometry_from_stepfile
from heat_battery.materials import materials
stp_path = "TwoWire.stp" # file generated in Autodesk Inventorw
points = {
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

bcs = {
    'outer_surface': [7,8,9,14,15,16,28,29,30,39,40,41,42,51,52,53,61,62,63,64,67,68,69,70,74,75,76,77,80,81,82,83,84,85,87,88,89,90]}

if __name__ == '__main__':
    build_geometry_from_stepfile(
        path = stp_path,
        name='mesh',
        dir='meshes/two_wire',
        mesh_size_max=0.0001,  
        points=points,
        mats=mats,
        bcs=bcs,
        fltk=False,
        extract_axisymetry=True,
        custom_data=custom_data,
    )