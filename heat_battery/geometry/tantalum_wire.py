from mpi4py import MPI
from math import pi
import gmsh
import os
from heat_battery import materials
from heat_battery.geometry.utilities import convert_to_legacy_fenics
from heat_battery.utilities import save_data
from inspect import getargspec

def add_cylinder(h0, h, r, dim=3, angle=2*pi):
    if dim == 3:
        return gmsh.model.occ.addCylinder(0, 0, h0, 0, 0, h, r, angle=angle)
    elif dim == 2:
        return gmsh.model.occ.addRectangle(0, h0, 0, r, h)
    
def add_anulus(h0, h, r_inner, t, dim=3, angle=2*pi):
    if dim == 3:
        outer_cylinder = gmsh.model.occ.addCylinder(0, 0, h0, 0, 0, h, r_inner+t, angle=angle)
        inner_cylinder = gmsh.model.occ.addCylinder(0, 0, h0, 0, 0, h, r_inner, angle=angle)
        anulus = gmsh.model.occ.cut([(3, outer_cylinder)], [(3, inner_cylinder)])
        return anulus[0][0][1]
    elif dim == 2:
        anulus = gmsh.model.occ.addRectangle(r_inner, h0, 0, t, h)
        return anulus
    
def add_bottom_plate(a, t, dim=3, angle=2*pi):
    if dim == 3:
        if angle == 2*pi:
            return gmsh.model.occ.addBox(-a/2, -a/2, -t, a, a, t)
        elif angle == pi:
            return gmsh.model.occ.addBox(-a/2, 0, -t, a, a/2, t)
        elif angle == pi/2:
            return gmsh.model.occ.addBox(0, 0, -t, a/2, a/2, t)
    elif dim == 2:
        return gmsh.model.occ.addRectangle(0, -t, 0, a/2, t)

def build_geometry(
        dim=3,
        dir='meshes/experiment_tanralum_wire',
        legacy_fenics=False,
        d_wire = 0.0002,         # diameter of the heating wire (m)
        l_wire = 0.10,           # length of the heating wire (m)
        mesh_size_max = 0.0005,   # priblizna max velikost elemenu (m)
        mesh_size_min = 0.001,   # priblizna min velikost elemenu (m)
        mesh_growth = 0.5,       # priblizna min velikost elemenu (-)
        fltk=False,
        symetry_3d=None,
        verbosity=0,
    ):

    if MPI.COMM_WORLD.rank == 0:
        
        file_path = dir + f'/mesh_{dim}d'
        gmsh_file = file_path + '.msh'
        add_data_file = file_path + '.ad'

        os.makedirs(dir, exist_ok=True)
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 1)
        gmsh.option.setNumber('General.Verbosity', verbosity)

        gmsh.model.add(f'Experiment_v2_{dim}d')
        gmsh.logger.start()

        if symetry_3d is None:
            angle = 2*pi
        elif symetry_3d == 'half':
            angle = pi
        elif symetry_3d == 'quarter':
            angle = pi/2

        # medium
        wire = add_cylinder(0, l_wire, d_wire/2, dim=dim, angle=angle)

        #f_tags, f_dim_tags = gmsh.model.occ.fragment([(dim, wire)])

        gmsh.model.occ.synchronize()

        # mark surfaces
        if dim == 3:
            if symetry_3d is None:
                bcs = {'outer_surface': [1, 2, 3]}
                jac_f = lambda x: 1
            elif symetry_3d == 'half':
                bcs = {'outer_surface': [1, 2, 3]}
                jac_f = lambda x: 2
            elif symetry_3d == 'quarter':
                bcs = {'outer_surface': [1, 2, 3]}
                jac_f = lambda x: 4
            boundary_list_type = 'SurfacesList'
        elif dim == 2:
            bcs = {'outer_surface': [1, 2, 3]}
            jac_f = lambda x: 2*pi*x[0]
            boundary_list_type = 'CurvesList'

        i = 1
        for bc_name, ents in bcs.items():
            gmsh.model.addPhysicalGroup(dim-1, ents, i, bc_name)
            i += 1

        mats = {
            'wire': (materials.TantalumWire, [wire]), 
        }

        i = 1
        for name, tuple_data in mats.items():
            entities = tuple_data[1]
            gmsh.model.addPhysicalGroup(dim, entities, i, name)
            i += 1
       
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size_max)
        gmsh.model.mesh.generate(dim)
        gmsh.write(gmsh_file)

        if fltk:
            gmsh.fltk.run()
        gmsh.finalize()

        h_mid = l_wire/2
        wire_r = d_wire/2
        probes_coords = [
            [0.0, 0.0, h_mid], [wire_r-1e-6, 0.0, h_mid]
        ]

        probes_names = [
            'T - wire mid', 'T - wire surf'
        ]

        if dim == 2:
            for i, item in enumerate(probes_coords):
                probes_coords[i] = [item[0], item[2], 0.0]

        spec = getargspec(build_geometry).args
        local_scope = locals()
        call_data = dict(zip(spec, [eval(arg, local_scope) for arg in spec]))

        add_data = {
            'call_data':call_data,
            'dim':dim,
            'symmetry':symetry_3d, 
            'probes_coords':probes_coords,
            'probes_names':probes_names,
            'materials':mats,
            'boundaries':bcs,
            'jac_f':jac_f,
            }
            
        if legacy_fenics:
            convert_to_legacy_fenics(gmsh_file)

        save_data(add_data_file, add_data)

    MPI.COMM_WORLD.Barrier()

    return None
