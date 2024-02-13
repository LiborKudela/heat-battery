from mpi4py import MPI
from math import pi
import gmsh
import os
from heat_battery import materials
from heat_battery.geometry.utilities import convert_to_legacy_fenics
from heat_battery.utilities import save_data

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
        dir='meshes/experiment_v2',
        legacy_fenics=False,
        d_wire = 0.003,          # diameter of the heating wire (m)
        wire_gap = 0.0001,       # gap bwtween wire and container (m)
        h_medium = 0.07,
        d_medium = 0.05,         # diameter 
        t_c_side=0.001,
        t_c_caps=0.001,
        mesh_size_max = 0.001,   # priblizna max velikost elemenu (m)
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
        wire = add_cylinder(-t_c_caps, h_medium+2*t_c_caps, d_wire/2, dim=dim, angle=angle)

        h_c = h_medium+2*t_c_caps
        d_c = d_medium+2*t_c_side
        _container = add_cylinder(-t_c_caps, h_c, d_c/2, dim=dim, angle=angle)
        _medium = add_cylinder(0, h_medium, d_medium/2, dim=dim, angle=angle)
        _wire = add_cylinder(-t_c_caps, h_medium+2*t_c_caps, d_wire/2+wire_gap, dim=dim, angle=angle)
        container = gmsh.model.occ.cut([(dim, _container)], [(dim, _medium), (dim, _wire)])

        _medium = add_cylinder(0, h_medium, d_medium/2, dim=dim, angle=angle)
        _wire = add_cylinder(-t_c_caps, h_medium+2*t_c_caps, d_wire/2, dim=dim, angle=angle)
        medium = gmsh.model.occ.cut([(dim, _medium)], [(dim, _wire)])


        f_tags, f_dim_tags = gmsh.model.occ.fragment([(dim, wire)], medium[0]+container[0])
        print(f_tags)

        gmsh.model.occ.synchronize()

        # mark subdomains
        gmsh.model.addPhysicalGroup(dim, [f_tags[0][1]], 1, 'wire')
        gmsh.model.addPhysicalGroup(dim, [f_tags[1][1]], 2, 'medium')
        gmsh.model.addPhysicalGroup(dim, [f_tags[2][1]], 3, 'container')

        mats = [
            (materials.Steel04, 'Wire'), 
            (materials.SandTheory, 'Medium'),
            (materials.Steel04, 'Container'),
        ]

        # mark surfaces
        if dim == 3:
            if symetry_3d is None:
                gmsh.model.addPhysicalGroup(dim-1, [25, 26, 27], 1, 'outer_surface')
                jac_f = lambda x: 1
            elif symetry_3d == 'half':
                gmsh.model.addPhysicalGroup(dim-1, [28, 29, 32], 1, 'outer_surface')
                jac_f = lambda x: 2
            elif symetry_3d == 'quarter':
                gmsh.model.addPhysicalGroup(dim-1, [35, 36, 38], 1, 'outer_surface')
                jac_f = lambda x: 4
            boundary_list_type = 'SurfacesList'
        elif dim == 2:
            gmsh.model.addPhysicalGroup(dim-1, [26, 27, 28], 1, 'outer_surface')
            volume_symm_coeff, surface_symm_coeff = 1, 1
            jac_f = lambda x: 2*pi*x[0]
            #boundary_list_type = 'CurvesList'

        
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size_max)
        gmsh.model.mesh.generate(dim)
        gmsh.write(gmsh_file)

        if fltk:
            gmsh.fltk.run()
        gmsh.finalize()

        h_mid = h_medium/2
        wire_r = d_wire/2
        medium_r = d_medium/2
        probes_coords = [
            [0.0, 0.0, h_medium/2], [wire_r-1e-6, 0.0, h_medium/2], [wire_r+1e-6, 0.0, h_medium/2],
            [medium_r-1e-6, 0.0, h_mid], [medium_r+1e-6, 0.0, h_mid], [medium_r+t_c_side, 0.0, h_mid], 
        ]

        probes_names = [
            'T - wire mid', 'T - wire surf', 'T - medium surf',
            'T - medium outer surf', 'T - container inner surf', 'T - container outer surf'
        ]

        if dim == 2:
            for i, item in enumerate(probes_coords):
                probes_coords[i] = [item[0], item[2], 0.0]

        add_data = {
            'symmetry':symetry_3d, 
            'probes_coords':probes_coords,
            'probes_names':probes_names,
            'materials':mats,
            'jac_f':jac_f,
            'outer_surface_index':1,
            'source_term_index':1}
            
        if legacy_fenics:
            convert_to_legacy_fenics(gmsh_file)

        save_data(add_data_file, add_data)

    MPI.COMM_WORLD.Barrier()

    return None
