from mpi4py import MPI
from math import pi
import gmsh
import os
from .. import materials
from .utilities import convert_to_legacy_fenics
from ..utilities import save_data
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

def     build_geometry(
        dim=3,
        dir='meshes/experiment_cartridge',
        legacy_fenics=False,
        h_b=0.118,           # vyska spodni casti (m)
        h_t=0.2485,          # vyska horni casti (m)
        h_d=0.067,           # vyska nasunuti (m)
        h_c=0.217,           # delka stopky patrony (m)
        h_c_unheated_top=0.01,    # delka ohrivane casti patrony (m)     
        h_c_unheated_bottom=0.01,    # delka ohrivane casti patrony (m)             
        h_unfill=0.021,    # vyska nezaplnena piskem (m)
        d_c=0.014,           # prumer partony (m)
        d_cG=0.021,          # prumer zavitu partony (m)
        d_c_bolt=0.024,      # prumer matice partony (m)
        h_cG=0.018,          # delka zavitu patrony (m)
        h_c_bolt=0.013,      # delka matice patrony (m)
        t_tp = 0.003,        # tloustka horni desky (m)
        mesh_size_max = 0.01,    # priblizna max velikost elemenu (m)
        mesh_size_min = 0.001,   # priblizna min velikost elemenu (m)
        mesh_growth = 0.5,   # priblizna min velikost elemenu (-)
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

        gmsh.model.add('Experiment_3d')
        gmsh.logger.start()

        r_c = d_c/2
        r_cG = d_cG/2
        r_c_bolt = d_c_bolt/2

        if symetry_3d is None:
            angle = 2*pi
        elif symetry_3d == 'half':
            angle = pi
        elif symetry_3d == 'quarter':
            angle = pi/2

        h0 = 0.0
        heated_h0 = h0 + h_c_unheated_bottom
        heated_h = h_c - h_c_unheated_bottom - h_c_unheated_top
        unheated_top_h0 = h0 + h_c_unheated_bottom + heated_h
        unheated_bottom = add_cylinder(h0, h_c_unheated_bottom, r_c, dim=dim, angle=angle)
        heated = add_cylinder(heated_h0, heated_h, r_c, dim=dim, angle=angle)
        unheated_top = add_cylinder(unheated_top_h0, h_c_unheated_top, r_c, dim=dim, angle=angle)
        
        h0_thread = h0 + h_c
        h0_bolt = h0_thread + h_cG
        thread = add_cylinder(h0_thread, h_cG, r_cG, dim=dim, angle=angle)
        bolt = add_cylinder(h0_bolt, h_c_bolt, r_c_bolt, dim=dim, angle=angle)
        thread = gmsh.model.occ.fuse([(dim, thread)], [(dim, bolt)])

        f_tags, f_dim_tags = gmsh.model.occ.fragment(thread[0], [(dim, unheated_top), (dim, heated), (dim, unheated_bottom)])

        gmsh.model.occ.synchronize()

        # mark subdomains
        gmsh.model.addPhysicalGroup(dim, [f_tags[0][1], f_tags[1][1], f_tags[3][1]], 1, 'cartridge_unheated')
        gmsh.model.addPhysicalGroup(dim, [f_tags[2][1]], 2, 'cartridge_heated')

        mats = [
            (materials.Cartridge_unheated, 'Unheated part of cartridge'),  
            (materials.Cartridge_heated, 'Heated part of cartridge'), 
        ]

        # mark surfaces
        if dim == 3:
            if symetry_3d is None:
                gmsh.model.addPhysicalGroup(dim-1, [1, 2, 3, 4, 5, 6, 8, 9], 1, 'outer_surface')
                jac_f = lambda x: 1
            elif symetry_3d == 'half':
                #gmsh.model.addPhysicalGroup(dim-1, [1, 3, 4, 5, 6, 9, 14, 15, 29, 30], 1, 'outer_surface')
                jac_f = lambda x: 2
            elif symetry_3d == 'quarter':
                #gmsh.model.addPhysicalGroup(dim-1, [5, 6, 7, 8, 23, 24, 25, 31, 32], 1, 'outer_surface')
                jac_f = lambda x: 4
            boundary_list_type = 'SurfacesList'
        elif dim == 2:
            gmsh.model.addPhysicalGroup(dim-1, [2, 3, 4, 5, 6, 7, 9, 10], 1, 'outer_surface')
            volume_symm_coeff, surface_symm_coeff = 1, 1
            jac_f = lambda x: 2*pi*x[0]
            boundary_list_type = 'CurvesList'

        bcs = [
            ('outer_surface'),
        ]

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size_max)
        gmsh.model.mesh.generate(dim)
        gmsh.model.mesh.optimize()
        gmsh.write(gmsh_file)

        if fltk:
            gmsh.fltk.run()
        gmsh.finalize()

        h_ref = h_b-h_d+h_t-h_unfill
        probes_coords = [
            [r_c, 0.0, h_ref-0.054],[r_c, 0.0, h_ref-2*0.054],[r_c, 0.0, h_ref-3*0.054], # cartridge surface
        ]

        probes_names = [
            '10 - A - Surface [°C]', '11 - B - Surface [°C]', '12 - C - Surface [°C]',
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


