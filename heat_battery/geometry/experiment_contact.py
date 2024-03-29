from mpi4py import MPI
from math import pi
import gmsh
import os
from .. import materials
from .utilities import convert_to_legacy_fenics
from ..utilities import save_data

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
        dir='meshes/experiment_contact',
        legacy_fenics=False,
        h_b=0.118,           # vyska spodni casti (m)
        h_t=0.2485,          # vyska horni casti (m)
        h_d=0.067,           # vyska nasunuti (m)
        h_c=0.217,           # delka stopky patrony (m)
        h_c_heated=0.191,    # delka ohrivane casti patrony (m)
        h_e=0.075,           # vyska spodni casti po dno (m)
        h_unfill=0.019,      # vyska nezaplnena piskem (m)
        d_c=0.014,           # prumer partony (m)
        d_cG=0.021,          # prumer zavitu partony (m)
        d_c_bolt=0.024,      # prumer matice partony (m)
        h_cG=0.018,          # delka zavitu patrony (m)
        h_c_bolt=0.013,      # delka matice patrony (m)
        r_s=0.125,           # polomer valce s piskem (m)
        t_w_in_b = 0.0006,   # tloustka spodni vnitrni steny (m)
        t_w_out_b = 0.0005,  # tloustka spodni vnejsi steny (m)
        t_w_in_t = 0.0006,   # tloustka horni vnitrni steny (m)
        t_w_out_t = 0.0005,  # tloustka horni vnejsi steny (m)
        t_i_b=0.029,         # tloustka izolace na spodni casti (m)
        t_i_t=0.005,          # tloustka izolace na horni casti (m)
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

        # outer wall bottom
        r_outer_b = r_s + t_w_in_b + t_i_b + t_w_out_b
        outer_wall_b = add_anulus(0, h_b, r_outer_b-t_w_out_b, t_w_out_b, dim=dim, angle=angle)
        r_outer_t = r_outer_b + t_w_out_t
        outer_wall_t = add_anulus(h_b-h_d, h_t, r_outer_t-t_w_out_t, t_w_out_t, dim=dim, angle=angle)
        outer_wall = gmsh.model.occ.fuse([(dim, outer_wall_t)], [(dim, outer_wall_b)])

        # inner wall
        inner_wall_b = add_anulus(0, h_b, r_s, t_w_in_b, dim=dim, angle=angle)
        inner_wall_t = add_anulus(h_b-h_d, h_t, r_s-t_w_in_t, t_w_in_t, dim=dim, angle=angle)
        inner_wall = gmsh.model.occ.fuse([(dim, inner_wall_t)], [(dim, inner_wall_b)])

        # bottom plate
        plate_b = add_bottom_plate(0.36, 0.002, dim=dim, angle=angle)

        # all steel
        steel = gmsh.model.occ.fuse([(dim, plate_b)], [outer_wall[0][0], inner_wall[0][0]]) 

        h0_heater = h_b-h_d+h_t-h_c-h_cG+t_tp-h_unfill
        h0_thread = h0_heater + h_c
        h0_bolt = h0_thread + h_cG
        
        # insulation wall bottom
        insulation_b = add_anulus(0, h_b, r_s+t_w_in_b, t_i_b, dim=dim, angle=angle)
        insulation_t = add_anulus(h_b, h_t-h_d, r_s, r_outer_t-r_s-t_w_out_t, dim=dim, angle=angle)
        insulation_plate_b = add_cylinder(0, h_b-h_e, r_s, dim=dim, angle=angle)
        insulation_plate_t = add_cylinder(h_b-h_d+h_t, t_i_t, r_outer_t, dim=dim, angle=angle)
        insulation_plate_above_sand = add_cylinder(h_b-h_d+h_t-h_unfill+t_tp, h_unfill-t_tp, r_s-t_w_in_t, dim=dim, angle=angle)
        cartridge_bolt = add_cylinder(h0_bolt, h_c_bolt, d_c_bolt/2, dim=dim, angle=angle)
        insulation_plate_above_sand = gmsh.model.occ.cut([(dim, insulation_plate_above_sand)], [(dim, cartridge_bolt)])
    
        insulation = gmsh.model.occ.fuse([(dim, insulation_b)],
                                        [(dim, insulation_t),
                                        (dim, insulation_plate_b),
                                        (dim, insulation_plate_t),
                                        insulation_plate_above_sand[0][0]])
        
        sand_b = add_cylinder(h_b-h_e, h_e-h_d, r_s, dim=dim, angle=angle)
        sand_t = add_cylinder(h_b-h_d, h_t-h_unfill, r_s-t_w_in_t, dim=dim, angle=angle)
        sand_full = gmsh.model.occ.fuse([(dim, sand_b)], [(dim, sand_t)])
        cartridge_heater = add_cylinder(h0_heater, h_c, r_c, dim=dim, angle=angle)
        cartridge_thread = add_cylinder(h0_thread, h_cG, r_cG, dim=dim, angle=angle)
        cartridge_bolt = add_cylinder(h0_bolt, h_c_bolt, r_c_bolt, dim=dim, angle=angle)
        cartridge = gmsh.model.occ.fuse([(dim, cartridge_heater)], [(dim, cartridge_thread), (dim, cartridge_bolt)])
        sand = gmsh.model.occ.cut(sand_full[0], cartridge[0])

        top_plate_full = add_cylinder(h_b-h_d+h_t-h_unfill, t_tp, r_s-t_w_in_t, dim=dim, angle=angle)
        cartridge_thread = add_cylinder(h0_thread, h_cG, r_cG, dim=dim, angle=angle)
        top_plate = gmsh.model.occ.cut([(dim, top_plate_full)], [(dim, cartridge_thread)])

        cartridge_heated = add_cylinder(h0_heater, h_c_heated, r_c, dim=dim, angle=angle)
        cartridge_unheated = add_cylinder(h0_heater+h_c_heated, h_c-h_c_heated, r_c, dim=dim, angle=angle)
        cartridge_thread = add_cylinder(h0_thread, h_cG, r_cG, dim=dim, angle=angle)
        cartridge_bolt = add_cylinder(h0_bolt, h_c_bolt, r_c_bolt, dim=dim, angle=angle)
        cartridge_unheated = gmsh.model.occ.fuse([(dim, cartridge_unheated)], [(dim, cartridge_thread), (dim, cartridge_bolt)])

        #TODO: find a way to use surface normal extrusions so it is more automatic
        #      and can be made for other surfacece easily
        dr = 0.0001
        contact_enlarged = add_cylinder(h0_heater-dr, h_c_heated+dr, r_c+dr, dim=dim, angle=angle)
        contact_hole = add_cylinder(h0_heater, h_c_heated, r_c, dim=dim, angle=angle)
        cartridge_contact = gmsh.model.occ.cut([(dim, contact_enlarged)], [(dim, contact_hole)])

        f_tags, f_dim_tags = gmsh.model.occ.fragment(steel[0], insulation[0]+sand[0]+cartridge_unheated[0]+[(dim, cartridge_heated)]+top_plate[0]+cartridge_contact[0])

        gmsh.model.occ.synchronize()

        # mark subdomains
        gmsh.model.addPhysicalGroup(dim, [f_tags[0][1], f_tags[5][1]], 1, 'steel')
        gmsh.model.addPhysicalGroup(dim, [f_tags[1][1]], 2, 'insulation')
        gmsh.model.addPhysicalGroup(dim, [f_tags[2][1]], 3, 'insulation bottom')
        gmsh.model.addPhysicalGroup(dim, [f_tags[3][1]], 4, 'cartridge_unheated')
        gmsh.model.addPhysicalGroup(dim, [f_tags[4][1]], 5, 'cartridge_heated')
        gmsh.model.addPhysicalGroup(dim, [f_tags[7][1]], 6, 'sand')
        gmsh.model.addPhysicalGroup(dim, [f_tags[6][1]], 7, 'cartridge_contact')
        
        mats = [
            (materials.Steel04, 'Steel parts'),                           #1
            (materials.Standard_insulation, 'Insulation'),                #2
            (materials.Standard_insulation, 'Insulation bottom'),         #3
            (materials.Cartridge_unheated, 'Unheated part of cartridge'), #4 
            (materials.Cartridge_heated, 'Heated part of cartridge'),     #5
            (materials.SandTheory, 'Sand'),                               #6
            (materials.Contact_sand, 'Contact_sand'),                     #7
        ]

        # mark surfaces
        if dim == 3:
            if symetry_3d is None:
                gmsh.model.addPhysicalGroup(dim-1, [12, 13, 14, 15, 16, 17, 18, 19, 20, 34, 35], 1, 'outer_surface')
                jac_f = lambda x: 1
            elif symetry_3d == 'half':
                gmsh.model.addPhysicalGroup(dim-1, [13, 15, 16, 17, 18, 21, 26, 27, 37, 38], 1, 'outer_surface')
                jac_f = lambda x: 2
            elif symetry_3d == 'quarter':
                gmsh.model.addPhysicalGroup(dim-1, [18, 19, 20, 21, 33, 34, 35, 40, 41], 1, 'outer_surface')
                jac_f = lambda x: 4
            boundary_list_type = 'SurfacesList'
        elif dim == 2:
            gmsh.model.addPhysicalGroup(dim-1, [13, 14, 15, 16, 17, 18, 33, 34], 1, 'outer_surface')
            volume_symm_coeff, surface_symm_coeff = 1, 1
            jac_f = lambda x: 2*pi*x[0]
            boundary_list_type = 'CurvesList'

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size_max)
        gmsh.model.mesh.field.add('Distance', 1)
        surface_list = [dimtag[1] for dimtag in gmsh.model.getBoundary([(dim, cartridge_heated)], oriented=False)]
        gmsh.model.mesh.field.setNumbers(1, boundary_list_type, surface_list)
        gmsh.model.mesh.field.add('MathEval', 3)
        gmsh.model.mesh.field.setString(3, 'F', f'{mesh_size_min}+{mesh_growth}*F1')
        gmsh.model.mesh.field.setAsBackgroundMesh(3)

        gmsh.model.mesh.generate(dim)
        gmsh.model.mesh.optimize()
        gmsh.write(gmsh_file)

        if fltk:
            gmsh.fltk.run()
        gmsh.finalize()

        h_ref = h_b-h_d+h_t-h_unfill
        
        probes_coords = [
            [r_c+0.022, 0.0, h_ref-0.06],[r_c+2*0.022, 0.0, h_ref-0.06],[r_c+3*0.022, 0.0, h_ref-0.06], # radial top sensors
            [r_c+0.022, 0.0, h_ref-2*0.06],[r_c+2*0.022, 0.0, h_ref-2*0.06],[r_c+3*0.022, 0.0, h_ref-2*0.06], # radial mid sensors
            [r_c+0.022, 0.0, h_ref-3*0.06],[r_c+2*0.022, 0.0, h_ref-3*0.06],[r_c+3*0.022, 0.0, h_ref-3*0.06], # radial bottom sensors
            [r_c, 0.0, h_ref-0.054],[r_c, 0.0, h_ref-2*0.054],[r_c, 0.0, h_ref-3*0.054], # cartridge surface
            [r_outer_t, 0.0, 0.065],[r_outer_t, 0.0, 0.065],[r_outer_t, 0.0, 0.065], # outer surface sensors
        ]

        probes_names = [
            '1 - Top [°C]', '2 - Top [°C]', '3 - Top [°C]',
            '4 - Middle [°C]', '5 - Middle [°C]', '6 - Middle [°C]',
            '7 - Bottom [°C]', '8 - Bottom [°C]', '9 - Bottom [°C]',
            '10 - A - Surface [°C]', '11 - B - Surface [°C]', '12 - C - Surface [°C]',
            '13 - I. Cover [°C]', '14 - II. Cover [°C]', '15 - III. Cover [°C]'
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
            'source_term_index':5}
            
        if legacy_fenics:
            convert_to_legacy_fenics(gmsh_file)

        save_data(add_data_file, add_data)

    MPI.COMM_WORLD.Barrier()

    return None

