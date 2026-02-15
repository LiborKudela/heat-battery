
from mpi4py import MPI
from heat_battery.geometry.utilities import save_mesh_add_data, find_closest_entity
from heat_battery import materials
import inspect
import gmsh
import os
import math
import numpy as np

def simple_cartridge(x, y, z, d, h):
    return (3, gmsh.model.occ.addCylinder(x, y, z, 0, 0, h, d/2))

def beam_spreader_cartridge(x, y, z, d, h, db, nb, tb):
    main_cylinder = [(3, gmsh.model.occ.addCylinder(x, y, z, 0, 0, h, d/2+tb))]
    beams = []
    delta_phi = math.pi/nb
    for i in range(nb):
        b = gmsh.model.occ.addBox(x-tb/2, y-db/2, z, tb, db, h)
        gmsh.model.occ.rotate([(3, b)], x, y, z, 0, 0, 1, i*delta_phi)
        beams.append((3, b))
    body = gmsh.model.occ.fuse(main_cylinder, beams)[0]
    hole = [(3, gmsh.model.occ.addCylinder(x, y, z, 0, 0, h, d/2))]
    spreader = gmsh.model.occ.cut(body, hole)
    return spreader[0][0]

def tht_beam_spreader(x, y, z0_pipe, z0_fins, d, h_pipe, h_fins, db, nb, tb):
    main_cylinder = [(3, gmsh.model.occ.addCylinder(x, y, z0_pipe, 0, 0, h_pipe, d/2+tb))]
    beams = []
    delta_phi = math.pi/nb
    for i in range(nb):
        b = gmsh.model.occ.addBox(x-tb/2, y-db/2, z0_fins, tb, db, h_fins)
        gmsh.model.occ.rotate([(3, b)], x, y, z0_pipe, 0, 0, 1, i*delta_phi)
        beams.append((3, b))
    body = gmsh.model.occ.fuse(main_cylinder, beams)[0]
    return body[0]

def cylinder_spreader(x, y, z, d, db, h, nb, tb):
    main_cylinder = [(3, gmsh.model.occ.addCylinder(x, y, z, 0, 0, h, d/2+tb))]
    hole = [(3, gmsh.model.occ.addCylinder(x, y, z, 0, 0, h, d/2))]
    spreader = gmsh.model.occ.cut(main_cylinder, hole)
    return spreader[0][0]

def build_geometry(
        name='mesh',
        dir="meshes/C3_passive",
        verbosity=0,
        mesh_size_max = 0.1,
        mesh_size_from_curvature=16,
        fltk=False,
        symmetry=False,
        size=1,
        t_insulation=0.1,
        cartridge_n=3, 
        cartridge_d_ratio=0.5,
        cartridge_diameter=0.014,
        cartridge_h_ratio=0.8,
        cartridge_spreader_lb=0.15,
        cartridge_spreader_nb=3,
        cartridge_spreader_tb=0.005,
        cartridge_spreader_mesh_size_min = 0.002,
        cartridge_spreader_mesh_grow_factor = 0.8,

        tht_in_sand=True,
        tht_d=0.05,
        tht_d_ratio=0.1,
        tht_n_ratio = 0.5,
        tht_spreader_h_ratio=0.8,
        tht_spreader_lb=0.15,
        tht_spreader_nb=3,
        tht_spreader_tb=0.005,
        thp_spreader_mesh_size_min = 0.002,
        thp_spreader_mesh_grow_factor = 0.8,
        thp_mesh_size_min = 0.002,
        thp_mesh_grow_factor = 0.8,
        thp_surface_segments = 1,
        
        sand_material="SandTheory",
        insulation_material="Standard_insulation",
        thp_spreader_material="Steel04",
        cartridge_material="Steel04",
        cartridge_spreader_material="Steel04",
        ):

    cartridge_spreader_db = cartridge_diameter+cartridge_spreader_tb+2*cartridge_spreader_lb
    tht_spreader_db = tht_d+tht_spreader_tb+2*tht_spreader_lb
    assert cartridge_d_ratio >= 0 and cartridge_d_ratio <= 1, "c_position must be between 0 and 1"
    assert tht_d_ratio >= 0 and tht_d_ratio <= 1, "m_position must be between 0 and 1"
    assert cartridge_h_ratio >= 0 and cartridge_h_ratio <= 1, "h_c_ratio must be between 0 and 1"
    assert tht_n_ratio >= 0 and tht_n_ratio <= 1, "n_m_ratio must be between 0 and 1"
    assert tht_d < t_insulation, "THT pipes cannot be bigger than insulation layer"
    if not tht_in_sand:
        assert tht_spreader_db < t_insulation, "THT spreader fins are too big to fit in insulation layer"

    d_sand=size
    h_sand=size

    gmsh_file = dir + f'/{name}.msh'
    step_file = dir + f'/{name}.step'
    add_data_file = dir + f'/{name}.ad'

    spec = inspect.getfullargspec(build_geometry).args
    local_scope = locals()
    call_data = dict(zip(spec, [eval(arg, local_scope) for arg in spec]))

    print("  Generating new model")
    os.makedirs(dir, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', verbosity)
    gmsh.option.setNumber("General.Terminal", 1 if verbosity > 0 else 0)

    gmsh.model.add("C3_passive")
    gmsh.logger.start()

    
    angle = 2*math.pi/cartridge_n
    dim = 3
    jac_f = lambda x: 4*cartridge_n

    # insulation
    r_max = d_sand/2 + t_insulation
    if symmetry:
        h_outer = h_sand/2 + t_insulation  
        h_inner = h_sand/2
        h_mid = h_outer
        if tht_in_sand:
            h_tht_pipe = h_outer
            h_tht_fins = h_sand*tht_spreader_h_ratio/2
            h0_tht_fins = h_mid-h_tht_fins
        else:
            h_tht_pipe = h_outer
            h_tht_fins = h_outer*tht_spreader_h_ratio
            h0_tht_fins = h_mid-h_tht_fins
    else:
        h_outer = h_sand + 2*t_insulation
        h_inner = h_sand
        h_mid = h_outer/2
        if tht_in_sand:
            h_tht_pipe = h_outer
            h_tht_fins = h_sand*tht_spreader_h_ratio
            h0_tht_fins = h_mid-h_tht_fins/2
        else:
            h_tht_pipe = h_outer
            h_tht_fins = h_outer*tht_spreader_h_ratio
            h0_tht_fins = h_mid-h_tht_fins/2
    outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, h_outer, r_max, angle=angle/2)
    inner_cylinder = gmsh.model.occ.addCylinder(0, 0, t_insulation, 0, 0, h_inner, d_sand/2, angle=angle/2)
    gmsh.model.occ.rotate([(3, outer_cylinder)], 0, 0, 0, 0, 0, 1, -angle/2)
    gmsh.model.occ.rotate([(3, inner_cylinder)], 0, 0, 0, 0, 0, 1, -angle/2)


    # how much angle one cartridge takes

    # cartridge
    cartridges = []
    spreaders = []
    cartridge_tc = []
    r_c = cartridge_d_ratio*(d_sand/2-cartridge_spreader_db/2-tht_d/2-0.001)
    assert cartridge_spreader_db/2/r_c < math.sin(angle/2), "Cartridge spreaders are too large and do not fit in this number and positions"
    h_c = cartridge_h_ratio*h_sand
    x, y = r_c*math.cos(0), r_c*math.sin(0)
    #c = gmsh.model.occ.addCylinder(x, y, t_insulation+h_sand-h_c, 0, 0, h_c, d_cartridge/2)
    c = simple_cartridge(x, y, t_insulation+h_sand-h_c-(h_sand-h_c)/2, cartridge_diameter, h_c)
    c = gmsh.model.occ.intersect([c], [(3, inner_cylinder)], removeObject=True, removeTool=False)[0][0]
    s = beam_spreader_cartridge(x, y, t_insulation+h_sand-h_c-(h_sand-h_c)/2, cartridge_diameter, h_c, db=cartridge_spreader_db, nb=cartridge_spreader_nb, tb=cartridge_spreader_tb)
    s = gmsh.model.occ.intersect([s], [(3, inner_cylinder)], removeObject=True, removeTool=False)[0][0]
    cartridges.append(c)
    spreaders.append(s)
    cartridge_tc.append([x, y, h_mid])
        
    # mem pipe positioning for mem in sand
    dr_m_tol = 1e-3
    # dr_m is radial distance from the interface between sand and insulation
    if tht_in_sand:
        dr_m = -(tht_spreader_db/2+dr_m_tol+tht_d_ratio*(d_sand/2-r_c-cartridge_spreader_db/2-tht_spreader_db-2*dr_m_tol))
    else:
        dr_m = tht_spreader_db/2+dr_m_tol+tht_d_ratio*(t_insulation-tht_spreader_db-2*dr_m_tol)
    d_m_angle = (2*math.asin((tht_spreader_db/2)/(d_sand/2+dr_m))) # how much angle one mem pipe takes
    n_m_request = (math.pi/cartridge_n*tht_n_ratio)/d_m_angle # float of how many mem pipes fit in
    n_m = max(int(round(n_m_request, 0)), 1) # at least one but integer

    # cut holes for membrane heating
    holes = []
    tht_spreaders = []
    m_phi = angle/2/(n_m*2)
    for j in range(n_m):
        x, y = (d_sand/2+dr_m)*math.cos(-j*2*m_phi-m_phi), (d_sand/2+dr_m)*math.sin(-j*2*m_phi-m_phi)

        segment_length = h_outer/thp_surface_segments
        mem_tube_segments = [gmsh.model.occ.addCylinder(x, y, segment_length*i, 0, 0, segment_length, tht_d/2) for i in range(thp_surface_segments)]
        if tht_in_sand:
            h_mem_spreader = tht_spreader_h_ratio*h_sand
            h0_mem = t_insulation+h_sand-h_mem_spreader-(h_sand-h_mem_spreader)/2
        else:
            h_mem_spreader = tht_spreader_h_ratio*(h_sand + 2*t_insulation)
            h0_mem = (h_sand + 2*t_insulation)-h_mem_spreader-((h_sand + 2*t_insulation)-h_mem_spreader)/2
        mem_spreader = tht_beam_spreader(x, y, 0, h0_tht_fins, tht_d, h_tht_pipe, h_tht_fins, db=tht_spreader_db, nb=tht_spreader_nb, tb=tht_spreader_tb)
        gmsh.model.occ.rotate([mem_spreader], x, y, 0, 0, 0, 1, -j*2*m_phi-m_phi-(j%2)*math.pi/cartridge_spreader_nb/2)
        for segment_tag in mem_tube_segments:
            if tht_in_sand:
                inner_cylinder = gmsh.model.occ.cut([(3, inner_cylinder)], [(3, segment_tag)], removeObject=True, removeTool=False)[0][0][1]
            outer_cylinder = gmsh.model.occ.cut([(3, outer_cylinder)], [(3, segment_tag)], removeObject=True, removeTool=True)[0][0][1]
            holes.append((3, segment_tag))
        if tht_in_sand:
            mem_spreader = gmsh.model.occ.intersect([mem_spreader], [(3, inner_cylinder), (3, outer_cylinder)], removeObject=True, removeTool=False)[0]
        else:
            mem_spreader = gmsh.model.occ.intersect([mem_spreader], [(3, outer_cylinder)], removeObject=True, removeTool=False)[0]
        tht_spreaders += mem_spreader

    f_tags, f_dim_tags = gmsh.model.occ.fragment(
        [(3, outer_cylinder)], # the main object
        [(3, inner_cylinder)]+cartridges+spreaders+tht_spreaders, #fragments inside main object
    )
    gmsh.model.occ.synchronize()

    mats = {
        'Sand': (materials.get_material_by_name(sand_material), [f_tags[-1][1]]),
        'Insulation': (materials.get_material_by_name(insulation_material), [f_tags[-2][1]]),
        'Cartridge': (materials.get_material_by_name(cartridge_material), [tag[1] for tag in cartridges]),
        'Cartridge_spreader': (materials.get_material_by_name(cartridge_spreader_material), [tag[1] for tag in spreaders]),
        'THT_spreader': (materials.get_material_by_name(thp_spreader_material), [tag[1] for tag in tht_spreaders]),
    }

    # create selected p-groups
    i = 1
    for mat_name, tuple_data in mats.items():
        entities = tuple_data[1]
        gmsh.model.addPhysicalGroup(dim, entities, i, mat_name)
        i += 1

    tol = 0.00001
    r = d_sand/2+t_insulation
    r_centroid = r*math.sin(angle/4)/(angle/4)
    if symmetry:
        vertical_wall = find_closest_entity(
            (r*math.cos(angle/4), -r*math.sin(angle/4), h_outer/2),
            gmsh.model.getEntities(2),
            dim=2,
            )
        bottom_wall = gmsh.model.getEntitiesInBoundingBox(
            -tol, -r*math.sin(angle/2)-tol, -tol,
            r+tol, tol, tol,
            2,
        )
        boundary_surfaces = [vertical_wall]+bottom_wall
    else:
        vertical_wall = find_closest_entity(
            (r_centroid*math.cos(angle/4), -r_centroid*math.sin(angle/4), h_outer/2),
            gmsh.model.getEntities(2),
            dim=2,
            )
        bottom_wall = gmsh.model.getEntitiesInBoundingBox(
            -tol, -r*math.sin(angle/2)-tol, -tol,
            r+tol, tol, tol,
            2,
        )
        top_wall = gmsh.model.getEntitiesInBoundingBox(
            -tol, -r*math.sin(angle/2)-tol, h_outer-tol,
            r+tol, tol, h_outer+tol,
            2,
        )
        boundary_surfaces = [vertical_wall]+bottom_wall+top_wall
    
    # find and categorize bcs surfaces
    bcs = {}

    tol = 0.00001
    m_surfs = []
    for j in range(n_m):
        x, y = (d_sand/2+dr_m)*math.cos(-j*2*m_phi-m_phi), (d_sand/2+dr_m)*math.sin(-j*2*m_phi-m_phi)
        for i in range(thp_surface_segments):
            z0 = segment_length*i
            x0 = x-tht_d/2
            y0 = y-tht_d/2
            x1 = x+tht_d/2
            y1 = y+tht_d/2
            z1 = segment_length*(i+1)
            new_surfs = gmsh.model.getEntitiesInBoundingBox(
                x0-tol, y0-tol, z0-tol,           # point min
                x1+tol, y1+tol, z1+tol,           # point max
                dim-1,                            # dim of ents
                )
            bcs[f'THP_surface_{j}_{i}'] = [tag[1] for tag in new_surfs]
        m_surfs.extend(new_surfs)
    thp_surface_flat_list = [tag[1] for tag in m_surfs]
        
    bcs['Outer_surface'] = [tag[1] for tag in boundary_surfaces]

    # mark physical surfaces
    i = 1
    for bc_name, ents in bcs.items():
        gmsh.model.addPhysicalGroup(dim-1, ents, i, bc_name)
        i += 1

    print("  Writing STEP file")
    gmsh.write(step_file)

    # mesh size field setting
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max) 
    #gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", mesh_size_from_curvature)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.add("Distance", 2)
    gmsh.model.mesh.field.add("Distance", 3)
    cartridge_spreader_surface_list = [dimtag[1] for dimtag in gmsh.model.getBoundary(spreaders[:3], oriented=False)]
    thp_spreader_surface_list = [dimtag[1] for dimtag in gmsh.model.getBoundary(tht_spreaders, oriented=False)]
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", cartridge_spreader_surface_list)
    gmsh.model.mesh.field.setNumbers(2, "SurfacesList", thp_surface_flat_list)
    gmsh.model.mesh.field.setNumbers(3, "SurfacesList", thp_spreader_surface_list)
    gmsh.model.mesh.field.add("MathEval", 4)
    gmsh.model.mesh.field.setString(
        4, 
        "F", 
        f"min({cartridge_spreader_mesh_size_min}+{cartridge_spreader_mesh_grow_factor}*F1,"
        f"{thp_mesh_size_min}+{thp_mesh_grow_factor}*F2,"
        f"{thp_spreader_mesh_size_min}+{thp_spreader_mesh_grow_factor}*F3)"
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(4)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

    if isinstance(fltk, str):
        fltk = [fltk]
    elif not fltk:
        fltk = []

    if 'premesh' in fltk:
        print("Starting premesh fltk window")
        gmsh.fltk.run()
    
    print("  Starting mesh algorithm")
    gmsh.model.mesh.generate(3)
    #gmsh.model.mesh.optimize()
    gmsh.write(gmsh_file)
    print("  Mesh written to disk")

    # Extract number of nodes
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    num_nodes = len(node_tags)
    print(f"  Number of nodes: {num_nodes}")

    # Extract number of elements
    element_types, element_tags, _ = gmsh.model.mesh.getElements(dim=3)
    num_elements = sum(len(tags) for tags in element_tags)  # Sum elements of all types
    print(f"  Number of elements: {num_elements}")

    if 'postmesh' in fltk:
        print("  Starting postmesh fltk")
        gmsh.fltk.run()

    gmsh.finalize()

    points = {
        'T':{"TC[i]":cartridge_tc[i] for i in range(len(cartridge_tc))},
    }

    custom_data = {
        'n_mem_pipes':n_m,
        'n_thp_surface_segments':thp_surface_segments,
    }

    save_mesh_add_data(
        add_data_file,
        call_data,
        dim,
        points,
        mats,
        bcs,
        jac_f,
        custom_data,
    )
    print("  Metadata written to disk")

    