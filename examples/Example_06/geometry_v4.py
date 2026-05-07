
from mpi4py import MPI
from heat_battery.geometry.utilities import save_mesh_add_data
from heat_battery import materials
import inspect
import gmsh
import os
import math

def simple_cartridge(x, y, z, d, h):
    return (3, gmsh.model.occ.addCylinder(x, y, z, 0, 0, h, d/2))

def beam_spreader(x, y, z, d, h, db, nb, tb):
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
        cartridge_mesh_size_min = 0.002,
        cartridge_mesh_grow_factor = 0.8,
        mem_mesh_size_min = 0.002,
        mem_mesh_grow_factor = 0.8,
        mesh_size_from_curvature=10,
        fltk=False,
        symmetry=False,
        size=1,
        t_insulation=0.1,
        n_c=3, 
        c_position=0.5,
        d_c=0.014,
        h_c_ratio=0.8,
        m_position=0.1,
        mem_in_sand=False,
        d_m=0.05,
        n_m_ratio = 0.5,
        spreader_db=0.15,
        spreader_nb=3,
        spreader_tb=0.005,
        ):

    assert c_position >= 0 and c_position <= 1, "c_position must be between 0 and 1"
    assert m_position >= 0 and m_position <= 1, "m_position must be between 0 and 1"
    assert h_c_ratio >= 0 and h_c_ratio <= 1, "h_c_ratio must be between 0 and 1"
    assert n_m_ratio >= 0 and n_m_ratio <= 1, "n_m_ratio must be between 0 and 1"
    assert d_m < t_insulation, "Mem pipes cannot be bigger than insulation layer"

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
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.add("C3_passive")
    gmsh.logger.start()

    if symmetry:
        angle = 2*math.pi/n_c
    else:
        angle = 2*math.pi

    dim = 3
    jac_f = lambda x: 4*n_c

    # insulation
    r_max = d_sand/2 + t_insulation
    h = h_sand/2 + t_insulation
    outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, h, r_max, angle=angle/2)
    inner_cylinder = gmsh.model.occ.addCylinder(0, 0, t_insulation, 0, 0, h_sand/2, d_sand/2, angle=angle/2)
    if symmetry:
        gmsh.model.occ.rotate([(3, outer_cylinder)], 0, 0, 0, 0, 0, 1, -angle/2)
        gmsh.model.occ.rotate([(3, inner_cylinder)], 0, 0, 0, 0, 0, 1, -angle/2)

    # cartridges
    cartridges = []
    spreaders = []
    cartridge_tc = []
    d_phi = 2*math.pi/n_c
    for j in range(n_c):
        r_c = c_position*(d_sand/2-spreader_db/2-d_m/2-0.001)
        h_c = h_c_ratio*h_sand
        x, y = r_c*math.cos(j*d_phi), r_c*math.sin(j*d_phi)
        #c = gmsh.model.occ.addCylinder(x, y, t_insulation+h_sand-h_c, 0, 0, h_c, d_cartridge/2)
        c = simple_cartridge(x, y, t_insulation+h_sand-h_c-(h_sand-h_c)/2, d_c, h_c)
        if symmetry:
            c = gmsh.model.occ.intersect([c], [(3, inner_cylinder)], removeObject=True, removeTool=False)[0][0]
        s = beam_spreader(x, y, t_insulation+h_sand-h_c-(h_sand-h_c)/2, d_c, h_c, db=spreader_db, nb=spreader_nb, tb=spreader_tb)
        if symmetry:
            s = gmsh.model.occ.intersect([s], [(3, inner_cylinder)], removeObject=True, removeTool=False)[0][0]
        cartridges.append(c)
        spreaders.append(s)
        cartridge_tc.append([x, y, h])
        
        if symmetry:
            break
    
    # mem pipe positioning for mem in sand
    dr_m_tol = 1e-3
    # dr_m is radial distance from the interface between sand and insulation
    if mem_in_sand:
        dr_m = -(d_m/2+dr_m_tol+m_position*(d_sand/2-r_c-spreader_db/2-d_m-2*dr_m_tol))
    else:
        dr_m = d_m/2+dr_m_tol+m_position*(t_insulation-d_m-2*dr_m_tol)
    d_m_angle = (2*math.asin((d_m/2)/(d_sand/2+dr_m))) # how much angle one mem pipe takes
    n_m_request = (math.pi/n_c*n_m_ratio)/d_m_angle # float of how many mem pipes fit in
    n_m = max(int(round(n_m_request, 0)), 1) # at least one but integer

    # cut holes for membrane heating
    holes = []
    m_phi = angle/2/(n_m*2)
    for j in range(n_m):
        x, y = (d_sand/2+dr_m)*math.cos(-j*2*m_phi-m_phi), (d_sand/2+dr_m)*math.sin(-j*2*m_phi-m_phi)
        mem_tube = gmsh.model.occ.addCylinder(x, y, 0, 0, 0, h+2*t_insulation, d_m/2)
        holes.append((3,mem_tube))
    if mem_in_sand:
        inner_cylinder = gmsh.model.occ.cut([(3, inner_cylinder)], holes, removeObject=True, removeTool=False)[0][0][1]
    outer_cylinder = gmsh.model.occ.cut([(3, outer_cylinder)], holes, removeObject=True, removeTool=True)[0][0][1]


    f_tags, f_dim_tags = gmsh.model.occ.fragment([(3, outer_cylinder)], 
                                                    [(3, inner_cylinder)]+cartridges+spreaders)
    gmsh.model.occ.synchronize()

    mats = {
        'Sand': (materials.SandTheory, [f_tags[-1][1]]),
        'Insulation': (materials.Standard_insulation, [f_tags[-2][1]]),
        'Cartridge': (materials.Steel04, [tag[1] for tag in cartridges]),
        'Spreader': (materials.Steel04, [tag[1] for tag in spreaders]),
    }

    # create selected p-groups
    i = 1
    for mat_name, tuple_data in mats.items():
        entities = tuple_data[1]
        gmsh.model.addPhysicalGroup(dim, entities, i, mat_name)
        i += 1

    if symmetry:
        boundary_surfaces = [1, 4]
    else:
        boundary_surfaces = [1,2,3]

    tol = 0.00001
    m_surfs = []
    for j in range(n_m):
        x, y = (d_sand/2+dr_m)*math.cos(-j*2*m_phi-m_phi), (d_sand/2+dr_m)*math.sin(-j*2*m_phi-m_phi)
        m_surfs += gmsh.model.getEntitiesInBoundingBox(
            x-d_m/2-tol, y-d_m/2-tol, 0.0-tol,              # point min
            x+d_m/2+tol, y+d_m/2+tol, h+2*t_insulation+tol, # point max
            dim-1,                                          # dim of ents
            )
        
    bcs = {'Outer_surface': boundary_surfaces,
            'Membrane_surface': [tag[1] for tag in m_surfs]}
    
    i = 1
    for bc_name, ents in bcs.items():
        gmsh.model.addPhysicalGroup(dim-1, ents, i, bc_name)
        i += 1

    print("  Writing STEP file")
    gmsh.write(step_file)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size_max)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", mesh_size_from_curvature)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.add("Distance", 2)
    cartridge_surface_list = [dimtag[1] for dimtag in gmsh.model.getBoundary(spreaders[:3], oriented=False)]
    mem_surface_list = [dimtag[1] for dimtag in m_surfs]
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", cartridge_surface_list)
    gmsh.model.mesh.field.setNumbers(2, "SurfacesList", mem_surface_list)
    gmsh.model.mesh.field.add("MathEval", 3)
    gmsh.model.mesh.field.setString(3, "F", f"min({cartridge_mesh_size_min}+{cartridge_mesh_grow_factor}*F1, {mem_mesh_size_min}+{mem_mesh_grow_factor}*F2)")
    #gmsh.model.mesh.field.add("MathEval", 4)
    #gmsh.model.mesh.field.setString(4, "F", f"{mem_mesh_size_min}+{mem_mesh_grow_factor}*F2")
    #gmsh.model.mesh.field.add("MathEval", 5)
    #gmsh.model.mesh.field.setString(5, "F", f"min(F3, F4)")
    gmsh.model.mesh.field.setAsBackgroundMesh(3)

    if isinstance(fltk, str):
        fltk = [fltk]
    elif not fltk:
        fltk = []

    if 'premesh' in fltk:
        print("Starting premesh fltk window")
        gmsh.fltk.run()
    
    print("  Starting mesh algorithm")
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize()
    gmsh.write(gmsh_file)
    print("  Mesh written to disk")

    if 'postmesh' in fltk:
        print("  Starting postmesh fltk")
        gmsh.fltk.run()

    gmsh.finalize()

    points = {
        'T':{"TC[i]":cartridge_tc[i] for i in range(len(cartridge_tc))},
    }

    save_mesh_add_data(
        add_data_file,
        call_data,
        dim,
        points,
        mats,
        bcs,
        jac_f,
    )
    print("  Metadata written to disk")