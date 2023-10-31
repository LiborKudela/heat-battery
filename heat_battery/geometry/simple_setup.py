import gmsh
import os
import math
import hashlib
import json
import meshio
from dolfinx import io
from mpi4py import MPI

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

def get_id(data):
        m = hashlib.sha256()
        d_str = json.dumps(data)
        m.update(d_str.encode('UTF-8'))
        return m.hexdigest()

def convert_to_legacy_fenics(msh_path):
    
    path, file = os.path.split(msh_path)
    name, ext = os.path.splitext(file)
    meshio_mesh_path = os.path.join(path, f"{name}_mesh.xdmf")
    meshio_subdomain_path = os.path.join(path, f"{name}_subdomains.xdmf")
    meshio_boundaries_path = os.path.join(path, f"{name}_boundaries.xdmf")
    msh = meshio.read(msh_path)
    print(meshio_mesh_path)

    # white mesh as xdmf
    points = msh.points
    cells = {"tetra" : msh.cells_dict["tetra"]}
    tetra_mesh = meshio.Mesh(points=points, cells=cells)
    meshio.write(meshio_mesh_path, tetra_mesh)

    # write subdomains as xdmf (mvc in fenics)
    cell_data = {"subdomains" : [msh.cell_data_dict["gmsh:physical"]["tetra"]]}
    tetra_data = meshio.Mesh(points=points, cells=cells, cell_data=cell_data)
    meshio.write(meshio_subdomain_path, tetra_data)

    # write boundaries as xdmf (mvc in fenics)
    cells = {"triangle" : msh.cells_dict["triangle"]}
    cell_data = {"triangle" : [msh.cell_data_dict["gmsh:physical"]["triangle"]]}
    triangle_data = meshio.Mesh(points=points, cells=cells, cell_data=cell_data)
    meshio.write(meshio_boundaries_path, triangle_data)

    print("legacy fenics mesh written")

def build_geometry(dir="meshes/simple_system",
                   d_sand=1,
                   h_sand=1,
                   t_insulation=0.1,
                   d_cartridge=0.014,
                   r_c=0.25,
                   h_c=0.8,
                   n_c=3, 
                   mesh_size = 0.01,
                   symmetry=False,
                   spreader_type='beam_spreader',
                   spreader_data={'db':0.15, 'nb':3, 'tb':0.004},
                   gmsh_gui=False,
                   legacy_fenics=False,
                   force_remesh=False,
                   ):
    
    # definate signature of the call
    signature_data = {
        'dir':dir, 
        'd_sand':d_sand,
        'h_sand':h_sand,
        't_insulation':t_insulation,
        'd_cartridge':d_cartridge,
        'r_c':r_c,
        'h_c':h_c,
        'n_c':n_c,
        'mesh_size':mesh_size,
        'symmetry':symmetry,
        'spreader_type':spreader_type,
        'spreader_data':spreader_data,
        }
    signature = get_id(signature_data)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Mesh Request: {signature}")
        print(f"    Signature data:")
        for k, v in signature_data.items():
            print(f"    {k}={v}")

    file_path = os.path.join(dir, signature)
    gmsh_file = os.path.join(file_path, "mesh.msh")

    exists = os.path.isdir(file_path)

    if MPI.COMM_WORLD.rank == 0:
        if not exists or force_remesh:
            print("Generating new mesh...")
            os.makedirs(file_path, exist_ok=True)
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)

            gmsh.model.add("SimpleSystem")
            gmsh.logger.start()

            if symmetry:
                angle = 2*math.pi/n_c
            else:
                angle = 2*math.pi

            # insulation
            r_max = d_sand/2 + t_insulation
            h = h_sand + 2*t_insulation
            outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, h, r_max, angle=angle)
            inner_cylinder = gmsh.model.occ.addCylinder(0, 0, t_insulation, 0, 0, h_sand, d_sand/2, angle=angle)
            if symmetry:
                gmsh.model.occ.rotate([(3, outer_cylinder)], 0, 0, 0, 0, 0, 1, -angle/2)
                gmsh.model.occ.rotate([(3, inner_cylinder)], 0, 0, 0, 0, 0, 1, -angle/2)

            # cartridges
            cartridges = []
            spreaders = []
            d_phi = 2*math.pi/n_c
            spreader_builder = eval(spreader_type)
            for i in range(n_c):
                x, y = r_c*math.cos(i*d_phi), r_c*math.sin(i*d_phi)
                #c = gmsh.model.occ.addCylinder(x, y, t_insulation+h_sand-h_c, 0, 0, h_c, d_cartridge/2)
                c = simple_cartridge(x, y, t_insulation+h_sand-h_c, d_cartridge, h_c)
                s = spreader_builder(x, y, t_insulation+h_sand-h_c, d_cartridge, h_c, **spreader_data)
                cartridges.append(c)
                spreaders.append(s)
                if symmetry:
                    break

            f_tags, f_dim_tags = gmsh.model.occ.fragment([(3, outer_cylinder)], 
                                                         [(3, inner_cylinder)]+cartridges+spreaders)
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(3, [tag[1] for tag in spreaders], 4, "spreader")
            gmsh.model.addPhysicalGroup(3, [tag[1] for tag in cartridges], 3, "cartridge")
            gmsh.model.addPhysicalGroup(3, [f_tags[-2][1]], 2, "insulation")
            gmsh.model.addPhysicalGroup(3, [f_tags[-1][1]], 1, "sand")

            if symmetry:
                boundary_surfaces = [1,2,4]
            else:
                boundary_surfaces = [1,2,3]
            gmsh.model.addPhysicalGroup(2, boundary_surfaces, 1, "boundary")

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
            #gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 3)

            gmsh.model.mesh.field.add("Distance", 1)
            surface_list = [dimtag[1] for dimtag in gmsh.model.getBoundary(spreaders[:3], oriented=False)]
            gmsh.model.mesh.field.setNumbers(1, "SurfacesList", surface_list)
            gmsh.model.mesh.field.add("MathEval", 3)
            gmsh.model.mesh.field.setString(3, "F", "0.004+0.5*F1")
            gmsh.model.mesh.field.setAsBackgroundMesh(3)
            
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.optimize()
            gmsh.write(gmsh_file)


            if gmsh_gui:
                gmsh.fltk.run()

            gmsh.finalize()

            if legacy_fenics:
                convert_to_legacy_fenics(gmsh_file)

            print("Mesh written to disk")
        else:
            print("Signature found!!")

    MPI.COMM_WORLD.Barrier()

    sim_coefficients = {'signature':signature, 'signature_data':signature_data}
    if symmetry:
        sim_coefficients['power'] = 1/n_c
        sim_coefficients['volume'] = n_c
        sim_coefficients['surface'] = n_c
    else:
        sim_coefficients['power'] = 1.0
        sim_coefficients['volume'] = 1.0
        sim_coefficients['surface'] = 1.0

    domain, cell_tags, facet_tags = io.gmshio.read_from_msh(gmsh_file, MPI.COMM_WORLD, 0, gdim=3)
    return domain, cell_tags, facet_tags, sim_coefficients

build_geometry()
