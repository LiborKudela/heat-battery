import os
import meshio
from typing import Callable
import cloudpickle
from mpi4py import MPI

def save_mesh_add_data(
    add_data_file_path: str, call_data: dict, dim: int, points: dict,
    mats: dict, bcs: dict, jac_f: Callable):
    
    add_data = {
        'call_data':call_data,
        'dim':dim,
        'points':points,
        'materials':mats,
        'boundaries':bcs,
        'jac_f':jac_f,
        }
    
    if MPI.COMM_WORLD.rank == 0:
        with open(add_data_file_path, 'wb') as fp:
            cloudpickle.dump(add_data, fp)
    return None

def convert_to_legacy_fenics(msh_path):
    #FIXME: this function is way behing the rest of the heat_battery module
    # it need to take argument 'dir' and 'name' and produce legacy FeniCS files
    # of mesh.xdmf, subdomains.xdmf and boundaries.xdmf
    
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
    