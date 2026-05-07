import os
import meshio
from typing import Callable
import cloudpickle
from mpi4py import MPI
import gmsh
import math
from pathlib import Path
from typing import Any
from dolfinx import fem



def save_mesh_add_data(
    add_data_file_path: str|Path, 
    call_data: dict[str, Any], 
    dim: int, 
    points: dict[str, dict[str, list[float]]],
    mats: dict[str, tuple[str, list[int]]], 
    bcs: dict[str, list[int]], 
    jac_f: Callable[[tuple[float, float, float]], float], 
    custom_data: dict|None=None):
    """
    Saves mesh metadata to binary file.
    
    Args:
        add_data_file_path: Path to the file to save the metadata to
        call_data: Dictionary containing the call data
        dim: Dimension of the mesh (2, 3)
        points: Dictionary containing the points (X, Y, Z) coordinates
        mats: Dictionary containing the volume materials groups
        bcs: Dictionary containing the surface boundaries groups

    Examples:
        >>> save_mesh_add_data(
        >>>     add_data_file_path='mesh_add_data.ad',
        >>>     call_data={'diameter': 1.0, 'height': 1.0},
        >>>     dim=2,
        >>>     points={'T': {'Tmid_1': [0.0, 0.0, 0.0]}},
        >>>     mats={'Sand': (heat_battery.materials.Sand, [1])},
        >>>     bcs={'outer_surface': [1]},
        >>>     jac_f=lambda x: 2*pi*x[0],
        >>>     custom_data={'master_key': {'subkey': 'value'}})
    """
    
    add_data = {
        'call_data':call_data,
        'dim':dim,
        'points':points,
        'materials':mats,
        'boundaries':bcs,
        'jac_f':jac_f,
        'custom_data':custom_data,
        }
    
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

def find_closest_entity(target_point, entities, dim=3):
    """Find the entity with center of mass closest to the target point.
    
    Args:
        target_point: Tuple (x, y, z) representing the target point
        entities: List of entity tags or (dim, tag) tuples
        dim: Dimension of entities if only tags are provided
        
    Returns:
        Tuple (dim, tag) of the closest entity and the distance
    """
    min_distance = float('inf')
    closest_entity = None
    
    for entity in entities:
        # Handle both tag-only and (dim, tag) formats
        if isinstance(entity, tuple) and len(entity) == 2:
            entity_dim, entity_tag = entity
        else:
            entity_dim, entity_tag = dim, entity
            
        com = gmsh.model.occ.getCenterOfMass(entity_dim, entity_tag)
        
        # Calculate Euclidean distance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(target_point, com)))
        
        if distance < min_distance:
            min_distance = distance
            closest_entity = (entity_dim, entity_tag)
    
    return closest_entity
    