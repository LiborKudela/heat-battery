from mpi4py import MPI
from math import pi
import gmsh

def build_geometry():

    if MPI.COMM_WORLD.rank == 0:
    
        gmsh.initialize()
        gmsh.finalize()

    MPI.COMM_WORLD.Barrier()

    return None

