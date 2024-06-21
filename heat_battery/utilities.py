import cloudpickle
from mpi4py import MPI

def save_data(filepath, data, only_root=True):
    if not only_root or MPI.COMM_WORLD.rank == 0:
        with open(filepath, 'wb') as fp:
            cloudpickle.dump(data, fp)
    return None

def load_data(filepath):
    with open(filepath, 'rb') as fp:
        data = cloudpickle.load(fp)
    return data

def test_package():
    #TODO: make this proper
    return 0