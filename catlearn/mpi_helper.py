import os
from mpi4py import MPI

def use_mpi4py():
    return os.environ.get("CATLEARN_USE_MPI4PY", "0") == "1"

def rank_size():
    if use_mpi4py():
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    from ase.parallel import world
    return world.rank, world.size

def rank():
    return rank_size()[0]

def size():
    return rank_size()[1]

def comm():
    comm = MPI.COMM_WORLD
    return comm

def bcast(obj, root=0):
    if use_mpi4py():
        return MPI.COMM_WORLD.bcast(obj, root=root)
    from ase.parallel import broadcast
    return broadcast(obj, root=root)
