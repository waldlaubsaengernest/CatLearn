import os

def use_mpi4py():
    return os.environ.get("CATLEARN_USE_MPI4PY", "0") == "1"

def comm():
    if use_mpi4py():
        from mpi4py import MPI
        return MPI.COMM_WORLD
    return None

def rank():
    c = comm()
    if c is not None:
        return c.Get_rank()
    from ase.parallel import world
    return world.rank

def size():
    c = comm()
    if c is not None:
        return c.Get_size()
    from ase.parallel import world
    return world.size

def bcast(obj, root=0, **kwargs):
    c = comm()
    if c is not None:
        return c.bcast(obj, root=root)
    from ase.parallel import broadcast
    return broadcast(obj, root=root)
