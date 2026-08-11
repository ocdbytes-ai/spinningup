import os, subprocess, sys
import numpy as np

# MPI is optional. It is only required to run with more than one process
# (num_cpu > 1). When mpi4py (or an underlying MPI runtime) is unavailable,
# we fall back to a single-process implementation where every collective is an
# identity operation. Install an MPI runtime (e.g. `brew install open-mpi`) to
# enable multi-process training.
try:
    from mpi4py import MPI
    HAS_MPI = True
    SUM, MIN, MAX = MPI.SUM, MPI.MIN, MPI.MAX
except Exception:
    # ImportError if mpi4py isn't installed; RuntimeError if it is installed but
    # can't locate a system MPI library at import time. Either way, fall back to
    # single-process execution.
    MPI = None
    HAS_MPI = False
    SUM = MIN = MAX = None


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1:
        return
    if not HAS_MPI:
        raise RuntimeError(
            "Running with num_cpu=%d requires MPI, but mpi4py could not load an "
            "MPI runtime. Install one (e.g. `brew install open-mpi` on macOS or "
            "`apt install libopenmpi-dev` on Debian/Ubuntu) and reinstall "
            "mpi4py, or run with num_cpu=1." % n)
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    print(('Message from %d: %s \t '%(proc_id(), string))+str(m))

def proc_id():
    """Get rank of calling process."""
    if not HAS_MPI:
        return 0
    return MPI.COMM_WORLD.Get_rank()

def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def num_procs():
    """Count active MPI processes."""
    if not HAS_MPI:
        return 1
    return MPI.COMM_WORLD.Get_size()

def broadcast(x, root=0):
    if not HAS_MPI:
        return
    MPI.COMM_WORLD.Bcast(x, root=root)

def mpi_op(x, op):
    # With a single process every reduction is the identity.
    if not HAS_MPI:
        return x
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(x):
    return mpi_op(x, SUM)

def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()

def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MAX)
        return mean, std, global_min, global_max
    return mean, std
