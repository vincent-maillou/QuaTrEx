from threadpoolctl import threadpool_limits, threadpool_info  # isort: skip
import os
import time
from pprint import pprint

import numba as nb
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from quatrex.core.quatrex_config import parse_config
from quatrex.core.scba import SCBA

PATH = os.path.dirname(__file__)
NUM_THREADS = 72

if __name__ == "__main__":

    nb.set_num_threads(NUM_THREADS)
    with threadpool_limits(limits=NUM_THREADS):
        pprint(threadpool_info()) if comm.rank == 0 else None
        config = parse_config(f"{PATH}/config.toml")
        scba = SCBA(config)
        tic = time.perf_counter()
        scba.run()
        toc = time.perf_counter()

    if comm.rank == 0:
        print(f"Leaving SCBA after: {(toc - tic):.2f} s")

        output_dir = f"{PATH}/outputs"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        np.save(f"{output_dir}/electron_ldos.npy", scba.observables.electron_ldos)
        np.save(f"{output_dir}/electron_density.npy", scba.observables.electron_density)
        np.save(f"{output_dir}/hole_density.npy", scba.observables.hole_density)
        np.save(f"{output_dir}/i_left.npy", scba.observables.electron_current["left"])
        np.save(f"{output_dir}/i_right.npy", scba.observables.electron_current["right"])
