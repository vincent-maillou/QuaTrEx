from threadpoolctl import threadpool_limits  # isort: skip
import os
import time

import numba
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from quatrex.core.quatrex_config import parse_config
from quatrex.core.scba import SCBA

numba.set_num_threads(1)
PATH = os.path.dirname(__file__)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        config = parse_config(f"{PATH}/config.toml")
        scba = SCBA(config)
        tic = time.perf_counter()
        scba.run()
        toc = time.perf_counter()

    if comm.rank == 0:
        print(f"Leaving SCBA after: {(toc - tic):.2f} s")

        output_dir = f"{PATH}/outputs"
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        np.save(f"{output_dir}/electron_ldos.npy", scba.observables.electron_ldos)
        np.save(f"{output_dir}/electron_density.npy", scba.observables.electron_density)
        np.save(f"{output_dir}/hole_density.npy", scba.observables.hole_density)
        np.save(f"{output_dir}/i_left.npy", scba.observables.electron_current["left"])
        np.save(f"{output_dir}/i_right.npy", scba.observables.electron_current["right"])
