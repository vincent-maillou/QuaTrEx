from threadpoolctl import threadpool_limits  # isort: skip
import os
import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from quatrex.core.quatrex_config import parse_config
from quatrex.core.scba import SCBA

PATH = os.path.dirname(__file__)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        config = parse_config(os.path.join(PATH, "config.toml"))
        scba = SCBA(config)
        tic = time.perf_counter()
        scba.run()
        toc = time.perf_counter()

    if comm.rank == 0:
        print(f"Leaving SCBA after: {(toc - tic):.2f} s")

        output_dir = f"{PATH}/outputs"
        os.mkdir(output_dir)

        np.save(f"{output_dir}/electron_ldos.npy", scba.observables.electron_ldos)
        np.save(f"{output_dir}/electron_density.npy", scba.observables.electron_density)
        np.save(f"{output_dir}/hole_density.npy", scba.observables.hole_density)
