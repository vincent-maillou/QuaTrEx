from threadpoolctl import threadpool_limits  # isort: skip
import os
import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from quatrex.core.quatrex_config import parse_config
from quatrex.core.qtbm import QTBM

from cupyx.scipy.sparse import SparseEfficiencyWarning

import warnings
warnings.simplefilter(action='ignore', category=SparseEfficiencyWarning)


PATH = os.path.dirname(__file__)

if __name__ == "__main__":
    with threadpool_limits(limits=128):
        config = parse_config(os.path.join(PATH, "config.toml"))
        qtbm = QTBM(config)
        tic = time.perf_counter()
        qtbm.run()
        toc = time.perf_counter()

    if comm.rank == 0:
        print(f"Leaving QTBM after: {(toc - tic):.2f} s")

        output_dir = f"{PATH}/outputs"
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        np.save(f"{output_dir}/transmission.npy", qtbm.observables.electron_transmission)
        np.save(f"{output_dir}/dos.npy", qtbm.observables.electron_DOS)