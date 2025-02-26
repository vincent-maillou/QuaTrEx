# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, xp
from qttools.datastructures import DSBSparse

from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy


def fft_correlate_kpoints(a: NDArray, b: NDArray) -> NDArray:
    """Computes the correlation of two arrays using FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The cross-correlation of the two arrays.

    """
    ne = a.shape[0] + b.shape[0] - 1
    nka = a.shape[1:-1]
    nkb = b.shape[1:-1]
    a_fft = xp.fft.fftn(a, (ne,) + nka, axes=(0,) + tuple(range(1, len(nka) + 1)))
    b_fft = xp.fft.fftn(
        xp.flip(b, axis=(0,) + tuple(range(1, len(nkb) + 1))),
        (ne,) + nkb,
        axes=(0,) + tuple(range(1, len(nkb) + 1)),
    )
    # Don't really know why I have to roll the result, but it works.
    return xp.roll(
        xp.fft.ifftn(a_fft * b_fft, axes=(0,) + tuple(range(1, len(nka) + 1))),
        shift=1,
        axis=tuple(range(1, len(nka) + 1)),
    )


class PCoulombScreening(ScatteringSelfEnergy):
    """Computes the dynamic polarization from the electronic system.

    Parameters
    ----------
    quatrex_config : Path
        Quatrex configuration file.
    coulomb_screening_energies : NDArray
        The energies for the Coulomb screening

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        coulomb_screening_energies: NDArray,
        number_of_kpoints: xp.ndarray,
    ) -> None:
        """Initializes the polarization."""
        self.energies = coulomb_screening_energies
        self.ne = len(self.energies)
        self.prefactor = (
            -1j
            / xp.pi
            * xp.abs(self.energies[1] - self.energies[0])
            / xp.prod(number_of_kpoints)
        )
        self.flatband = quatrex_config.electron.flatband

    def compute(
        self, g_lesser: DSBSparse, g_greater: DSBSparse, out: tuple[DSBSparse, ...]
    ) -> None:
        """Computes the polarization.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        g_greater : DSBSparse
            The greater Green's function.
        out : tuple[DSBSparse, ...]
            The output matrices for the polarization. The order is
            p_lesser, p_greater, p_retarded.

        """
        p_lesser, p_greater, p_retarded = out

        # Transpose the matrices to nnz distribution.
        t0 = time.perf_counter()
        for m in (g_lesser, g_greater):
            # These should ideally already be in nnz-distribution.
            m.dtranspose() if m.distribution_state != "nnz" else None

        p_g_full = self.prefactor * fft_correlate_kpoints(
            g_greater.data, -g_lesser.data.conj()
        )
        p_l_full = -p_g_full[::-1].conj()
        # Fill the matrices with the data. Take second part of the
        # energy convolution.
        p_lesser.data = p_l_full[self.ne - 1 :]
        p_greater.data = p_g_full[self.ne - 1 :]

        # Transpose the matrices to stack distribution.
        t0 = time.perf_counter()
        for m in (p_lesser, p_greater):
            m.dtranspose() if m.distribution_state != "stack" else None
        # NOTE: The Green's functions must not be transposed back to
        # stack distribution, as they are needed in nnz distribution for
        # the other interactions.

        t1 = time.perf_counter()
        if comm.rank == 0:
            print(f"PCoulombScreening: nnz->stack transpose time: {t1-t0}", flush=True)

        # Enforce anti-Hermitian symmetry and calculate Pr.
        t0 = time.perf_counter()
        p_lesser.data = (p_lesser.data - p_lesser.ltranspose(copy=True).data.conj()) / 2
        p_greater.data = (
            p_greater.data - p_greater.ltranspose(copy=True).data.conj()
        ) / 2

        # Discard the real part.
        p_lesser._data.real = 0
        p_greater._data.real = 0

        p_retarded.data = (p_greater.data - p_lesser.data) / 2
        t1 = time.perf_counter()
        if comm.rank == 0:
            print(f"PCoulombScreening: Symmetrization time: {t1-t0}", flush=True)
