# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.utils import assemble_kpoint_dsb


def fft_convolve(a: NDArray, b: NDArray) -> NDArray:
    """Computes the convolution of two arrays using FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The convolution of the two arrays.

    """
    ne = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (ne,), axes=(0,))
    b_fft = xp.fft.fftn(b, (ne,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


def fft_circular_convolve(a: xp.ndarray, b: xp.ndarray, axes: tuple[int]) -> xp.ndarray:
    """Computes the circular convolution of two arrays using the FFT."""
    # Extract the shapes of the arrays along the axes as tuples.
    nka = tuple(a.shape[i] for i in axes)
    nkb = tuple(b.shape[i] for i in axes)
    a_fft = xp.fft.fftn(a, nka, axes=axes)
    b_fft = xp.fft.fftn(b, nkb, axes=axes)
    return xp.fft.ifftn(a_fft * b_fft, axes=axes)


def fft_convolve_kpoints(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Computes the convolution of two arrays using the FFT."""
    ne = a.shape[0] + b.shape[0] - 1
    nka = a.shape[1:-1]
    nkb = b.shape[1:-1]
    a_fft = xp.fft.fftn(a, (ne,) + nka, axes=(0,) + tuple(range(1, len(nka) + 1)))
    b_fft = xp.fft.fftn(b, (ne,) + nkb, axes=(0,) + tuple(range(1, len(nkb) + 1)))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,) + tuple(range(1, len(nka) + 1)))


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


def hilbert_transform(a: NDArray, energies: NDArray, eta=1e-8) -> NDArray:
    """Computes the Hilbert transform of the array a.

    Assumes that the first axis corresponds to the energy axis.

    Parameters
    ----------
    a : NDArray
        The array to transform.
    energies : NDArray
        The energy values corresponding to the first axis of a.
    eta : float, optional
        For the principle part. Small part to avoid singularity, by
        default 1e-8.

    Returns
    -------
    NDArray
         The Hilbert transform of a.

    """
    energy_differences = xp.expand_dims(energies - energies[0], tuple(range(1, a.ndim)))
    ne = energies.size
    # eta for removing the singularity. See Cauchy principal value.
    b = (
        fft_convolve(a, 1 / (energy_differences + eta))[:ne]
        + fft_convolve(a, 1 / (-energy_differences[::-1] - eta))[ne - 1 :]
    )
    # Not sure about the prefactor. Currently gives the same value as the old code.
    return b / (2 * xp.pi) * (energies[1] - energies[0])


class SigmaCoulombScreening(ScatteringSelfEnergy):
    """Computes the scattering self-energy from the Coulomb screening.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        electron_energies: NDArray,
    ):
        """Initializes the scattering self-energy."""
        self.energies = electron_energies
        number_of_kpoints = quatrex_config.electron.number_of_kpoints
        self.num_energies = self.energies.size
        self.prefactor = (
            1j
            / (2 * xp.pi)
            * (self.energies[1] - self.energies[0])
            / xp.prod(number_of_kpoints)
        )
        self.big_block_sizes = None

    def compute(
        self,
        g_lesser: DSBSparse,
        g_greater: DSBSparse,
        w_lesser: DSBSparse,
        w_greater: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Computes the GW self-energy.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        g_greater : DSBSparse
            The greater Green's function.
        w_lesser : DSBSparse
            The lesser screened Coulomb interaction.
        w_greater : DSBSparse
            The greater screened Coulomb interaction.
        out : tuple[DSBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.

        """
        if w_lesser.nnz != g_lesser.nnz:
            raise ValueError(
                "The sparsity pattern of w_lesser and g_lesser must match."
                "Something went horribly wrong."
            )

        # Save the block sizes for later.
        if self.big_block_sizes is None:
            self.big_block_sizes = w_lesser.block_sizes

        # Enforce that the block sizes are the same. NOTE: This triggers
        # a block-reordering in the DSBSparse object.
        w_lesser.block_sizes = g_lesser.block_sizes
        w_greater.block_sizes = g_greater.block_sizes

        sigma_lesser, sigma_greater, sigma_retarded = out
        t0 = time.perf_counter()
        # Transpose the matrices to nnz distribution.
        for m in (
            w_lesser,
            w_greater,
            g_lesser,
            g_greater,
            sigma_lesser,
            sigma_greater,
            sigma_retarded,
        ):
            # The electron Green's functions and self-energies should
            # ideally already be in nnz-distribution. We cannot discard
            # the data here, as we cannot be sure that this is the
            # first/only interaction.
            m.dtranspose() if m.distribution_state != "nnz" else None
        t1 = time.perf_counter()
        if comm.rank == 0:
            print(
                f"SigmaCoulombScreening: stack->nnz transpose time: {t1-t0}", flush=True
            )

        # Compute the full self-energy using the convolution theorem.
        # Second term are corrections for negative frequencies that
        # where cut off by the polarization calculation.
        sigma_lesser.data += self.prefactor * (
            fft_convolve_kpoints(g_lesser.data, w_lesser.data)[: self.num_energies]
            - fft_correlate_kpoints(g_lesser.data, w_greater.data.conj())[
                self.num_energies - 1 :
            ]
        )
        sigma_greater.data += self.prefactor * (
            fft_convolve_kpoints(g_greater.data, w_greater.data)[: self.num_energies]
            - fft_correlate_kpoints(g_greater.data, w_lesser.data.conj())[
                self.num_energies - 1 :
            ]
        )

        # Compute retarded self-energy with a Hilbert transform.
        sigma_antihermitian = 1j * xp.imag(sigma_greater.data - sigma_lesser.data)
        sigma_hermitian = hilbert_transform(sigma_antihermitian, self.energies)
        sigma_retarded.data += 1j * sigma_hermitian + sigma_antihermitian / 2

        # Transpose the matrices to stack distribution.
        t0 = time.perf_counter()
        for m in (w_lesser, w_greater):
            m.dtranspose(discard=True) if m.distribution_state != "stack" else None
        # NOTE: The electron Green's functions and self-energies must
        # not be transposed back to stack distribution, as they are
        # needed in nnz distribution for the other interactions.

        t1 = time.perf_counter()
        if comm.rank == 0:
            print(
                f"SigmaCoulombScreening: nnz->stack transpose time: {t1-t0}", flush=True
            )

        # Recover original block sizes.
        w_lesser.block_sizes = self.big_block_sizes
        w_greater.block_sizes = self.big_block_sizes


class SigmaFock(ScatteringSelfEnergy):
    """Computes the bare Fock self-energy.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        electron_energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ):
        """Initializes the bare Fock self-energy."""
        self.energies = electron_energies
        number_of_kpoints = quatrex_config.electron.number_of_kpoints
        self.prefactor = (
            1j
            / (2 * xp.pi)
            * (self.energies[1] - self.energies[0])
            / xp.prod(number_of_kpoints)
        )
        try:
            coulomb_matrix_sparray = distributed_load(
                quatrex_config.input_dir / "coulomb_matrix.npz"
            ).astype(xp.complex128)
            coulomb_matrix_dict = None
        except FileNotFoundError:
            coulomb_matrix_dict = distributed_load(
                quatrex_config.input_dir / "coulomb_matrix.pkl"
            )
            coulomb_matrix_sparray = coulomb_matrix_dict[(0, 0, 0)]
        # Make sure that the Coulomb matrix is Hermitian.
        coulomb_matrix_sparray = (
            0.5 * (coulomb_matrix_sparray + coulomb_matrix_sparray.conj().T)
        ).tocoo()

        # Scale the coulomb matrix by the relative permittivity.
        coulomb_matrix_sparray *= (
            1 / quatrex_config.coulomb_screening.relative_permittivity
        )
        if coulomb_matrix_dict is not None:
            coulomb_matrix_dict = {
                k: v * 1 / quatrex_config.coulomb_screening.relative_permittivity
                for k, v in coulomb_matrix_dict.items()
            }
        # Load block sizes for the coulomb matrix.
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")

        # Create the DSBSparse object.
        # TODO: This is pretty wasteful memory-wise.
        # Workaround: Use comm size as global stack shape.
        coulomb_matrix = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=(comm.size,)
            + tuple([k for k in number_of_kpoints if k > 1]),
        )
        coulomb_matrix.data = 0.0
        coulomb_matrix += coulomb_matrix_sparray

        if coulomb_matrix_dict is not None:
            coulomb_matrix.data = 0
            number_of_kpoints = xp.array(
                [1 if k <= 1 else k for k in number_of_kpoints]
            )
            assemble_kpoint_dsb(
                coulomb_matrix,
                coulomb_matrix_dict,
                number_of_kpoints,
                -(number_of_kpoints // 2),
            )
            # Change the sign of the Coulomb matrix.
            coulomb_matrix.data *= -1
        coulomb_matrix.dtranspose()
        self.coulomb_matrix_data = coulomb_matrix.data[0]

    def compute(self, g_lesser: DSBSparse, out: tuple[DSBSparse, ...]) -> None:
        """Computes the Fock self-energy.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        out : tuple[DSBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded.

        """
        # TODO: Check again if we really need to transpose the matrices
        # here.
        (sigma_retarded,) = out
        t0 = time.perf_counter()
        for m in (g_lesser, sigma_retarded):
            # These should both already be in nnz-distribution.
            m.dtranspose() if m.distribution_state != "nnz" else None
        t1 = time.perf_counter()
        if comm.rank == 0:
            print(f"SigmaFock: stack->nnz transpose time: {t1-t0}", flush=True)

        # Compute the electron density by summing over energies.
        gl_density = self.prefactor * g_lesser.data.sum(axis=0)
        sigma_retarded.data += fft_circular_convolve(
            gl_density,
            self.coulomb_matrix_data,
            axes=tuple(range(gl_density.ndim - 1)),
        )

        # NOTE: The electron Green's functions and self-energies must
        # not be transposed back to stack distribution, as they are
        # needed in nnz distribution for the other interactions.
