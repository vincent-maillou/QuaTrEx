# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy


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
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b, (n,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


def fft_correlate(a: NDArray, b: NDArray) -> NDArray:
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
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b[::-1], (n,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


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
    # Add a small imaginary part to avoid singularity.
    energy_differences = (energies - energies[0]).reshape(-1, 1)
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
        self.ne = self.energies.size
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])
        # Load the Hamiltonian and block sizes for sparsity pattern.
        hamiltonian_sparray = distributed_load(
            quatrex_config.input_dir / "hamiltonian.npz"
        )
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")
        self.w_lesser_reduced = compute_config.dsbsparse_type.from_sparray(
            hamiltonian_sparray,
            block_sizes,
            (self.ne,),
            densify_blocks=[(i, i) for i in range(len(block_sizes))],
        )
        self.w_greater_reduced = compute_config.dsbsparse_type.zeros_like(
            self.w_lesser_reduced
        )

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

        # If the w_lesser and w_greater don't have the same sparsity
        # pattern as g_lesser and g_greater, we have to reduce them to
        # the same sparsity pattern.
        if w_lesser.nnz != self.w_lesser_reduced.nnz:
            self.w_lesser_reduced.data[:] = w_lesser[*self.w_lesser_reduced.spy()]
        else:
            self.w_lesser_reduced.data[:] = w_lesser.data
        if w_greater.nnz != self.w_greater_reduced.nnz:
            self.w_greater_reduced.data[:] = w_greater[*self.w_greater_reduced.spy()]
        else:
            self.w_greater_reduced.data[:] = w_greater.data

        sigma_lesser, sigma_greater, sigma_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (
            g_lesser,
            g_greater,
            self.w_lesser_reduced,
            self.w_greater_reduced,
            sigma_lesser,
            sigma_greater,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "nnz" else None

        # Compute the full self-energy using the convolution theorem.
        # Second term are corrections for negative frequencies that
        # where cut off by the polarization calculation.
        sigma_lesser._data[
            sigma_lesser._stack_padding_mask,
            ...,
            : sigma_lesser.nnz_section_sizes[comm.rank],
        ] += self.prefactor * (
            fft_convolve(g_lesser.data, self.w_lesser_reduced.data)[: self.ne]
            - fft_correlate(g_lesser.data, self.w_greater_reduced.data.conj())[
                self.ne - 1 :
            ]
        )
        sigma_greater._data[
            sigma_greater._stack_padding_mask,
            ...,
            : sigma_greater.nnz_section_sizes[comm.rank],
        ] += self.prefactor * (
            fft_convolve(g_greater.data, self.w_greater_reduced.data)[: self.ne]
            - fft_correlate(g_greater.data, self.w_lesser_reduced.data.conj())[
                self.ne - 1 :
            ]
        )

        # Compute retarded self-energy with a Hilbert transform.
        sigma_antihermitian = sigma_greater.data - sigma_lesser.data
        sigma_hermitian = hilbert_transform(sigma_antihermitian, self.energies)
        sigma_retarded._data[
            sigma_retarded._stack_padding_mask,
            ...,
            : sigma_retarded.nnz_section_sizes[comm.rank],
        ] += (
            1j * sigma_hermitian + sigma_antihermitian / 2
        )

        # Transpose the matrices to stack distribution.
        for m in (
            g_lesser,
            g_greater,
            self.w_lesser_reduced,
            self.w_greater_reduced,
            sigma_lesser,
            sigma_greater,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "stack" else None


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
    ):
        """Initializes the bare Fock self-energy."""
        self.energies = electron_energies
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])
        coulomb_matrix_sparray = distributed_load(
            quatrex_config.input_dir / "coulomb_matrix.npz"
        ).astype(xp.complex128)
        # Load block sizes for the coulomb matrix.
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")
        # Load the Hamiltonian to get the correct sparsity pattern.
        hamiltonian_sparray = distributed_load(
            quatrex_config.input_dir / "hamiltonian.npz"
        )
        # Create the DSBSparse object.
        self.coulomb_matrix = compute_config.dsbsparse_type.from_sparray(
            sparse.coo_matrix(
                (
                    xp.asarray(
                        coulomb_matrix_sparray.tocsr()[
                            hamiltonian_sparray.row, hamiltonian_sparray.col
                        ]
                    )[0],
                    (hamiltonian_sparray.row, hamiltonian_sparray.col),
                )
            ),
            block_sizes,
            (self.energies.size,),
            densify_blocks=[(i, i) for i in range(len(block_sizes))],
        )

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
        (sigma_retarded,) = out
        for m in (g_lesser, sigma_retarded, self.coulomb_matrix):
            m.dtranspose() if m.distribution_state != "nnz" else None
        # Compute the electron density by summing over energies.
        gl_density = self.prefactor * g_lesser.data.sum(axis=0)
        sigma_retarded._data[
            sigma_retarded._stack_padding_mask,
            ...,
            : sigma_retarded.nnz_section_sizes[comm.rank],
        ] += (
            gl_density * self.coulomb_matrix.data
        )
        for m in (g_lesser, sigma_retarded, self.coulomb_matrix):
            m.dtranspose() if m.distribution_state != "stack" else None
