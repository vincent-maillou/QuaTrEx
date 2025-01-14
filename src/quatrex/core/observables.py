# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse

from quatrex.electron import ElectronSolver

if xp.__name__ == "numpy":
    from scipy.signal import find_peaks
elif xp.__name__ == "cupy":
    from cupyx.scipy.signal import find_peaks
else:
    raise ImportError("Unknown backend.")


def get_block(
    coo: sparse.coo_matrix,
    block_sizes: NDArray,
    block_offsets: NDArray,
    index: tuple,
) -> NDArray:
    """Gets a block from a COO matrix.

    Parameters
    ----------
    coo : sparse.coo_matrix
        The COO matrix.
    block_sizes : NDArray
        The block sizes.
    block_offsets : NDArray
        The block offsets.
    index : tuple
        The index of the block to extract.

    Returns
    -------
    block : NDArray
        The requested, dense block.

    """
    row, col = index

    mask = (
        (block_offsets[row] <= coo.row)
        & (coo.row < block_offsets[row + 1])
        & (block_offsets[col] <= coo.col)
        & (coo.col < block_offsets[col + 1])
    )
    block = xp.zeros((int(block_sizes[row]), int(block_sizes[col])), dtype=coo.dtype)
    block[
        coo.row[mask] - block_offsets[row],
        coo.col[mask] - block_offsets[col],
    ] = coo.data[mask]

    return block


def density(x: DSBSparse, overlap: sparse.spmatrix | None = None) -> NDArray:
    """Computes the density from Green's function and overlap matrix.

    Parameters
    ----------
    x : DSBSparse
        The Green's function.
    overlap : sparse.spmatrix, optional
        The overlap matrix, by default None.

    Returns
    -------
    NDArray
        The density, i.e. the imaginary part of the diagonal of the
        Green's function.

    """
    if overlap is None:
        local_density = x.diagonal().imag
        return xp.vstack(comm.allgather(local_density))

    local_density = []
    overlap = overlap.tocoo()
    for i in range(x.num_blocks):
        overlap_diag = get_block(overlap, x.block_sizes, x.block_offsets, (i, i))
        local_density_slice = xp.diagonal(
            x.blocks[i, i] @ overlap_diag, axis1=-2, axis2=-1
        ).copy()
        if i < x.num_blocks - 1:
            overlap_upper = get_block(
                overlap, x.block_sizes, x.block_offsets, (i + 1, i)
            )
            local_density_slice += xp.diagonal(
                x.blocks[i, i + 1] @ overlap_upper, axis1=-2, axis2=-1
            )
        if i > 0:
            overlap_lower = get_block(
                overlap, x.block_sizes, x.block_offsets, (i - 1, i)
            )
            local_density_slice += xp.diagonal(
                x.blocks[i, i - 1] @ overlap_lower, axis1=-2, axis2=-1
            )

        local_density.append(local_density_slice.imag)

    return xp.vstack(comm.allgather(xp.hstack(local_density)))


def contact_currents(solver: ElectronSolver) -> tuple[NDArray, NDArray]:
    """Computes the contact currents.

    Parameters
    ----------
    solver : ElectronSolver
        The electron solver.

    Returns
    -------
    NDArray
        The contact currents, gathered across all participating ranks.

    """
    i_left = xp.hstack(comm.allgather(solver.i_left))
    i_right = xp.hstack(comm.allgather(solver.i_right))
    return i_left, i_right


def band_edges(
    electron_ldos: NDArray, energies: NDArray, mid_gap_energies: NDArray
) -> tuple[NDArray, NDArray]:
    """Computes the band edges from the local density of states.

    Parameters
    ----------
    ldos : NDArray
        The local density of states.
    energies : NDArray
        The energies corresponding to the LDOS.
    mid_gap_energies : NDArray
        The mid-gap energies through the whole device.

    Returns
    -------
    valence_band_edges : NDArray
        The valence band edges.
    conduction_band_edges : NDArray
        The conduction band edges.

    """

    conduction_band_edges = xp.zeros_like(mid_gap_energies)
    valence_band_edges = xp.zeros_like(mid_gap_energies)
    for i in range(electron_ldos.shape[1]):
        peaks = find_peaks(xp.abs(electron_ldos[:, i]), height=1e-8)[0]
        peak_energies = energies[peaks]
        mid_gap_energy = mid_gap_energies[i]
        mask = (peak_energies - mid_gap_energy) < 0
        valence_band_edges[i] = peak_energies[mask].max()
        conduction_band_edges[i] = peak_energies[~mask].min()

    return valence_band_edges, conduction_band_edges
