from qttools import xp
from qttools.kernels.numba.dsbcoo import compute_block_sort_index
from qttools.utils.mpi_utils import distributed_load

from quatrex.coulomb_screening import CoulombScreeningSolver


def _block_canonicalize(rows, cols, block_sizes):
    """Canonicalize and block order for indices to have a consistent ordering."""
    # Canonical ordering of the transpose.
    inds_bcoo2canonical = xp.lexsort(xp.vstack((cols, rows)))
    canonical_rows = rows[inds_bcoo2canonical]
    canonical_cols = cols[inds_bcoo2canonical]
    # Compute index for sorting the transpose by block.
    inds_canonical2bcoo = compute_block_sort_index(
        canonical_rows, canonical_cols, block_sizes
    )
    # Mapping directly from original ordering to transpose
    # block-ordering is achieved by chaining the two mappings.
    inds_bcoo2bcoo = inds_bcoo2canonical[inds_canonical2bcoo]
    # Cache the necessary objects.
    return inds_bcoo2bcoo


def test_solve(
    datadir, quatrex_config, compute_config, block_sizes, coulomb_screening_energies
):
    """Test the computation of the polarization."""
    # Load the data
    pl_data = distributed_load(datadir.joinpath("pl_data_iter0.npy"))
    pg_data = distributed_load(datadir.joinpath("pg_data_iter0.npy"))
    # TODO: Also load data for p_retarded
    wl_data = distributed_load(datadir.joinpath("wl_data_iter0.npy"))
    wg_data = distributed_load(datadir.joinpath("wg_data_iter0.npy"))
    rows = distributed_load(datadir.joinpath("rows.npy"))
    cols = distributed_load(datadir.joinpath("columns.npy"))
    reordering = _block_canonicalize(rows, cols, block_sizes)
    # Reorder the data
    pl_data = pl_data[..., reordering]
    pg_data = pg_data[..., reordering]
    wl_data = wl_data[..., reordering]
    wg_data = wg_data[..., reordering]
    rows = rows[reordering]
    cols = cols[reordering]
    # Create the DSBSparse objects
    p_lesser = compute_config.dbsparse_type(
        pl_data, rows, cols, block_sizes, (pl_data.shape[0],)
    )
    p_greater = compute_config.dbsparse_type(
        pg_data, rows, cols, block_sizes, (pg_data.shape[0],)
    )
    p_retarded = compute_config.dbsparse_type.zeros_like(p_lesser)
    p_retarded.data[:] = (p_greater.data - p_lesser.data) / 2
    w_lesser = compute_config.dbsparse_type.zeros_like(p_lesser)
    w_greater = compute_config.dbsparse_type.zeros_like(p_greater)
    w_retarded = compute_config.dbsparse_type.zeros_like(p_greater)
    new_block_sizes = block_sizes[: len(block_sizes) // 3] * 3
    w_lesser.block_sizes = new_block_sizes
    w_greater.block_sizes = new_block_sizes
    w_retarded.block_sizes = new_block_sizes
    # Create the expected results
    w_lesser_expected = compute_config.dbsparse_type(
        wl_data, rows, cols, block_sizes, (wl_data.shape[0],)
    )
    w_greater_expected = compute_config.dbsparse_type(
        wg_data, rows, cols, block_sizes, (wg_data.shape[0],)
    )
    # Initialize the polarization object
    coulomb_screening_solver = CoulombScreeningSolver(
        quatrex_config, compute_config, coulomb_screening_energies
    )
    # Compute the polarization
    coulomb_screening_solver.solve(
        p_lesser,
        p_greater,
        p_retarded,
        out=(w_lesser, w_greater, w_retarded),
    )
    # Compare the results
    assert xp.allclose(w_lesser.data, w_lesser_expected.data)
    assert xp.allclose(w_greater.data, w_greater_expected.data)
