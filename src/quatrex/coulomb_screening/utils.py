from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp


def assemble_boundary_blocks(
    diag_left: xp.ndarray,
    right: xp.ndarray,
    below: xp.ndarray,
    diag_right: xp.ndarray,
    above: xp.ndarray,
    left: xp.array,
    mat: "DSBSparse",
) -> None:
    """Assembles the boundary blocks from the smaller blocks of input Hamiltonian."""
    diag_left[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[0, 0],
                    mat.blocks[0, 1],
                    mat.blocks[0, 2],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[1, 0],
                    mat.blocks[0, 0],
                    mat.blocks[0, 1],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[2, 0],
                    mat.blocks[1, 0],
                    mat.blocks[0, 0],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    right[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[0, 3],
                    xp.zeros_like(mat.blocks[0, 3]),
                    xp.zeros_like(mat.blocks[0, 3]),
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[0, 2],
                    mat.blocks[0, 3],
                    xp.zeros_like(mat.blocks[0, 2]),
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[0, 1],
                    mat.blocks[0, 2],
                    mat.blocks[0, 3],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    below[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[3, 0],
                    xp.zeros_like(mat.blocks[3, 0]),
                    xp.zeros_like(mat.blocks[3, 0]),
                ),
                axis=1,
            ),
            xp.concatenate(
                (
                    mat.blocks[2, 0],
                    mat.blocks[3, 0],
                    xp.zeros_like(mat.blocks[2, 0]),
                ),
                axis=1,
            ),
            xp.concatenate(
                (
                    mat.blocks[1, 0],
                    mat.blocks[2, 0],
                    mat.blocks[3, 0],
                ),
                axis=1,
            ),
        ),
        axis=2,
    )
    diag_right[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[-1, -1],
                    mat.blocks[-2, -1],
                    mat.blocks[-3, -1],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[-1, -2],
                    mat.blocks[-1, -1],
                    mat.blocks[-2, -1],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[-1, -3],
                    mat.blocks[-1, -2],
                    mat.blocks[-1, -1],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    above[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[-4, -1],
                    xp.zeros_like(mat.blocks[-4, -1]),
                    xp.zeros_like(mat.blocks[-4, -1]),
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[-3, -1],
                    mat.blocks[-4, -1],
                    xp.zeros_like(mat.blocks[-3, -1]),
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[-2, -1],
                    mat.blocks[-3, -1],
                    mat.blocks[-4, -1],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    left[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[-1, -4],
                    mat.blocks[-1, -3],
                    mat.blocks[-1, -2],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    xp.zeros_like(mat.blocks[-1, -3]),
                    mat.blocks[-1, -4],
                    mat.blocks[-1, -3],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    xp.zeros_like(mat.blocks[-1, -4]),
                    xp.zeros_like(mat.blocks[-1, -4]),
                    mat.blocks[-1, -4],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )


def special_obc_multiply(
    boundary_system_matrix: DSBSparse, dsbs1: DSBSparse, dsb2: DSBSparse
):
    """Special case of multiply for calculating the boundary_system_matrix."""
    # Left side
    boundary_system_matrix.blocks[0, 0] -= (
        dsbs1.blocks[0, 0] @ dsb2.blocks[0, 0]
        + dsbs1.blocks[0, 1] @ dsb2.blocks[1, 0]
        + dsbs1.blocks[1, 0] @ dsb2.blocks[0, 1]
    )
    boundary_system_matrix.blocks[0, 1] -= (
        dsbs1.blocks[0, 0] @ dsb2.blocks[0, 1] + dsbs1.blocks[0, 1] @ dsb2.blocks[0, 0]
    )
    boundary_system_matrix.blocks[0, 2] -= dsbs1.blocks[0, 1] @ dsb2.blocks[0, 1]
    boundary_system_matrix.blocks[1, 0] -= (
        dsbs1.blocks[1, 0] @ dsb2.blocks[0, 0] + dsbs1.blocks[0, 0] @ dsb2.blocks[1, 0]
    )
    boundary_system_matrix.blocks[2, 0] -= dsbs1.blocks[1, 0] @ dsb2.blocks[1, 0]
    # Right side
    boundary_system_matrix.blocks[-1, -1] -= (
        dsbs1.blocks[-1, -1] @ dsb2.blocks[-1, -1]
        + dsbs1.blocks[-1, -2] @ dsb2.blocks[-2, -1]
        + dsbs1.blocks[-2, -1] @ dsb2.blocks[-1, -2]
    )
    boundary_system_matrix.blocks[-1, -2] -= (
        dsbs1.blocks[-1, -1] @ dsb2.blocks[-1, -2]
        + dsbs1.blocks[-1, -2] @ dsb2.blocks[-1, -1]
    )
    boundary_system_matrix.blocks[-1, -3] -= dsbs1.blocks[-1, -2] @ dsb2.blocks[-1, -2]
    boundary_system_matrix.blocks[-2, -1] -= (
        dsbs1.blocks[-2, -1] @ dsb2.blocks[-1, -1]
        + dsbs1.blocks[-1, -1] @ dsb2.blocks[-2, -1]
    )
    boundary_system_matrix.blocks[-3, -1] -= dsbs1.blocks[-2, -1] @ dsb2.blocks[-2, -1]
