from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp


def assemble_boundary_blocks(
    diag_left: xp.ndarray,
    right: xp.ndarray,
    below: xp.ndarray,
    diag_right: xp.ndarray,
    above: xp.ndarray,
    left: xp.ndarray,
    mat: "DSBSparse",
    nbc: int,
) -> None:
    """Assembles the boundary blocks from the smaller blocks of input Hamiltonian.

    Parameters
    ----------
    diag_left : xp.ndarray
        The diagonal left block.
    right : xp.ndarray
        The right block of the diagonal left block.
    below : xp.ndarray
        The below block of the diagonal left block.
    diag_right : xp.ndarray
        The diagonal right block.
    above : xp.ndarray
        The above block of the diagonal right block.
    left : xp.ndarray
        The left block of the diagonal right block.
    mat : DSBSparse
        The boundary system matrix.
    nbc : int
        The number of small blocks used for the boundary blocks.
    """
    # TODO: Below is not fault-tolerant. Add checks.
    bs = diag_left.shape[1] // nbc
    num_blocks = mat.block_sizes.size
    for i in range(nbc):
        for j in range(nbc):
            k = i - j
            diag_left[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                max(0, k), -min(0, k)
            ]
            diag_right[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                min(0, k) - 1, -max(0, k) - 1
            ]
            if -k + nbc < num_blocks:
                right[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                    0, -k + nbc
                ]
                above[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                    k - nbc - 1, -1
                ]
            if k + nbc < num_blocks:
                below[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                    k + nbc, 0
                ]
                left[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                    -1, -k - nbc - 1
                ]
