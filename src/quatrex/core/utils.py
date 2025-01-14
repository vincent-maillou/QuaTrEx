from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp
from scipy import sparse


def assemble_kpoint_dsb(
    buffer: DSBSparse,
    lattice_matrix: dict[tuple, sparse.csr_matrix],
    number_of_kpoints: xp.ndarray,
    roll_index: int | xp.ndarray,
) -> DSBSparse:
    """Assembles a DSBSparse with the k-point distribution."""
    if isinstance(roll_index, int):
        roll_index = xp.array([roll_index, roll_index, roll_index])
    for i, ii in enumerate(xp.roll(range(number_of_kpoints[0]), roll_index[0])):
        for j, jj in enumerate(xp.roll(range(number_of_kpoints[1]), roll_index[1])):
            for k, kk in enumerate(xp.roll(range(number_of_kpoints[2]), roll_index[2])):
                stack_index = tuple(
                    [i]
                    if number_of_kpoints[0] > 1
                    else (
                        [] + [j]
                        if number_of_kpoints[1] > 1
                        else [] + [k]
                        if number_of_kpoints[2] > 1
                        else []
                    )
                )
                ik = (ii - number_of_kpoints[0] // 2) / number_of_kpoints[0]
                jk = (jj - number_of_kpoints[1] // 2) / number_of_kpoints[1]
                kk = (kk - number_of_kpoints[2] // 2) / number_of_kpoints[2]
                for cell_index in lattice_matrix.keys():
                    buffer.stack[(...,) + stack_index] -= (
                        xp.exp(
                            2
                            * xp.pi
                            * 1j
                            * (
                                ik * cell_index[0]
                                + jk * cell_index[1]
                                + kk * cell_index[2]
                            )
                        )
                        * lattice_matrix[cell_index]
                    )
