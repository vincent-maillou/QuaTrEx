from scipy.sparse import coo_array

from quatrex.core.config import QuatrexConfig
from qttools.datastructures.coogroup import COOGroup


class SigmaPhonon:
    """Computes the lesser electron-photon self-energy.

    Parameters
    ----------
    config : QuasiConfig
        Dictionary containing the configuration parameters.
    contour_order : str
        The contour order of the self-energy. Either "lesser" or
        "greater".

    Attributes
    ----------
    M : np.ndarray
        The electron-photon interaction tensor.
    G : np.ndarray
        The electron Green's function.
    D : np.ndarray
        The photon Green's function.
    num_electron_energies : int
        The number of electron energies.
    num_photon_energies : int
        The number of photon energies.
    num_sites : int
        The number of sites.
    num_diagonals : int
        The number of off-diagonals to compute.

    Methods
    -------
    compute()
        Compute the electron-photon self-energy.

    """

    def __init__(
        self,
        config: QuatrexConfig,
        g_lesser: coo_array,
        g_greater: coo_array,
        d_lesser: coo_array | None = None,
        d_greater: coo_array | None = None,
    ) -> None:
        """Initializes the self-energy."""
        self.model = config.phonon.model
        self.deformation_potential = config.phonon.deformation_potential
        self.energy = config.phonon.energy

        self.g_lesser = g_lesser
        self.g_greater = g_greater

    def pseudo_scattering(self) -> COOGroup:
        """Computes the electron-photon self-energy using the pseudo-scattering
        model.

        Returns
        -------
        sigma : np.ndarray
            The electron-photon self-energy.

        """
        # TODO: Implement the pseudo-scattering model.
        ...

    def greens_function(self, parallel: bool = False) -> COOGroup:

        raise NotImplementedError("Greens function method not implemented.")
