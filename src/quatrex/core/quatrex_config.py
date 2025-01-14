# Copyright 2023-2025 ETH Zurich and the QuaTrEx authors. All rights reserved.

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self


class SCSPConfig(BaseModel):
    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)


class SCBAConfig(BaseModel):
    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)

    observables_interval: PositiveInt | None = None

    coulomb_screening: bool = False
    photon: bool = False
    phonon: bool = False


class PoissonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["point-charge", "orbital"] = "point-charge"
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5
    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)

    num_orbitals_per_atom: dict[str, int] = Field(default_factory=dict)


class MemoizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable: bool = True
    num_ref_iterations: PositiveInt = 2
    convergence_tol: PositiveFloat = 1e-5


class OBCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sancho-rubio", "spectral"] = "spectral"
    nevp_solver: Literal["beyn", "full"] = "beyn"

    # Parameters for spectral OBC algorithms.
    block_sections: PositiveInt = 1
    min_decay: PositiveFloat = 1e-6
    max_decay: PositiveFloat | None = None
    num_ref_iterations: PositiveInt = 2
    x_ii_formula: Literal["self-energy", "direct", "stabilized"] = "self-energy"

    # Parameters for iterative OBC algorithms.
    max_iterations: PositiveInt = 1000
    convergence_tol: PositiveFloat = 1e-7

    # Parameters for subspace NEVP solvers.
    r_o: PositiveFloat = 10.0
    r_i: PositiveFloat = 0.9
    c_hat: PositiveInt = 10
    num_quad_points: PositiveInt = 20

    # Parameters for reusing surface Green's functions from previous
    # SCBA iterations.
    memoizer: MemoizerConfig = MemoizerConfig()


class LyapunovConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["spectral", "doubling", "vectorize"] = "spectral"

    # Parameters for iterative Lyapunov algorithms.
    max_iterations: PositiveInt = 1000
    convergence_tol: PositiveFloat = 1e-7

    memoizer: MemoizerConfig = MemoizerConfig()


class ElectronConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solver: Literal["rgf", "inv"] = "rgf"

    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    eta_obc: NonNegativeFloat = 1e-6  # eV
    eta: NonNegativeFloat = 1e-12  # eV

    fermi_level: float | None = None

    left_fermi_level: float | None = None
    right_fermi_level: float | None = None

    temperature: PositiveFloat = 300.0  # K

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None

    @model_validator(mode="after")
    def set_left_right_fermi_levels(self) -> Self:
        if (self.left_fermi_level is None) != (self.right_fermi_level is None):
            raise ValueError(
                "Either both left and right Fermi levels must be set or neither."
            )

        if self.left_fermi_level is None and self.right_fermi_level is None:
            if self.fermi_level is None:
                raise ValueError("Fermi level must be set.")

            self.left_fermi_level = self.fermi_level
            self.right_fermi_level = self.fermi_level

        return self

    @model_validator(mode="after")
    def set_left_right_temperatures(self) -> Self:
        if (self.left_temperature is None) != (self.right_temperature is None):
            raise ValueError(
                "Either both left and right temperatures must be set or neither."
            )

        if self.left_temperature is None and self.right_temperature is None:
            self.left_temperature = self.temperature
            self.right_temperature = self.temperature

        return self


class CoulombScreeningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solver: Literal["rgf", "inv"] = "rgf"
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    temperature: PositiveFloat = 300.0  # K

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None


class PhotonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solver: Literal["rgf", "inv"] = "rgf"
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()


class PhononConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solver: Literal["rgf", "inv"] = "rgf"
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    model: Literal["pseudo-scattering", "negf"] = "pseudo-scattering"
    phonon_energy: NonNegativeFloat | None = None
    deformation_potential: NonNegativeFloat | None = None
    temperature: PositiveFloat = 300.0  # K

    @model_validator(mode="after")
    def check_phonon_energy_or_deformation_potential(self):
        if self.model == "pseudo-scattering" and (
            self.phonon_energy is None or self.deformation_potential is None
        ):
            raise ValueError("'phonon_energy' and 'deformation_potential' must be set.")

        return self


class QuatrexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ---------------------------------------
    scsp: SCSPConfig = SCSPConfig()
    scba: SCBAConfig = SCBAConfig()
    poisson: PoissonConfig = PoissonConfig()

    electron: ElectronConfig

    phonon: PhononConfig | None = None
    coulomb_screening: CoulombScreeningConfig | None = None
    photon: PhotonConfig | None = None

    # --- Directory paths ----------------------------------------------
    config_dir: Path
    simulation_dir: Path = Path("./quatrex/")

    @model_validator(mode="after")
    def resolve_config_path(self) -> Self:
        self.config_dir = Path(self.config_dir).resolve()
        return self

    @model_validator(mode="after")
    def resolve_simulation_dir(self):
        self.simulation_dir = (self.config_dir / self.simulation_dir).resolve()
        return self

    @property
    def input_dir(self) -> Path:
        return self.simulation_dir / "inputs/"


def parse_config(config_file: Path) -> QuatrexConfig:
    """Reads the TOML config file."""

    config_file = Path(config_file).resolve()

    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    
    if 'simulation_dir' in config:
        simulation_dir = config['simulation_dir']
        if not os.path.isabs(simulation_dir):
            parent_dir = os.path.dirname(os.path.abspath(config_file))
            simulation_dir = Path(os.path.join(parent_dir, simulation_dir))
            config['simulation_dir'] = simulation_dir

    config["config_dir"] = config_file.parent

    return QuatrexConfig(**config)
