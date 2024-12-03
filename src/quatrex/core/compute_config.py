# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator
from qttools.datastructures import DSBCOO, DSBCSR, DSBSparse


class ComputeConfig(BaseModel):
    """All configurations concerning computational details."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dsbsparse_type: DSBSparse = DSBCOO

    @field_validator("dbsparse_type", mode="before")
    def set_dbsparse(cls, value) -> DSBSparse:
        """Converts the string value to the corresponding DSBSparse object."""
        if value == "DSBCSR":
            return DSBCSR
        elif value == "DSBCOO":
            return DSBCOO
        raise ValueError(f"Invalid value '{value}' for dbsparse")


def parse_config(config_file: Path) -> ComputeConfig:
    """Reads the TOML config file.

    Parameters
    ----------
    config_file : Path
        Path to the TOML config file.

    Returns
    -------
    ComputeConfig
        The parsed compute config.

    """
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return ComputeConfig(**config)
