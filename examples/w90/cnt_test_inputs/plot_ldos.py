import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quatrex.core.quatrex_config import parse_config
from quatrex.post_processing.plot_ldos import get_averages, spectral_plot

PATH = os.path.dirname(__file__)

if __name__ == "__main__":
    input_dir = Path(f"{PATH}/inputs")
    output_dir = Path(f"{PATH}/outputs")
    num_cells = len(np.load(input_dir / "block_sizes.npy"))

    energies = np.load(input_dir / "electron_energies.npy")
    electron_ldos = np.load(output_dir / "electron_ldos.npy")

    config = parse_config(f"{PATH}/config.toml")

    left_fermi_level = config.electron.left_fermi_level
    right_fermi_level = config.electron.right_fermi_level

    fig, ax_ldos = plt.subplots(figsize=(8, 5))

    num_energies = len(energies)
    average_ldos = get_averages(
        electron_ldos, num_energies, num_cells, method="rolling"
    )

    spectral_plot(
        ax_ldos, average_ldos, energies, cmap="viridis", colorbar_label="LDOS"
    )

    plt.show()
