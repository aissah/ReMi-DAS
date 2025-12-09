from __future__ import annotations

import numpy as np
from .geometry import ArrayGeometry


def calculate_strain_rate_from_displacement(
    data_displacement: np.ndarray, array: ArrayGeometry, fs: float
) -> np.ndarray:
    """Finite-difference estimate of strain rate from displacement.

    Strain:      ∂u/∂x via central differences in space
    Strain rate: ∂²u/∂x∂t via central differences in time
    """

    num_sensors, num_samples = data_displacement.shape

    strain = np.zeros_like(data_displacement)
    for i in range(1, num_sensors - 1):
        strain[i] = (data_displacement[i + 1] - data_displacement[i - 1]) / (2 * array.dx)

    if num_sensors > 1:
        strain[0] = (data_displacement[1] - data_displacement[0]) / array.dx
        strain[-1] = (data_displacement[-1] - data_displacement[-2]) / array.dx

    strain_rate = np.zeros_like(strain)
    dt = 1.0 / fs

    for j in range(1, num_samples - 1):
        strain_rate[:, j] = (strain[:, j + 1] - strain[:, j - 1]) / (2 * dt)

    if num_samples > 1:
        strain_rate[:, 0] = (strain[:, 1] - strain[:, 0]) / dt
        strain_rate[:, -1] = (strain[:, -1] - strain[:, -2]) / dt

    return strain_rate






