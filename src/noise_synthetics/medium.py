from dataclasses import dataclass
import numpy as np


@dataclass
class MediumModel:
    """Simple surface-wave dispersion model.

    Provides phase velocity and wavenumber interpolation over frequency.
    """

    freqs: np.ndarray  # Hz, strictly increasing
    velocities: np.ndarray  # m/s, same length as freqs

    def phase_velocity(self, f: np.ndarray | float) -> np.ndarray:
        return np.interp(f, self.freqs, self.velocities)

    def wavenumber(self, f: np.ndarray | float) -> np.ndarray:
        c = self.phase_velocity(f)
        omega = 2 * np.pi * np.asarray(f)
        eps = 1e-9
        return omega / np.maximum(c, eps)






