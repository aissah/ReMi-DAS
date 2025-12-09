from dataclasses import dataclass
import numpy as np


@dataclass
class ArrayGeometry:
    """Uniform linear array geometry.

    Parameters
    ----------
    x0 : float
        First sensor position in meters.
    dx : float
        Sensor spacing in meters.
    n_sensors : int
        Number of sensors.
    """

    x0: float
    dx: float
    n_sensors: int

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.dx * np.arange(self.n_sensors)






