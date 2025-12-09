from __future__ import annotations

from datetime import datetime
import numpy as np

try:
    import dascore as dc
    from dascore.utils.time import to_timedelta64
except Exception as exc:  # pragma: no cover - optional dependency
    dc = None
    to_timedelta64 = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def to_dascore_patch(arr2d: np.ndarray, data_name: str, distance_step: float, fs: float):
    """Convert a 2D array into a DASCore Patch.

    Raises a clear error if dascore is not installed.
    """
    if dc is None:
        raise RuntimeError(
            "dascore is not installed. Install with `pip install dascore` or add the optional dependency.") from _IMPORT_ERROR

    attrs = dict(category="Synthetic", id=data_name)

    time_start = dc.to_datetime64(datetime.now())
    time_step = to_timedelta64(1 / fs)
    time = time_start + np.arange(arr2d.shape[1]) * time_step

    distance_start = 0
    distance = distance_start + np.arange(arr2d.shape[0]) * distance_step

    coords = dict(distance=distance, time=time)
    dims = ("distance", "time")

    return dc.Patch(data=arr2d, coords=coords, attrs=attrs, dims=dims)






