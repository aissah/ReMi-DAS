from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import numpy.random as npr


@dataclass
class TrafficField:
    azimuth_mean_deg: float
    azimuth_std_deg: float
    rate_per_min: float
    duration_mean_s: float
    # Optional cone-based control: (center_deg, halfwidth_deg, weight[, alpha])
    azimuth_cones: Optional[List[Tuple[float, float, float, float]]] = None
    source_radii_m: Optional[List[float]] = None
    ring_min_radius_m: Optional[float] = None
    ring_max_radius_m: Optional[float] = None
    ring_sampling: str = "area"
    spectral_alpha_default: float = 0.0
    spectral_ref_hz: float = 1.0
    distance_decay_exponent: float = 0.5
    reference_radius_m: float = 1000.0
    wavelet_type: str = "colored_noise"
    ricker_fc_range_hz: Tuple[float, float] = (5.0, 25.0)
    ricker_cycles_range: Tuple[float, float] = (10.0, 30.0)
    user_event_times_s: Optional[List[float]] = None


def make_even_cones(
    n_cones: int,
    halfwidth_deg: float,
    weights: Optional[List[float]] = None,
    alphas: Optional[List[float]] = None,
) -> List[Tuple[float, float, float, float]]:
    centers = np.linspace(0.0, 360.0, n_cones, endpoint=False)
    if weights is None:
        weights = [1.0] * n_cones
    if alphas is None:
        alphas = [0.0] * n_cones
    return [
        (float(c), float(halfwidth_deg), float(w), float(a))
        for c, w, a in zip(centers, weights, alphas)
    ]


def sample_azimuth_deg(field: TrafficField) -> tuple[float, float]:
    if field.azimuth_cones and len(field.azimuth_cones) > 0:
        weights = np.array([w for (_, _, w, *_) in field.azimuth_cones], dtype=float)
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        idx = npr.choice(len(field.azimuth_cones), p=weights)
        entry = field.azimuth_cones[idx]
        if len(entry) == 4:
            center, halfw, _, alpha = entry
        else:
            center, halfw, _ = entry
            alpha = field.spectral_alpha_default
        az = npr.uniform(center - halfw, center + halfw)
        return (az + 360.0) % 360.0, float(alpha)
    return float(npr.normal(field.azimuth_mean_deg, field.azimuth_std_deg)), float(
        field.spectral_alpha_default
    )


def sample_radius_m(field: TrafficField, default_radius: float) -> float:
    if field.ring_min_radius_m is not None and field.ring_max_radius_m is not None:
        rmin = float(field.ring_min_radius_m)
        rmax = float(field.ring_max_radius_m)
        rmin, rmax = max(0.0, min(rmin, rmax)), max(rmin, rmax)
        if rmax <= rmin:
            return float(default_radius)
        if field.ring_sampling.lower() == "area":
            u = npr.uniform(0.0, 1.0)
            return float(np.sqrt(u * (rmax * rmax - rmin * rmin) + rmin * rmin))
        return float(npr.uniform(rmin, rmax))
    if field.source_radii_m and len(field.source_radii_m) > 0:
        return float(npr.choice(field.source_radii_m))
    return float(default_radius)






