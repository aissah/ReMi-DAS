"""Synthetic ambient noise generation for displacement and DAS strain rate.

This package provides small, focused modules for array geometry, medium
dispersion, source field configuration, signal utilities, synthesis routines,
and basic derivative-based estimators.
"""

from .geometry import ArrayGeometry
from .medium import MediumModel
from .fields import (
    TrafficField,
    make_even_cones,
    sample_azimuth_deg,
    sample_radius_m,
)
from .signals import raised_cosine_taper, traffic_band_envelope
from .synthesis import synthesize_displacement, synthesize_strain_rate
from .derivatives import calculate_strain_rate_from_displacement

__all__ = [
    "ArrayGeometry",
    "MediumModel",
    "TrafficField",
    "make_even_cones",
    "sample_azimuth_deg",
    "sample_radius_m",
    "raised_cosine_taper",
    "traffic_band_envelope",
    "synthesize_displacement",
    "synthesize_strain_rate",
    "calculate_strain_rate_from_displacement",
]






