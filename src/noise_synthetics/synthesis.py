from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.random as npr

from .geometry import ArrayGeometry
from .medium import MediumModel
from .fields import TrafficField, sample_azimuth_deg, sample_radius_m
from .signals import raised_cosine_taper, traffic_band_envelope


@dataclass
class SimulationConfig:
    fs: float
    T: float
    fade_s: float = 2.0


def _build_source_time_function(
    field: TrafficField, cfg: SimulationConfig, t0: float
) -> tuple[np.ndarray, int, int, float, Optional[float], Optional[float]]:
    if field.wavelet_type.lower() == "ricker":
        fc = float(npr.uniform(*field.ricker_fc_range_hz))
        cycles = float(npr.uniform(*field.ricker_cycles_range))
        T_wav = max(1.0 / cfg.fs, cycles / max(fc, 1e-6))
        n_evt = max(int(np.ceil(T_wav * cfg.fs)), 64)
        tvec = (np.arange(n_evt) - n_evt / 2) / cfg.fs
        arg = np.pi * fc * tvec
        s = (1.0 - 2.0 * (arg**2)) * np.exp(-(arg**2))
        s = s / (np.std(s) + 1e-9)
        dur = T_wav
        start = int(np.floor((t0 - 0.5 * T_wav) * cfg.fs))
        return s, n_evt, start, dur, fc, cycles

    # Colored noise with band shaping and spectral tilt
    dur = float(np.maximum(0.3, npr.exponential(field.duration_mean_s)))
    n_evt = max(int(np.ceil(dur * cfg.fs)), 128)
    w_evt = raised_cosine_taper(n_evt, p=0.2)
    s = npr.normal(0, 1, n_evt) * w_evt
    freqs_evt = np.fft.rfftfreq(n_evt, d=1 / cfg.fs)
    bp_evt = traffic_band_envelope(freqs_evt)
    f0 = max(0.1, float(field.spectral_ref_hz))
    # alpha_spec is applied outside using the selected azimuth cone
    start = int(np.floor(t0 * cfg.fs))
    return s, n_evt, start, dur, None, None


def synthesize_displacement(
    array: ArrayGeometry,
    medium: MediumModel,
    field: TrafficField,
    cfg: SimulationConfig,
    *,
    collect_events: bool = False,
    enforce_bidirectional: bool = False,
):
    n = int(round(cfg.fs * cfg.T))
    t = np.arange(n) / cfg.fs

    freqs_full = np.fft.rfftfreq(n, d=1 / cfg.fs)
    k0_full = medium.wavenumber(freqs_full)

    x = array.x
    data = np.zeros((array.n_sensors, n), dtype=np.float64)

    events = [] if collect_events else None

    lam = field.rate_per_min * cfg.T / 60.0
    lam_base = lam * 0.5 if enforce_bidirectional else lam
    if field.user_event_times_s:
        base_times = [float(tt) for tt in field.user_event_times_s if 0.0 <= float(tt) < cfg.T]
    else:
        n_events = npr.poisson(lam_base)
        base_times = list(npr.uniform(0.0, cfg.T, size=n_events))

    for t0 in base_times:
        A = np.abs(npr.normal(1.0, 0.3))
        az_base_deg, alpha_spec = sample_azimuth_deg(field)
        az_list_deg = [az_base_deg, (az_base_deg + 180.0) % 360.0] if enforce_bidirectional else [az_base_deg]
        A_per = A / np.sqrt(2.0) if enforce_bidirectional else A

        r_sample = sample_radius_m(field, default_radius=field.reference_radius_m)
        if field.distance_decay_exponent is not None and field.reference_radius_m > 0:
            A_per *= (max(r_sample, 1.0) / field.reference_radius_m) ** (
                -float(field.distance_decay_exponent)
            )

        s, n_evt, start, dur, fc, cycles = _build_source_time_function(field, cfg, t0)

        # Apply band/tilt if colored noise
        if fc is None:
            freqs_evt = np.fft.rfftfreq(n_evt, d=1 / cfg.fs)
            bp_evt = traffic_band_envelope(freqs_evt)
            f0 = max(0.1, float(field.spectral_ref_hz))
            tilt = (np.maximum(freqs_evt, f0) / f0) ** (-float(alpha_spec))
            S = np.fft.rfft(s)
            S *= bp_evt * tilt
            s = np.fft.irfft(S, n_evt)
            s = s / (np.std(s) + 1e-9)

        end = min(max(start, 0) + n_evt, n)
        seg_start = 0 if start >= 0 else -start
        start_clip = max(start, 0)
        seg = s[seg_start : seg_start + (end - start_clip)]

        evt_full = np.zeros(n, dtype=float)
        evt_full[start_clip:end] = seg
        Sfull = np.fft.rfft(evt_full)

        for az_deg in az_list_deg:
            az = np.deg2rad(az_deg)
            cos_theta = np.cos(az)
            # per-channel phase factor
            for ich, xi in enumerate(x):
                phase = np.exp(-1j * k0_full * xi * cos_theta)
                Si = Sfull * phase
                ei = np.fft.irfft(Si, n)
                data[ich] += A_per * ei

            if collect_events:
                ev = {
                    "t0_s": float(t0),
                    "duration_s": float(dur),
                    "amplitude": float(A_per),
                    "azimuth_deg": float(az_deg),
                    "radius_m": float(r_sample),
                    "alpha_spec": float(alpha_spec),
                    "wavelet_type": field.wavelet_type,
                }
                if fc is not None:
                    ev.update({"fc_hz": float(fc), "cycles": float(cycles)})
                events.append(ev)

    data *= raised_cosine_taper(n, p=min(cfg.fade_s / cfg.T, 0.45))
    if collect_events:
        return t, data, events
    return t, data


def synthesize_strain_rate(
    array: ArrayGeometry,
    medium: MediumModel,
    field: TrafficField,
    cfg: SimulationConfig,
    *,
    collect_events: bool = False,
    enforce_bidirectional: bool = False,
):
    n = int(round(cfg.fs * cfg.T))
    t = np.arange(n) / cfg.fs

    freqs_full = np.fft.rfftfreq(n, d=1 / cfg.fs)
    k0_full = medium.wavenumber(freqs_full)
    omega_full = 2 * np.pi * freqs_full

    x = array.x
    data_displacement = np.zeros((array.n_sensors, n), dtype=np.float64)
    data_strain_rate = np.zeros((array.n_sensors, n), dtype=np.float64)

    events = [] if collect_events else None

    lam = field.rate_per_min * cfg.T / 60.0
    lam_base = lam * 0.5 if enforce_bidirectional else lam
    if field.user_event_times_s:
        base_times = [float(tt) for tt in field.user_event_times_s if 0.0 <= float(tt) < cfg.T]
    else:
        n_events = npr.poisson(lam_base)
        base_times = list(npr.uniform(0.0, cfg.T, size=n_events))

    for t0 in base_times:
        A = np.abs(npr.normal(1.0, 0.3))
        az_base_deg, alpha_spec = sample_azimuth_deg(field)
        az_list_deg = [az_base_deg, (az_base_deg + 180.0) % 360.0] if enforce_bidirectional else [az_base_deg]
        A_per = A / np.sqrt(2.0) if enforce_bidirectional else A

        r_sample = sample_radius_m(field, default_radius=field.reference_radius_m)
        if field.distance_decay_exponent is not None and field.reference_radius_m > 0:
            A_per *= (max(r_sample, 1.0) / field.reference_radius_m) ** (
                -float(field.distance_decay_exponent)
            )

        s, n_evt, start, dur, fc, cycles = _build_source_time_function(field, cfg, t0)

        if fc is None:
            freqs_evt = np.fft.rfftfreq(n_evt, d=1 / cfg.fs)
            bp_evt = traffic_band_envelope(freqs_evt)
            f0 = max(0.1, float(field.spectral_ref_hz))
            tilt = (np.maximum(freqs_evt, f0) / f0) ** (-float(alpha_spec))
            S = np.fft.rfft(s)
            S *= bp_evt * tilt
            s = np.fft.irfft(S, n_evt)
            s = s / (np.std(s) + 1e-9)

        end = min(max(start, 0) + n_evt, n)
        seg_start = 0 if start >= 0 else -start
        start_clip = max(start, 0)
        seg = s[seg_start : seg_start + (end - start_clip)]

        evt_full = np.zeros(n, dtype=float)
        evt_full[start_clip:end] = seg
        Sfull = np.fft.rfft(evt_full)

        for az_deg in az_list_deg:
            az = np.deg2rad(az_deg)
            cos_theta = np.cos(az)
            strain_rate_factor = -1j * omega_full * k0_full * cos_theta
            for ich, xi in enumerate(x):
                phase = np.exp(-1j * k0_full * xi * cos_theta)
                Si = Sfull * phase
                ei = np.fft.irfft(Si, n)
                data_displacement[ich] += A_per * ei

                Si_str = strain_rate_factor * Si
                ei_str = np.fft.irfft(Si_str, n)
                data_strain_rate[ich] += A_per * ei_str

            if collect_events:
                ev = {
                    "t0_s": float(t0),
                    "duration_s": float(dur),
                    "amplitude": float(A_per),
                    "azimuth_deg": float(az_deg),
                    "radius_m": float(r_sample),
                    "alpha_spec": float(alpha_spec),
                    "wavelet_type": field.wavelet_type,
                }
                if fc is not None:
                    ev.update({"fc_hz": float(fc), "cycles": float(cycles)})
                events.append(ev)

    taper = raised_cosine_taper(n, p=min(cfg.fade_s / cfg.T, 0.45))
    data_displacement *= taper
    data_strain_rate *= taper

    if collect_events:
        return t, data_displacement, data_strain_rate, events
    return t, data_displacement, data_strain_rate






