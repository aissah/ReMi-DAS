# **ReMi-DAS**  
**Refraction Microtremor Processing for Distributed Acoustic Sensing Data**

**ReMi-DAS** is an open-source toolkit for applying Refraction Microtremor (ReMi) analysis to Distributed Acoustic Sensing (DAS) data.  
It extends conventional ReMi workflows to accommodate strain/strain-rate measurements from DAS arrays, enabling efficient and scalable shear-wave velocity (Vs) profiling—particularly in urban or infrastructure-constrained environments.

Author: Shihao Yuan (`syuan@mines.edu`)

DISCLAIMER: This is a development build. The code may contain errors or unstable functionality. Contributions and feedback are welcome.

This package is developed using core functionality from the [**DASCore**](https://github.com/DASDAE/dascore/tree/master) project — a flexible Python library for reading, processing, and visualizing DAS data.

---

### Features

- A Jupyter notebook demonstrating the complete ReMi workflow
- Modular Python scripts for:
  - Preprocessing DAS data (e.g., tapering, filtering)
  - Slowness-frequency transformation using Tau-P methods
  - Dispersion curve picking and visualization
- Tools for slowness-frequency image plotting and Rayleigh-wave dispersion analysis

---

### Synthetic study

The notebook `notebooks/ReMi-DAS-synthetic.ipynb` provides a synthetic study that:
- Generates ambient traffic-noise fields and simulates DAS strain-rate along a fiber using the `src/noise_synthetics` module.
- Exports synthetic data to a DASCore `Patch` and runs the same ReMi workflow end-to-end (tapering, f–p transform, dispersion picking).

This is useful for checking parameters and understanding expected dispersion features before applying the workflow to field datasets.

---
  
#### How synthetic traffic noise is generated
- Event process: vehicle pass-bys are modeled as a Poisson process (rate per minute), with random amplitudes and durations (exponential about a mean).
- Source geometry: each event samples an azimuth (Gaussian or weighted cones) and a source distance (fixed list or uniform/area-weighted ring). Amplitude optionally decays with distance.
- Source time function: either colored noise shaped to a traffic band (≈1–49 Hz) with optional spectral tilt, or a Ricker pulse; windowed with a raised-cosine taper.
- Propagation: events are projected as plane waves using a dispersion relation k(f)=ω/c(f) from a user-specified medium; phase delays are applied across the linear array.
- Strain-rate: derived in the frequency domain by multiplying by −i·ω·k·cos(θ); both displacement and strain-rate can be returned.

### References

- McMechan, G.A. and Yedlin, M.J., 1981. Analysis of dispersive waves by wave field transformation. *Geophysics*, 46(6), pp.869-874.  
- Louie, J.N., 2001. Faster, better: shear-wave velocity to 100 meters depth from refraction microtremor arrays. *BSSA*, 91(2), pp.347-364. 
- Chambers, D., Jin, G., Tourei, A., Issah, A.H.S., Lellouch, A., Martin, E.R., Zhu, D., Girard, A.J., Yuan, S., Cullison, T. and Snyder, T., 2024. Dascore: A python library for distributed fiber optic sensing. *Seismica*, 3(2), pp.10-26443.
