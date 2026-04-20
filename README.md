# Dual Quaternion Unscented Kalman Filter (DQ-UKF) for 6-DOF Pose Estimation

Code accompanying the paper:

> *Observability and Filter Design for Visual Pose Estimation via Dual Quaternions*  
> Nicholas B. Andrews and Kristi A. Morgansen
<!-- > *[Venue, Year]*  
> [[Paper]](#) | [[arXiv]](#) -->

## Overview

This repository implements a **Dual Quaternion Unscented Kalman Filter (DQ-UKF)** for
estimating the 6-DOF relative pose between a camera and a target object equipped with
visual markers.

Key features:

- Dual quaternion representation of rigid-body pose on the SE(3) manifold.
- Unscented Kalman filter with manifold-aware retract/lift operations so that
  the unit-norm and vector-dual constraints on the state are maintained throughout.
- Numerical dynamics integration via `scipy.integrate.solve_ivp`.
- OpenCV `solvePnP` baseline for comparison.
- Simulation of dynamic occlusion (variable number of visible markers per step).

## Repository structure

```
.
├── quaternion.py       # Quaternion and dual quaternion algebra library
├── dqukf.py            # DQ-UKF filter implementation
├── pose_estimation.py  # Simulation and pose estimation demo (main script)
└── plotters.py         # Visualization utilities
```

## Installation

Requires Python 3.11.

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate dqukf_pose_estimation
```

**pip:**

```bash
pip install -r requirements.txt
```

## Usage

Run the full simulation and generate comparison plots:

```bash
python pose_estimation.py
```

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{[key],
  title   = {[Title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
``` -->

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
