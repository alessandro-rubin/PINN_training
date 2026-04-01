# PINN Training Repo

Self-directed learning path for Physics-Informed Neural Networks (PINNs) and Scientific ML,
focused on powertrain / embedded digital twin applications.

## Learning Path

| # | Module | Physics | PINN Concept |
|---|--------|---------|--------------|
| 01 | [Damped Oscillator](01_ode_basics/) | Spring-mass-damper ODE | Autograd, collocation, IC loss |
| 02 | 1D Heat Equation | Parabolic PDE | Boundary conditions, space-time domain |
| 03 | Burgers' Equation | Nonlinear PDE | Shock capturing, viscosity |
| 04 | Inverse Problem | Parameter ID from data | Data loss + physics loss, estimating k/c |
| 05 | Hybrid Physics+Data | Residual dynamics | Neural ODE, grey-box modeling |

## Setup

```bash
uv sync
```

## References

- Raissi, Perdikaris, Karniadakis (2019) — *Physics-informed neural networks*. JCP.
- Karniadakis et al. (2021) — *Physics-informed machine learning*. Nature Reviews Physics.
