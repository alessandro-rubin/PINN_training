# PINN Training Repo

Self-directed learning path for Physics-Informed Neural Networks (PINNs) and Scientific ML,
focused on powertrain / embedded digital twin applications.

## Learning Path

| # | Module | Physics | PINN Concept |
|---|--------|---------|--------------|
| 01 | [Damped Oscillator](01_ode_basics/) | Spring-mass-damper ODE | Autograd, collocation, IC loss |
| 02 | [1D Heat Equation](02_heat_equation/) | Parabolic PDE | 2D input, Dirichlet BCs, weighted loss terms — [next steps](02_heat_equation/NEXT_STEPS.md) |
| 03 | [Robust Training](03_robust_training/) | 1D Heat Equation (revisited) | Hard constraints, adaptive loss weights, L-BFGS |
| 04 | Burgers' Equation | Nonlinear PDE | Shock capturing, viscosity |
| 05 | Inverse Problem | Parameter ID from data | Data loss + physics loss, estimating k/c |
| 06 | Hybrid Physics+Data | Residual dynamics | Neural ODE, grey-box modeling |

## Setup

```bash
uv sync
```

## References

- Raissi, Perdikaris, Karniadakis (2019) — *Physics-informed neural networks*. JCP.
- Karniadakis et al. (2021) — *Physics-informed machine learning*. Nature Reviews Physics.
