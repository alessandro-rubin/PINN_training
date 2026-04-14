# Module 02 — 1D Heat Equation

## Physics

| | |
|---|---|
| PDE | ∂u/∂t = α ∂²u/∂x² |
| IC | u(x, 0) = sin(πx/L) |
| BCs | u(0, t) = 0,  u(L, t) = 0 |
| Analytical | u(x,t) = sin(πx/L) · exp(−α(π/L)²t) |

Models heat conduction in a rod (analogous to battery cell thermal diffusion or brake-disc warm-up).

## New PINN concepts vs module 01

| Concept | Module 01 | Module 02 |
|---------|-----------|-----------|
| Network input | t (1D) | (x, t) (2D) |
| Constraints | IC only | IC + two Dirichlet BCs |
| Loss terms | L_ode + L_ic | L_pde + λ_ic · L_ic + λ_bc · L_bc |
| Collocation | time points in [0,T] | (x,t) pairs in [0,L]×[0,T] |

The `lambda_ic` / `lambda_bc` weights upscale the boundary/IC loss relative to
the interior physics residual — a standard trick when constraints are hard to satisfy.

## Run

```bash
uv run python pinn_heat.py
mlflow ui   # http://localhost:5000
```

## Output

`heat_result.png` — three-panel figure: analytical solution, PINN prediction, and pointwise absolute error as 2D colour maps over the (x, t) domain.
