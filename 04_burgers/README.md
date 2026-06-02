# Module 04 — Viscous Burgers' Equation

## Physics

| | |
|---|---|
| PDE | ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² |
| IC | u(x, 0) = −sin(πx) |
| BCs | u(−1, t) = 0,  u(1, t) = 0 |
| Domain | x ∈ [−1, 1],  t ∈ [0, 1] |
| Viscosity | ν = 0.01/π ≈ 3.18 × 10⁻³ |
| Reference | high-resolution finite-difference solver (no clean closed form) |

Burgers' equation is the simplest PDE combining **nonlinear advection**
(`u ∂u/∂x`) with **diffusion** (`ν ∂²u/∂x²`). From the smooth sine initial
profile, the negative slope at the centre steepens until — around t ≈ 0.5 — it
forms a near-discontinuity, a **shock**, whose width is set by ν. This is the
canonical PINN benchmark from Raissi, Perdikaris & Karniadakis (2019).

In powertrain terms it is the prototype for any convection-dominated transport:
pressure-wave steepening in an exhaust manifold, 1D gas dynamics in an intake
runner, or fuel-film advection — wherever a quantity is carried by its own
velocity field and steep fronts emerge.

## New PINN concepts vs module 03

| Concept | Modules 01–03 | Module 04 |
|---------|---------------|-----------|
| PDE linearity | linear residual | **nonlinear** — `u · u_x` couples u to its own gradient |
| Solution regularity | smooth | sharp **shock** at x ≈ 0 for t ≳ 0.5 |
| Validation target | analytical formula | **numerical reference** (finite differences) |
| Network size | 4 × 32 | 8 × 40 (the shock is far stiffer) |
| Collocation count | ~2 000 | 10 000 (front needs resolution) |

The nonlinear term is the whole story: the same advection that sharpens the front
also makes the loss landscape stiffer, so this module leans on the **L-BFGS polish**
introduced in module 03 to drive the residual down after Adam stalls.

## Why no analytical solution?

The viscous Burgers equation admits a semi-analytical Cole–Hopf solution, but it
reduces to an awkward improper integral that is itself solved numerically. It is
cleaner and more transparent to validate against a **method-of-lines** reference:
second-order central differences in space on a fine grid (1000 points), integrated
with a stiff BDF solver to survive the thin viscous layer. The PINN's quality is
reported as a relative L2 error against this field.

## Run

```bash
cd 04_burgers
uv run python pinn_burgers.py
mlflow ui   # http://localhost:5000
```

## Output (`media/`, regenerated on each run)

- `result_burgers.png` — three-panel space-time maps: reference, PINN, absolute error
- `snapshots_burgers.png` — `u(x, ·)` slices at t = 0, 0.25, 0.5, 0.75, 1.0 over the
  reference (grey), showing the shock sharpen
- `loss_curve.png` — per-term loss diagnostics (Technique 1 from module 03)

## Experiments to try

1. **Sharpen the shock**: drop ν to 0.001/π. Does the PINN smear the front? How much
   more collocation / capacity does it need?
2. **Residual-adaptive sampling**: resample collocation points where the PDE residual
   is largest (i.e. near the shock) instead of uniformly. This is the single biggest
   accuracy lever for Burgers.
3. **Hard IC constraint**: borrow the output transform from module 03 to bake in
   `u(x,0) = −sin(πx)` exactly and drop `L_ic`.
4. **Causality**: the shock at late t depends on early-t dynamics. Try a
   time-marching / causal weighting scheme that trains earlier times first.

## References

- Raissi, Perdikaris, Karniadakis (2019) — *Physics-informed neural networks*. JCP.
- Basdevant et al. (1986) — spectral/FD reference solutions for Burgers' equation.
