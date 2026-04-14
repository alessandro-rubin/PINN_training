# Module 03 — Robust PINN Training

## Why this module exists

Modules 01 and 02 produce working PINNs on clean problems, but two failure modes lurk
beneath the surface:

1. **Loss imbalance** — one term dominates; others stagnate. You don't notice because
   total loss decreases.
2. **Soft constraint violation** — BCs and IC are approximated, not enforced. They can
   drift, especially early in training or with a bad λ guess.

This module introduces four techniques that address these issues directly. The physics
is identical to module 02 (1D heat equation) so the comparison is clean.

## Four techniques

| # | Technique | Problem it solves |
|---|-----------|-------------------|
| 1 | **Loss diagnostics** | Blind training — you can't see which term is failing |
| 2 | **Adaptive weights** | Fixed λ is a guess; one term can silently dominate |
| 3 | **Hard constraints** | Soft BCs/IC can be violated; architecture makes them exact |
| 4 | **L-BFGS polish** | Adam plateaus near saddle points; second-order escapes them |

## Hard constraint formulation

The key idea is to encode the constraints directly into the network's output layer.

For this problem (zero Dirichlet BCs + sinusoidal IC), the transformation is:

```
u(x, t) = sin(πx/L)  +  t · x · (L−x) · net(x, t)
```

Check each constraint:

| Constraint | Why it holds |
|------------|-------------|
| IC: u(x, 0) = sin(πx/L) | The `t` factor zeroes the correction at t=0 |
| BC: u(0, t) = 0 | The `x` factor zeroes the correction at x=0; sin(0)=0 |
| BC: u(L, t) = 0 | The `(L−x)` factor zeroes the correction at x=L; sin(π)=0 |

Because all constraints are satisfied for any network weights, the only remaining
loss term is the PDE residual — no λ tuning needed at all.

## Adaptive weights

The `AdaptiveWeights` class tracks an exponential moving average of each loss
component and returns weights inversely proportional to their magnitude:

```
w_i  ∝  1 / EMA(L_i)
```

Weights are normalised so their mean equals 1, preserving the overall loss scale.
Effect: if L_pde is momentarily 100× larger than L_ic, its weight is scaled down
proportionally, preventing it from overwhelming the smaller terms.

## Three MLflow runs

| Run | Constraints | Weights | Optimiser |
|-----|-------------|---------|-----------|
| `soft_fixed`    | soft (penalty) | fixed λ=10   | Adam |
| `soft_adaptive` | soft (penalty) | adaptive     | Adam |
| `hard_lbfgs`    | hard (architecture) | none    | Adam → L-BFGS |

The `hard_lbfgs` run uses 6 000 Adam steps followed by 200 L-BFGS steps.

## New concepts vs module 02

| Concept | Module 02 | Module 03 |
|---------|-----------|-----------|
| BC/IC enforcement | soft penalty | hard (architecture) or adaptive soft |
| Loss weights | fixed λ=10 | adaptive EMA-based |
| Optimiser | Adam only | Adam + L-BFGS polish |
| Diagnostics | total loss only | per-term loss curves |

## Run

```bash
uv run python pinn_robust.py
mlflow ui   # http://localhost:5000
```

## Output

- `result_soft_fixed.png` — error maps for the baseline run
- `result_soft_adaptive.png` — error maps for the adaptive-weight run
- `result_hard_lbfgs.png` — error maps for the hard-constraint run
- `loss_curves.png` — per-term loss diagnostics for all three runs side by side
