# Module 04 — Viscous Burgers' Equation

## The physics

$$u_t + u\,u_x = \nu\,u_{xx}, \qquad x \in [-1, 1],\; t \in [0, 1]$$

$$u(x, 0) = -\sin(\pi x), \qquad u(-1, t) = u(1, t) = 0, \qquad \nu = 0.01/\pi$$

Burgers' equation is the simplest PDE that combines **nonlinear advection** (`u u_x`)
with **diffusion** (`ν u_xx`). It is a 1D caricature of the momentum equation in
Navier-Stokes and the standard benchmark from the original PINN paper (Raissi et al., 2019).

What happens physically: points where `u > 0` move right, points where `u < 0` move
left. With the initial condition `-sin(πx)`, the fluid on both sides moves *towards*
`x = 0`, the wave front steepens and, around `t ≈ 1/π ≈ 0.32`, forms a **viscous
shock**: a transition from `u ≈ +1` to `u ≈ -1` over a very thin layer.

### How thin is the shock?

For a stationary shock between `u = +1` and `u = -1` the exact profile is

```
u(x) = -tanh(x / (2ν))
```

so the transition width is of order `2ν ≈ 0.006`. Compare this with the spacing of
6 000 uniformly random collocation points in the `2 × 1` domain: about
`sqrt(2 / 6000) ≈ 0.018`. On average **the shock falls between collocation points**,
so the PDE residual is barely sampled where it matters most.

Lowering `ν` makes the shock thinner and the problem harder. With `ν → 0` the solution
becomes a true discontinuity and the strong form of the PDE (the thing a PINN
minimises) no longer holds there.

## Reference solution: Cole-Hopf

The substitution `u = -2ν φ_x / φ` turns Burgers into the linear heat equation
`φ_t = ν φ_xx`, which has a closed-form convolution solution. After the change of
variables `η = sqrt(4νt)·z` the integrals take the form `∫ g(z) e^{-z²} dz`, which
Gauss-Hermite quadrature evaluates to machine precision (`reference_solution()`).

Two numerical details worth understanding:

- The integrand contains `exp(-cos(πy) / (2πν))`, which spans about `e^{±50}`.
  The exponent is shifted by its maximum before `exp` (log-sum-exp trick); the shift
  cancels in the ratio.
- The reference was cross-checked against an independent finite-volume solver
  (4 001 cells, SSP-RK3): mean difference ~3e-4, with the maximum (~1e-2) located at
  the shock, where the finite-volume scheme adds numerical diffusion.

## New technique: Residual-based Adaptive Refinement (RAR)

Fixed collocation points are a *static* guess about where the physics is difficult.
RAR makes this guess adaptive (Lu et al., 2021; Wu et al., 2023):

```
every rar_every Adam steps:
    draw 50 000 random candidate points
    evaluate |r(x, t)| = |u_t + u u_x - nu u_xx| on all of them
    sample n_add candidates with probability  p ~ |r|^k / mean(|r|^k) + c
    append them to the collocation set
```

This is the **RAR-D** variant (residual-based *density*), with `k = 1, c = 1` as the
default recommended by Wu et al. (2023). The original greedy RAR simply takes the
`n_add` candidates with the largest `|r|` (the limit `k -> inf`). On Burgers that
places hundreds of almost identical points on the shock line, which over-weights it
in the mean-squared loss and, in our tests, made training unstable. Sampling from a
density still concentrates points at the shock, but spreads them out and keeps a
uniform floor (`c`) elsewhere.

Evaluating the residual is cheap (no backward pass through the weights), so
the candidate pool can be 10x larger than the training set.

## Two MLflow runs

Both runs share the network (8 × 20, Tanh), the hard-constrained output from module 03,
the optimiser schedule (10 000 Adam steps with cosine decay, then L-BFGS) and the
**same final number of collocation points** (6 000). Only the *placement* differs.

| Run | Initial points | Refinement | Final points |
|-----|----------------|------------|--------------|
| `uniform` | 6 000 uniform | none | 6 000 |
| `rar`     | 3 000 uniform | 9 rounds × 333 RAR-D points | 5 997 |

Hard constraint used here (compare with module 03):

```
u(x, t) = -sin(πx) + t · (1 − x²) · net(x̂, t̂)
```

`x̂`, `t̂` are the inputs rescaled to `[-1, 1]`. The IC holds because of the `t`
factor, the BCs because `(1 − x²)` vanishes at `x = ±1` and `sin(±π) = 0`.

Metrics logged every 250 steps: `L_pde`, `rel_l2` and `max_abs_err` against the
Cole-Hopf reference on a 257 × 101 grid, `n_collocation`, and (RAR only)
`candidate_mean_residual`, an unbiased estimate of the domain-average residual.

## What to look for

- **`residual_and_points.png`**: in the `rar` panel the added points pile up along
  the shock line `x = 0, t > 0.3`. In the `uniform` panel the residual is largest
  exactly there.
- **Time slices in `result_*.png`**: the error is concentrated at the shock; away
  from it both runs are accurate. A smeared or offset shock is the typical failure.
- **`L_pde` vs `rel_l2` in `convergence.png`**: the training loss is measured on
  *different point sets* in the two runs (RAR deliberately adds hard points, which
  makes its loss jump up after each refinement). Only `rel_l2` is a fair comparison.
  This is a general lesson: **a low PINN training loss is not a certificate of
  accuracy**; in real applications without a reference, check the residual on fresh
  points.

## New concepts vs module 03

| Concept | Module 03 | Module 04 |
|---------|-----------|-----------|
| PDE | linear (heat) | nonlinear (advection + diffusion) |
| Solution features | smooth, decaying | steepening front, thin internal layer |
| Collocation | fixed uniform | fixed uniform vs. residual-adaptive (RAR) |
| Reference | closed form | Cole-Hopf + Gauss-Hermite quadrature |
| Error metric | max / mean abs error | relative L2 (standard in the literature) |
| Input scaling | none | inputs mapped to `[-1, 1]` |

## Run

```bash
cd 04_burgers
uv run python pinn_burgers.py            # full runs, several minutes on a laptop CPU
uv run python pinn_burgers.py --quick    # smoke test, results not meaningful
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

## Output (written to `media/`)

- `result_uniform.png`, `result_rar.png` — reference / PINN / error maps, plus
  slices at `t = 0.25, 0.5, 0.75` (error map overlaid with the collocation points)
- `residual_and_points.png` — `log10 |residual|` with the final collocation points
- `convergence.png` — `L_pde` and relative L2 error over the whole training

## Exercises

1. **Viscosity sweep.** Run with `ν ∈ {0.1/π, 0.01/π, 0.003/π}` (change `NU`). At
   which `ν` does the uniform run break down? Does RAR push that limit?
2. **Soft vs hard constraints.** Replace `BurgersPINN.forward` with a raw network plus
   IC/BC penalty terms (module 03 `PINNSoft`). How does `rel_l2` change?
3. **Greedy vs density.** Set `rar_k` very large and `rar_c = 0` (close to greedy
   top-k) and compare with the default `k = 1, c = 1`. Also try `k = 2, c = 0`.
   Watch `rel_l2` right after each refinement step in MLflow.
4. **Fixed-budget resampling.** Keep the number of points constant and *replace*
   the lowest-residual points instead of adding new ones. Compare cost and accuracy.
5. **Causality.** Plot `rel_l2` restricted to `t < 0.3` and `t > 0.3` separately.
   PINNs often fit late times before early ones are correct; see Wang et al. (2024)
   on causal training.

## References

- Raissi, Perdikaris, Karniadakis (2019) — *Physics-informed neural networks*. JCP 378.
- Lu, Meng, Mao, Karniadakis (2021) — *DeepXDE: A deep learning library for solving
  differential equations*. SIAM Review 63(1). (Introduces RAR.)
- Wu, Zhu, Tan, Kartha, Lu (2023) — *A comprehensive study of non-adaptive and
  residual-based adaptive sampling for physics-informed neural networks*. CMAME 403.
- Wang, Sankaran, Perdikaris (2024) — *Respecting causality for training
  physics-informed neural networks*. CMAME 421.
- Basdevant et al. (1986) — *Spectral and finite difference solutions of the Burgers
  equation*. Computers & Fluids 14(1). (Cole-Hopf reference values.)
