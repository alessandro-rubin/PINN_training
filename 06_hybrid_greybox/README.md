# Module 06 — Hybrid Physics + Data: Grey-box Universal Differential Equations

## A change of paradigm

Modules 01–05 were all **PINNs**: the network *was* the solution, `t → x(t)`,
and the physics lived in the loss via autograd derivatives. This module flips
the role of the network.

| | PINN (01–05) | Neural ODE / UDE (06) |
|---|---|---|
| What the network represents | the **solution** `x(t)` | the **dynamics** `dx/dt = f(x)` |
| How physics enters | residual penalty in the loss | structure of `f` (known terms kept) |
| How we get a trajectory | one forward pass | **integrate** `f` with an ODE solver |
| Gradient path | autograd on the output field | backprop **through the integrator** |

Here the network learns the *vector field* (or the unknown part of it), and a
differentiable RK4 solver rolls it forward in time. Training matches the rolled-out
trajectory to data — "discretise then optimise".

## The system: a Duffing oscillator with an unmodelled term

The true plant has a **hardening (nonlinear) spring**:

$$m\ddot{x} + c\dot{x} + kx + \beta x^3 = 0,\qquad x(0)=x_0,\ \dot{x}(0)=0$$

Our engineer's first-principles model knows the linear part (`m, c, k` are the
true datasheet values) but is **missing the** $\beta x^3$ **term** — nobody
modelled the stiffening. The grey-box question: can we *keep* the trusted linear
physics and *learn only the missing force* from data?

## Three models, head-to-head

| Run | Dynamics | Idea |
|-----|----------|------|
| `physics_only` | $\ddot{x} = (-c\dot{x} - kx)/m$ | naive first-principles; **no learning**. Structurally incomplete → wrong frequency. |
| `blackbox_node` | $\ddot{x} = a_{NN}(x,\dot{x})$ | pure Neural ODE: learn the **whole** acceleration. Flexible, but data-hungry, not interpretable, generalises poorly. |
| `greybox_ude` | $\ddot{x} = (-c\dot{x} - kx - g_{NN}(x))/m$ | **keep physics, learn the residual** $g_{NN}(x)$. Data-efficient, generalises, and $g_{NN}$ is interpretable. |

The grey-box residual is hypothesised to depend on **displacement only** — a
modelling choice that injects domain knowledge (a position-dependent restoring
force) and makes $g_{NN}(x)$ directly comparable to the true $\beta x^3$.

## Two test scenarios

1. **Training fit** — integrate from the training IC, `x0 = 1.2`.
2. **Generalisation** — integrate from a **new, larger** IC, `x0 = 2.0`, never
   seen during training. This is the digital-twin question: *identified on one
   experiment, does it hold at a different operating point?* The black box has
   never visited this region of phase space; the grey box leans on exact linear
   physics and degrades gracefully.

## What to look for

- **`greybox_residual.png`** — the headline result. The learned $g_{NN}(x)$
  should overlay the true $\beta x^3$ **inside the training x-range** (shaded),
  and visibly deviate outside it — an honest picture of neural-network
  extrapolation limits.
- **`greybox_trajectories.png`** — `physics_only` drifts out of phase (wrong
  frequency); both learned models fit the training window; on the new IC the
  grey box tracks longest.
- **`greybox_summary.png`** — training-fit vs generalisation MSE. The grey box
  wins on generalisation by the largest margin.

## Differentiable integration (no `torchdiffeq`)

The RK4 stepper is written in plain torch ops so gradients flow from the
trajectory back to the weights. Seeing the solver explicitly — rather than
hiding it behind a library — is half the point. For long horizons or stiff
systems you would reach for the adjoint method (`torchdiffeq.odeint_adjoint`),
which trades recompute for constant memory; for these short rollouts plain
backprop-through-the-solver is simpler and exact.

## New concepts vs module 05

| Concept | Module 05 (inverse) | Module 06 (grey-box) |
|---------|---------------------|----------------------|
| Network role | solution field `x(t)` | vector field `f(x)` |
| Unknown | scalar parameters `c, k` | a **function** `g(x)` (the missing physics) |
| Forward model | autograd residual | **ODE solver** rollout |
| Validation | identified params vs truth | residual `g_NN(x)` vs `βx³` + generalisation |

## Run

```bash
uv run python 06_hybrid_greybox/greybox_ude.py
mlflow ui   # http://localhost:5000

# quick smoke run (fewer epochs):
GREYBOX_EPOCHS=300 uv run python 06_hybrid_greybox/greybox_ude.py
```

## Output (`media/`, regenerated on each run)

- `greybox_residual.png` — learned `g_NN(x)` vs true `βx³` (the interpretability payoff)
- `greybox_trajectories.png` — training fit + generalisation to a new IC
- `greybox_phase.png` — phase portrait (x vs v) for the generalisation scenario
- `greybox_loss.png` — training loss for the two learned models
- `greybox_summary.png` — fit vs generalisation MSE across all three models

## Experiments to try

1. **Train on position only.** Drop the velocity term from the loss (measure `x`
   alone). Neural-ODE training gets harder — does the physics prior in the grey
   box make it more robust than the black box?
2. **Sparser / shorter data.** Halve the training horizon or subsample the data.
   The grey box should degrade far more gracefully — quantify the data-efficiency gap.
3. **Wrong residual hypothesis.** Let `g_NN` depend on `(x, v)` instead of `x`.
   Does it still recover `βx³`, or does it smear the term across spurious
   velocity dependence?
4. **Discover the term symbolically.** Fit a small polynomial / SINDy-style
   library to the learned `g_NN(x)` and check it recovers a clean `≈1.0·x³`.
5. **Adjoint integration.** Swap the hand-written RK4 for `torchdiffeq.odeint_adjoint`
   and push to a long horizon where backprop-through-the-solver runs out of memory.
6. **Identify β as a parameter.** Combine with module 05: assume you *know* the
   term is cubic but not its coefficient, and fit a single scalar `β` instead of
   a network — the most data-efficient grey box of all.
