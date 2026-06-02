# Module 05 — Inverse Problem: Parameter Identification

## The Setup

We return to the spring-mass-damper of module 01:

$$m\ddot{x} + c\,\dot{x} + k\,x = 0, \quad x(0)=1,\; \dot{x}(0)=0, \quad m=1$$

But now the roles are reversed:

| | Module 01 (forward) | Module 05 (inverse) |
|---|---|---|
| Inputs | m, c, k (known) | sparse, noisy measurements of x(t) |
| Output | trajectory x(t) | the unknown c and k |

The network maps `t → x(t)` exactly as before. The difference is that **c and k
are trainable scalar parameters** of the model, initialised far from the true
values and recovered through joint optimisation.

## Loss function

$$\mathcal{L} = \lambda_\text{data}\,\mathcal{L}_\text{data} + \lambda_\text{ode}\,\mathcal{L}_\text{ode} + \lambda_\text{ic}\,\mathcal{L}_\text{ic}$$

| Term | Expression | Role |
|------|-----------|------|
| L_data | mean squared error at observed points | pulls c, k to fit the measurements |
| L_ode  | mean squared PDE residual on collocation points | enforces physics globally, regularises the inverse problem |
| L_ic   | (x(0)−1)² + (ẋ(0)−0)² | pins the trajectory at t=0 |

**Physics acts as a regulariser.** Without L_ode, fitting 5 noisy points would
be hopelessly under-constrained. The ODE residual anchors the entire trajectory,
so even a handful of observations suffices to identify two parameters.

## Positivity constraints on c, k

c and k must be positive (physically: real damping, stable spring). Rather than
a hard clamp (which zeros gradients), the raw parameters pass through `softplus`:

```
c = softplus(c_raw) ≈ log(1 + exp(c_raw))
```

This is smooth, always positive, and lets the optimiser freely explore ℝ.

## Three experiments

| Run | Observations | Noise | What you learn |
|-----|-------------|-------|----------------|
| `clean_20` | 20 | none | best-case accuracy — how close can a PINN get? |
| `noisy_20` | 20 | 5% Gaussian | realistic sensor noise; does the physics regularise effectively? |
| `noisy_5`  | 5  | 5% Gaussian | sparse regime — the ODE residual matters most here |

True values: **c = 0.4, k = 4.0**. Initial guesses: c_init = 1.0, k_init = 2.0
(2.5× and 2× off, respectively).

## Powertrain context

| Problem | Analogue |
|---------|---------|
| Identify c, k from vibration data | Torsional stiffness + damping from coast-down log |
| Identify α from temperature sensors | Thermal diffusivity (inverse of module 02) |
| Identify ν from flow measurements | Fluid viscosity (inverse of module 04) |

The key message: when a physics model is available, embedding it in the loss
function transforms an under-determined regression into a well-posed identification
problem.

## New concepts vs module 04

| Concept | Modules 01–04 | Module 05 |
|---------|---------------|-----------|
| Unknowns | network weights | weights **+** physical parameters c, k |
| Loss terms | physics only | physics **+** data fit |
| Validation | compare to ground truth trajectory | compare *identified parameters* to true values |
| Positivity | — | `softplus` reparameterisation |

## Run

```bash
cd 05_inverse_problem
uv run python pinn_inverse.py
mlflow ui   # http://localhost:5000
```

## Output (`media/`, regenerated on each run)

- `result_clean_20.png` — trajectory comparison + parameter convergence curve
- `result_noisy_20.png` — same for the noisy experiment
- `result_noisy_5.png`  — same for the sparse experiment
- `loss_clean_20.png` etc. — per-term loss diagnostics
- `summary.png` — bar chart of c/k identification error across all three runs

## Experiments to try

1. **Worse initial guesses**: start c_init=5.0, k_init=0.5. Does the optimiser still converge?
2. **Higher noise**: push noise to 20%. Where does the physics regularisation break down?
3. **Single observation**: reduce to n_obs=1. The ODE has a unique solution — is 1 measurement
   plus physics sufficient to identify two parameters?
4. **Identify α in the heat equation**: adapt the approach to module 02. The parameter enters
   linearly in the PDE, making convergence much faster.
5. **Uncertainty quantification**: replace the point estimate with a Bayesian PINN
   (dropout or ensemble) to get confidence intervals on c and k.
