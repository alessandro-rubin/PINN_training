# Module 02 — Next Steps

Suggested improvements, ordered roughly from quick wins to deeper experiments.

---

## Quick wins

### 1. Hard-enforce BCs architecturally
Instead of the `λ_bc` penalty term, multiply the network output by `x·(L−x)`.
This forces `u(0,t) = u(L,t) = 0` exactly by construction, removing one loss term and its tuning knob.

```python
def forward(self, x, t):
    xt = torch.cat([x, t], dim=1)
    raw = self.net(xt)
    return raw * x * (L - x)   # hard BC satisfaction
```

For the IC, a correction form works: `u(x,t) = sin(πx/L) + t · network(x,t)`, which satisfies the IC exactly at `t=0`.

### 2. Add L-BFGS fine-tuning after Adam
A standard PINN trick — Adam gets you into a good basin, L-BFGS tightens it.
Often halves the error with ~500 extra steps and negligible extra code.

### 3. Plot the loss history
`history` is collected in `train()` but never visualised. A log-scale plot of
`L_pde`, `L_ic`, `L_bc` over epochs reveals training dynamics — e.g. if BC loss
plateaus early, `λ_bc` is too low.

---

## Deeper experiments (good MLflow sweep candidates)

### 4. Adaptive loss weights
Fixed `λ = 10` is a guess. A simple improvement: rescale each loss term by the
inverse of its gradient norm at each step (GradNorm-style). Reduces sensitivity
to manual tuning and often improves convergence stability.

### 5. Residual-based adaptive refinement (RAR)
After N epochs, evaluate the PDE residual over a dense grid, identify high-residual
regions, and add extra collocation points there. Particularly effective when the
solution has sharp gradients in time or space.

### 6. Fourier feature input encoding
Networks struggle to learn multi-frequency functions from raw `(x, t)` inputs.
Replace the input layer with sinusoidal encodings:

```python
# [x, t] → [sin(k₁x), cos(k₁x), ..., sin(k₁t), cos(k₁t), ...]
```

Big accuracy improvement for almost no code change. See Tancik et al. (2020),
*Fourier Features Let Networks Learn High Frequency Functions*.

### 7. Sweep `λ_ic` / `λ_bc` as hyperparameters
The current MLflow sweep varies network depth but keeps loss weights fixed.
A 2D grid `λ ∈ {1, 10, 100}` crossed with `hidden_layers ∈ {2, 4, 6}` would
show how sensitive final accuracy is to this choice.

---

## Pedagogically useful extensions

### 8. Non-uniform diffusivity α(x)
Replace constant `ALPHA` with a spatially varying field, e.g.:

```python
alpha = 0.1 + 0.05 * torch.sin(np.pi * x / L)
```

No closed-form solution exists — validate against a finite-difference reference.
This is much closer to real battery cell or brake-disc thermal models where
material properties vary across the domain.

### 9. Extend to 2D: `(x, y, t) → u`
Add a second spatial dimension to get a plate instead of a rod. Computational
cost increases but the network change is minimal (3D input instead of 2D).
Natural stepping stone before cylindrical geometries relevant to powertrain
components.

### 10. Inverse problem: infer α from measurements
Make `ALPHA` a `nn.Parameter` and provide synthetic noisy observations of `u(x,t)`.
The PINN then jointly fits the field *and* identifies the diffusivity.
This is the use case where PINNs offer a genuine advantage over classical solvers —
and it is a small code change from the current forward problem.

---

## Suggested order of attack

| Priority | Task | Why |
|----------|------|-----|
| First | Plot loss history (#3) | Diagnose what is actually happening during training |
| Second | L-BFGS fine-tuning (#2) | Easy win, standard PINN practice |
| Third | Inverse problem (#10) | Most industrially relevant; shows PINNs' unique value |
| Later | Fourier features (#6) | Bigger accuracy gain once the basics are solid |
| Later | Non-uniform α (#8) | Bridges to real-world thermal models |
