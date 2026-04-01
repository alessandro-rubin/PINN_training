# Module 01 — Damped Harmonic Oscillator PINN

## The Physics

$$m\ddot{x} + c\dot{x} + kx = 0, \quad x(0)=1,\; \dot{x}(0)=0$$

A spring-mass-damper — structurally identical to torsional drivetrain modes (replace x with angular displacement θ, k with torsional stiffness, c with damping coefficient).

Parameters used: m=1, c=0.4, k=4 → underdamped (ζ≈0.1), ω_n=2 rad/s.

## The PINN Approach

Instead of integrating the ODE numerically, we train a network `x_nn(t; θ)` to:

1. **Satisfy the ODE** at N random collocation points in [0, T]:
   - Loss: `mean( (m*x'' + c*x' + k*x)² )`
   - Derivatives from autograd — exact, no discretization

2. **Match initial conditions** at t=0:
   - Loss: `(x_nn(0) - 1)² + (x'_nn(0) - 0)²`

Total loss = L_ODE + L_IC → minimize with Adam.

## Key concepts to understand before moving on

- Why `requires_grad=True` on input `t`?
- Why `create_graph=True` in `grad()`?
- What happens if you use ReLU instead of Tanh?
- How does the loss landscape change with more/fewer collocation points?

## Experiments to try

1. **Increase damping** to c=2.0 (critically damped) — does the PINN still converge?
2. **Reduce collocation points** to 20 — how does accuracy degrade?
3. **Longer horizon** T=20 — PINNs often struggle with long time horizons. Why?
4. **Forced oscillation**: add a forcing term `F*cos(omega_f * t)` on the right-hand side.

## Run

```bash
cd 01_ode_basics
uv run python pinn_oscillator.py
```

Output: `oscillator_result.png`
