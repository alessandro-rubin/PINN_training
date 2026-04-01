"""
PINN for the Damped Harmonic Oscillator
========================================

Physics:
    m * x''(t) + c * x'(t) + k * x(t) = 0
    x(0) = 1,  x'(0) = 0

This ODE describes a spring-mass-damper system — the same structure you see in
drivetrain torsional modes, suspension dynamics, and actuator resonance.

Analytical solution (underdamped case, zeta < 1):
    x(t) = exp(-zeta * omega_n * t) * cos(omega_d * t + phi)

PINN idea:
    A neural network x_nn(t; theta) is trained to satisfy:
      1. The ODE residual at N_f "collocation" points scattered in [0, T]
      2. The initial conditions x(0) = x0, x'(0) = v0

    Derivatives are computed via automatic differentiation — no finite differences,
    no numerical integration. The network learns the solution by minimizing
    the physics residual.

Run:
    uv run python pinn_oscillator.py
    mlflow ui   # then open http://localhost:5000
"""

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import mlflow.pytorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────
# 1. Physical parameters
# ──────────────────────────────────────────────────
M   = 1.0    # mass [kg]
C   = 0.4    # damping coefficient [N·s/m]
K   = 4.0    # spring stiffness [N/m]
T   = 10.0   # simulation horizon [s]

X0  = 1.0    # initial displacement [m]
V0  = 0.0    # initial velocity [m/s]

# Derived (useful for reference)
omega_n = np.sqrt(K / M)
zeta    = C / (2 * np.sqrt(M * K))
omega_d = omega_n * np.sqrt(1 - zeta**2)
print(f"System: omega_n={omega_n:.3f} rad/s, zeta={zeta:.3f}, omega_d={omega_d:.3f} rad/s")


# ──────────────────────────────────────────────────
# 2. Analytical solution (ground truth)
# ──────────────────────────────────────────────────
def analytical(t_np):
    """Closed-form solution for underdamped oscillator with x(0)=1, v(0)=0."""
    phi = np.arctan(zeta / np.sqrt(1 - zeta**2))
    A   = X0 / np.cos(phi)
    return A * np.exp(-zeta * omega_n * t_np) * np.cos(omega_d * t_np - phi)


# ──────────────────────────────────────────────────
# 3. Neural network architecture
# ──────────────────────────────────────────────────
class PINN(nn.Module):
    """
    Fully-connected network: t -> x(t)

    Input:  scalar time t  (1D)
    Output: displacement x (1D)

    Architecture notes:
    - Tanh activations are standard for PINNs: they are smooth (infinitely
      differentiable), which makes autograd-computed ODE residuals well-behaved.
    - Xavier init keeps gradients in a good range at the start.
    """
    def __init__(self, hidden_layers=4, hidden_size=32):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


def grad(y, x):
    """First derivative dy/dx via autograd. x must require_grad=True."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,   # keep graph so we can differentiate again
    )[0]


# ──────────────────────────────────────────────────
# 4. Loss functions
# ──────────────────────────────────────────────────
def ode_residual(model, t_f):
    """
    Evaluate the ODE: m*x'' + c*x' + k*x = 0

    We need x, x', x'' at each collocation point t_f.
    Autograd gives exact derivatives of the network output.
    """
    x    = model(t_f)
    x_t  = grad(x, t_f)       # dx/dt
    x_tt = grad(x_t, t_f)     # d²x/dt²
    residual = M * x_tt + C * x_t + K * x
    return residual


def ic_loss(model, t0):
    """
    Penalize deviation from initial conditions x(0)=X0, x'(0)=V0.
    """
    x0_pred  = model(t0)
    x0_t     = grad(x0_pred, t0)
    loss_x   = (x0_pred - X0) ** 2
    loss_v   = (x0_t   - V0) ** 2
    return loss_x + loss_v


# ──────────────────────────────────────────────────
# 5. Training
# ──────────────────────────────────────────────────
def train(n_collocation=200, n_epochs=5000, lr=1e-3):
    model = PINN(hidden_layers=4, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    # Collocation points: random draws in [0, T], require grad for autograd
    t_f = torch.FloatTensor(n_collocation, 1).uniform_(0, T).requires_grad_(True)

    # IC point
    t0  = torch.zeros(1, 1, requires_grad=True)

    history = []

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        # Physics loss: ODE residual should be zero everywhere
        res   = ode_residual(model, t_f)
        L_ode = torch.mean(res ** 2)

        # IC loss: match x(0)=1, x'(0)=0
        L_ic  = ic_loss(model, t0).mean()

        # Total loss — IC is weighted more heavily early on; here equal weights
        # are fine for this simple problem. In harder cases you'd tune lambda_ic.
        loss = L_ode + L_ic

        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(loss.item())

        mlflow.log_metrics({
            "loss":  loss.item(),
            "L_ode": L_ode.item(),
            "L_ic":  L_ic.item(),
            "lr":    scheduler.get_last_lr()[0],
        }, step=epoch)

        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | loss={loss.item():.2e}  "
                  f"L_ode={L_ode.item():.2e}  L_ic={L_ic.item():.2e}")

    return model, history


# ──────────────────────────────────────────────────
# 6. Evaluate and plot
# ──────────────────────────────────────────────────
def evaluate(model):
    t_np   = np.linspace(0, T, 500)
    x_true = analytical(t_np)

    t_tensor = torch.FloatTensor(t_np).unsqueeze(1)
    with torch.no_grad():
        x_pred = model(t_tensor).numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(t_np, x_true, label="Analytical", lw=2)
    ax.plot(t_np, x_pred, "--", label="PINN", lw=2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("x(t) [m]")
    ax.set_title("Damped Oscillator — PINN vs Analytical")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(t_np, np.abs(x_pred - x_true))
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|error| [m]")
    ax.set_title("Pointwise absolute error")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("oscillator_result.png", dpi=150)
    plt.show()
    plt.close()

    max_err = float(np.max(np.abs(x_pred - x_true)))
    print(f"\nMax absolute error: {max_err:.4f} m")
    return max_err


# ──────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    N_COLLOCATION = 400
    N_EPOCHS      = 5000
    LR            = 1e-3
    HIDDEN_LAYERS_OPTIONS = [2,4,6]
    HIDDEN_SIZE   = 32

    torch.manual_seed(42)

    mlflow.set_experiment("pinn-oscillator")
    for HIDDEN_LAYERS in HIDDEN_LAYERS_OPTIONS:
        with mlflow.start_run():
            mlflow.log_params({
                "m": M, "c": C, "k": K, "T": T,
                "n_collocation": N_COLLOCATION,
                "n_epochs":      N_EPOCHS,
                "lr":            LR,
                "hidden_layers": HIDDEN_LAYERS,
                "hidden_size":   HIDDEN_SIZE,
                "optimizer":     "Adam",
                "scheduler":     "StepLR(step=2000, gamma=0.5)",
                # derived physics
                "omega_n": round(omega_n, 4),
                "zeta":    round(zeta, 4),
            })

            model, history = train(N_COLLOCATION, N_EPOCHS, LR)

            max_err = evaluate(model)
            mlflow.log_metric("max_abs_error", max_err)

            mlflow.log_artifact("oscillator_result.png")
            mlflow.pytorch.log_model(model, "model")
