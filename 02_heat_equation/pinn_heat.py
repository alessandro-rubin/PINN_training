"""
PINN for the 1D Heat Equation
==============================

Physics:
    ∂u/∂t = α * ∂²u/∂x²
    u(x, 0) = sin(π x / L)          (initial condition)
    u(0, t) = 0,  u(L, t) = 0       (Dirichlet boundary conditions)

This is the canonical parabolic PDE — it models heat conduction in a rod,
battery cell thermal diffusion, or brake-disc warm-up in powertrain contexts.

Analytical solution (exact, by separation of variables):
    u(x, t) = sin(π x / L) * exp(−α (π/L)² t)

What's new vs module 01:
    - 2D input: the network maps (x, t) → u instead of t → x
    - Boundary conditions: we now enforce u=0 at both ends of the rod
    - Space-time collocation: random (x, t) pairs scatter across the whole domain
    - Three loss terms: PDE residual + IC + two BCs

Run:
    uv run python pinn_heat.py
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
ALPHA = 0.1    # thermal diffusivity [m²/s]  (steel ≈ 1.2e-5, scaled for convenience)
L     = 1.0    # rod length [m]
T     = 2.0    # simulation horizon [s]

# Analytical decay rate — useful to check the solution decays at the right rate
decay_rate = ALPHA * (np.pi / L) ** 2
print(f"System: α={ALPHA}, L={L}, T={T}")
print(f"Analytical decay rate: {decay_rate:.4f} s⁻¹  "
      f"(u halves every {np.log(2)/decay_rate:.2f} s)")


# ──────────────────────────────────────────────────
# 2. Analytical solution (ground truth)
# ──────────────────────────────────────────────────
def analytical(x_np, t_np):
    """Exact solution u(x,t) = sin(πx/L) * exp(−α(π/L)²t)."""
    return np.sin(np.pi * x_np / L) * np.exp(-decay_rate * t_np)


# ──────────────────────────────────────────────────
# 3. Neural network architecture
# ──────────────────────────────────────────────────
class PINN(nn.Module):
    """
    Fully-connected network: (x, t) → u(x, t)

    Input:  2D — spatial coordinate x and time t
    Output: scalar temperature u

    The 2D input is the key structural difference from module 01.
    Everything else (Tanh activations, Xavier init) stays the same.
    """
    def __init__(self, hidden_layers=4, hidden_size=32):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
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

    def forward(self, x, t):
        """Accept x and t as separate tensors, concatenate internally."""
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def grad(y, x):
    """First derivative dy/dx via autograd. x must have requires_grad=True."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


# ──────────────────────────────────────────────────
# 4. Loss functions
# ──────────────────────────────────────────────────
def pde_residual(model, x_f, t_f):
    """
    Evaluate the PDE: ∂u/∂t − α * ∂²u/∂x² = 0

    x_f, t_f: collocation points in the interior, both require_grad=True.
    """
    u     = model(x_f, t_f)
    u_t   = grad(u, t_f)          # ∂u/∂t
    u_x   = grad(u, x_f)          # ∂u/∂x
    u_xx  = grad(u_x, x_f)        # ∂²u/∂x²
    residual = u_t - ALPHA * u_xx
    return residual


def ic_loss(model, x_ic, t_ic):
    """
    Penalize deviation from the initial condition u(x,0) = sin(πx/L).
    """
    u_pred    = model(x_ic, t_ic)
    u_exact   = torch.sin(np.pi * x_ic / L)
    return (u_pred - u_exact) ** 2


def bc_loss(model, t_bc):
    """
    Penalize deviation from Dirichlet BCs: u(0,t) = 0 and u(L,t) = 0.

    t_bc: time samples along the boundary.
    """
    x_left  = torch.zeros_like(t_bc)
    x_right = torch.full_like(t_bc, L)

    u_left  = model(x_left,  t_bc)
    u_right = model(x_right, t_bc)

    return (u_left ** 2).mean() + (u_right ** 2).mean()


# ──────────────────────────────────────────────────
# 5. Training
# ──────────────────────────────────────────────────
def train(n_collocation=2000, n_ic=200, n_bc=200,
          n_epochs=8000, lr=1e-3,
          hidden_layers=4, hidden_size=32,
          lambda_ic=10.0, lambda_bc=10.0):
    """
    Train the PINN.

    lambda_ic / lambda_bc: loss weights for IC and BC terms.
    Upweighting boundary/IC losses is a common trick to enforce constraints
    more strongly than the interior physics residual.
    """
    model     = PINN(hidden_layers=hidden_layers, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

    # Interior collocation points: random (x, t) ∈ [0,L] × [0,T]
    x_f = torch.FloatTensor(n_collocation, 1).uniform_(0, L).requires_grad_(True)
    t_f = torch.FloatTensor(n_collocation, 1).uniform_(0, T).requires_grad_(True)

    # Initial condition points: t=0, x ∈ [0,L]
    x_ic = torch.FloatTensor(n_ic, 1).uniform_(0, L).requires_grad_(False)
    t_ic = torch.zeros(n_ic, 1, requires_grad=False)

    # Boundary condition points: t ∈ [0,T] (used for both x=0 and x=L)
    t_bc = torch.FloatTensor(n_bc, 1).uniform_(0, T).requires_grad_(False)

    history = []

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        L_pde = torch.mean(pde_residual(model, x_f, t_f) ** 2)
        L_ic  = ic_loss(model, x_ic, t_ic).mean()
        L_bc  = bc_loss(model, t_bc)

        loss = L_pde + lambda_ic * L_ic + lambda_bc * L_bc

        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(loss.item())

        mlflow.log_metrics({
            "loss":  loss.item(),
            "L_pde": L_pde.item(),
            "L_ic":  L_ic.item(),
            "L_bc":  L_bc.item(),
            "lr":    scheduler.get_last_lr()[0],
        }, step=epoch)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | loss={loss.item():.2e}  "
                  f"L_pde={L_pde.item():.2e}  L_ic={L_ic.item():.2e}  "
                  f"L_bc={L_bc.item():.2e}")

    return model, history


# ──────────────────────────────────────────────────
# 6. Evaluate and plot
# ──────────────────────────────────────────────────
def evaluate(model):
    Nx, Nt = 100, 100
    x_np = np.linspace(0, L, Nx)
    t_np = np.linspace(0, T, Nt)
    XX, TT = np.meshgrid(x_np, t_np)   # shape (Nt, Nx)

    x_flat = torch.FloatTensor(XX.ravel()).unsqueeze(1)
    t_flat = torch.FloatTensor(TT.ravel()).unsqueeze(1)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat).numpy().reshape(Nt, Nx)

    u_true = analytical(XX, TT)
    err    = np.abs(u_pred - u_true)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Analytical
    im0 = axes[0].contourf(XX, TT, u_true, levels=50, cmap="RdBu_r")
    axes[0].set_title("Analytical solution")
    axes[0].set_xlabel("x [m]");  axes[0].set_ylabel("t [s]")
    plt.colorbar(im0, ax=axes[0])

    # PINN prediction
    im1 = axes[1].contourf(XX, TT, u_pred, levels=50, cmap="RdBu_r")
    axes[1].set_title("PINN prediction")
    axes[1].set_xlabel("x [m]");  axes[1].set_ylabel("t [s]")
    plt.colorbar(im1, ax=axes[1])

    # Absolute error
    im2 = axes[2].contourf(XX, TT, err, levels=50, cmap="hot_r")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x [m]");  axes[2].set_ylabel("t [s]")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("1D Heat Equation — PINN vs Analytical", fontsize=13)
    plt.tight_layout()
    plt.savefig("media/heat_result.png", dpi=150)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    max_err  = float(np.max(err))
    mean_err = float(np.mean(err))
    print(f"\nMax absolute error:  {max_err:.4e}")
    print(f"Mean absolute error: {mean_err:.4e}")
    return max_err, mean_err


# ──────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    N_COLLOCATION = 2000
    N_IC          = 200
    N_BC          = 200
    N_EPOCHS      = 8000
    LR            = 1e-3
    HIDDEN_SIZE   = 32
    HIDDEN_LAYERS_OPTIONS = [2, 4, 6]

    torch.manual_seed(42)

    mlflow.set_experiment("pinn-heat-equation")

    for HIDDEN_LAYERS in HIDDEN_LAYERS_OPTIONS:
        with mlflow.start_run():
            mlflow.log_params({
                "alpha":         ALPHA,
                "L":             L,
                "T":             T,
                "n_collocation": N_COLLOCATION,
                "n_ic":          N_IC,
                "n_bc":          N_BC,
                "n_epochs":      N_EPOCHS,
                "lr":            LR,
                "hidden_layers": HIDDEN_LAYERS,
                "hidden_size":   HIDDEN_SIZE,
                "lambda_ic":     10.0,
                "lambda_bc":     10.0,
                "optimizer":     "Adam",
                "scheduler":     "StepLR(step=3000, gamma=0.5)",
                "decay_rate":    round(decay_rate, 6),
            })

            model, history = train(
                n_collocation=N_COLLOCATION,
                n_ic=N_IC,
                n_bc=N_BC,
                n_epochs=N_EPOCHS,
                lr=LR,
                hidden_layers=HIDDEN_LAYERS,
                hidden_size=HIDDEN_SIZE,
            )

            max_err, mean_err = evaluate(model)
            mlflow.log_metrics({
                "max_abs_error":  max_err,
                "mean_abs_error": mean_err,
            })

            mlflow.log_artifact("media/heat_result.png")
            mlflow.pytorch.log_model(model, "model")
