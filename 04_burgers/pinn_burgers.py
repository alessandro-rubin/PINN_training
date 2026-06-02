"""
PINN for the Viscous Burgers' Equation
======================================

Physics:
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²          (1D viscous Burgers)
    u(x, 0)  = −sin(π x)                  (initial condition)
    u(−1, t) = u(1, t) = 0                (Dirichlet boundary conditions)

Domain:  x ∈ [−1, 1],  t ∈ [0, 1],  ν = 0.01/π ≈ 3.18e-3

Burgers' equation is the simplest model that combines **nonlinear advection**
(u ∂u/∂x) with **diffusion** (ν ∂²u/∂x²). It is the canonical testbed for shock
capturing: starting from a smooth sine profile, the negative slope at x=0
steepens until — around t ≈ 0.5 — it forms a near-discontinuity (a shock) whose
width is set by the viscosity ν. Smaller ν → sharper shock → harder for a PINN.

In powertrain terms this is the prototype for any convection-dominated transport
problem: 1D gas dynamics in an intake runner, pressure-wave steepening in an
exhaust manifold, or fuel-film advection — anywhere a quantity is carried along
by its own velocity field and steep fronts emerge.

What's new vs modules 01–03:
    - Nonlinear PDE: the u ∂u/∂x term couples the solution to its own gradient,
      so the residual is no longer linear in u.
    - Shock formation: steep gradients concentrate error; collocation density and
      network capacity matter far more than in the smooth heat equation.
    - No clean analytical solution: we validate against a high-resolution
      finite-difference reference solver (method of lines + stiff integrator)
      instead of a closed form.

Run:
    uv run python pinn_burgers.py
    mlflow ui   # then open http://localhost:5000
"""

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import mlflow.pytorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; no interactive display needed
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────
# 1. Physical parameters
# ──────────────────────────────────────────────────
NU     = 0.01 / np.pi          # viscosity — sets the shock width
X_MIN  = -1.0
X_MAX  =  1.0
T_MAX  =  1.0

print(f"ν={NU:.5e}  |  domain x∈[{X_MIN},{X_MAX}], t∈[0,{T_MAX}]")


# ──────────────────────────────────────────────────
# 2. Reference solution — finite-difference, method of lines
# ──────────────────────────────────────────────────
def reference_solution(nx=1000, nt=200):
    """
    High-resolution numerical reference for validation.

    Solves Burgers in conservative form
        u_t = −∂/∂x(u²/2) + ν u_xx
    on a fine uniform grid using second-order central differences for both the
    flux and the diffusion term, integrated in time with a stiff BDF solver.

    The fine spatial grid resolves the shock; the stiff integrator handles the
    severe time-step restriction that the thin viscous layer would otherwise
    impose. Returns (x_grid, t_grid, U) with U shaped (nt, nx).
    """
    x  = np.linspace(X_MIN, X_MAX, nx)
    dx = x[1] - x[0]

    def rhs(t, u):
        # enforce Dirichlet BCs by holding the endpoints at zero
        u = u.copy()
        u[0] = u[-1] = 0.0
        du = np.zeros_like(u)
        # interior nodes
        flux = 0.5 * u**2
        # central difference for the convective flux  −∂(u²/2)/∂x
        conv = -(flux[2:] - flux[:-2]) / (2 * dx)
        # central difference for the diffusion  ν u_xx
        diff = NU * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        du[1:-1] = conv + diff
        return du

    u0  = -np.sin(np.pi * x)
    t_eval = np.linspace(0, T_MAX, nt)
    sol = solve_ivp(rhs, (0, T_MAX), u0, t_eval=t_eval,
                    method="BDF", rtol=1e-6, atol=1e-8)
    U = sol.y.T            # (nt, nx)
    U[:, 0] = U[:, -1] = 0.0
    return x, t_eval, U


# ──────────────────────────────────────────────────
# 3. Network architecture
# ──────────────────────────────────────────────────
class PINN(nn.Module):
    """(x, t) → u via a fully-connected Tanh network.

    The shock makes this problem far stiffer than the heat equation, so the
    default network is deeper/wider (8 layers × 40) than module 02–03.
    """
    def __init__(self, hidden_layers=8, hidden_size=40):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ──────────────────────────────────────────────────
# 4. Autograd helpers and loss terms
# ──────────────────────────────────────────────────
def grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def pde_residual(model, x_f, t_f):
    """u_t + u u_x − ν u_xx = 0   (the nonlinear convective term is u * u_x)."""
    u    = model(x_f, t_f)
    u_t  = grad(u, t_f)
    u_x  = grad(u, x_f)
    u_xx = grad(u_x, x_f)
    return u_t + u * u_x - NU * u_xx


def ic_loss(model, x_ic, t_ic):
    u_pred  = model(x_ic, t_ic)
    u_exact = -torch.sin(np.pi * x_ic)
    return ((u_pred - u_exact) ** 2).mean()


def bc_loss(model, t_bc):
    x_left  = torch.full_like(t_bc, X_MIN)
    x_right = torch.full_like(t_bc, X_MAX)
    u_left  = model(x_left,  t_bc)
    u_right = model(x_right, t_bc)
    return (u_left ** 2).mean() + (u_right ** 2).mean()


# ──────────────────────────────────────────────────
# 5. Collocation sampling
# ──────────────────────────────────────────────────
def make_collocation(n_collocation, n_ic, n_bc):
    x_f  = torch.FloatTensor(n_collocation, 1).uniform_(X_MIN, X_MAX).requires_grad_(True)
    t_f  = torch.FloatTensor(n_collocation, 1).uniform_(0, T_MAX).requires_grad_(True)
    x_ic = torch.FloatTensor(n_ic, 1).uniform_(X_MIN, X_MAX)
    t_ic = torch.zeros(n_ic, 1)
    t_bc = torch.FloatTensor(n_bc, 1).uniform_(0, T_MAX)
    return x_f, t_f, x_ic, t_ic, t_bc


# ──────────────────────────────────────────────────
# 6. Training — Adam warm-up then L-BFGS polish (Technique 4 from module 03)
# ──────────────────────────────────────────────────
def train(n_collocation=10000, n_ic=400, n_bc=400,
          n_adam=10000, n_lbfgs=500, lr=1e-3,
          lambda_ic=1.0, lambda_bc=1.0,
          hidden_layers=8, hidden_size=40):
    model     = PINN(hidden_layers, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    x_f, t_f, x_ic, t_ic, t_bc = make_collocation(n_collocation, n_ic, n_bc)

    history = {"loss": [], "L_pde": [], "L_ic": [], "L_bc": []}

    def total_loss():
        L_pde = pde_residual(model, x_f, t_f).pow(2).mean()
        L_ic  = ic_loss(model, x_ic, t_ic)
        L_bc  = bc_loss(model, t_bc)
        loss  = L_pde + lambda_ic * L_ic + lambda_bc * L_bc
        return loss, L_pde, L_ic, L_bc

    # — Adam phase —
    for epoch in range(1, n_adam + 1):
        optimizer.zero_grad()
        loss, L_pde, L_ic, L_bc = total_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
        _log_step(history, epoch, loss, L_pde, L_ic, L_bc,
                  scheduler.get_last_lr()[0])

    # — L-BFGS polish phase —
    print(f"\nStarting L-BFGS polish ({n_lbfgs} steps)…")
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe",
    )
    for step in range(1, n_lbfgs + 1):
        def closure():
            lbfgs.zero_grad()
            loss, *_ = total_loss()
            loss.backward()
            return loss
        lbfgs.step(closure)
        loss, L_pde, L_ic, L_bc = total_loss()
        _log_step(history, n_adam + step, loss, L_pde, L_ic, L_bc, lr=0.0)
        if step % 100 == 0:
            print(f"  L-BFGS step {step:4d} | loss={loss.item():.2e}")

    return model, history


def _log_step(history, epoch, loss, L_pde, L_ic, L_bc, lr):
    history["loss"].append(loss.item())
    history["L_pde"].append(L_pde.item())
    history["L_ic"].append(L_ic.item())
    history["L_bc"].append(L_bc.item())
    mlflow.log_metrics({
        "loss":  loss.item(), "L_pde": L_pde.item(),
        "L_ic":  L_ic.item(), "L_bc":  L_bc.item(), "lr": lr,
    }, step=epoch)
    if epoch % 1000 == 0:
        print(f"  epoch {epoch:5d} | loss={loss.item():.2e}  "
              f"L_pde={L_pde.item():.2e}  "
              f"L_ic={L_ic.item():.2e}  L_bc={L_bc.item():.2e}")


# ──────────────────────────────────────────────────
# 7. Evaluation against the finite-difference reference
# ──────────────────────────────────────────────────
def evaluate(model, x_ref, t_ref, U_ref, tag="burgers"):
    XX, TT = np.meshgrid(x_ref, t_ref)
    x_flat = torch.FloatTensor(XX.ravel()).unsqueeze(1)
    t_flat = torch.FloatTensor(TT.ravel()).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).numpy().reshape(U_ref.shape)

    err      = np.abs(u_pred - U_ref)
    max_err  = float(np.max(err))
    mean_err = float(np.mean(err))
    rel_l2   = float(np.linalg.norm(u_pred - U_ref) / np.linalg.norm(U_ref))

    # ── Panel A: space-time colour maps ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title, cmap in zip(
        axes,
        [U_ref, u_pred, err],
        ["Reference (FD)", f"PINN ({tag})", "Absolute error"],
        ["RdBu_r", "RdBu_r", "hot_r"],
    ):
        im = ax.contourf(XX, TT, data, levels=80, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        plt.colorbar(im, ax=ax)
    plt.suptitle(f"Viscous Burgers — {tag}  (rel-L2={rel_l2:.2e})", fontsize=13)
    plt.tight_layout()
    map_f = f"media/result_{tag}.png"
    plt.savefig(map_f, dpi=150)
    plt.close()

    # ── Panel B: solution snapshots showing the shock ──
    fig, ax = plt.subplots(figsize=(7, 5))
    for t_snap in [0.0, 0.25, 0.5, 0.75, 1.0]:
        i = int(round(t_snap / T_MAX * (len(t_ref) - 1)))
        ax.plot(x_ref, U_ref[i], color="lightgray", linewidth=4, zorder=1)
        ax.plot(x_ref, u_pred[i], linewidth=1.5, zorder=2, label=f"t={t_ref[i]:.2f}")
    ax.plot([], [], color="lightgray", linewidth=4, label="reference")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(f"Shock formation — {tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    snap_f = f"media/snapshots_{tag}.png"
    plt.savefig(snap_f, dpi=150)
    plt.close()

    print(f"  [{tag}] max_err={max_err:.4e}  mean_err={mean_err:.4e}  "
          f"rel_L2={rel_l2:.4e}")
    return max_err, mean_err, rel_l2, map_f, snap_f


def plot_loss_curve(history, fname="media/loss_curve.png"):
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, history["L_pde"], label="L_pde", color="steelblue")
    ax.semilogy(epochs, history["L_ic"],  label="L_ic",  color="tomato",
                linestyle="--")
    ax.semilogy(epochs, history["L_bc"],  label="L_bc",  color="seagreen",
                linestyle=":")
    ax.semilogy(epochs, history["loss"],  label="total", color="black",
                linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Burgers PINN — loss components")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ──────────────────────────────────────────────────
# 8. Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    N_COLLOCATION = 10000
    N_IC          = 400
    N_BC          = 400
    N_ADAM        = 10000
    N_LBFGS       = 500
    LR            = 1e-3
    HIDDEN_LAYERS = 8
    HIDDEN_SIZE   = 40

    torch.manual_seed(42)
    np.random.seed(42)

    print("\nComputing finite-difference reference solution…")
    x_ref, t_ref, U_ref = reference_solution()
    print(f"  reference grid: {U_ref.shape[1]} x-points × {U_ref.shape[0]} t-steps")

    mlflow.set_experiment("pinn-burgers")
    with mlflow.start_run(run_name="burgers_adam_lbfgs"):
        mlflow.log_params({
            "nu": NU, "n_collocation": N_COLLOCATION,
            "n_adam": N_ADAM, "n_lbfgs": N_LBFGS,
            "hidden_layers": HIDDEN_LAYERS, "hidden_size": HIDDEN_SIZE,
            "lr": LR,
        })
        model, history = train(
            n_collocation=N_COLLOCATION, n_ic=N_IC, n_bc=N_BC,
            n_adam=N_ADAM, n_lbfgs=N_LBFGS, lr=LR,
            hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
        )
        max_err, mean_err, rel_l2, map_f, snap_f = evaluate(
            model, x_ref, t_ref, U_ref, tag="burgers")
        loss_f = plot_loss_curve(history)

        mlflow.log_metrics({
            "max_abs_error": max_err, "mean_abs_error": mean_err,
            "rel_l2_error": rel_l2,
        })
        for f in (map_f, snap_f, loss_f):
            mlflow.log_artifact(f)
        mlflow.pytorch.log_model(model, "model")

    print(f"\nDone. rel-L2 error = {rel_l2:.4e}")
    print("Run `mlflow ui` to inspect the run.")
