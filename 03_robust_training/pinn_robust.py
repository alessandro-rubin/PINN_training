"""
Module 03 — Robust PINN Training
==================================

Same physics as module 02 (1D heat equation, ∂u/∂t = α ∂²u/∂x²).
Focus: four techniques that turn a working PINN into a reliable one.

    Technique 1 — Loss diagnostics   : plot each loss term separately
    Technique 2 — Adaptive weights   : balance loss terms automatically
    Technique 3 — Hard constraints   : encode BCs and IC into the architecture
    Technique 4 — L-BFGS polish      : second-order optimiser after Adam

Three MLflow runs compare the approaches head-to-head:

    soft_fixed    : module 02 baseline (soft BCs/IC, fixed λ=10)
    soft_adaptive : soft constraints + adaptive λ
    hard_lbfgs    : hard constraints (BCs + IC) + Adam + L-BFGS

Run:
    uv run python pinn_robust.py
    mlflow ui   # http://localhost:5000
"""

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import mlflow.pytorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────
# 1. Physical parameters  (identical to module 02)
# ──────────────────────────────────────────────────
ALPHA = 0.1
L     = 1.0
T     = 2.0

decay_rate = ALPHA * (np.pi / L) ** 2
print(f"α={ALPHA}, L={L}, T={T}  |  decay rate={decay_rate:.4f} s⁻¹")


# ──────────────────────────────────────────────────
# 2. Analytical solution
# ──────────────────────────────────────────────────
def analytical(x_np, t_np):
    return np.sin(np.pi * x_np / L) * np.exp(-decay_rate * t_np)


# ──────────────────────────────────────────────────
# 3. Network architectures
# ──────────────────────────────────────────────────
class _Base(nn.Module):
    """Shared backbone: (x, t) → scalar via fully-connected + Tanh."""
    def __init__(self, hidden_layers=4, hidden_size=32):
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


class PINNSoft(_Base):
    """
    Module 02 style: raw network output, constraints enforced by loss penalties.
    """
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


class PINNHard(_Base):
    """
    Hard constraint output transformation.

    The network output is lifted to:
        u(x, t) = sin(πx/L)  +  t · x · (L − x) · net(x, t)

    This satisfies, for free, regardless of network weights:
        IC :  u(x, 0) = sin(πx/L)           ← the t factor is zero
        BC0:  u(0, t) = 0                    ← the x factor is zero
        BCL:  u(L, t) = sin(π) + ... = 0     ← (L−x) factor is zero at x=L,
                                                 and sin(π)=0

    The PDE residual is the only remaining loss term, which simplifies training
    and removes the need for λ weights entirely.
    """
    def forward(self, x, t):
        raw = self.net(torch.cat([x, t], dim=1))
        base = torch.sin(np.pi * x / L)          # satisfies IC
        correction = t * x * (L - x) * raw        # zero at t=0, x=0, x=L
        return base + correction


# ──────────────────────────────────────────────────
# 4. Autograd helpers and loss functions
# ──────────────────────────────────────────────────
def grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def pde_residual(model, x_f, t_f):
    """∂u/∂t − α ∂²u/∂x² = 0"""
    u    = model(x_f, t_f)
    u_t  = grad(u, t_f)
    u_x  = grad(u, x_f)
    u_xx = grad(u_x, x_f)
    return u_t - ALPHA * u_xx


def ic_loss(model, x_ic, t_ic):
    u_pred  = model(x_ic, t_ic)
    u_exact = torch.sin(np.pi * x_ic / L)
    return (u_pred - u_exact) ** 2


def bc_loss(model, t_bc):
    x_left  = torch.zeros_like(t_bc)
    x_right = torch.full_like(t_bc, L)
    u_left  = model(x_left,  t_bc)
    u_right = model(x_right, t_bc)
    return (u_left ** 2).mean() + (u_right ** 2).mean()


# ──────────────────────────────────────────────────
# 5. Adaptive weight helper  (Technique 2)
# ──────────────────────────────────────────────────
class AdaptiveWeights:
    """
    Tracks an exponential moving average of each loss component and returns
    weights inversely proportional to their magnitude.

    Effect: if L_pde is 100× larger than L_ic, its weight is scaled down
    proportionally, preventing it from drowning out the smaller terms.
    Weights are normalised so their mean stays at 1 (total loss scale preserved).

    Args:
        n          : number of loss terms
        ema_alpha  : smoothing factor (higher = slower adaptation)
    """
    def __init__(self, n: int, ema_alpha: float = 0.98):
        self.ema      = torch.ones(n)
        self.alpha    = ema_alpha

    def __call__(self, losses: list) -> torch.Tensor:
        vals     = torch.stack([l.detach() for l in losses])
        self.ema = self.alpha * self.ema + (1.0 - self.alpha) * vals
        weights  = 1.0 / (self.ema + 1e-8)
        return weights / weights.mean()   # mean = 1 → scale-preserving


# ──────────────────────────────────────────────────
# 6. Training functions
# ──────────────────────────────────────────────────
def _make_collocation(n_collocation, n_ic, n_bc):
    """Sample all collocation point sets (shared across training functions)."""
    x_f  = torch.FloatTensor(n_collocation, 1).uniform_(0, L).requires_grad_(True)
    t_f  = torch.FloatTensor(n_collocation, 1).uniform_(0, T).requires_grad_(True)
    x_ic = torch.FloatTensor(n_ic, 1).uniform_(0, L)
    t_ic = torch.zeros(n_ic, 1)
    t_bc = torch.FloatTensor(n_bc, 1).uniform_(0, T)
    return x_f, t_f, x_ic, t_ic, t_bc


def train_soft_fixed(n_collocation=2000, n_ic=200, n_bc=200,
                     n_epochs=8000, lr=1e-3,
                     hidden_layers=4, hidden_size=32,
                     lambda_ic=10.0, lambda_bc=10.0):
    """
    Baseline: module 02 approach.
    Soft constraints with fixed loss weights.
    """
    model     = PINNSoft(hidden_layers, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    x_f, t_f, x_ic, t_ic, t_bc = _make_collocation(n_collocation, n_ic, n_bc)

    history = {"loss": [], "L_pde": [], "L_ic": [], "L_bc": []}

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        L_pde = pde_residual(model, x_f, t_f).pow(2).mean()
        L_ic  = ic_loss(model, x_ic, t_ic).mean()
        L_bc  = bc_loss(model, t_bc)
        loss  = L_pde + lambda_ic * L_ic + lambda_bc * L_bc

        loss.backward()
        optimizer.step()
        scheduler.step()

        _log_step(history, epoch, loss, L_pde, L_ic, L_bc,
                  scheduler.get_last_lr()[0])
        for k, v in history.items():
            v.append(locals()[k].item() if k != "loss" else loss.item())

    return model, history


def train_soft_adaptive(n_collocation=2000, n_ic=200, n_bc=200,
                        n_epochs=8000, lr=1e-3,
                        hidden_layers=4, hidden_size=32):
    """
    Adaptive weights: loss terms are balanced automatically each step.
    No manual λ tuning required.
    """
    model     = PINNSoft(hidden_layers, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    x_f, t_f, x_ic, t_ic, t_bc = _make_collocation(n_collocation, n_ic, n_bc)
    aw = AdaptiveWeights(n=3)

    history = {"loss": [], "L_pde": [], "L_ic": [], "L_bc": []}

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        L_pde = pde_residual(model, x_f, t_f).pow(2).mean()
        L_ic  = ic_loss(model, x_ic, t_ic).mean()
        L_bc  = bc_loss(model, t_bc)

        w    = aw([L_pde, L_ic, L_bc])
        loss = w[0] * L_pde + w[1] * L_ic + w[2] * L_bc

        loss.backward()
        optimizer.step()
        scheduler.step()

        _log_step(history, epoch, loss, L_pde, L_ic, L_bc,
                  scheduler.get_last_lr()[0],
                  extra={"w_pde": w[0].item(), "w_ic": w[1].item(),
                         "w_bc": w[2].item()})
        for k, v in history.items():
            v.append(locals()[k].item() if k != "loss" else loss.item())

    return model, history


def train_hard_lbfgs(n_collocation=2000, n_ic=200, n_bc=200,
                     n_adam=6000, n_lbfgs=200,
                     lr=1e-3,
                     hidden_layers=4, hidden_size=32):
    """
    Hard constraints + L-BFGS polish.

    Because BCs and IC are satisfied exactly by the PINNHard architecture,
    the only loss term is the PDE residual — no λ tuning at all.
    Adam trains for n_adam steps, then L-BFGS polishes for n_lbfgs steps.
    """
    model     = PINNHard(hidden_layers, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    x_f, t_f, *_ = _make_collocation(n_collocation, n_ic, n_bc)

    history = {"loss": [], "L_pde": [], "L_ic": [], "L_bc": []}

    # — Adam phase —
    for epoch in range(1, n_adam + 1):
        optimizer.zero_grad()
        L_pde = pde_residual(model, x_f, t_f).pow(2).mean()
        L_pde.backward()
        optimizer.step()
        scheduler.step()

        _log_step(history, epoch, L_pde, L_pde,
                  torch.tensor(0.0), torch.tensor(0.0),
                  scheduler.get_last_lr()[0])
        for k, v in history.items():
            v.append(L_pde.item() if k in ("loss", "L_pde") else 0.0)

    # — L-BFGS polish phase  (Technique 4) —
    print(f"\nStarting L-BFGS polish ({n_lbfgs} steps)…")
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    for step in range(1, n_lbfgs + 1):
        def closure():
            lbfgs.zero_grad()
            loss = pde_residual(model, x_f, t_f).pow(2).mean()
            loss.backward()
            return loss

        loss = lbfgs.step(closure)
        epoch = n_adam + step

        _log_step(history, epoch, loss, loss,
                  torch.tensor(0.0), torch.tensor(0.0), lr=0.0)
        for k, v in history.items():
            v.append(loss.item() if k in ("loss", "L_pde") else 0.0)

        if step % 50 == 0:
            print(f"  L-BFGS step {step:4d} | L_pde={loss.item():.2e}")

    return model, history


# ──────────────────────────────────────────────────
# 7. Shared logging helper
# ──────────────────────────────────────────────────
def _log_step(history, epoch, loss, L_pde, L_ic, L_bc, lr, extra=None):
    metrics = {
        "loss":  loss.item(),
        "L_pde": L_pde.item(),
        "L_ic":  L_ic.item(),
        "L_bc":  L_bc.item(),
        "lr":    lr,
    }
    if extra:
        metrics.update(extra)
    mlflow.log_metrics(metrics, step=epoch)
    if epoch % 1000 == 0:
        print(f"  epoch {epoch:5d} | loss={loss.item():.2e}  "
              f"L_pde={L_pde.item():.2e}  "
              f"L_ic={L_ic.item():.2e}  L_bc={L_bc.item():.2e}")


# ──────────────────────────────────────────────────
# 8. Evaluation
# ──────────────────────────────────────────────────
def evaluate(model, tag: str):
    Nx, Nt = 100, 100
    x_np   = np.linspace(0, L, Nx)
    t_np   = np.linspace(0, T, Nt)
    XX, TT = np.meshgrid(x_np, t_np)

    x_flat = torch.FloatTensor(XX.ravel()).unsqueeze(1)
    t_flat = torch.FloatTensor(TT.ravel()).unsqueeze(1)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat).numpy().reshape(Nt, Nx)

    u_true = analytical(XX, TT)
    err    = np.abs(u_pred - u_true)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title, cmap in zip(
        axes,
        [u_true, u_pred, err],
        ["Analytical", f"PINN ({tag})", "Absolute error"],
        ["RdBu_r", "RdBu_r", "hot_r"],
    ):
        im = ax.contourf(XX, TT, data, levels=50, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("t [s]")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"1D Heat Equation — {tag}", fontsize=13)
    plt.tight_layout()
    fname = f"result_{tag}.png"
    plt.savefig(fname, dpi=150)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    max_err  = float(np.max(err))
    mean_err = float(np.mean(err))
    print(f"  [{tag}] max_err={max_err:.4e}  mean_err={mean_err:.4e}")
    return max_err, mean_err, fname


def plot_loss_curves(histories: dict, fname="loss_curves.png"):
    """
    Technique 1 — Loss diagnostics.

    Plot L_pde, L_ic, L_bc for each run on a log scale.
    This is the primary tool for diagnosing training failures:
    a term that stops decreasing early signals the corresponding λ is too low.
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4),
                             sharey=False)
    if len(histories) == 1:
        axes = [axes]

    for ax, (tag, history) in zip(axes, histories.items()):
        epochs = range(1, len(history["loss"]) + 1)
        ax.semilogy(epochs, history["L_pde"], label="L_pde", color="steelblue")
        ax.semilogy(epochs, history["L_ic"],  label="L_ic",  color="tomato",
                    linestyle="--")
        ax.semilogy(epochs, history["L_bc"],  label="L_bc",  color="seagreen",
                    linestyle=":")
        ax.semilogy(epochs, history["loss"],  label="total", color="black",
                    linewidth=1.5)
        ax.set_title(tag)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (log scale)")
        ax.legend(fontsize=8)

    plt.suptitle("Loss component diagnostics", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    return fname


# ──────────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    N_COLLOCATION  = 2000
    N_IC           = 200
    N_BC           = 200
    N_EPOCHS       = 8000   # Adam steps for soft variants
    N_ADAM         = 6000   # Adam steps for hard variant
    N_LBFGS        = 200    # L-BFGS polish steps
    LR             = 1e-3
    HIDDEN_LAYERS  = 4
    HIDDEN_SIZE    = 32

    torch.manual_seed(42)
    mlflow.set_experiment("pinn-robust-training")

    histories   = {}
    run_results = {}

    # ── Run A: soft constraints, fixed λ (module 02 baseline) ──
    print("\n=== Run A: soft_fixed ===")
    with mlflow.start_run(run_name="soft_fixed"):
        mlflow.log_params({
            "variant": "soft_fixed", "lambda_ic": 10.0, "lambda_bc": 10.0,
            "n_epochs": N_EPOCHS, "hidden_layers": HIDDEN_LAYERS,
        })
        model_a, hist_a = train_soft_fixed(
            n_collocation=N_COLLOCATION, n_ic=N_IC, n_bc=N_BC,
            n_epochs=N_EPOCHS, lr=LR,
            hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
        )
        max_err, mean_err, fig_f = evaluate(model_a, "soft_fixed")
        mlflow.log_metrics({"max_abs_error": max_err, "mean_abs_error": mean_err})
        mlflow.log_artifact(fig_f)
        mlflow.pytorch.log_model(model_a, "model")
    histories["soft_fixed"] = hist_a

    # ── Run B: soft constraints, adaptive λ ──
    print("\n=== Run B: soft_adaptive ===")
    with mlflow.start_run(run_name="soft_adaptive"):
        mlflow.log_params({
            "variant": "soft_adaptive", "lambda": "adaptive",
            "n_epochs": N_EPOCHS, "hidden_layers": HIDDEN_LAYERS,
        })
        model_b, hist_b = train_soft_adaptive(
            n_collocation=N_COLLOCATION, n_ic=N_IC, n_bc=N_BC,
            n_epochs=N_EPOCHS, lr=LR,
            hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
        )
        max_err, mean_err, fig_f = evaluate(model_b, "soft_adaptive")
        mlflow.log_metrics({"max_abs_error": max_err, "mean_abs_error": mean_err})
        mlflow.log_artifact(fig_f)
        mlflow.pytorch.log_model(model_b, "model")
    histories["soft_adaptive"] = hist_b

    # ── Run C: hard constraints + L-BFGS ──
    print("\n=== Run C: hard_lbfgs ===")
    with mlflow.start_run(run_name="hard_lbfgs"):
        mlflow.log_params({
            "variant": "hard_lbfgs", "lambda": "none (hard constraints)",
            "n_adam": N_ADAM, "n_lbfgs": N_LBFGS,
            "hidden_layers": HIDDEN_LAYERS,
        })
        model_c, hist_c = train_hard_lbfgs(
            n_collocation=N_COLLOCATION, n_ic=N_IC, n_bc=N_BC,
            n_adam=N_ADAM, n_lbfgs=N_LBFGS, lr=LR,
            hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
        )
        max_err, mean_err, fig_f = evaluate(model_c, "hard_lbfgs")
        mlflow.log_metrics({"max_abs_error": max_err, "mean_abs_error": mean_err})
        mlflow.log_artifact(fig_f)
        mlflow.pytorch.log_model(model_c, "model")
    histories["hard_lbfgs"] = hist_c

    # ── Loss diagnostic plot (all three runs) ──
    loss_fig = plot_loss_curves(histories)
    print(f"\nLoss curves saved to {loss_fig}")
    print("Done. Run `mlflow ui` to compare the three runs.")
