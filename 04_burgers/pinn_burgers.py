"""
Module 04 — Viscous Burgers' Equation
======================================

    u_t + u u_x = nu u_xx,      x in [-1, 1],  t in [0, 1]
    u(x, 0)  = -sin(pi x)
    u(-1, t) = u(1, t) = 0
    nu = 0.01 / pi

The classic PINN benchmark (Raissi et al., 2019). The nonlinear advection term
u u_x steepens the initial sine wave into a near-discontinuity (a viscous shock)
at x = 0 around t ~ 0.3. The shock width scales with nu (roughly 2 nu ~ 0.006),
which is much smaller than the typical spacing of uniformly sampled collocation
points. A PINN trained on uniform points therefore "sees" the shock only through
a handful of residual samples and tends to smear it.

New technique in this module:

    Residual-based Adaptive Refinement (RAR) -- periodically evaluate the PDE
    residual on a large pool of random candidates and add the worst points to
    the collocation set. Points migrate to where the physics is hardest.

Both runs reuse the lessons of module 03: hard-constrained IC/BCs and an
Adam -> L-BFGS schedule. They use the same final number of collocation points,
so the comparison isolates *where* the points are, not *how many*.

    uniform : fixed uniform random collocation set
    rar     : smaller initial set, grown by RAR to the same final size

The reference solution is computed from the exact Cole-Hopf transform with
Gauss-Hermite quadrature (no numerical PDE solver involved).

Run:
    uv run python pinn_burgers.py            # full runs  (~ minutes on CPU)
    uv run python pinn_burgers.py --quick    # smoke test (~ seconds)
    mlflow ui                                # http://localhost:5000
"""

import argparse
import os
from dataclasses import asdict, dataclass

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import mlflow.pytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

MEDIA_DIR = "media"


# ──────────────────────────────────────────────────
# 1. Physical parameters
# ──────────────────────────────────────────────────
NU    = 0.01 / np.pi
X_MIN = -1.0
X_MAX = 1.0
T_MAX = 1.0


# ──────────────────────────────────────────────────
# 2. Reference solution (Cole-Hopf transform)
# ──────────────────────────────────────────────────
def reference_solution(x, t, nu=NU, n_quad=100):
    """
    Exact solution of the viscous Burgers' equation for u(x, 0) = -sin(pi x).

    The Cole-Hopf substitution u = -2 nu phi_x / phi turns Burgers into the
    heat equation phi_t = nu phi_xx, whose solution is a Gaussian convolution.
    Changing variables eta = sqrt(4 nu t) z gives integrals of the form
    int g(z) exp(-z^2) dz, which Gauss-Hermite quadrature handles directly:

        u(x, t) = - sum_i w_i sin(pi y_i) f(y_i) / sum_i w_i f(y_i)
        y_i     = x - sqrt(4 nu t) z_i
        f(y)    = exp(-cos(pi y) / (2 pi nu))

    f spans ~ e^{+-50} for nu = 0.01/pi, so the exponent is shifted by its
    maximum before exponentiating (log-sum-exp trick) to avoid overflow.
    The shift cancels in the ratio.

    Args:
        x, t   : arrays broadcastable to a common shape
        nu     : kinematic viscosity
        n_quad : Gauss-Hermite nodes (100 is converged to ~1e-12 here)
    """
    z, w = np.polynomial.hermite.hermgauss(n_quad)
    x, t = np.broadcast_arrays(np.asarray(x, dtype=float),
                               np.asarray(t, dtype=float))
    xq = x[..., None]
    tq = np.maximum(t, 1e-12)[..., None]

    y     = xq - np.sqrt(4.0 * nu * tq) * z
    log_f = -np.cos(np.pi * y) / (2.0 * np.pi * nu)
    log_f = log_f - log_f.max(axis=-1, keepdims=True)
    f     = w * np.exp(log_f)
    u     = -(np.sin(np.pi * y) * f).sum(axis=-1) / f.sum(axis=-1)

    return np.where(t > 0.0, u, -np.sin(np.pi * x))


# ──────────────────────────────────────────────────
# 3. Network (hard constraints, see module 03)
# ──────────────────────────────────────────────────
class BurgersPINN(nn.Module):
    """
    Hard-constrained PINN:

        u(x, t) = -sin(pi x)  +  t (1 - x^2) net(x_hat, t_hat)

        IC : u(x, 0)   = -sin(pi x)    <- the t factor vanishes at t = 0
        BC : u(+-1, t) = 0             <- (1 - x^2) vanishes, sin(+-pi) = 0

    The inputs are rescaled to [-1, 1] before entering the MLP so that both
    coordinates live on the same scale as the Tanh activations expect.
    """
    def __init__(self, hidden_layers=8, hidden_size=20):
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
        x_hat = 2.0 * (x - X_MIN) / (X_MAX - X_MIN) - 1.0
        t_hat = 2.0 * t / T_MAX - 1.0
        raw   = self.net(torch.cat([x_hat, t_hat], dim=1))
        return -torch.sin(np.pi * x) + t * (1.0 - x ** 2) * raw


# ──────────────────────────────────────────────────
# 4. PDE residual
# ──────────────────────────────────────────────────
def grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def pde_residual(model, x, t):
    """r = u_t + u u_x - nu u_xx"""
    u    = model(x, t)
    u_t  = grad(u, t)
    u_x  = grad(u, x)
    u_xx = grad(u_x, x)
    return u_t + u * u_x - NU * u_xx


def residual_magnitude(model, x, t, batch_size=20_000):
    """|r| at arbitrary points, evaluated in batches without keeping the graph."""
    out = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].clone().requires_grad_(True)
        tb = t[i:i + batch_size].clone().requires_grad_(True)
        out.append(pde_residual(model, xb, tb).detach().abs())
    return torch.cat(out)


# ──────────────────────────────────────────────────
# 5. Collocation sampling and RAR
# ──────────────────────────────────────────────────
def sample_uniform(n):
    x = torch.empty(n, 1).uniform_(X_MIN, X_MAX)
    t = torch.empty(n, 1).uniform_(0.0, T_MAX)
    return x, t


def rar_refine(model, x_f, t_f, n_add, n_candidates):
    """
    Residual-based Adaptive Refinement (greedy variant, Lu et al. 2021).

    1. Draw n_candidates uniform random points.
    2. Evaluate |residual| on all of them.
    3. Append the n_add points with the largest residual to the training set.

    Returns the enlarged collocation set and the mean candidate residual
    before refinement (a cheap, unbiased estimate of the domain-wide residual).
    """
    x_c, t_c = sample_uniform(n_candidates)
    r        = residual_magnitude(model, x_c, t_c).squeeze(1)
    idx      = torch.topk(r, n_add).indices
    x_new    = torch.cat([x_f.detach(), x_c[idx]]).requires_grad_(True)
    t_new    = torch.cat([t_f.detach(), t_c[idx]]).requires_grad_(True)
    return x_new, t_new, r.mean().item()


# ──────────────────────────────────────────────────
# 6. Evaluation grid (shared by training-time validation and final plots)
# ──────────────────────────────────────────────────
class EvalGrid:
    """
    Dense (x, t) grid with the reference solution precomputed once.
    Odd Nx puts a node exactly on x = 0, where the shock forms.
    """
    def __init__(self, nx=257, nt=101):
        self.x = np.linspace(X_MIN, X_MAX, nx)
        self.t = np.linspace(0.0, T_MAX, nt)
        self.XX, self.TT = np.meshgrid(self.x, self.t)
        self.u_ref  = reference_solution(self.XX, self.TT)
        self.x_flat = torch.tensor(self.XX.ravel(), dtype=torch.float32).unsqueeze(1)
        self.t_flat = torch.tensor(self.TT.ravel(), dtype=torch.float32).unsqueeze(1)

    def predict(self, model):
        with torch.no_grad():
            return model(self.x_flat, self.t_flat).numpy().reshape(self.XX.shape)

    def errors(self, model):
        u_pred = self.predict(model)
        diff   = u_pred - self.u_ref
        rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(self.u_ref))
        return rel_l2, float(np.abs(diff).max()), u_pred


# ──────────────────────────────────────────────────
# 7. Training
# ──────────────────────────────────────────────────
@dataclass
class Config:
    variant: str
    n_collocation_final: int = 6000
    n_collocation_init: int = 6000    # < final only for RAR
    rar_every: int = 1000             # Adam steps between refinements
    rar_candidates: int = 50_000
    n_adam: int = 10_000
    n_lbfgs: int = 50                 # outer L-BFGS steps (x max_iter each)
    lbfgs_max_iter: int = 50
    lr: float = 1e-3
    hidden_layers: int = 8
    hidden_size: int = 20
    log_every: int = 250


def train(cfg: Config, grid: EvalGrid):
    model     = BurgersPINN(cfg.hidden_layers, cfg.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_adam, eta_min=cfg.lr * 0.05
    )

    x_f, t_f = sample_uniform(cfg.n_collocation_init)
    x_f.requires_grad_(True)
    t_f.requires_grad_(True)

    use_rar = cfg.n_collocation_final > cfg.n_collocation_init
    if use_rar:
        # Spread the added points evenly over the refinement rounds, stopping
        # before the last rar_every steps so the final set is also trained on.
        n_rounds  = max(1, cfg.n_adam // cfg.rar_every - 1)
        n_per_add = (cfg.n_collocation_final - cfg.n_collocation_init) // n_rounds
        print(f"  RAR: {n_rounds} rounds x {n_per_add} points "
              f"(from {cfg.n_collocation_init} to "
              f"{cfg.n_collocation_init + n_rounds * n_per_add})")

    history = {"step": [], "L_pde": [], "rel_l2": []}

    def log(step, loss, lr):
        rel_l2, max_err, _ = grid.errors(model)
        history["step"].append(step)
        history["L_pde"].append(loss)
        history["rel_l2"].append(rel_l2)
        mlflow.log_metrics({"L_pde": loss, "rel_l2": rel_l2, "max_abs_err": max_err,
                            "lr": lr, "n_collocation": x_f.shape[0]}, step=step)
        return rel_l2

    # — Adam phase —
    for step in range(1, cfg.n_adam + 1):
        optimizer.zero_grad()
        loss = pde_residual(model, x_f, t_f).pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0:
            rel_l2 = log(step, loss.item(), scheduler.get_last_lr()[0])
            if step % (cfg.log_every * 4) == 0:
                print(f"  adam {step:6d} | L_pde={loss.item():.2e}  "
                      f"rel_l2={rel_l2:.2e}  n_f={x_f.shape[0]}")

        if (use_rar and step % cfg.rar_every == 0
                and x_f.shape[0] + n_per_add <= cfg.n_collocation_final):
            x_f, t_f, r_mean = rar_refine(model, x_f, t_f,
                                          n_per_add, cfg.rar_candidates)
            mlflow.log_metric("candidate_mean_residual", r_mean, step=step)

    # — L-BFGS polish phase —
    print(f"  L-BFGS polish ({cfg.n_lbfgs} x {cfg.lbfgs_max_iter} iterations)")
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=cfg.lbfgs_max_iter,
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        loss = pde_residual(model, x_f, t_f).pow(2).mean()
        loss.backward()
        return loss

    for k in range(1, cfg.n_lbfgs + 1):
        loss = lbfgs.step(closure).item()
        step = cfg.n_adam + k * cfg.lbfgs_max_iter
        if not np.isfinite(loss):
            print("  L-BFGS diverged, stopping polish early")
            break
        rel_l2 = log(step, loss, 0.0)
        if k % 10 == 0:
            print(f"  lbfgs {k:5d} | L_pde={loss:.2e}  rel_l2={rel_l2:.2e}")

    return model, history, (x_f.detach().numpy(), t_f.detach().numpy())


# ──────────────────────────────────────────────────
# 8. Plots
# ──────────────────────────────────────────────────
SLICE_TIMES = (0.25, 0.5, 0.75)


def plot_solution(model, grid: EvalGrid, tag: str, colloc=None):
    """Top row: reference / PINN / error maps. Bottom row: time slices."""
    rel_l2, max_err, u_pred = grid.errors(model)
    err    = np.abs(u_pred - grid.u_ref)
    extent = (X_MIN, X_MAX, 0.0, T_MAX)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, data, title, cmap in zip(
        axes[0],
        [grid.u_ref, u_pred, err],
        ["Reference (Cole-Hopf)", f"PINN ({tag})", "Absolute error"],
        ["RdBu_r", "RdBu_r", "hot_r"],
    ):
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        plt.colorbar(im, ax=ax)

    if colloc is not None:
        axes[0, 2].scatter(colloc[0], colloc[1], s=0.3, c="k", alpha=0.3)

    for ax, ts in zip(axes[1], SLICE_TIMES):
        j = int(np.argmin(np.abs(grid.t - ts)))
        ax.plot(grid.x, grid.u_ref[j], "k-", lw=2, label="reference")
        ax.plot(grid.x, u_pred[j], "r--", lw=1.5, label="PINN")
        ax.set_title(f"t = {grid.t[j]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8)

    plt.suptitle(f"Burgers' equation, nu = 0.01/pi -- {tag}  "
                 f"(rel L2 = {rel_l2:.2e}, max err = {max_err:.2e})", fontsize=12)
    plt.tight_layout()
    fname = os.path.join(MEDIA_DIR, f"result_{tag}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return rel_l2, max_err, fname


def plot_residual_and_points(models: dict, colloc: dict, grid: EvalGrid):
    """
    |residual| map for each run with the final collocation points overlaid.
    Shows where each PINN struggles and where RAR decided to put its points.
    """
    x_t = grid.x_flat
    t_t = grid.t_flat
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.5))
    for ax, (tag, model) in zip(np.atleast_1d(axes), models.items()):
        r  = residual_magnitude(model, x_t, t_t).numpy().reshape(grid.XX.shape)
        im = ax.imshow(np.log10(r + 1e-12), origin="lower", aspect="auto",
                       extent=(X_MIN, X_MAX, 0.0, T_MAX), cmap="viridis")
        xs, ts = colloc[tag]
        ax.scatter(xs, ts, s=0.4, c="w", alpha=0.35)
        ax.set_title(f"{tag}: log10 |residual|, {len(xs)} collocation pts")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fname = os.path.join(MEDIA_DIR, "residual_and_points.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_convergence(histories: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for tag, h in histories.items():
        axes[0].semilogy(h["step"], h["L_pde"], label=tag)
        axes[1].semilogy(h["step"], h["rel_l2"], label=tag)
    for ax, title in zip(axes, ["Training loss L_pde", "Relative L2 error vs reference"]):
        ax.set_title(title)
        ax.set_xlabel("iteration (Adam steps, then L-BFGS inner iterations)")
        ax.legend()
    plt.tight_layout()
    fname = os.path.join(MEDIA_DIR, "convergence.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


# ──────────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true",
                        help="tiny budget for a smoke test (results are not meaningful)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(MEDIA_DIR, exist_ok=True)

    budget = dict(n_collocation_final=6000, n_adam=10_000,
                  n_lbfgs=50, lbfgs_max_iter=50, rar_every=1000)
    if args.quick:
        budget = dict(n_collocation_final=600, n_adam=400,
                      n_lbfgs=2, lbfgs_max_iter=10, rar_every=100, log_every=50,
                      rar_candidates=5000)

    configs = [
        Config(variant="uniform", n_collocation_init=budget["n_collocation_final"],
               **budget),
        Config(variant="rar", n_collocation_init=budget["n_collocation_final"] // 2,
               **budget),
    ]

    grid = EvalGrid()
    mlflow.set_experiment("pinn-burgers")

    models, histories, collocs = {}, {}, {}
    for cfg in configs:
        print(f"\n=== Run: {cfg.variant} ===")
        torch.manual_seed(args.seed)   # identical network initialisation for both runs
        with mlflow.start_run(run_name=cfg.variant):
            mlflow.log_params({**asdict(cfg), "nu": NU, "seed": args.seed})
            model, history, colloc = train(cfg, grid)
            rel_l2, max_err, fig_f = plot_solution(model, grid, cfg.variant, colloc)
            mlflow.log_metrics({"final_rel_l2": rel_l2, "final_max_abs_err": max_err})
            mlflow.log_artifact(fig_f)
            if not args.quick:
                mlflow.pytorch.log_model(model, name="model")
            print(f"  [{cfg.variant}] rel_l2={rel_l2:.3e}  max_err={max_err:.3e}")
        models[cfg.variant]    = model
        histories[cfg.variant] = history
        collocs[cfg.variant]   = colloc

    print(f"\nSaved {plot_residual_and_points(models, collocs, grid)}")
    print(f"Saved {plot_convergence(histories)}")
    print("Done. Run `mlflow ui` to compare the runs.")


if __name__ == "__main__":
    main()
