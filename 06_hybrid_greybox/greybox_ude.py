"""
Hybrid Physics + Data — Grey-box Universal Differential Equations
=================================================================

Every first-principles model is wrong somewhere: a friction law is linearised,
a stiffness is assumed constant, a loss term is ignored. The grey-box idea is to
*keep the physics you trust and learn only the part you don't*.

This module is a deliberate change of paradigm from modules 01–05.

    PINN (modules 01–05):  network IS the solution.   t → x(t),
                           derivatives via autograd, physics in the loss.

    Neural ODE / UDE (06): network is the (unknown part of the) DYNAMICS.
                           dx/dt = f(x);  we integrate f forward with a solver
                           and fit the resulting trajectory to data.

True system — a Duffing oscillator (nonlinear *hardening* spring):

    m x'' + c x' + k x + β x³ = 0,      x(0)=x0,  x'(0)=0

Our engineer's first-principles model knows the linear part (m, c, k are the
true values from a datasheet) but is **missing the β x³ stiffening term** — the
spring gets stiffer as it deflects, and nobody modelled that.

Three models are trained and compared head-to-head:

    physics_only :  m x'' + c x' + k x = 0           (no learning; structurally
                    incomplete — wrong frequency because it ignores stiffening)

    blackbox_node:  x'' = a_NN(x, x')                (pure Neural ODE: learn the
                    whole acceleration; flexible but data-hungry, not
                    interpretable, generalises poorly to new operating points)

    greybox_ude  :  m x'' + c x' + k x = −g_NN(x)    (keep known physics, learn
                    only the residual force; data-efficient, generalises, and
                    g_NN(x) is INTERPRETABLE — it should recover β x³)

Two test scenarios expose the difference:

    1. Training fit            — integrate from the training IC, x0 = 1.2
    2. Generalisation          — integrate from a NEW, larger IC, x0 = 2.0
                                 ("identified on one experiment, deployed at a
                                  different operating point" — the digital-twin
                                  question). The black box has never seen this
                                  region of phase space; the grey box leans on
                                  exact linear physics and degrades gracefully.

Powertrain context: this is the core workflow of a hybrid digital twin —
a calibrated physical model (engine map, driveline torsional model, thermal
network) augmented with a learned residual that mops up the un-modelled
dynamics (friction nonlinearities, blow-by, heat-transfer corrections), while
staying physically interpretable and extrapolating beyond the training cycle.

Run:
    uv run python greybox_ude.py
    mlflow ui   # http://localhost:5000
"""

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import mlflow.pytorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────
# 1. True physical system  (Duffing oscillator)
# ──────────────────────────────────────────────────
M    = 1.0
C    = 0.2     # damping        — known to the engineer
K    = 1.5     # linear stiffness — known to the engineer
BETA = 1.0     # cubic stiffening — UNKNOWN; this is what the residual must learn

H        = 0.1     # integrator step size [s] — coarse enough to train fast,
                   # fine enough for RK4 (≈17–26 steps per oscillation period)
T_TRAIN  = 10.0    # training horizon
T_TEST   = 12.0    # generalisation horizon
X0_TRAIN = 1.2     # training initial displacement
X0_TEST  = 2.0     # generalisation initial displacement (new operating point)
NOISE    = 0.02    # std of additive Gaussian sensor noise on the training data

print(f"True Duffing system: m={M}, c={C}, k={K}, beta={BETA}  "
      f"(missing term: beta * x^3)")


def true_rhs(t, s):
    """Right-hand side of the full Duffing system — used only to synthesise data."""
    x, v = s
    a = (-C * v - K * x - BETA * x**3) / M
    return [v, a]


def reference(s0, T):
    """High-accuracy ground-truth trajectory on the integrator grid."""
    n = round(T / H)
    t_grid = np.arange(n + 1) * H
    sol = solve_ivp(true_rhs, [0.0, T], s0, t_eval=t_grid,
                    method="RK45", rtol=1e-9, atol=1e-9)
    return t_grid, sol.y.T            # states: [n+1, 2]


# ──────────────────────────────────────────────────
# 2. Differentiable RK4 integrator
# ──────────────────────────────────────────────────
# A classic 4th-order Runge–Kutta written in pure torch ops, so gradients flow
# from the trajectory all the way back to the network weights ("discretise then
# optimise"). No torchdiffeq dependency — and seeing the solver explicitly is
# the whole point of this module.
def integrate(deriv, s0, n_steps, h):
    """deriv: s[2] → s'[2].  Returns trajectory [n_steps+1, 2]."""
    s = s0
    traj = [s]
    for _ in range(n_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * h * k1)
        k3 = deriv(s + 0.5 * h * k2)
        k4 = deriv(s + h * k3)
        s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(s)
    return torch.stack(traj)


def rollout(deriv, s0_np, T):
    """Integrate a model under no_grad and return a numpy trajectory."""
    n = round(T / H)
    s0 = torch.tensor(s0_np, dtype=torch.float32)
    with torch.no_grad():
        traj = integrate(deriv, s0, n, H)
    return traj.numpy()


# ──────────────────────────────────────────────────
# 3. The three dynamics models
# ──────────────────────────────────────────────────
def mlp(in_dim, hidden_size=32, hidden_layers=2):
    layers = [nn.Linear(in_dim, hidden_size), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
    layers += [nn.Linear(hidden_size, 1)]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class PhysicsOnly(nn.Module):
    """Naive first-principles model: linear oscillator, no learnable parameters."""
    def deriv(self, s):
        x, v = s[0:1], s[1:2]
        a = (-C * v - K * x) / M
        return torch.cat([v, a])


class BlackBoxNODE(nn.Module):
    """Pure Neural ODE: keep only the kinematic relation x'=v and learn the
    entire acceleration a_NN(x, v) from scratch. No physics whatsoever."""
    def __init__(self, hidden_size=32, hidden_layers=2):
        super().__init__()
        self.net = mlp(2, hidden_size, hidden_layers)

    def deriv(self, s):
        x, v = s[0:1], s[1:2]
        a = self.net(torch.cat([x, v]))
        return torch.cat([v, a])


class GreyBoxUDE(nn.Module):
    """Grey box: keep the trusted linear physics, add a learned residual force
    g_NN(x). The dynamics are

        x'' = (−c x' − k x − g_NN(x)) / m

    so a perfectly trained g_NN(x) recovers the missing term β x³. We hypothesise
    the residual depends on displacement only — a modelling choice that bakes in
    domain knowledge (a position-dependent restoring force) and makes g_NN(x)
    directly comparable to the true β x³."""
    def __init__(self, hidden_size=32, hidden_layers=2):
        super().__init__()
        self.net = mlp(1, hidden_size, hidden_layers)

    def residual(self, x):
        return self.net(x)

    def deriv(self, s):
        x, v = s[0:1], s[1:2]
        a = (-C * v - K * x - self.residual(x)) / M
        return torch.cat([v, a])


# ──────────────────────────────────────────────────
# 4. Training
# ──────────────────────────────────────────────────
def train(model, data_states, s0, n_steps, run_tag,
          n_epochs=2000, lr=1e-2):
    """Fit a model by integrating from s0 and matching the (noisy) trajectory.

    We match BOTH states (x and v): assuming the measured trajectory includes
    velocity (encoder + tacho, or numerically differentiated position)
    stabilises Neural-ODE training considerably. Matching position only is left
    as an experiment.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)

    history = {"loss": []}
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        pred = integrate(model.deriv, s0, n_steps, H)
        loss = ((pred - data_states) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(loss.item())
        mlflow.log_metrics({"loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0]}, step=epoch)

        if epoch % 250 == 0 or epoch == 1:
            print(f"  [{run_tag}] epoch {epoch:5d} | loss={loss.item():.3e}")

    return history


def trajectory_mse(model_deriv, s0_np, T):
    """MSE of a model's rollout against the clean ground-truth trajectory."""
    _, true_states = reference(s0_np, T)
    pred = rollout(model_deriv, s0_np, T)
    return float(np.mean((pred - true_states) ** 2))


# ──────────────────────────────────────────────────
# 5. Plotting
# ──────────────────────────────────────────────────
COLORS = {"physics_only": "darkorange",
          "blackbox_node": "seagreen",
          "greybox_ude": "tomato"}
LABELS = {"physics_only": "physics-only (linear)",
          "blackbox_node": "black-box Neural ODE",
          "greybox_ude": "grey-box UDE"}


def plot_trajectories(models, data_states_np):
    """Two panels: training-window fit, and generalisation to a new IC."""
    t_tr, true_tr = reference([X0_TRAIN, 0.0], T_TRAIN)
    t_te, true_te = reference([X0_TEST, 0.0], T_TEST)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # ── Panel 1: training fit ──
    ax = axes[0]
    ax.plot(t_tr, true_tr[:, 0], color="black", lw=2, label="true x(t)")
    ax.scatter(t_tr, data_states_np[:, 0], s=8, color="gray", alpha=0.5,
               zorder=1, label=f"noisy data (n={len(t_tr)})")
    for tag, m in models.items():
        pred = rollout(m.deriv, [X0_TRAIN, 0.0], T_TRAIN)
        ax.plot(t_tr, pred[:, 0], color=COLORS[tag], lw=1.6, ls="--",
                label=LABELS[tag])
    ax.set_ylabel("x [m]")
    ax.set_title(f"Training window  —  fit from IC x0={X0_TRAIN}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: generalisation to a new operating point ──
    ax = axes[1]
    ax.plot(t_te, true_te[:, 0], color="black", lw=2, label="true x(t)")
    for tag, m in models.items():
        pred = rollout(m.deriv, [X0_TEST, 0.0], T_TEST)
        ax.plot(t_te, pred[:, 0], color=COLORS[tag], lw=1.6, ls="--",
                label=LABELS[tag])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("x [m]")
    ax.set_title(f"Generalisation  —  NEW IC x0={X0_TEST} (never seen in training)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = "media/greybox_trajectories.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_residual(greybox):
    """The interpretability payoff: does g_NN(x) recover β x³?"""
    x = np.linspace(-X0_TEST, X0_TEST, 400)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        g = greybox.residual(x_t).numpy().ravel()
    g_true = BETA * x**3

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, g_true, color="black", lw=2, label=r"true missing force  $\beta x^3$")
    ax.plot(x, g, color="tomato", lw=1.8, ls="--", label=r"learned $g_{NN}(x)$")
    ax.axvspan(-X0_TRAIN, X0_TRAIN, color="steelblue", alpha=0.12,
               label="training x-range")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("residual force")
    ax.set_title("Grey-box recovers the missing physics term")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = "media/greybox_residual.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_phase(models):
    """Phase portrait (x vs v) for the generalisation scenario."""
    _, true_te = reference([X0_TEST, 0.0], T_TEST)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(true_te[:, 0], true_te[:, 1], color="black", lw=2, label="true")
    for tag, m in models.items():
        pred = rollout(m.deriv, [X0_TEST, 0.0], T_TEST)
        ax.plot(pred[:, 0], pred[:, 1], color=COLORS[tag], lw=1.3, ls="--",
                label=LABELS[tag])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("v [m/s]")
    ax.set_title(f"Phase portrait — generalisation (x0={X0_TEST})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = "media/greybox_phase.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_loss(histories):
    fig, ax = plt.subplots(figsize=(7, 4))
    for tag, h in histories.items():
        ax.semilogy(range(1, len(h["loss"]) + 1), h["loss"],
                    color=COLORS[tag], label=LABELS[tag])
    ax.set_xlabel("epoch")
    ax.set_ylabel("trajectory MSE (log scale)")
    ax.set_title("Training loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = "media/greybox_loss.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_summary(results):
    """Bar chart: training-fit MSE vs generalisation MSE for each model."""
    tags = list(results.keys())
    train_mse = [results[t]["train_mse"] for t in tags]
    gen_mse   = [results[t]["gen_mse"] for t in tags]

    x = np.arange(len(tags))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, train_mse, w, label="training-fit MSE", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, gen_mse,   w, label="generalisation MSE", color="tomato", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in tags], fontsize=8)
    ax.set_ylabel("trajectory MSE (log scale)")
    ax.set_title("Fit vs generalisation across models")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fname = "media/greybox_summary.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ──────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    N_EPOCHS = int(os.environ.get("GREYBOX_EPOCHS", "2000"))  # lower for a quick run
    LR       = 1e-2

    os.makedirs("media", exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    mlflow.set_experiment("pinn-hybrid-greybox")

    n_steps_train = round(T_TRAIN / H)

    # ── Synthesise the (noisy) training trajectory ──
    _, true_train = reference([X0_TRAIN, 0.0], T_TRAIN)
    data_np = true_train + NOISE * np.random.randn(*true_train.shape)
    data_states = torch.tensor(data_np, dtype=torch.float32)
    s0_train = torch.tensor([X0_TRAIN, 0.0], dtype=torch.float32)

    models = {}
    histories = {}
    results = {}

    # ──────────────────────────────────────────────
    # Model A — physics only (no training)
    # ──────────────────────────────────────────────
    print("\n=== Model A: physics_only (linear, no learning) ===")
    with mlflow.start_run(run_name="physics_only"):
        mlflow.log_params({"model": "physics_only", "trainable": False,
                           "c": C, "k": K, "knows_beta": False})
        phys = PhysicsOnly()
        models["physics_only"] = phys
        tr_mse  = trajectory_mse(phys.deriv, [X0_TRAIN, 0.0], T_TRAIN)
        gen_mse = trajectory_mse(phys.deriv, [X0_TEST, 0.0], T_TEST)
        results["physics_only"] = {"train_mse": tr_mse, "gen_mse": gen_mse}
        mlflow.log_metrics({"train_mse": tr_mse, "gen_mse": gen_mse})
        print(f"  train_mse={tr_mse:.3e}  gen_mse={gen_mse:.3e}")

    # ──────────────────────────────────────────────
    # Model B — black-box Neural ODE
    # ──────────────────────────────────────────────
    print("\n=== Model B: blackbox_node (learn full acceleration) ===")
    with mlflow.start_run(run_name="blackbox_node"):
        mlflow.log_params({"model": "blackbox_node", "trainable": True,
                           "n_epochs": N_EPOCHS, "lr": LR, "knows_physics": False})
        bb = BlackBoxNODE()
        histories["blackbox_node"] = train(
            bb, data_states, s0_train, n_steps_train, "blackbox_node",
            n_epochs=N_EPOCHS, lr=LR)
        models["blackbox_node"] = bb
        tr_mse  = trajectory_mse(bb.deriv, [X0_TRAIN, 0.0], T_TRAIN)
        gen_mse = trajectory_mse(bb.deriv, [X0_TEST, 0.0], T_TEST)
        results["blackbox_node"] = {"train_mse": tr_mse, "gen_mse": gen_mse}
        mlflow.log_metrics({"train_mse": tr_mse, "gen_mse": gen_mse})
        mlflow.pytorch.log_model(bb, "model")
        print(f"  train_mse={tr_mse:.3e}  gen_mse={gen_mse:.3e}")

    # ──────────────────────────────────────────────
    # Model C — grey-box UDE
    # ──────────────────────────────────────────────
    print("\n=== Model C: greybox_ude (physics + learned residual) ===")
    with mlflow.start_run(run_name="greybox_ude"):
        mlflow.log_params({"model": "greybox_ude", "trainable": True,
                           "n_epochs": N_EPOCHS, "lr": LR, "knows_physics": True,
                           "residual_input": "x"})
        gb = GreyBoxUDE()
        histories["greybox_ude"] = train(
            gb, data_states, s0_train, n_steps_train, "greybox_ude",
            n_epochs=N_EPOCHS, lr=LR)
        models["greybox_ude"] = gb
        tr_mse  = trajectory_mse(gb.deriv, [X0_TRAIN, 0.0], T_TRAIN)
        gen_mse = trajectory_mse(gb.deriv, [X0_TEST, 0.0], T_TEST)
        results["greybox_ude"] = {"train_mse": tr_mse, "gen_mse": gen_mse}
        mlflow.log_metrics({"train_mse": tr_mse, "gen_mse": gen_mse})
        mlflow.pytorch.log_model(gb, "model")
        print(f"  train_mse={tr_mse:.3e}  gen_mse={gen_mse:.3e}")

    # ──────────────────────────────────────────────
    # Figures  (collected on a dedicated comparison run)
    # ──────────────────────────────────────────────
    print("\n=== Generating figures ===")
    f_traj  = plot_trajectories(models, data_np)
    f_res   = plot_residual(gb)
    f_phase = plot_phase(models)
    f_loss  = plot_loss(histories)
    f_sum   = plot_summary(results)
    figs = [f_traj, f_res, f_phase, f_loss, f_sum]
    for f in figs:
        print(f"  wrote {f}")

    with mlflow.start_run(run_name="comparison"):
        mlflow.log_metrics({
            f"{tag}_{k}": results[tag][k]
            for tag in results for k in ("train_mse", "gen_mse")
        })
        for f in figs:
            mlflow.log_artifact(f)

    # ──────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────
    print("\n--- Fit vs generalisation summary (trajectory MSE) ---")
    print(f"  {'model':<16}  {'train_mse':>12}  {'gen_mse':>12}")
    for tag in ("physics_only", "blackbox_node", "greybox_ude"):
        r = results[tag]
        print(f"  {tag:<16}  {r['train_mse']:12.3e}  {r['gen_mse']:12.3e}")
    print("\nDone. Run `mlflow ui` to compare all three runs.")
