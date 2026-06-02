"""
PINN Inverse Problem — Identifying Damping and Stiffness
=========================================================

We return to the spring-mass-damper from module 01, but with the roles reversed:

    Forward problem (module 01):  given m, c, k  →  predict x(t)
    Inverse problem (module 05):  given noisy x(t) measurements  →  recover c, k

The ODE is still:
    m x''(t) + c x'(t) + k x(t) = 0,  x(0)=1,  x'(0)=0

but now c and k are **trainable parameters** of the PINN rather than constants.
The network is trained simultaneously on:

    L_data  :  match the (noisy) observations
    L_ode   :  satisfy the physics residual everywhere in [0, T]
    L_ic    :  satisfy x(0)=1, x'(0)=0

Because the physics constrains the solution globally, only a handful of
measurements is enough to recover the two unknowns — far fewer than a
purely data-driven regression would need.  This is the core value proposition
of the PINN inverse paradigm.

Three experiments are run head-to-head in MLflow:

    clean_20    :  20 clean (noiseless) measurements
    noisy_20    :  20 measurements with 5% Gaussian noise
    noisy_5     :  5 noisy measurements (sparse regime)

In powertrain terms this maps to:
    - Identifying torsional stiffness and damping from a coast-down vibration log
    - Estimating thermal conductivity from temperature sensors (inverse of module 02)
    - Recovering suspension parameters from road-load acceleration data

Run:
    uv run python pinn_inverse.py
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

# ──────────────────────────────────────────────────
# 1. True physical parameters  (the targets we want to recover)
# ──────────────────────────────────────────────────
M     = 1.0
C_TRUE = 0.4   # damping coefficient — to be identified
K_TRUE = 4.0   # spring stiffness    — to be identified
T     = 10.0
X0    = 1.0
V0    = 0.0

omega_n = np.sqrt(K_TRUE / M)
zeta    = C_TRUE / (2 * np.sqrt(M * K_TRUE))
omega_d = omega_n * np.sqrt(1 - zeta**2)
print(f"True system: m={M}, c={C_TRUE}, k={K_TRUE}")
print(f"  omega_n={omega_n:.3f} rad/s, zeta={zeta:.4f} (underdamped)")


# ──────────────────────────────────────────────────
# 2. Analytical solution  (used only to generate synthetic data)
# ──────────────────────────────────────────────────
def analytical(t_np):
    phi = np.arctan(zeta / np.sqrt(1 - zeta**2))
    return (X0 / np.sqrt(1 - zeta**2)) * np.exp(-zeta * omega_n * t_np) \
           * np.cos(omega_d * t_np - phi)


# ──────────────────────────────────────────────────
# 3. Network architecture
# ──────────────────────────────────────────────────
class InversePINN(nn.Module):
    """
    t → x(t)  with c and k as learnable scalar parameters.

    c and k are stored as raw (unconstrained) parameters; positivity is enforced
    via softplus so the optimiser can freely explore ℝ without sign-flipping.

    softplus(raw) ≈ raw for large raw, ≈ exp(raw) near zero — smooth and
    always positive, unlike a hard clamp that would zero out gradients.
    """
    def __init__(self, hidden_layers=4, hidden_size=32,
                 c_init=1.0, k_init=2.0):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)

        # Raw parameters — not the physical c, k directly
        self._c_raw = nn.Parameter(torch.tensor(np.log(np.expm1(c_init))))
        self._k_raw = nn.Parameter(torch.tensor(np.log(np.expm1(k_init))))

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def c(self):
        return torch.nn.functional.softplus(self._c_raw)

    @property
    def k(self):
        return torch.nn.functional.softplus(self._k_raw)

    def forward(self, t):
        return self.net(t)


# ──────────────────────────────────────────────────
# 4. Loss terms
# ──────────────────────────────────────────────────
def grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def ode_residual(model, t_f):
    """m x'' + c x' + k x = 0   (all three terms contribute)."""
    x    = model(t_f)
    x_t  = grad(x, t_f)
    x_tt = grad(x_t, t_f)
    return M * x_tt + model.c * x_t + model.k * x


def ic_loss(model, t0):
    """x(0) = X0,  x'(0) = V0."""
    x   = model(t0)
    x_t = grad(x, t0)
    return (x - X0) ** 2 + (x_t - V0) ** 2


def data_loss(model, t_data, x_data):
    """Match the (noisy) measurement set."""
    return ((model(t_data) - x_data) ** 2).mean()


# ──────────────────────────────────────────────────
# 5. Training
# ──────────────────────────────────────────────────
def train(t_obs_np, x_obs_np, run_tag,
          n_collocation=2000, n_epochs=20000, lr=1e-3,
          lambda_data=1.0, lambda_ode=1.0, lambda_ic=10.0,
          c_init=1.0, k_init=2.0,
          hidden_layers=4, hidden_size=32):
    """
    Train the inverse PINN and recover c, k from the measurement set.

    lambda weights:
        lambda_data  : how much to trust the data vs. the physics
        lambda_ode   : weight on the interior physics residual
        lambda_ic    : high weight on ICs because there are only 2 IC points
                       but they pin the solution uniquely
    """
    model = InversePINN(hidden_layers, hidden_size, c_init=c_init, k_init=k_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    t_f   = torch.FloatTensor(n_collocation, 1).uniform_(0, T).requires_grad_(True)
    t0    = torch.zeros(1, 1, requires_grad=True)
    t_data = torch.FloatTensor(t_obs_np).unsqueeze(1)
    x_data = torch.FloatTensor(x_obs_np).unsqueeze(1)

    history = {
        "loss": [], "L_data": [], "L_ode": [], "L_ic": [],
        "c_est": [], "k_est": [],
    }

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        L_data = data_loss(model, t_data, x_data)
        L_ode  = ode_residual(model, t_f).pow(2).mean()
        L_ic   = ic_loss(model, t0).mean()
        loss   = lambda_data * L_data + lambda_ode * L_ode + lambda_ic * L_ic

        loss.backward()
        optimizer.step()
        scheduler.step()

        c_est = model.c.item()
        k_est = model.k.item()

        history["loss"].append(loss.item())
        history["L_data"].append(L_data.item())
        history["L_ode"].append(L_ode.item())
        history["L_ic"].append(L_ic.item())
        history["c_est"].append(c_est)
        history["k_est"].append(k_est)

        mlflow.log_metrics({
            "loss": loss.item(), "L_data": L_data.item(),
            "L_ode": L_ode.item(), "L_ic": L_ic.item(),
            "c_est": c_est, "k_est": k_est,
            "c_err_pct": abs(c_est - C_TRUE) / C_TRUE * 100,
            "k_err_pct": abs(k_est - K_TRUE) / K_TRUE * 100,
            "lr": scheduler.get_last_lr()[0],
        }, step=epoch)

        if epoch % 2000 == 0:
            print(f"  epoch {epoch:5d} | loss={loss.item():.2e}  "
                  f"c={c_est:.4f} (true={C_TRUE})  k={k_est:.4f} (true={K_TRUE})")

    return model, history


# ──────────────────────────────────────────────────
# 6. Evaluation and plotting
# ──────────────────────────────────────────────────
def evaluate(model, history, t_obs_np, x_obs_np, run_tag):
    t_plot = np.linspace(0, T, 500)
    x_true = analytical(t_plot)

    t_tensor = torch.FloatTensor(t_plot).unsqueeze(1)
    with torch.no_grad():
        x_pred = model(t_tensor).numpy().ravel()

    c_est = model.c.item()
    k_est = model.k.item()
    c_err = abs(c_est - C_TRUE) / C_TRUE * 100
    k_err = abs(k_est - K_TRUE) / K_TRUE * 100

    # ── Panel 1: trajectory comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(t_plot, x_true,  color="steelblue",  linewidth=2, label="True x(t)")
    ax.plot(t_plot, x_pred,  color="tomato", linewidth=1.5, linestyle="--",
            label=f"PINN  c={c_est:.3f}, k={k_est:.3f}")
    ax.scatter(t_obs_np, x_obs_np, color="black", s=30, zorder=5,
               label=f"Observations (n={len(t_obs_np)})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("x [m]")
    ax.set_title(f"Trajectory — {run_tag}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: parameter convergence ──
    ax = axes[1]
    epochs = range(1, len(history["c_est"]) + 1)
    ax.plot(epochs, history["c_est"], color="tomato",   linewidth=1.5, label="c estimate")
    ax.plot(epochs, history["k_est"], color="seagreen", linewidth=1.5, label="k estimate")
    ax.axhline(C_TRUE, color="tomato",   linestyle=":", linewidth=1, label=f"c true={C_TRUE}")
    ax.axhline(K_TRUE, color="seagreen", linestyle=":", linewidth=1, label=f"k true={K_TRUE}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("parameter value")
    ax.set_title(f"Parameter convergence — {run_tag}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{run_tag} | c: {C_TRUE} → {c_est:.4f} ({c_err:.1f}% err)  |  "
        f"k: {K_TRUE} → {k_est:.4f} ({k_err:.1f}% err)",
        fontsize=11,
    )
    plt.tight_layout()
    fname = f"media/result_{run_tag}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"  [{run_tag}]  c={c_est:.4f} (err={c_err:.2f}%)  "
          f"k={k_est:.4f} (err={k_err:.2f}%)")
    return c_est, k_est, c_err, k_err, fname


def plot_loss_curve(history, run_tag):
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, history["L_data"], label="L_data",  color="black",
                linewidth=2)
    ax.semilogy(epochs, history["L_ode"],  label="L_ode",   color="steelblue")
    ax.semilogy(epochs, history["L_ic"],   label="L_ic",    color="tomato",
                linestyle="--")
    ax.semilogy(epochs, history["loss"],   label="total",   color="gray",
                linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"Loss diagnostics — {run_tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"media/loss_{run_tag}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_summary(results):
    """Bar chart comparing c/k identification error across all experiments."""
    tags   = list(results.keys())
    c_errs = [results[t]["c_err"] for t in tags]
    k_errs = [results[t]["k_err"] for t in tags]

    x = np.arange(len(tags))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, c_errs, w, label="c error %", color="tomato",   alpha=0.8)
    ax.bar(x + w/2, k_errs, w, label="k error %", color="seagreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("identification error [%]")
    ax.set_title("Parameter identification accuracy across experiments")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fname = "media/summary.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ──────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    N_EPOCHS      = 20000
    N_COLLOCATION = 2000
    LR            = 1e-3
    HIDDEN_LAYERS = 4
    HIDDEN_SIZE   = 32

    # Deliberately perturbed initial guesses — the optimiser must find the true values
    C_INIT = 1.0   # 2.5× off from true 0.4
    K_INIT = 2.0   # 2× off from true 4.0

    torch.manual_seed(42)
    np.random.seed(42)
    mlflow.set_experiment("pinn-inverse-problem")

    # ── Pre-compute clean reference trajectory ──
    t_dense = np.linspace(0, T, 1000)
    x_dense = analytical(t_dense)

    # ──────────────────────────────────────────────
    # Experiment A — 20 clean measurements
    # ──────────────────────────────────────────────
    N_OBS  = 20
    t_obs  = np.linspace(0.5, T, N_OBS)   # skip t=0 (that's an IC, not data)
    x_obs  = analytical(t_obs)

    print("\n=== Experiment A: clean_20 ===")
    with mlflow.start_run(run_name="clean_20"):
        mlflow.log_params({
            "n_obs": N_OBS, "noise": 0.0, "c_true": C_TRUE, "k_true": K_TRUE,
            "c_init": C_INIT, "k_init": K_INIT,
            "n_epochs": N_EPOCHS, "n_collocation": N_COLLOCATION,
        })
        model_a, hist_a = train(
            t_obs, x_obs, run_tag="clean_20",
            n_collocation=N_COLLOCATION, n_epochs=N_EPOCHS, lr=LR,
            c_init=C_INIT, k_init=K_INIT,
            hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
        )
        c_a, k_a, ce_a, ke_a, fig_a = evaluate(model_a, hist_a, t_obs, x_obs, "clean_20")
        lf_a = plot_loss_curve(hist_a, "clean_20")
        mlflow.log_metrics({"c_err_pct_final": ce_a, "k_err_pct_final": ke_a,
                             "c_final": c_a, "k_final": k_a})
        for f in (fig_a, lf_a):
            mlflow.log_artifact(f)
        mlflow.pytorch.log_model(model_a, "model")

    # ──────────────────────────────────────────────
    # Experiment B — 20 noisy measurements (5% noise)
    # ──────────────────────────────────────────────
    NOISE = 0.05
    x_obs_noisy = x_obs + NOISE * np.random.randn(N_OBS)

    print("\n=== Experiment B: noisy_20 ===")
    with mlflow.start_run(run_name="noisy_20"):
        mlflow.log_params({
            "n_obs": N_OBS, "noise": NOISE, "c_true": C_TRUE, "k_true": K_TRUE,
            "c_init": C_INIT, "k_init": K_INIT,
            "n_epochs": N_EPOCHS, "n_collocation": N_COLLOCATION,
        })
        model_b, hist_b = train(
            t_obs, x_obs_noisy, run_tag="noisy_20",
            n_collocation=N_COLLOCATION, n_epochs=N_EPOCHS, lr=LR,
            c_init=C_INIT, k_init=K_INIT,
            hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
        )
        c_b, k_b, ce_b, ke_b, fig_b = evaluate(model_b, hist_b, t_obs, x_obs_noisy, "noisy_20")
        lf_b = plot_loss_curve(hist_b, "noisy_20")
        mlflow.log_metrics({"c_err_pct_final": ce_b, "k_err_pct_final": ke_b,
                             "c_final": c_b, "k_final": k_b})
        for f in (fig_b, lf_b):
            mlflow.log_artifact(f)
        mlflow.pytorch.log_model(model_b, "model")

    # ──────────────────────────────────────────────
    # Experiment C — 5 noisy measurements (sparse regime)
    # ──────────────────────────────────────────────
    N_SPARSE = 5
    t_sparse = np.linspace(0.5, T, N_SPARSE)
    x_sparse = analytical(t_sparse) + NOISE * np.random.randn(N_SPARSE)

    print("\n=== Experiment C: noisy_5 ===")
    with mlflow.start_run(run_name="noisy_5"):
        mlflow.log_params({
            "n_obs": N_SPARSE, "noise": NOISE, "c_true": C_TRUE, "k_true": K_TRUE,
            "c_init": C_INIT, "k_init": K_INIT,
            "n_epochs": N_EPOCHS, "n_collocation": N_COLLOCATION,
        })
        model_c, hist_c = train(
            t_sparse, x_sparse, run_tag="noisy_5",
            n_collocation=N_COLLOCATION, n_epochs=N_EPOCHS, lr=LR,
            c_init=C_INIT, k_init=K_INIT,
            hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
        )
        c_c, k_c, ce_c, ke_c, fig_c = evaluate(model_c, hist_c, t_sparse, x_sparse, "noisy_5")
        lf_c = plot_loss_curve(hist_c, "noisy_5")
        mlflow.log_metrics({"c_err_pct_final": ce_c, "k_err_pct_final": ke_c,
                             "c_final": c_c, "k_final": k_c})
        for f in (fig_c, lf_c):
            mlflow.log_artifact(f)
        mlflow.pytorch.log_model(model_c, "model")

    # ──────────────────────────────────────────────
    # Summary comparison
    # ──────────────────────────────────────────────
    results = {
        "clean_20":  {"c_err": ce_a, "k_err": ke_a},
        "noisy_20":  {"c_err": ce_b, "k_err": ke_b},
        "noisy_5":   {"c_err": ce_c, "k_err": ke_c},
    }
    summary_fig = plot_summary(results)

    print("\n─── Final parameter identification summary ───")
    print(f"  {'Experiment':<12}  {'c_est':>8}  {'c_err%':>8}  {'k_est':>8}  {'k_err%':>8}")
    for tag, (c_e, k_e, ce, ke) in zip(
        ["clean_20", "noisy_20", "noisy_5"],
        [(c_a,k_a,ce_a,ke_a),(c_b,k_b,ce_b,ke_b),(c_c,k_c,ce_c,ke_c)],
    ):
        print(f"  {tag:<12}  {c_e:8.4f}  {ce:8.2f}  {k_e:8.4f}  {ke:8.2f}")
    print(f"\nSummary chart: {summary_fig}")
    print("Done. Run `mlflow ui` to compare all three runs.")
