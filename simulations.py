"""
Simulations for: When Does OLS Break?
Eigenvalue Spectra of Random Design Matrices and Their Impact on Linear Model Inference

Generates all figures for the coursework.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

SEED = 42
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Marchenko-Pastur density
# ---------------------------------------------------------------------------
def mp_density(lam, gamma, sigma2=1.0):
    """Evaluate the MP density at points lam for ratio gamma."""
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    density = np.zeros_like(lam, dtype=float)
    mask = (lam >= lam_minus) & (lam <= lam_plus) & (lam > 0)
    density[mask] = (
        np.sqrt((lam_plus - lam[mask]) * (lam[mask] - lam_minus))
        / (2 * np.pi * gamma * sigma2 * lam[mask])
    )
    return density


def mp_support(gamma, sigma2=1.0):
    """Return (lambda_minus, lambda_plus) for the MP law."""
    return sigma2 * (1 - np.sqrt(gamma)) ** 2, sigma2 * (1 + np.sqrt(gamma)) ** 2


# ===================================================================
# FIGURE 1: Eigenvalue histograms vs MP density for four gamma values
# ===================================================================
def figure1_eigenvalue_histograms():
    print("Generating Figure 1: Eigenvalue histograms vs MP density...")
    gammas = [0.2, 0.5, 0.8, 0.95]
    n = 2000

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, gamma in enumerate(gammas):
        p = int(gamma * n)
        X = rng.standard_normal((n, p))
        S = X.T @ X / n
        eigenvalues = np.linalg.eigvalsh(S)

        lm, lp = mp_support(gamma)
        lam_grid = np.linspace(max(0, lm - 0.1), lp + 0.3, 500)
        density = mp_density(lam_grid, gamma)

        ax = axes[idx]
        ax.hist(eigenvalues, bins=80, density=True, alpha=0.6,
                color="#85B7EB", edgecolor="none", label="Empirical")
        ax.plot(lam_grid, density, color="#185FA5", linewidth=2,
                label="MP density")
        ax.axvline(lm, color="#993C1D", linewidth=1, linestyle="--", alpha=0.7,
                   label=rf"$\lambda_- = {lm:.3f}$")
        ax.axvline(lp, color="#993C1D", linewidth=1, linestyle="--", alpha=0.7,
                   label=rf"$\lambda_+ = {lp:.3f}$")
        ax.set_title(rf"$\gamma = {gamma}$  ($n={n},\, p={p}$)")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(left=0)

    fig.suptitle(
        r"Eigenvalue distribution of $\frac{1}{n}X^\top X$ vs. Marchenko–Pastur density",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig("fig1_mp_histograms.png")
    plt.close(fig)
    print("Saved fig1_mp_histograms.png")


# ===================================================================
# FIGURE 2: Universality check — Gaussian vs Uniform vs Rademacher
# ===================================================================
def figure2_universality():
    print("Generating Figure 2: Universality check...")
    n, gamma = 2000, 0.5
    p = int(gamma * n)

    fig, ax = plt.subplots(figsize=(8, 5))

    distributions = {
        r"Gaussian $\mathcal{N}(0,1)$": lambda: rng.standard_normal((n, p)),
        r"Uniform $(-\sqrt{3}, \sqrt{3})$": lambda: rng.uniform(
            -np.sqrt(3), np.sqrt(3), (n, p)
        ),
        "Rademacher (±1)": lambda: rng.choice([-1.0, 1.0], size=(n, p)),
    }
    colors = ["#85B7EB", "#5DCAA5", "#F0997B"]

    for (label, gen_fn), col in zip(distributions.items(), colors):
        X = gen_fn()
        S = X.T @ X / n
        eigs = np.linalg.eigvalsh(S)
        ax.hist(eigs, bins=80, density=True, alpha=0.45, color=col,
                edgecolor="none", label=label)

    lm, lp = mp_support(gamma)
    lam_grid = np.linspace(max(0, lm - 0.05), lp + 0.2, 500)
    ax.plot(lam_grid, mp_density(lam_grid, gamma), color="#26215C",
            linewidth=2.5, label="MP density")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Density")
    ax.set_title(
        rf"Universality of the MP law at $\gamma = {gamma}$  —  three entry distributions"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig2_universality.png")
    plt.close(fig)
    print("Saved fig2_universality.png")


# ===================================================================
# FIGURE 3: Total variance / MSE of β̂ vs gamma
# ===================================================================
def figure3_mse_vs_gamma():
    print("Generating Figure 3: MSE of β̂ vs γ...")
    n = 500
    sigma2 = 1.0
    n_reps = 200
    gammas = np.arange(0.05, 0.96, 0.05)
    empirical_mse = []
    theoretical_mse = []

    for gamma in gammas:
        p = int(gamma * n)
        beta_true = np.zeros(p)
        beta_true[0] = 1.0  # sparse true signal
        mse_vals = []

        for _ in range(n_reps):
            X = rng.standard_normal((n, p))
            eps = rng.standard_normal(n) * np.sqrt(sigma2)
            Y = X @ beta_true + eps
            beta_hat = np.linalg.solve(X.T @ X, X.T @ Y)
            mse_vals.append(np.mean((beta_hat - beta_true) ** 2))

        empirical_mse.append(np.mean(mse_vals))
        # Theoretical: trace(cov(β̂))/p = σ² γ / (n(1-γ))  ... per component = σ²/(n(1-γ))
        theoretical_mse.append(sigma2 / (n * (1 - gamma)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, empirical_mse, "o", color="#534AB7", markersize=5,
            label=r"Empirical MSE  $\frac{1}{p}\|\hat{\beta}-\beta\|^2$")
    ax.plot(gammas, theoretical_mse, "-", color="#D85A30", linewidth=2,
            label=r"MP prediction  $\sigma^2 / [n(1-\gamma)]$")
    ax.set_xlabel(r"$\gamma = p/n$")
    ax.set_ylabel("Average MSE per component")
    ax.set_title(
        r"Mean squared error of $\hat{\beta}$ diverges as $\gamma \to 1$"
    )
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("fig3_mse_vs_gamma.png")
    plt.close(fig)
    print("Saved fig3_mse_vs_gamma.png")


# ===================================================================
# FIGURE 4: Confidence ellipsoid shape — condition number & max axis
# ===================================================================
def figure4_ellipsoid_volume():
    print("Generating Figure 4: Ellipsoid shape vs γ...")
    from scipy.stats import f as f_dist

    n = 500
    sigma2 = 1.0
    n_reps = 100
    gammas = np.arange(0.05, 0.96, 0.05)

    median_cond = []
    median_max_axis = []
    theoretical_max_axis = []

    for gamma in gammas:
        p = int(gamma * n)
        conds = []
        max_axes = []
        F_crit = f_dist.ppf(0.95, p, n - p)

        for _ in range(n_reps):
            X = rng.standard_normal((n, p))
            eps = rng.standard_normal(n) * np.sqrt(sigma2)
            Y = X @ np.zeros(p) + eps

            XtX = X.T @ X
            eigs = np.linalg.eigvalsh(XtX)

            conds.append(eigs[-1] / eigs[0])

            resid = Y - X @ np.linalg.solve(XtX, X.T @ Y)
            RSS = resid @ resid
            s2 = RSS / (n - p)
            # Max half-axis of the ellipsoid ∝ sqrt(p * s2 * F_crit / lambda_min)
            max_axis = np.sqrt(p * s2 * F_crit / eigs[0])
            max_axes.append(max_axis)

        median_cond.append(np.median(conds))
        median_max_axis.append(np.median(max_axes))
        # Theoretical: lambda_min ≈ n*(1-sqrt(gamma))^2, s2 ≈ sigma2
        lam_min_theory = n * (1 - np.sqrt(gamma)) ** 2
        theoretical_max_axis.append(
            np.sqrt(p * sigma2 * F_crit / max(lam_min_theory, 1e-10))
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: condition number
    ax1.semilogy(gammas, median_cond, "s-", color="#534AB7", markersize=5,
                 linewidth=1.5, label="Empirical (median)")
    # Theoretical: lambda_max/lambda_min = (1+sqrt(gamma))^2 / (1-sqrt(gamma))^2
    theory_cond = [(1 + np.sqrt(g)) ** 2 / (1 - np.sqrt(g)) ** 2 for g in gammas]
    ax1.semilogy(gammas, theory_cond, "--", color="#D85A30", linewidth=2,
                 label=r"MP prediction $\lambda_+/\lambda_-$")
    ax1.set_xlabel(r"$\gamma = p/n$")
    ax1.set_ylabel("Condition number (log scale)")
    ax1.set_title(r"Condition number of $X^\top X$")
    ax1.legend()

    # Right: max half-axis of confidence ellipsoid
    ax2.semilogy(gammas, median_max_axis, "o-", color="#0F6E56", markersize=5,
                 linewidth=1.5, label="Empirical (median)")
    ax2.semilogy(gammas, theoretical_max_axis, "--", color="#D85A30", linewidth=2,
                 label="MP-based prediction")
    ax2.set_xlabel(r"$\gamma = p/n$")
    ax2.set_ylabel("Max half-axis length (log scale)")
    ax2.set_title("Longest axis of 95% confidence ellipsoid")
    ax2.legend()

    fig.suptitle(
        r"Confidence ellipsoid degrades as $\gamma \to 1$: ill-conditioning of $X^\top X$",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("fig4_ellipsoid_shape.png")
    plt.close(fig)
    print("Saved fig4_ellipsoid_shape.png")


# ===================================================================
# FIGURE 5: Leverage distribution for various gamma
# ===================================================================
def figure5_leverage_distribution():
    print("Generating Figure 5: Leverage distributions...")
    n = 1000
    gammas = [0.1, 0.3, 0.5, 0.8, 0.95]
    colors = ["#97C459", "#85B7EB", "#AFA9EC", "#F0997B", "#E24B4A"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for gamma, col in zip(gammas, colors):
        p = int(gamma * n)
        X = rng.standard_normal((n, p))
        # Hat matrix diagonals: P_{ii} = (X (XtX)^{-1} Xt)_{ii}
        # Efficient: compute X @ inv(XtX) @ Xt row by row
        XtX_inv = np.linalg.inv(X.T @ X)
        H_diag = np.sum((X @ XtX_inv) * X, axis=1)

        ax.hist(H_diag, bins=60, density=True, alpha=0.5, color=col,
                edgecolor="none",
                label=rf"$\gamma={gamma}$, mean $= {np.mean(H_diag):.2f}$")
        # Vertical line at 2gamma (rule of thumb threshold)
        ax.axvline(2 * gamma, color=col, linewidth=1.2, linestyle="--", alpha=0.8)

    ax.set_xlabel(r"Leverage $P_{ii}$")
    ax.set_ylabel("Density")
    ax.set_title(
        r"Distribution of leverage values $P_{ii}$  —  dashed lines at $2\gamma$ threshold"
    )
    ax.legend()
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig("fig5_leverage_distribution.png")
    plt.close(fig)
    print("Saved fig5_leverage_distribution.png")


# ===================================================================
# FIGURE 6: The γ ≥ 1 cliff — eigenvalue histogram at γ = 2
# ===================================================================
def figure6_gamma_above_one():
    print("Generating Figure 6: γ > 1 cliff...")
    n = 1000
    gamma = 2.0
    p = int(gamma * n)

    X = rng.standard_normal((n, p))
    # Use XXT/n instead (n x n, same nonzero eigenvalues as XtX/n)
    S = X @ X.T / n
    eigenvalues_nonzero = np.linalg.eigvalsh(S)

    # The full spectrum of (1/n)XtX has p eigenvalues:
    #   n nonzero ones (same as eigenvalues of XXT/n) and p-n zeros
    n_zeros = p - n
    all_eigenvalues = np.concatenate([np.zeros(n_zeros), eigenvalues_nonzero])

    lm, lp = mp_support(gamma)
    lam_grid = np.linspace(max(0.001, lm - 0.1), lp + 0.5, 500)
    density = mp_density(lam_grid, gamma)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1, 2.5]})

    # Left panel: bar showing the point mass at zero
    mass_at_zero = 1 - 1 / gamma
    axes[0].bar([0], [mass_at_zero], width=0.08, color="#E24B4A", alpha=0.8,
                edgecolor="#791F1F")
    axes[0].set_xlim(-0.2, 0.4)
    axes[0].set_ylim(0, 0.65)
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("Point mass")
    axes[0].set_title(rf"Point mass at 0: $1 - 1/\gamma = {mass_at_zero:.2f}$")
    axes[0].text(0, mass_at_zero + 0.02, f"{mass_at_zero:.2f}", ha="center",
                 fontsize=11, fontweight="bold", color="#791F1F")

    # Right panel: histogram of nonzero eigenvalues + MP density
    axes[1].hist(eigenvalues_nonzero, bins=80, density=True, alpha=0.6,
                 color="#85B7EB", edgecolor="none", label="Nonzero eigenvalues")
    # Scale the MP density by 1/gamma to account for the fraction that is nonzero
    # Actually the density of the continuous part already integrates to 1/gamma
    # when gamma > 1, so if we histogram only the nonzero eigenvalues (n of them),
    # we need to compare against the density rescaled: f_gamma * (p/n) = f_gamma * gamma
    # No — the MP density integrates to min(1, 1/gamma) over the continuous part.
    # For the n nonzero eigenvalues, their empirical density (normalised over n values)
    # needs to be compared with f_gamma * gamma (since f_gamma integrates to 1/gamma).
    axes[1].plot(lam_grid, density * gamma, color="#185FA5", linewidth=2,
                 label="MP density (rescaled)")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("Density")
    axes[1].set_title(
        rf"Continuous part of the spectrum ($\gamma = {gamma}$, $n={n}$, $p={p}$)"
    )
    axes[1].legend()

    fig.suptitle(
        r"The $\gamma \geq 1$ cliff: $X^\top X$ is singular, OLS is undefined",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("fig6_gamma_above_one.png")
    plt.close(fig)
    print("Saved fig6_gamma_above_one.png")


# ===================================================================
# FIGURE 7: Coverage of 95% CI for a single β_j vs gamma
# ===================================================================
def figure7_coverage():
    print("Generating Figure 7: CI coverage and width vs γ...")
    from scipy.stats import t as t_dist

    n = 300
    sigma2 = 1.0
    n_reps = 2000
    gammas = np.arange(0.05, 0.96, 0.05)
    coverages = []
    median_widths = []

    for gamma in gammas:
        p = int(gamma * n)
        beta_true = np.zeros(p)
        beta_true[0] = 1.0
        covered = 0
        widths = []

        t_crit = t_dist.ppf(0.975, df=n - p)

        for _ in range(n_reps):
            X = rng.standard_normal((n, p))
            eps = rng.standard_normal(n) * np.sqrt(sigma2)
            Y = X @ beta_true + eps

            XtX = X.T @ X
            beta_hat = np.linalg.solve(XtX, X.T @ Y)
            resid = Y - X @ beta_hat
            s2 = (resid @ resid) / (n - p)

            XtX_inv_diag = np.linalg.inv(XtX)[0, 0]
            se = np.sqrt(s2 * XtX_inv_diag)
            w = 2 * t_crit * se

            lower = beta_hat[0] - t_crit * se
            upper = beta_hat[0] + t_crit * se

            if lower <= beta_true[0] <= upper:
                covered += 1
            widths.append(w)

        coverages.append(covered / n_reps)
        median_widths.append(np.median(widths))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(gammas, coverages, "o-", color="#534AB7", markersize=5, linewidth=1.5)
    ax1.axhline(0.95, color="#993C1D", linestyle="--", linewidth=1, label="Nominal 95%")
    ax1.set_xlabel(r"$\gamma = p/n$")
    ax1.set_ylabel("Empirical coverage")
    ax1.set_title(r"Coverage stays at 95% (exact under normality)")
    ax1.set_ylim(0.90, 1.0)
    ax1.set_xlim(0, 1)
    ax1.legend()

    ax2.semilogy(gammas, median_widths, "s-", color="#D85A30", markersize=5,
                 linewidth=1.5, label="Empirical (median)")
    # Theoretical width ≈ 2 * t_crit * sigma * sqrt((XtX)^{-1}_{11})
    # ≈ 2 * t_crit * sigma / sqrt(n(1-gamma)) approximately
    theory_widths = []
    for gamma in gammas:
        p = int(gamma * n)
        tc = t_dist.ppf(0.975, df=n - p)
        theory_widths.append(2 * tc * np.sqrt(sigma2 / (n * (1 - gamma))))
    ax2.semilogy(gammas, theory_widths, "--", color="#0F6E56", linewidth=2,
                 label=r"MP approximation: $\propto 1/\sqrt{1-\gamma}$")
    ax2.set_xlabel(r"$\gamma = p/n$")
    ax2.set_ylabel("CI width (log scale)")
    ax2.set_title(r"...but the CI width diverges as $\gamma \to 1$")
    ax2.set_xlim(0, 1)
    ax2.legend()

    fig.suptitle(
        r"The 95% CI for $\beta_1$ remains valid but becomes useless ($n=300$)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("fig7_coverage.png")
    plt.close(fig)
    print("Saved fig7_coverage.png")


if __name__ == "__main__":
    figure1_eigenvalue_histograms()
    figure2_universality()
    figure3_mse_vs_gamma()
    figure4_ellipsoid_volume()
    figure5_leverage_distribution()
    figure6_gamma_above_one()
    figure7_coverage()
    print("All figures generated successfully.")