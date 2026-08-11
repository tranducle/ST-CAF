"""Assumption-stress experiments for the ST-CAF competing-risk model.

This module deliberately does *not* estimate a real-world ST-CAF effect size.
The available studies measure different estimands and populations, so combining
them into a single ``calibrated`` probability would create false precision.
Instead, the script performs three reproducible model checks:

1. paired, dimensionless scenario sweeps under conservative/reference/wide
   intervention ranges;
2. Jansen total-order sensitivity analysis for each declared range set; and
3. a semi-Markov robustness check in which matched-mean Weibull event times
   replace the CTMC's exponential races.

All reported intervals are simulation intervals over declared assumptions, not
confidence intervals for an empirical treatment effect.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

ROOT: Final[Path] = Path(__file__).parent
FIGURE_PATH: Final[Path] = ROOT / "figures" / "ST_CAF_MC_Distributions.pdf"
REPORT_PATH: Final[Path] = ROOT / "mc_report.json"


@dataclass(frozen=True)
class RangeSet:
    """Dimensionless uncertainty ranges used by one scenario sweep."""

    name: str
    alpha_over_mu: tuple[float, float]
    beta_over_mu: tuple[float, float]
    alpha_reduction: tuple[float, float]
    reporting_multiplier: tuple[float, float]
    gamma_over_mu_eff: tuple[float, float]


RANGE_SETS: Final[tuple[RangeSet, ...]] = (
    RangeSet("conservative", (0.5, 2.0), (0.5, 2.0), (1.0, 1.5), (1.0, 2.0), (0.1, 1.0)),
    RangeSet("reference", (0.25, 4.0), (0.25, 4.0), (1.0, 2.0), (1.0, 4.0), (0.1, 5.0)),
    RangeSet("wide", (0.1, 10.0), (0.1, 10.0), (1.0, 4.0), (1.0, 10.0), (0.1, 20.0)),
)


def p_sys_closed_form(alpha: np.ndarray | float, mu: np.ndarray | float,
                      beta: np.ndarray | float, gamma: np.ndarray | float) -> np.ndarray:
    """Return the limiting probability of absorption into systemic compromise."""
    alpha_a = np.asarray(alpha, dtype=float)
    mu_a = np.asarray(mu, dtype=float)
    beta_a = np.asarray(beta, dtype=float)
    gamma_a = np.asarray(gamma, dtype=float)
    if np.any(alpha_a <= 0) or np.any(mu_a <= 0) or np.any(beta_a <= 0) or np.any(gamma_a < 0):
        raise ValueError("alpha, mu, and beta must be positive; gamma must be non-negative")
    return alpha_a / (alpha_a + mu_a) * beta_a / (beta_a + mu_a + gamma_a)


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float], n: int) -> np.ndarray:
    lo, hi = bounds
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def _transform_unit_cube(u: np.ndarray, ranges: RangeSet) -> tuple[np.ndarray, ...]:
    """Map a unit-cube design to the declared dimensionless ranges."""
    if u.ndim != 2 or u.shape[1] != 5:
        raise ValueError("unit-cube design must have shape (n, 5)")

    def log_map(col: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        lo, hi = bounds
        return np.exp(np.log(lo) + col * (np.log(hi) - np.log(lo)))

    def linear_map(col: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        lo, hi = bounds
        return lo + col * (hi - lo)

    return (
        log_map(u[:, 0], ranges.alpha_over_mu),
        log_map(u[:, 1], ranges.beta_over_mu),
        linear_map(u[:, 2], ranges.alpha_reduction),
        linear_map(u[:, 3], ranges.reporting_multiplier),
        log_map(u[:, 4], ranges.gamma_over_mu_eff),
    )


def evaluate_design(u: np.ndarray, ranges: RangeSet) -> dict[str, np.ndarray]:
    """Evaluate paired baseline and layer interventions for a unit-cube design."""
    a_ratio, b_ratio, k_alpha, k_mu, g_ratio = _transform_unit_cube(u, ranges)
    mu_base = np.ones(len(u))
    alpha_base = a_ratio * mu_base
    beta = b_ratio * mu_base

    alpha_full = alpha_base / k_alpha
    mu_full = mu_base * k_mu
    gamma_full = g_ratio * mu_full

    p_base = p_sys_closed_form(alpha_base, mu_base, beta, 0.0)
    p_l2 = p_sys_closed_form(alpha_full, mu_base, beta, 0.0)
    p_l3 = p_sys_closed_form(alpha_base, mu_full, beta, 0.0)
    p_l4 = p_sys_closed_form(alpha_base, mu_base, beta, g_ratio * mu_base)
    p_full = p_sys_closed_form(alpha_full, mu_full, beta, gamma_full)

    return {
        "p_baseline": p_base,
        "p_layer2_only": p_l2,
        "p_layer3_only": p_l3,
        "p_layer4_only": p_l4,
        "p_full": p_full,
        "reduction_full": p_base / p_full,
    }


def _summary(x: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "simulation_interval_95_low": float(np.quantile(x, 0.025)),
        "simulation_interval_95_high": float(np.quantile(x, 0.975)),
    }


def run_scenario_sweeps(n: int, seed: int) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Run paired scenario sweeps for all declared range sets."""
    rng = np.random.default_rng(seed)
    report: dict[str, dict] = {}
    raw: dict[str, dict[str, np.ndarray]] = {}
    for ranges in RANGE_SETS:
        u = rng.random((n, 5))
        values = evaluate_design(u, ranges)
        raw[ranges.name] = values
        report[ranges.name] = {
            "range_set": {
                "alpha_over_mu": list(ranges.alpha_over_mu),
                "beta_over_mu": list(ranges.beta_over_mu),
                "alpha_reduction": list(ranges.alpha_reduction),
                "reporting_multiplier": list(ranges.reporting_multiplier),
                "gamma_over_mu_eff": list(ranges.gamma_over_mu_eff),
            },
            **{name: _summary(array) for name, array in values.items()},
        }
    return report, raw


def jansen_total_order(ranges: RangeSet, n_base: int, seed: int,
                       n_bootstrap: int = 200) -> dict:
    """Estimate total-order indices for log10(P_sys) with Jansen's estimator."""
    rng = np.random.default_rng(seed)
    a = rng.random((n_base, 5))
    b = rng.random((n_base, 5))
    y_a = np.log10(evaluate_design(a, ranges)["p_full"])
    y_b = np.log10(evaluate_design(b, ranges)["p_full"])
    variance = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
    if variance <= 0:
        raise RuntimeError("zero model-output variance in sensitivity design")

    names = ["alpha_over_mu", "beta_over_mu", "alpha_reduction", "reporting_multiplier", "gamma_over_mu_eff"]
    estimates: dict[str, float] = {}
    intervals: dict[str, list[float]] = {}
    for i, name in enumerate(names):
        ab = a.copy()
        ab[:, i] = b[:, i]
        y_ab = np.log10(evaluate_design(ab, ranges)["p_full"])
        squared = 0.5 * (y_a - y_ab) ** 2
        estimate = float(np.mean(squared) / variance)
        estimates[name] = estimate

        boot = np.empty(n_bootstrap)
        for j in range(n_bootstrap):
            idx = rng.integers(0, n_base, n_base)
            boot_var = float(np.var(np.concatenate([y_a[idx], y_b[idx]]), ddof=1))
            boot[j] = float(np.mean(squared[idx]) / boot_var) if boot_var > 0 else np.nan
        intervals[name] = [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))]

    return {
        "method": "Jansen total-order estimator on log10(P_sys)",
        "n_base": n_base,
        "bootstrap_replicates": n_bootstrap,
        "total_order": estimates,
        "bootstrap_interval_95": intervals,
    }


def _weibull_times(rng: np.random.Generator, rate: float, shape: float, n: int) -> np.ndarray:
    """Draw Weibull times with mean exactly equal to 1/rate."""
    scale = (1.0 / rate) / math.gamma(1.0 + 1.0 / shape)
    return scale * rng.weibull(shape, n)


def _race_probability(rng: np.random.Generator, alpha: float, mu: float,
                      beta: float, gamma: float, shapes: tuple[float, ...], n: int) -> float:
    click_shape, report1_shape, lateral_shape, report2_shape, auto_shape = shapes
    click = _weibull_times(rng, alpha, click_shape, n)
    report1 = _weibull_times(rng, mu, report1_shape, n)
    stage1 = click < report1
    lateral = _weibull_times(rng, beta, lateral_shape, n)
    report2 = _weibull_times(rng, mu, report2_shape, n)
    if gamma > 0:
        auto = _weibull_times(rng, gamma, auto_shape, n)
        stage2 = (lateral < report2) & (lateral < auto)
    else:
        stage2 = lateral < report2
    return float(np.mean(stage1 & stage2))


def run_semimarkov_stress(n_per_shape: int, seed: int) -> dict:
    """Replace exponential races by matched-mean Weibull races."""
    rng = np.random.default_rng(seed)
    shapes = (0.5, 1.0, 2.0)
    rows = []
    for shape_tuple in itertools.product(shapes, repeat=5):
        p_base = _race_probability(rng, 1.0, 1.0, 1.0, 0.0, shape_tuple, n_per_shape)
        p_full = _race_probability(rng, 0.5, 2.0, 1.0, 1.0, shape_tuple, n_per_shape)
        rows.append((shape_tuple, p_base, p_full, p_base / p_full if p_full > 0 else math.inf))

    reductions = np.asarray([row[3] for row in rows], dtype=float)
    finite = reductions[np.isfinite(reductions)]
    exp_row = next(row for row in rows if row[0] == (1.0, 1.0, 1.0, 1.0, 1.0))
    return {
        "design": {
            "baseline_rates": {"alpha": 1.0, "mu": 1.0, "beta": 1.0, "gamma": 0.0},
            "layered_rates": {"alpha": 0.5, "mu": 2.0, "beta": 1.0, "gamma": 1.0},
            "weibull_shapes": list(shapes),
            "n_shape_combinations": len(rows),
            "n_per_combination": n_per_shape,
            "matched_mean_policy": "Each Weibull mean equals the reciprocal of its corresponding CTMC rate.",
        },
        "exponential_case": {
            "p_baseline_simulated": exp_row[1],
            "p_layered_simulated": exp_row[2],
            "reduction_simulated": exp_row[3],
            "p_baseline_closed_form": float(p_sys_closed_form(1.0, 1.0, 1.0, 0.0)),
            "p_layered_closed_form": float(p_sys_closed_form(0.5, 2.0, 1.0, 1.0)),
        },
        "shape_stress": {
            "fraction_layered_lower_risk": float(np.mean([row[2] < row[1] for row in rows])),
            "reduction_min": float(np.min(finite)),
            "reduction_median": float(np.median(finite)),
            "reduction_max": float(np.max(finite)),
            "simulation_interval_95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
        },
        "rows": [
            {
                "shapes": list(shape_tuple),
                "p_baseline": p_base,
                "p_layered": p_full,
                "reduction": reduction,
            }
            for shape_tuple, p_base, p_full, reduction in rows
        ],
    }


def render_figure(sweeps: dict, raw: dict[str, dict[str, np.ndarray]], sobol: dict,
                  semimarkov: dict) -> None:
    """Render a three-panel figure that exposes assumption sensitivity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    names = [ranges.name for ranges in RANGE_SETS]
    med = [sweeps[name]["reduction_full"]["median"] for name in names]
    lo = [sweeps[name]["reduction_full"]["simulation_interval_95_low"] for name in names]
    hi = [sweeps[name]["reduction_full"]["simulation_interval_95_high"] for name in names]
    axes[0].errorbar(names, med, yerr=[np.asarray(med) - np.asarray(lo), np.asarray(hi) - np.asarray(med)],
                     fmt="o", color="#1f4e9c", capsize=4, linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Paired reduction factor (scenario output)")
    axes[0].set_title("(a) Dependence on declared ranges")
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    st = sobol["reference"]["total_order"]
    labels = [r"$\alpha/\mu$", r"$\beta/\mu$", r"$k_\alpha$", r"$k_\mu$", r"$\gamma/\mu_{eff}$"]
    axes[1].barh(labels, list(st.values()), color=["#4e79a7", "#e15759", "#76b7b2", "#59a14f", "#af7aa1"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Jansen total-order index")
    axes[1].set_title("(b) Reference-range sensitivity")
    axes[1].grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    reductions = np.asarray([row["reduction"] for row in semimarkov["rows"] if np.isfinite(row["reduction"])])
    axes[2].boxplot([reductions], tick_labels=["Weibull\nshape sweep"], patch_artist=True,
                    boxprops={"facecolor": "#f28e2b", "alpha": 0.65}, showfliers=False)
    axes[2].axhline(semimarkov["exponential_case"]["reduction_simulated"], color="#1f4e9c",
                    linestyle="--", linewidth=1.5, label="Exponential case")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Paired reduction factor")
    axes[2].set_title("(c) Memoryless-assumption stress")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50000, help="Samples per scenario range set.")
    parser.add_argument("--sobol-n", type=int, default=4096, help="Base samples for Jansen indices.")
    parser.add_argument("--semimarkov-n", type=int, default=20000, help="Samples per Weibull-shape combination.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.n, args.sobol_n, args.semimarkov_n) <= 0:
        raise SystemExit("all sample sizes must be positive")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sweeps, raw = run_scenario_sweeps(args.n, args.seed)
    sobol = {
        ranges.name: jansen_total_order(ranges, args.sobol_n, args.seed + 100 + i)
        for i, ranges in enumerate(RANGE_SETS)
    }
    semimarkov = run_semimarkov_stress(args.semimarkov_n, args.seed + 200)

    report = {
        "analysis_type": "assumption_stress_not_empirical_effect_estimation",
        "claim_boundary": (
            "Outputs are conditional on declared dimensionless ranges and matched-mean event-time shapes. "
            "They do not estimate an organizational treatment effect or validate ST-CAF in the field."
        ),
        "seed": args.seed,
        "n_per_range_set": args.n,
        "scenario_sweeps": sweeps,
        "sobol": sobol,
        "semi_markov": semimarkov,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    render_figure(sweeps, raw, sobol, semimarkov)

    logger.info("=== ST-CAF assumption-stress analysis ===")
    for name in (ranges.name for ranges in RANGE_SETS):
        stats = sweeps[name]["reduction_full"]
        logger.info("%-12s median %.3f; 95%% simulation interval [%.3f, %.3f]",
                    name, stats["median"], stats["simulation_interval_95_low"],
                    stats["simulation_interval_95_high"])
    shape = semimarkov["shape_stress"]
    logger.info("Semi-Markov: layered risk lower in %.1f%% of shape combinations; reduction range [%.3f, %.3f]",
                100.0 * shape["fraction_layered_lower_risk"], shape["reduction_min"], shape["reduction_max"])
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", FIGURE_PATH)


if __name__ == "__main__":
    main()
