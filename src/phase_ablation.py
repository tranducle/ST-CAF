"""Phase-comparison stress test for the ST-CAF competing-risk model.

The earlier manuscript inferred a cross-phase theorem from a partial derivative
that held the other phase-varying rates fixed.  This script implements the exact
two-phase comparison, supplies a counterexample to the former theorem, and
quantifies how often Layer-4 absolute risk reduction increases under explicitly
declared monotone scenario draws.  The frequency is a scenario result, not a
population probability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
# In the packaged deposit the code sits in src/ beside sibling outputs/ and
# figures/ directories; in the working tree it sits at the project root. Resolve
# the destinations against whichever layout is present so both reproduce.
_BASE = ROOT.parent if (ROOT.parent / "outputs").is_dir() else ROOT
OUT_FIG = (_BASE / "figures" if (_BASE / "figures").is_dir() else _BASE / "outputs") / "ST_CAF_Phase_Exposure.pdf"
OUT_REPORT = (_BASE / "outputs" if (_BASE / "outputs").is_dir() else _BASE) / "phase_analysis_report.json"


@dataclass(frozen=True)
class PhaseRates:
    alpha: float
    mu: float
    beta: float
    gamma: float

    def validate(self) -> None:
        if self.alpha <= 0 or self.mu <= 0 or self.beta <= 0 or self.gamma < 0:
            raise ValueError("alpha, mu, beta must be positive and gamma non-negative")


@dataclass(frozen=True)
class MultiplierSpec:
    """Per-phase multiplier distribution for the lateral-movement rate beta.

    ``log_uniform`` draws exp(U(log low, log high)); ``uniform`` draws U(low, high).
    Both consume exactly one uniform variate per sample, so switching between them
    preserves the seed and the draw order of every other rate.
    """

    kind: str
    low: float
    high: float
    label: str

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.low <= 0 or self.high <= self.low:
            raise ValueError(f"invalid multiplier support [{self.low}, {self.high}]")
        if self.kind == "log_uniform":
            return np.exp(rng.uniform(np.log(self.low), np.log(self.high), n))
        if self.kind == "uniform":
            return rng.uniform(self.low, self.high, n)
        raise ValueError(f"unknown multiplier kind: {self.kind}")


BETA_VARIANTS: dict[str, MultiplierSpec] = {
    "stipulated": MultiplierSpec("uniform", 1.0, 3.0, "uniform on [1, 3] (non-decreasing beta)"),
    "two_sided": MultiplierSpec("log_uniform", 1.0 / 3.0, 3.0, "log-uniform on [1/3, 3]"),
    "mirrored": MultiplierSpec("uniform", 1.0 / 3.0, 1.0, "uniform on [1/3, 1] (non-increasing beta)"),
}


REPRESENTATIVE = {
    "D1": PhaseRates(1.20, 0.80, 0.50, 0.00),
    "D2": PhaseRates(0.80, 1.00, 1.00, 1.00),
    "D3": PhaseRates(0.60, 1.20, 2.00, 2.00),
}

# All former direction and regularity conditions hold, yet Delta4 falls from
# D2 to D3.  This demonstrates why the former cross-phase theorem was invalid.
COUNTEREXAMPLE = {
    "D1": PhaseRates(0.20, 0.05, 0.01, 0.00),
    "D2": PhaseRates(0.10, 0.10, 0.05, 1.00),
    "D3": PhaseRates(0.05, 0.20, 0.10, 2.00),
}


def p_sys(rates: PhaseRates, gamma_override: float | None = None) -> float:
    rates.validate()
    gamma = rates.gamma if gamma_override is None else gamma_override
    if gamma < 0:
        raise ValueError("gamma override must be non-negative")
    return rates.alpha / (rates.alpha + rates.mu) * rates.beta / (rates.beta + rates.mu + gamma)


def delta4(rates: PhaseRates) -> float:
    """Absolute model-risk reduction attributable to gamma at fixed other rates."""
    return p_sys(rates, 0.0) - p_sys(rates)


def exact_phase_decrease_condition(left: PhaseRates, right: PhaseRates) -> bool:
    """Return the exact condition P_sys(right) < P_sys(left)."""
    left.validate()
    right.validate()
    lhs = right.alpha * right.beta * (left.alpha + left.mu) * (left.beta + left.mu + left.gamma)
    rhs = left.alpha * left.beta * (right.alpha + right.mu) * (right.beta + right.mu + right.gamma)
    return lhs < rhs


def _summarize(phases: dict[str, PhaseRates]) -> dict:
    return {
        name: {
            **asdict(rates),
            "p_sys": p_sys(rates),
            "p_sys_without_layer4": p_sys(rates, 0.0),
            "delta4": delta4(rates),
            "exposure_factor": rates.alpha / (rates.alpha + rates.mu),
            "former_regularity_margin": rates.mu**2 + rates.gamma * rates.mu - rates.beta**2,
        }
        for name, rates in phases.items()
    }


def run_monotone_stress(n: int, seed: int, beta_variant: str = "stipulated") -> dict:
    """Sample monotone phase sequences and record ordering frequencies."""
    if beta_variant not in BETA_VARIANTS:
        raise ValueError(f"unknown beta variant: {beta_variant}")
    beta_spec = BETA_VARIANTS[beta_variant]
    rng = np.random.default_rng(seed)
    alpha1 = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
    mu1 = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
    beta1 = np.exp(rng.uniform(np.log(0.25), np.log(1.0), n))

    alpha2 = alpha1 * rng.uniform(0.5, 1.0, n)
    alpha3 = alpha2 * rng.uniform(0.5, 1.0, n)
    mu2 = mu1 * rng.uniform(1.0, 2.0, n)
    mu3 = mu2 * rng.uniform(1.0, 2.0, n)
    beta2 = beta1 * beta_spec.draw(rng, n)
    beta3 = beta2 * beta_spec.draw(rng, n)
    gamma2 = mu2 * np.exp(rng.uniform(np.log(0.1), np.log(3.0), n))
    gamma3 = gamma2 * rng.uniform(1.0, 3.0, n)

    def vector_delta(alpha: np.ndarray, mu: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        f = alpha / (alpha + mu)
        return f * (beta / (beta + mu) - beta / (beta + mu + gamma))

    d2 = vector_delta(alpha2, mu2, beta2, gamma2)
    d3 = vector_delta(alpha3, mu3, beta3, gamma3)
    ordering = d3 > d2
    former_regularity = (
        (mu2**2 + gamma2 * mu2 > beta2**2)
        & (mu3**2 + gamma3 * mu3 > beta3**2)
    )
    risk1 = alpha1 / (alpha1 + mu1) * beta1 / (beta1 + mu1)
    risk2 = alpha2 / (alpha2 + mu2) * beta2 / (beta2 + mu2 + gamma2)
    risk3 = alpha3 / (alpha3 + mu3) * beta3 / (beta3 + mu3 + gamma3)

    # Relative Layer-4 contribution: P(gamma=0)/P(gamma) = 1 + gamma/(beta+mu).
    # The exposure factor alpha/(alpha+mu) cancels, so this ratio is orderable
    # across phases by a criterion involving only (beta, mu, gamma).
    relative2 = 1.0 + gamma2 / (beta2 + mu2)
    relative3 = 1.0 + gamma3 / (beta3 + mu3)

    return {
        "n": n,
        "seed": seed,
        "beta_multiplier_variant": beta_variant,
        "beta_multiplier_description": beta_spec.label,
        "declared_sampling_ranges": {
            "D1_alpha": [0.5, 2.0],
            "D1_mu": [0.5, 2.0],
            "D1_beta": [0.25, 1.0],
            "phase_multipliers": {
                "alpha": [0.5, 1.0],
                "mu": [1.0, 2.0],
                "beta": [beta_spec.low, beta_spec.high],
                "beta_kind": beta_spec.kind,
                "gamma_D2_over_mu_D2": [0.1, 3.0],
                "gamma_D3_over_gamma_D2": [1.0, 3.0],
            },
        },
        "fraction_delta4_D3_greater_D2": float(np.mean(ordering)),
        "fraction_satisfying_former_regularity": float(np.mean(former_regularity)),
        "fraction_ordered_given_former_regularity": float(np.mean(ordering[former_regularity])) if np.any(former_regularity) else None,
        "fraction_system_risk_strictly_decreases_each_phase": float(np.mean((risk2 < risk1) & (risk3 < risk2))),
        "fraction_relative_layer4_D3_greater_D2": float(np.mean(relative3 > relative2)),
        "claim_boundary": "Frequencies describe the declared scenario generator, not organizations or a population distribution.",
    }


def render(representative: dict, counterexample: dict) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=False)
    names = ("D1", "D2", "D3")
    x = np.arange(3)
    labels = ["Digitization\n(D1)", "Digitalization\n(D2)", "Transformation\n(D3)"]

    for ax, data, title, color in (
        (axes[0], representative, "(a) One admissible increasing sequence", "#1f4e9c"),
        (axes[1], counterexample, "(b) Admissible counterexample", "#b22222"),
    ):
        values = [data[name]["delta4"] for name in names]
        ax.bar(x, values, color=color, alpha=0.82, width=0.55)
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(r"Layer-4 model-risk difference $\Delta^{(4)}(p)$")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n <= 0:
        raise SystemExit("n must be positive")

    representative = _summarize(REPRESENTATIVE)
    counterexample = _summarize(COUNTEREXAMPLE)
    stress = run_monotone_stress(args.n, args.seed, "stipulated")
    beta_sensitivity = {
        name: run_monotone_stress(args.n, args.seed, name)
        for name in ("two_sided", "mirrored")
    }
    report = {
        "analysis_type": "conditional_phase_comparison_not_cross_phase_theorem",
        "exact_condition": (
            "P(right)<P(left) iff alpha_right*beta_right*(alpha_left+mu_left)*"
            "(beta_left+mu_left+gamma_left) < alpha_left*beta_left*"
            "(alpha_right+mu_right)*(beta_right+mu_right+gamma_right)."
        ),
        "representative_sequence": representative,
        "counterexample_to_former_theorem": counterexample,
        "monotone_stress": stress,
        "beta_stipulation_sensitivity": beta_sensitivity,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    render(representative, counterexample)

    print("=== Phase comparison stress test ===")
    print(f"Representative Delta4: {[round(representative[n]['delta4'], 6) for n in ('D1','D2','D3')]}")
    print(f"Counterexample Delta4: {[round(counterexample[n]['delta4'], 6) for n in ('D1','D2','D3')]}")
    print(f"Delta4(D3)>Delta4(D2) in {100*stress['fraction_delta4_D3_greater_D2']:.1f}% of declared monotone draws")
    print(f"...and in {100*stress['fraction_ordered_given_former_regularity']:.1f}% of draws satisfying the former regularity condition")
    print(f"Relative Layer-4 contribution ordered in {100*stress['fraction_relative_layer4_D3_greater_D2']:.1f}% of draws")
    for name, result in beta_sensitivity.items():
        print(
            f"beta multiplier {result['beta_multiplier_description']}: "
            f"Delta4(D3)>Delta4(D2) in {100*result['fraction_delta4_D3_greater_D2']:.1f}%, "
            f"relative ordered in {100*result['fraction_relative_layer4_D3_greater_D2']:.1f}%"
        )
    print(f"Wrote {OUT_REPORT}")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
