"""Numerical verification of the ST-CAF competing-risk CTMC.

Uses a normalized illustrative scenario as a five-state continuous-time Markov
chain and renders ``ST_CAF_Simulation.pdf``.  The rates are not empirical
calibrations and the output is not a treatment-effect estimate.

State space
-----------
``S0`` Secure            normal clinical operations, campaign not yet delivered
``S1`` Exposed           spear-phishing email delivered to the physician's inbox
``S2`` Local Failure     physician clicks the malicious link
``S3`` Systemic Compromise  ransomware lateral movement (absorbing)
``SC`` Contained         campaign neutralised by reporting or automated
                         isolation (absorbing)

A single campaign is modelled, so ``S1``/``S2`` recoveries are absorbed into
``SC`` rather than fed back into ``S0``. Under that construction the limiting
absorption probability is exactly the closed form quoted as Eq. (1) of the
manuscript::

    P_sys(inf) = alpha / (alpha + mu) * beta / (beta + mu + gamma)

with ``gamma = gamma_auto * exp(-delta * beta)`` for ST-CAF and ``gamma = 0``
for the traditional model. The script asserts this identity, so the figure and
the analytical bound cannot drift apart.

Usage
-----
    python3 simulate_stcaf_ctmc.py

Requires numpy, scipy and matplotlib. Deterministic: no random sampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

logger = logging.getLogger(__name__)

HORIZON_HOURS: Final[float] = 24.0
OUTPUT_PDF: Final[Path] = Path(__file__).with_name("ST_CAF_Simulation.pdf")


@dataclass(frozen=True)
class ScenarioParameters:
    """Transition rates for one modelled configuration, in units of h^-1.

    Attributes:
        label: Legend entry for the rendered curve.
        lam: Campaign delivery rate, ``S0 -> S1``.
        alpha: Exposure-to-failure (click) rate, ``S1 -> S2``. Layer 2
            affordances reduce this rate.
        mu: Human reporting rate. Layer 3 self-efficacy training raises it.
        beta: Ransomware lateral-movement rate, ``S2 -> S3``.
        gamma_auto: Layer 4 automated endpoint-isolation rate. Zero for
            models without an automated adaptation loop.
        delta: Telemetry latency in hours, damping the effective automation
            rate by ``exp(-delta * beta)``.
    """

    label: str
    lam: float
    alpha: float
    mu: float
    beta: float
    gamma_auto: float
    delta: float

    @property
    def gamma_effective(self) -> float:
        """Latency-damped automated isolation rate."""
        return self.gamma_auto * float(np.exp(-self.delta * self.beta))

    @property
    def p_sys_limit(self) -> float:
        """Closed-form limiting probability of systemic compromise, Eq. (1)."""
        exposure = self.alpha / (self.alpha + self.mu)
        escalation = self.beta / (self.beta + self.mu + self.gamma_effective)
        return exposure * escalation


# Normalized illustrative rates.  They test numerical agreement with the
# analytical absorption probability without mixing heterogeneous empirical
# estimands.
TRADITIONAL: Final[ScenarioParameters] = ScenarioParameters(
    label="Reference configuration",
    lam=1.0,
    alpha=1.0,
    mu=1.0,
    beta=1.0,
    gamma_auto=0.0,
    delta=0.0,
)

ST_CAF: Final[ScenarioParameters] = ScenarioParameters(
    label="Illustrative layered configuration",
    lam=1.0,
    alpha=0.5,
    mu=2.0,
    beta=1.0,
    gamma_auto=1.0,
    delta=0.0,
)


def _derivatives(_t: float, state: np.ndarray, p: ScenarioParameters) -> list[float]:
    """Right-hand side of the Kolmogorov forward equations.

    Args:
        _t: Time in hours; the chain is time-homogeneous so this is unused.
        state: Occupancy probabilities ordered ``[S0, S1, S2, S3, SC]``.
        p: Transition rates for the configuration being integrated.

    Returns:
        Time derivatives of the occupancy probabilities.
    """
    s0, s1, s2, _s3, _sc = state
    gamma = p.gamma_effective
    return [
        -p.lam * s0,
        p.lam * s0 - (p.alpha + p.mu) * s1,
        p.alpha * s1 - (p.beta + p.mu + gamma) * s2,
        p.beta * s2,
        p.mu * s1 + (p.mu + gamma) * s2,
    ]


def integrate(p: ScenarioParameters, horizon: float = HORIZON_HOURS) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the CTMC and return the systemic-compromise trajectory.

    Args:
        p: Transition rates for the configuration.
        horizon: Attack window in hours.

    Returns:
        Tuple of the time grid and ``P_sys(t)``, the occupancy of state ``S3``.

    Raises:
        RuntimeError: If the ODE solver fails or probability mass is not
            conserved to within 1e-6.
    """
    grid = np.linspace(0.0, horizon, 2001)
    solution = solve_ivp(
        _derivatives,
        (0.0, horizon),
        y0=[1.0, 0.0, 0.0, 0.0, 0.0],
        t_eval=grid,
        args=(p,),
        method="Radau",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed for {p.label}: {solution.message}")

    mass_error = float(np.max(np.abs(solution.y.sum(axis=0) - 1.0)))
    if mass_error > 1e-6:
        raise RuntimeError(f"Probability mass not conserved for {p.label}: {mass_error:.2e}")

    return solution.t, solution.y[3]


def render(scenarios: tuple[tuple[ScenarioParameters, np.ndarray, np.ndarray], ...]) -> None:
    """Render the comparison figure to ``OUTPUT_PDF``.

    Publication-grade styling: serif math, greyscale-distinguishable line styles,
    endpoint markers, reference gridlines at the two plateau values, an inset
    callout for the residual ST-CAF risk, and a self-contained legend that does
    not overlap the steady-state region of either curve.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
    })

    fig, ax = plt.subplots(figsize=(6.6, 3.7))

    color_trad = "#b22222"   # firebrick (distinguishable in greyscale via dashed)
    color_caf = "#1f4e9c"    # deep blue (solid)

    for params, times, p_sys in scenarios:
        is_traditional = params.gamma_auto == 0.0
        colour = color_trad if is_traditional else color_caf
        linestyle = "--" if is_traditional else "-"
        ax.plot(
            times, p_sys, linestyle, color=colour, linewidth=2.2,
            label=params.label, zorder=3,
        )
        # Endpoint marker
        ax.plot(times[-1], p_sys[-1], "o", color=colour, markersize=5, zorder=4)

    # Reference gridlines at the two plateau values (dotted, behind curves)
    p_trad_end = scenarios[0][2][-1]
    p_caf_end = scenarios[1][2][-1]
    ax.axhline(p_trad_end, color=color_trad, linestyle=":", linewidth=0.8, alpha=0.55, zorder=1)
    ax.axhline(p_caf_end, color=color_caf, linestyle=":", linewidth=0.8, alpha=0.55, zorder=1)

    # Endpoint annotations with leader-friendly placement
    ax.annotate(
        f"$P_{{sys}}\\to {p_trad_end:.3f}$",
        xy=(times[-1], p_trad_end),
        xytext=(-8, 11), textcoords="offset points",
        ha="right", va="bottom", fontsize=10, color=color_trad,
        arrowprops=dict(arrowstyle="-", color=color_trad, lw=0.6),
    )
    # ST-CAF plateau sits near zero; place its callout in the empty lower-middle
    # region (left of the endpoint, above the flat blue curve, below the red one)
    # and route the connector through that empty band so it crosses neither curve
    # nor the legend (legend is moved to the upper-left empty quadrant below).
    ax.annotate(
        f"$P_{{sys}}\\to {p_caf_end:.3f}$",
        xy=(times[-1], p_caf_end),
        xytext=(0.22, 0.34), textcoords="axes fraction",
        ha="center", va="center", fontsize=9.5, color=color_caf,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color_caf, lw=0.7, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=color_caf, lw=0.8,
                        connectionstyle="arc3,rad=0.15"),
    )

    ax.set_xlabel("Time since campaign launch (hours)")
    ax.set_ylabel("Probability of systemic compromise, $P_{sys}$")
    ax.set_xlim(0.0, HORIZON_HOURS)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_ylim(-0.02, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    # Legend in the upper-left empty quadrant (above the rising curves), so it
    # cannot collide with the ST-CAF callout's connector arrow.
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.7")

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Integrate both configurations, verify Eq. (1) and emit the figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = []

    for params in (TRADITIONAL, ST_CAF):
        times, p_sys = integrate(params)
        terminal = float(p_sys[-1])
        limit = params.p_sys_limit
        # The 24 h horizon must be long enough for the numerical trajectory to
        # have reached the closed form quoted as Eq. (1).
        if abs(terminal - limit) > 5e-3:
            raise RuntimeError(
                f"{params.label}: P_sys(24 h)={terminal:.6f} has not converged "
                f"to the Eq. (1) limit {limit:.6f}"
            )
        logger.info(
            "%-45s P_sys(24 h) = %.4f   Eq.(1) limit = %.4f   gamma_eff = %.4f",
            params.label,
            terminal,
            limit,
            params.gamma_effective,
        )
        results.append((params, times, p_sys))

    reduction = results[0][2][-1] / results[1][2][-1]
    logger.info("Relative reduction in P_sys(24 h): %.1f-fold", reduction)
    logger.info("Absolute risk reduction: %.4f", results[0][2][-1] - results[1][2][-1])

    render(tuple(results))
    logger.info("Wrote %s", OUTPUT_PDF)


if __name__ == "__main__":
    main()
