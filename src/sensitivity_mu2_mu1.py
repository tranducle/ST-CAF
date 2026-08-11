"""Sensitivity of the pathway compromise probability to the reporting-rate
asymmetry mu2/mu1 (MTH-S2-02).

The manuscript's main analysis uses a single reporting rate mu at both S1 and
S2. With separate rates, the pathway probability generalizes to

    P = alpha/(alpha+mu1) * beta/(beta+mu2+gamma).

This script evaluates how P and the layered/no-containment reduction ratio
depend on mu2/mu1 at a representative dimensionless reference configuration.
It reproduces the numbers reported in the manuscript's shared-mu discussion.
"""

from __future__ import annotations

import argparse


def p_path(a: float, mu1: float, b: float, mu2: float, g: float) -> float:
    """Generalized two-reporting-rate pathway probability."""
    return (a / (a + mu1)) * (b / (b + mu2 + g))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=float, default=1.0, help="click rate alpha")
    ap.add_argument("--mu1", type=float, default=1.0, help="pre-failure reporting rate")
    ap.add_argument("--b", type=float, default=0.5, help="lateral-movement rate beta")
    ap.add_argument("--g", type=float, default=0.5, help="containment rate gamma")
    args = ap.parse_args()

    base = p_path(args.a, args.mu1, args.b, args.mu1, args.g)
    print(f"Reference configuration: a={args.a}, mu1={args.mu1}, b={args.b}, g={args.g}")
    print(f"Baseline (mu2/mu1 = 1): P = {base:.4f}")
    print()
    print("mu2/mu1 | P_path  | layered/no-g reduction ratio")
    ratios = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for r in ratios:
        mu2 = args.mu1 * r
        p = p_path(args.a, args.mu1, args.b, mu2, args.g)
        p_nog = p_path(args.a, args.mu1, args.b, mu2, 0.0)
        print(f"  {r:5.2f}  | {p:.4f} | {p_nog / p:.4f}")


if __name__ == "__main__":
    main()
