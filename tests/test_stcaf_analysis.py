"""Regression tests for the revised ST-CAF analytical experiments."""

from __future__ import annotations

import unittest

import numpy as np

import phase_ablation as phase
import simulate_stcaf_mc as mc


class ClosedFormTests(unittest.TestCase):
    def test_known_closed_form(self) -> None:
        self.assertAlmostEqual(float(mc.p_sys_closed_form(1.0, 1.0, 1.0, 0.0)), 0.25)
        self.assertAlmostEqual(float(mc.p_sys_closed_form(0.5, 2.0, 1.0, 1.0)), 0.05)

    def test_monotone_comparative_statics_in_positive_domain(self) -> None:
        base = float(mc.p_sys_closed_form(1.0, 1.0, 1.0, 1.0))
        self.assertGreater(float(mc.p_sys_closed_form(1.1, 1.0, 1.0, 1.0)), base)
        self.assertGreater(float(mc.p_sys_closed_form(1.0, 1.0, 1.1, 1.0)), base)
        self.assertLess(float(mc.p_sys_closed_form(1.0, 1.1, 1.0, 1.0)), base)
        self.assertLess(float(mc.p_sys_closed_form(1.0, 1.0, 1.0, 1.1)), base)

    def test_invalid_boundary_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mc.p_sys_closed_form(0.0, 1.0, 1.0, 0.0)


class ScenarioSweepTests(unittest.TestCase):
    def test_paired_interventions_never_raise_model_risk(self) -> None:
        rng = np.random.default_rng(7)
        values = mc.evaluate_design(rng.random((2000, 5)), mc.RANGE_SETS[1])
        self.assertTrue(np.all(values["p_full"] <= values["p_baseline"]))
        self.assertTrue(np.all(values["reduction_full"] >= 1.0))

    def test_sweep_is_seed_reproducible(self) -> None:
        first, _ = mc.run_scenario_sweeps(1000, 19)
        second, _ = mc.run_scenario_sweeps(1000, 19)
        self.assertEqual(first, second)


class PhaseLogicTests(unittest.TestCase):
    def test_exact_phase_condition_matches_direct_comparison(self) -> None:
        for left, right in ((phase.REPRESENTATIVE["D1"], phase.REPRESENTATIVE["D2"]),
                            (phase.REPRESENTATIVE["D2"], phase.REPRESENTATIVE["D3"]),
                            (phase.COUNTEREXAMPLE["D2"], phase.COUNTEREXAMPLE["D3"])):
            self.assertEqual(phase.exact_phase_decrease_condition(left, right), phase.p_sys(right) < phase.p_sys(left))

    def test_former_theorem_counterexample(self) -> None:
        d2 = phase.delta4(phase.COUNTEREXAMPLE["D2"])
        d3 = phase.delta4(phase.COUNTEREXAMPLE["D3"])
        self.assertLess(d3, d2)
        for rates in (phase.COUNTEREXAMPLE["D2"], phase.COUNTEREXAMPLE["D3"]):
            self.assertGreater(rates.mu**2 + rates.gamma * rates.mu, rates.beta**2)

    def test_representative_sequence_obeys_declared_directions(self) -> None:
        d1, d2, d3 = (phase.REPRESENTATIVE[name] for name in ("D1", "D2", "D3"))
        self.assertGreaterEqual(d1.alpha, d2.alpha)
        self.assertGreaterEqual(d2.alpha, d3.alpha)
        self.assertLessEqual(d1.mu, d2.mu)
        self.assertLessEqual(d2.mu, d3.mu)
        self.assertLessEqual(d1.beta, d2.beta)
        self.assertLessEqual(d2.beta, d3.beta)
        self.assertLessEqual(d1.gamma, d2.gamma)
        self.assertLessEqual(d2.gamma, d3.gamma)


if __name__ == "__main__":
    unittest.main()
