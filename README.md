# ST-CAF

**Socio-Technical Cybersecurity Awareness Framework**

Companion replication code for the manuscript:

> *"Beyond Technological Defenses: A Phase-Aware Socio-Technical Framework for Cybersecurity Awareness in Digital Transformation"*

> **Anonymized for review.** Author and affiliation information is withheld to preserve double-blind review. Identifying metadata will be added on acceptance.

## Scope and scientific boundary

This repository provides the analysis code, frozen inputs, tests, and
machine-readable outputs behind the manuscript's quantitative component.

**The models are analytical and assumption-conditional — they are not
empirically calibrated.** The paper proposes the Socio-Technical Cybersecurity
Awareness Framework (ST-CAF) and formalizes an escalation pathway as a
phase-indexed continuous-time Markov chain (CTMC). The scripts here:

1. reproduce the within-model comparative statics;
2. run the declared dimensionless scenario sweeps (Monte Carlo reduction);
3. compute Jansen total-order sensitivity indices with bootstrap intervals;
4. run a matched-mean semi-Markov (Weibull) robustness stress test;
5. run the phase-stress simulation over three digital-transformation phases,
   including the sensitivity of the cross-phase result to the stipulated
   direction of the lateral-movement rate.

All results are scenario-conditional and **do not estimate a real-world effect
size**. The incident illustration and standards crosswalk in the manuscript use
only cited public sources and are not reproduced here.

## Repository structure

```text
ST-CAF/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── simulate_stcaf_mc.py            # MC reduction, total-order sensitivity, bootstrap, semi-Markov
│   ├── simulate_stcaf_ctmc_journal.py  # closed-form P_path + counterexample computations
│   ├── phase_ablation.py               # phase-stress generator + scenario ablation
│   └── sensitivity_mu2_mu1.py          # reporting-rate asymmetry (mu2/mu1) sensitivity
├── conftest.py                         # puts src/ on sys.path for pytest
├── tests/
│   └── test_stcaf_analysis.py          # 8 regression tests (invariants + outputs)
└── outputs/
    ├── mc_report.json                  # frozen MC/sensitivity/semi-Markov results (manuscript source of truth)
    └── phase_analysis_report.json      # phase-stress results
```

## Requirements

- Python 3.8+
- NumPy

```bash
pip install -r requirements.txt
```

## Reproduction

### One-command tests

```bash
PYTHONPATH=src python tests/test_stcaf_analysis.py   # standalone runner (8 regression tests, NumPy only)
# or, if pytest is available:
pytest tests/ -v
```

The standalone runner executes the same 8 regression tests (analytical
invariants, numeric consistency, and output checks) and requires NumPy only.
It needs `PYTHONPATH=src` because the scripts live under `src/` and the test
imports them as top-level modules; under pytest, `conftest.py` does this for
you.

### Regenerate results

All scripts default to deterministic seeds (seed 42) so a rerun reproduces the
frozen outputs within declared precision.

```bash
# Full Monte Carlo reduction + Sobol sensitivity + semi-Markov stress (writes outputs/mc_report.json)
python src/simulate_stcaf_mc.py --seed 42

# Phase-stress generator + ablation (writes outputs/phase_analysis_report.json)
python src/phase_ablation.py --seed 42 --n 100000

# Closed-form pathway probability + Layer-4 counterexample
python src/simulate_stcaf_ctmc_journal.py

# Reporting-rate asymmetry sensitivity (mu2/mu1) — supports the shared-mu discussion
python src/sensitivity_mu2_mu1.py
```

### Documented command-line options

- `--envelope {conservative,reference,wide}` — scenario envelope (MC reduction)
- `--n` / `-n` — number of draws (default 100,000 for phase stress)
- `--seed` — PRNG seed (default 42)
- `--n-base` — Sobol base sample (default 4,096)
- `--boot-reps` — bootstrap replicates (default 200)

## Outputs and manuscript mapping

All scripts write their JSON and plots to `outputs/`.

| Script | Output | Manuscript reference |
|---|---|---|
| `simulate_stcaf_mc.py` | `outputs/mc_report.json` | Paired scenario reductions, Jansen total-order indices, matched-mean semi-Markov stress; three-panel range/reference/memoryless figure |
| `phase_ablation.py` | `outputs/phase_analysis_report.json` | Layer-4 cross-phase section: counterexample, phase-stress frequencies, beta-stipulation sensitivity; phase-exposure figure |
| `simulate_stcaf_ctmc_journal.py` | (prints; writes a plot) | Closed-form absorption proposition + Layer-4 counterexample |
| `sensitivity_mu2_mu1.py` | (prints) | Reporting-rate asymmetry discussion |

### Keys inside `outputs/phase_analysis_report.json`

| Key | What it reports |
|---|---|
| `exact_condition` | the exact cross-phase comparison criterion |
| `representative_sequence` | an admissible sequence in which the Layer-4 absolute contribution rises |
| `counterexample_to_former_theorem` | an admissible sequence in which it falls |
| `monotone_stress` | frequencies over 100,000 monotone phase sequences, including `fraction_delta4_D3_greater_D2` (absolute scale) and `fraction_relative_layer4_D3_greater_D2` (multiplicative scale) |
| `beta_stipulation_sensitivity` | the same frequencies when the per-phase beta multiplier is drawn log-uniformly on [1/3, 3] (`two_sided`) or uniformly on [1/3, 1] (`mirrored`) instead of the stipulated [1, 3], at the same seed and draw order |

The `beta_stipulation_sensitivity` block exists because the stipulated
non-decreasing beta is the direction most favourable to cross-phase ordering on
the absolute scale: relaxing it lowers that frequency rather than raising it.
Neither the absolute nor the multiplicative scale orders universally, and the
manuscript claims no ordering theorem on either.

## Randomness and precision

- All draws use a repeatable PRNG seeded deterministically (default seed 42).
- Monte Carlo results are reported as medians/quantiles over declared input
  draws; these are **simulation intervals, not confidence intervals for a
  population parameter**.
- Bootstrap intervals (200 replicates) quantify estimator uncertainty in the
  Jansen indices and are reported in the manuscript table notes.

## License and citation

See `LICENSE`. When citing this code, please cite the accompanying manuscript
(author metadata will be added on acceptance).

---

*This repository contains no empirical data. It documents reproducible
analytical and assumption-stress computations only.*
