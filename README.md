# ST-CAF

**Socio-Technical Cybersecurity Awareness Framework**

Companion replication code for the manuscript:

> *"A Phase-Aware Socio-Technical Framework for Cybersecurity Awareness in Digital Transformation: Analytical Model and Assumption-Stress Evaluation"*

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
5. run the phase-stress simulation over three digital-transformation phases.

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
│   └── phase_ablation.py               # phase-stress generator + scenario ablation
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
PYTHONPATH=src pytest tests/ -v
```

The standalone runner executes the same 8 regression tests (analytical
invariants, numeric consistency, and output checks) and requires
NumPy only. `PYTHONPATH=src` is needed because the scripts live under `src/`
and the test imports them as top-level modules.

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
```

### Documented command-line options

- `--envelope {conservative,reference,wide}` — scenario envelope (MC reduction)
- `--n` / `-n` — number of draws (default 100,000 for phase stress)
- `--seed` — PRNG seed (default 42)
- `--n-base` — Sobol base sample (default 4,096)
- `--boot-reps` — bootstrap replicates (default 200)

## Outputs and manuscript mapping

| Script | Output | Manuscript reference |
|---|---|---|
| `simulate_stcaf_mc.py` | `mc_report.json` | Scenario sweeps §(reduction factors), Table (Jansen indices), semi-Markov stress |
| `phase_ablation.py` | `phase_analysis_report.json` | Phase-stress generator and ablation summary |
| `simulate_stcaf_ctmc_journal.py` | (prints) | Closed-form absorption proposition + Layer-4 counterexample |

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
