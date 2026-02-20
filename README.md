

<p align="left">
  <img src="logo2.png" width="620" title="KilonovaSCORER Logo">
</p>

# KilonovaScorer

**A Simulation-Based Scoring Framework for Early Identification of Kilonova Candidates**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]
[![License](https://img.shields.io/badge/license-MIT-green.svg)]
[![Status](https://img.shields.io/badge/status-beta-orange.svg)]

`KilonovaScorer` is an open-source statistical framework designed to **score and rank transient candidates** according to their consistency with physically motivated kilonova light-curve simulations. The package is optimized for the **early-time, sparse-data regime** encountered in multimessenger astronomy follow-up campaigns.

The framework enables rapid candidate vetting using simulation-based statistical diagnostics, making it suitable for alert-driven environments such as LSST brokers and gravitational-wave follow-up infrastructures.

---

## Scientific Motivation

Kilonovae evolve rapidly, often reaching peak brightness within a few days after merger. Under realistic survey cadences (e.g., LSST), only a small number of photometric observations are typically available before the transient fades.

In this low-data regime:

- Bayesian parameter inference becomes weakly constrained,
- model comparison is dominated by prior assumptions,
- rapid follow-up prioritization becomes challenging.

`KilonovaScorer` addresses this problem by comparing observations directly against ensembles of simulated light curves using **prior predictive checks**, producing quantitative compatibility scores suitable for real-time decision making.


## Installation

Install directly with pip:

```bash
pip install git+https://github.com/phelipedarc/KilonovaSCORER.git
```
or from source:
```bash
git clone https://github.com/phelipedarc/KilonovaSCORER
cd KilonovaSCORER
pip install -e .
```

## Core Functionality

The current beta release provides the following high-level interface:

| Function | Description |
|---|---|
| `kilonovascorer_v1` | **Main scoring engine.** Compares observational data against simulation grids and returns scoring metrics and diagnostics. |
| `load_observations` | **Data ingestion utility.** Cleans and formats CSV/JSON light-curve data for compatibility. |
| `preprocess_lsst_like` | **Cadence preprocessing.** Converts arbitrary light curves into LSST-like follow-up cadence. |
| `plot_final_all_metrics` | **Visualization module.** Produces multi-panel diagnostic plots of scoring results. |
| `binned_stats_cumulative_ptail` | **Statistical diagnostics.** Computes cumulative scores and ABC-style consistency metrics. |
| `plotting.plot_survivor_param_kde_grid` | **Visualization diagnostics.** Produces a multi-panel for each kilonova parameter constraint over time from the ABC-diagnostic. |

---




#### Project Development

This project was developed at **Northwestern University** (December 2025 – March 2026) in collaboration with **LAB-IA — Laboratório de Inteligência Artificial (CBPF)**.

- Northwestern University — https://ciera.northwestern.edu/  
- LAB-IA — https://labia.cbpf.br/

Corresponding email: phelipedarc@gmail.com

About me: https://phelipedarc.github.io/




