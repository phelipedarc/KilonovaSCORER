# KilonovaSCORER

**Rank transient candidates by how well they match kilonova simulations — using only a few early observations.**

KilonovaSCORER compares each observation of a candidate with a large set of simulated kilonova light curves. It returns two per-observation scores (`Ptail` and `Pnear`), one cumulative ranking score, and a temporal-consistency check (the ABC diagnostic).

It was built for the early, sparse-data phase of gravitational-wave follow-up (for example Rubin/LSST Target-of-Opportunity campaigns), where full Bayesian inference is not practical.

```{admonition} It ranks, it does not classify
:class: important
The score tells you *which candidates to follow up first* and should not be used as a classifier.
```

## 30-second example

```python
import KilonovaScorer as ks

obs = ks.load_observations("candidate.csv", merger_mjd=60905.06,
                           dist_mpc=259.0, dist_err_mpc=62.0)
obs["filter_mapped"] = obs["band"].str.lower().map(FILTER_LOOKUP)

metrics, _ = ks.kilonovascorer_v1(obs, sims, "my_candidate", k_near=3.0, overlap_k=1.5)
scores = ks.binned_stats_cumulative_ptail(metrics, bin_size=0.2)
ks.plot_final_all_metrics(metrics, sims, scores, candidate_name="my_candidate")
```

The full, runnable version is in the {doc}`quickstart`.

## Start here

| If you want to… | Read |
|---|---|
| Understand the idea | {doc}`introduction` |
| Install the package | {doc}`installation` |
| Get the simulation table | {doc}`simulations` |
| Score your first candidate | {doc}`quickstart` |
| Understand the numbers | {doc}`ptail`, {doc}`pnear`, {doc}`abc` |

```{toctree}
:hidden:
:caption: Learn

introduction
installation
simulations
quickstart
```

```{toctree}
:hidden:
:caption: The scores

ptail
pnear
abc
```

```{toctree}
:hidden:
:caption: Reference

Notebook walkthrough <tutorials/getting_started>
api
citing
```
