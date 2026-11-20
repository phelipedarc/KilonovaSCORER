# Quickstart

Score a candidate in five steps. You need two things:

1. a **simulation table** ({doc}`simulations` — the 96 MB Zenodo sample is enough), and
2. a **photometry file** for your candidate.

## Step 1 — Load the simulations

```python
import numpy as np
import pandas as pd
import KilonovaScorer as ks

# Map every filter name you might meet to the four LSST-like bands
FILTER_LOOKUP = {
    "lsstg": "g-band", "g-ztf": "g-band", "ztfg": "g-band", "g-p1": "g-band", "g": "g-band",
    "lsstr": "r-band", "r-ztf": "r-band", "ztfr": "r-band", "r-p1": "r-band", "r": "r-band",
    "lssti": "i-band", "i-ztf": "i-band", "ztfi": "i-band", "i-p1": "i-band", "i": "i-band",
    "lsstz": "z-band", "z-ztf": "z-band", "ztfz": "z-band", "z-p1": "z-band", "z": "z-band",
}

sims = pd.read_csv("mednumber_simulation_sample_1e3.csv")
sims["filter_mapped"] = sims["band"].str.lower().map(FILTER_LOOKUP)
```

The scorer looks for a column called `filter_mapped`. Rows whose band is not in the dictionary get `NaN` and are ignored.

You can plot the grid to check it loaded correctly:

```python
ks.plot_simulations_LCS(sims)
```

## Step 2 — Load your observations

Your file can be a `.csv` or a `.json`.

**CSV** needs these columns:

| Column | Meaning |
|---|---|
| `time` | Observation time (MJD) |
| `magnitude` | Apparent AB magnitude |
| `e_magnitude` | Magnitude error |
| `band` | Filter name, e.g. `g`, `ztfr`, `lssti` |

Then give the merger time and the distance from the gravitational-wave alert:

```python
MERGER_MJD = 60905.055961     # merger/trigger time (MJD)
DL_MPC     = 259.0            # luminosity distance (Mpc)
DL_ERR_MPC = 62.0             # its uncertainty (Mpc)

obs = ks.load_observations("my_candidate.csv",
                           merger_mjd=MERGER_MJD,
                           dist_mpc=DL_MPC,
                           dist_err_mpc=DL_ERR_MPC)
obs["filter_mapped"] = obs["band"].str.lower().map(FILTER_LOOKUP)
```

This adds `time_after_gw`, `absolute_magnitude` and `absolute_magnitude_error`. The distance and photometric errors are both folded into the absolute-magnitude error.

```{tip}
Very dense light curves (like AT 2017gfo) are unrealistic for a real search. To mimic a survey with one visit per night and band, thin them first:

`obs = ks.preprocess_lsst_like(obs, bands=("g-band", "r-band"), strategy="snr")`
```

## Step 3 — Score

```python
np.random.seed(0)     # optional: makes the run repeatable

metrics, overlap = ks.kilonovascorer_v3(
    obs, sims, "my_candidate",
    time_bin_width=0.2,                                  # days
    band_list=("g-band", "r-band", "i-band", "z-band"),
    k_near=3.0,                                          # Pnear window (paper value)
    n_kde_sim=5000,                                      # samples; 50000 is slower but smoother
    min_sim_points=20,
    overlap_k=1.5,                                       # ABC window (paper value)
)
```

`metrics` has one row per observation: `p_tail_mean`, `p_tail_std`, `prob_near`, `n_consistent_lcs`, `running_survivors_n`, and more.

## Step 4 — Get the cumulative score

```python
scores = ks.binned_stats_cumulative_ptail(metrics, bin_size=0.2)
scores[["time_mid", "running_mean", "running_mean_penalized", "t_collapse"]]
```

`running_mean` ± `running_std` is the cumulative score. `running_mean_penalized` is the same score set to 0 once no simulation fits all epochs (`t_collapse`). See {doc}`ptail` and {doc}`abc`.

## Step 5 — Plot the diagnostic report

```python
ks.plot_final_all_metrics(metrics, sims, scores, candidate_name="my_candidate")
```

The four panels are:

| Panel | Shows |
|---|---|
| a (top) | Light curve over the surviving simulations, coloured by `Pnear` |
| b | `Pnear` for each detection |
| c | ABC survival fraction (a red line marks where it reaches zero) |
| d (bottom) | `Ptail` per bin and the cumulative score |

Optional: which model parameters survive?

```python
ks.plotting.plot_survivor_param_kde_grid(
    metrics, sims,
    params=("mej_1", "vej_1", "kappa_1", "temperature_floor_1",
            "mej_2", "vej_2", "kappa_2", "temperature_floor_2"))
```

## Common problems

| Symptom | Likely cause |
|---|---|
| `metrics` is empty | `filter_mapped` is `NaN` (band not in the dictionary), or `band_list` does not match the names `g-band`, `r-band`, … |
| Fewer rows than observations | A time bin had fewer than `min_sim_points` simulations, or an error was missing/zero |
| Different numbers each run | Random sampling; set `np.random.seed(...)` |

For the full walkthrough with real data (SN 2025ulz and AT 2017gfo) see the {doc}`tutorials/getting_started` notebook.
