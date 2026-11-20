# ABC diagnostic

**Question:** is there *any* simulated kilonova that explains **all** the observations at once?

## Why we need it

`Ptail` and `Pnear` look at one observation at a time. A supernova can pass every single-epoch test and still behave in a way no kilonova does over time.

**Example — SN 2025ulz.** For its first ~4 days it looked like a kilonova. Then it re-brightened in the $i$ band (about 4–6 days), which kilonovae do not do. Each re-brightening point still had a non-zero score on its own. Only the history-aware check caught it.

## How it works

1. For each observation, **accept** every simulated light curve that comes within $k_\mathrm{ABC}\,\sigma_\mathrm{obs}$ of it in that time bin:

$$
\lvert M_\mathrm{sim} - M_\mathrm{obs}\rvert \le k_\mathrm{ABC}\,\sigma_\mathrm{obs}
$$

   A light curve is accepted if **any** of its points in the bin falls inside this window (a deliberately generous rule).

2. **Intersect** the accepted sets over time. The *survivors* $S_t$ are the light curves that were accepted at **every** observation up to time $t$:

$$
S_t = S_{t-1} \,\cap\, \{\text{accepted at } t\}
$$

   The number of survivors can only go down.

3. Track the **survival fraction**, comparing the survivors with the number accepted at that epoch alone:

$$
f_\mathrm{surv}(t) = \frac{\lvert S_t\rvert}{\lvert A_t\rvert + \epsilon}
$$

## What it tells you

| What you see | What it means |
|---|---|
| $\lvert S_t\rvert$ falls then levels off | Normal for a kilonova: data narrow down the models, but some always fit |
| $f_\mathrm{surv}$ close to 1 | The new point agrees with the whole earlier history |
| $\lvert S_t\rvert = 0$ | **No** simulation explains all epochs. The candidate is flagged as temporally inconsistent |

When $\lvert S_t\rvert = 0$, the cumulative score is **set to zero** from that point on, whatever the individual `Ptail` values are. This stops a chance match at one epoch from keeping a bad candidate high in the ranking.

## Choosing $k_\mathrm{ABC}$

- The paper uses **$k_\mathrm{ABC} = 1.5$**. In the code this argument is `overlap_k`.
- A smaller simulation grid has fewer light curves, so fewer pass a tight window. For AT 2017gfo the smallest value that did not collapse was 1.5 for $10^5$ simulations, 2.0 for $10^4$, 2.5 for $10^3$ and 2.0 for $10^2$.
- **Rule of thumb:** if you use a small grid, try a larger `overlap_k` and check that a known kilonova does not collapse.

The defaults are `k_near=3.0` and `overlap_k=1.5`, the paper's values.

## In code

`kilonovascorer_v1` stores the ABC results for each observation:

| Column | Meaning |
|---|---|
| `consistent_ids` | IDs of simulations accepted at this observation |
| `n_consistent_lcs` | How many were accepted ($\lvert A_t\rvert$) |
| `running_survivors_n` | Survivors so far, **within this band only** |
| `running_survivors_ids` | Their IDs |

Plots (`plot_final_all_metrics`) combine **all bands** in time order to find the survivors across the whole light curve.

### The zero-score penalty

`binned_stats_cumulative_ptail` returns both versions of the cumulative score:

| Column | Meaning |
|---|---|
| `running_mean`, `running_std` | Cumulative score **without** the penalty |
| `running_mean_penalized`, `running_std_penalized` | Same, but set to 0 from the bin that contains the collapse onward |
| `t_collapse` | Time (days) when the survivors reach zero, across all bands. `NaN` if they never do |

```python
scores = ks.binned_stats_cumulative_ptail(metrics, bin_size=0.2)
scores[["time_mid", "running_mean_penalized", "t_collapse"]]
```

`plot_final_all_metrics` draws both curves.

### See which models survive

```python
ks.plotting.plot_survivor_param_kde_grid(
    metrics, sims,
    params=("mej_1", "vej_1", "kappa_1", "temperature_floor_1",
            "mej_2", "vej_2", "kappa_2", "temperature_floor_2"),
)
```

This shows, epoch by epoch, which ejecta masses, velocities, opacities and temperatures are still allowed.

## Limits

- It cannot reject what it cannot see. If observations stop before the supernova diverges (SN 2025ulz before ~2.5 days), the score stays high.
- If the true transient is outside your simulation grid, it can collapse even though it is a real kilonova.
- Use it to *rank*, not as a standalone reject rule.
