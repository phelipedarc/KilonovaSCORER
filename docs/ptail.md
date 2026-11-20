# Ptail score

**Question:** is this observation in the extreme tail of what the kilonova simulations predict?

`Ptail` is a number between 0 and 1.

- Close to **1** → the observation sits in the middle of the simulated distribution (typical).
- Close to **0** → it sits far in a tail (unlikely under the model).

It works like a two-sided tail-area probability, but it measures *plausibility under the simulations*, not statistical significance.

## How it is computed

For one observation (time $t$, band $b$, absolute magnitude $M_\mathrm{obs} \pm \sigma_\mathrm{obs}$):

**1. Collect the simulated magnitudes** in the same band, in the 0.2-day time bin around $t$. This set is the *prior predictive distribution* (PPD).

**2. Add the observation noise** to each simulated value:

$$
M_\mathrm{rep} = M_\mathrm{model} + \epsilon, \qquad \epsilon \sim \mathcal{N}(0,\sigma_\mathrm{obs}^2)
$$

This makes the comparison fair: a noisy measurement can land further from the model than a perfect one.

**3. Find where the observation falls.** With $F$ the cumulative distribution of $M_\mathrm{rep}$:

$$
F(M_\mathrm{obs}) = \Pr(M_\mathrm{rep} \le M_\mathrm{obs})
$$

**4. Take the two-sided tail area:**

$$
P_\mathrm{tail} = 2\,\min\big[F(M_\mathrm{obs}),\; 1 - F(M_\mathrm{obs})\big]
$$

**5. Get an uncertainty.** Draw 100 values of $M_\mathrm{obs}$ from $\mathcal{N}(M_\mathrm{obs},\sigma_\mathrm{obs}^2)$, recompute $P_\mathrm{tail}$ for each, and use the mean and standard deviation. These are the columns `p_tail_mean` and `p_tail_std`.

```{note}
In magnitudes, a smaller number is brighter. So $F(M_\mathrm{obs})$ is the fraction of simulated values that are *brighter than or equal to* the observation.
```

## From many observations to one score

You usually have several observations. They are combined in three steps.

**Step A — Work in logit space.** Scores near 0 or 1 have unstable errors. So each score $x_i$ is transformed first:

$$
z_i = \ln\frac{x_i}{1-x_i}, \qquad \sigma_{z_i} \approx \frac{\sigma_i}{x_i(1-x_i)}
$$

**Step B — Weight by precision.** Observations with small errors count more:

$$
\bar z = \frac{\sum_i z_i/\sigma_{z_i}^2}{\sum_i 1/\sigma_{z_i}^2}, \qquad
\sigma_{\bar z}^2 = \Big(\sum_i 1/\sigma_{z_i}^2\Big)^{-1}
$$

**Step C — Go back to a probability** with $\bar x = 1/(1+e^{-\bar z})$.

This is done first for all observations in the same 0.2-day bin, then **sequentially** over bins, so the cumulative score updates as each night arrives.

### Why logit space? A worked example

Two observations in one night: $P_\mathrm{tail} = [0.8,\ 0.01]$ with $\sigma = [0.1,\ 0.05]$.

| Method | Result |
|---|---|
| Weighted mean in probability space | $0.168 \pm 0.044$ — dominated by the tiny score with the small absolute error |
| Weighted mean in logit space | $0.785 \pm 0.104$ — the score with the small *relative* error wins |

We checked that the package reproduces this: `ivw_stats_logit` returns 0.785 ± 0.105.

## In code

```python
metrics, _ = ks.kilonovascorer_v1(obs, sims, "my_candidate", k_near=3.0, overlap_k=1.5)

# per-observation scores
metrics[["band", "obs_time", "p_tail_mean", "p_tail_std"]]

# cumulative score, one row per 0.2 d bin
scores = ks.binned_stats_cumulative_ptail(metrics, bin_size=0.2)
scores[["time_mid", "mean", "std", "running_mean", "running_std"]]
```

| Column | Meaning |
|---|---|
| `mean`, `std` | Combined score for that time bin |
| `running_mean`, `running_std` | **Cumulative score** up to and including that bin |
| `running_mean_penalized`, `running_std_penalized` | Cumulative score with the ABC penalty applied ({doc}`abc`) |

## Reading the result

- A single high `Ptail` does not make a kilonova, and a single low one does not rule it out. Look at the cumulative score and the {doc}`abc`.
- In the paper's AT 2017gfo test, the cumulative score was about 0.6 on night 1 and settled near 0.44 later.
- Results vary slightly between runs because the scorer draws random samples. Call `np.random.seed(0)` before scoring if you need identical numbers.

```{note}
If every observation in a time bin has `Ptail` exactly 0 (a point far outside all simulations), that bin carries no weight in the cumulative score. It does not crash, and the {doc}`abc` is what rejects such a candidate.
```
