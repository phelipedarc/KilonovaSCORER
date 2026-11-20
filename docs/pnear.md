# Pnear score

**Question:** how many simulations sit close to this observation?

`Pnear` is a local density check. It is a number between 0 and 1: the fraction of the simulated distribution that falls inside a small window around the observation.

## How it is computed

Use the same noise-added simulated values $M_\mathrm{rep}$ as for {doc}`ptail`. Define a window (the *Region of Practical Equivalence*, ROPE) centred on the observation:

$$
R = \big[M_\mathrm{obs} - k_\mathrm{near}\,\sigma_\mathrm{obs},\;\; M_\mathrm{obs} + k_\mathrm{near}\,\sigma_\mathrm{obs}\big]
$$

Then:

$$
P_\mathrm{near} = \Pr\big(M_\mathrm{rep} \in R\big) \approx \frac{\#\{M_\mathrm{rep} \in R\}}{N_\mathrm{samples}}
$$

- $k_\mathrm{near}$ sets the window width. The paper uses **$k_\mathrm{near} = 3.0$**.
- A larger $k_\mathrm{near}$ gives larger `Pnear` values for every observation.

## Ptail vs Pnear

| | `Ptail` | `Pnear` |
|---|---|---|
| Looks at | Position in the whole distribution | Density right around the point |
| High means | Observation is typical | Many simulations lie nearby |
| Low means | Observation is in a tail | Observation is in a sparse region |
| Combined over time? | Yes (cumulative score) | **No**, one value per observation |
| Has an error bar? | Yes | No (the noise is already in the window and the distribution) |

They answer different questions. A point can be in the *middle* of a distribution that is very wide (high `Ptail`, low `Pnear`).

## In code

`Pnear` is computed in the same call as `Ptail`:

```python
metrics, _ = ks.kilonovascorer_v1(obs, sims, "my_candidate", k_near=3.0)
metrics[["band", "obs_time", "prob_near"]]
```

```{note}
`kilonovascorer_v1` names the column `prob_near`. `kilonovascorer_v3` returns the same values under both `p_near_KNe` and `prob_near`.
```

## Reading the result

- In the paper's AT 2017gfo test, every detection before about 2 days had $P_\mathrm{near} > 0.6$.
- The diagnostic plot draws a dashed line at 0.2, an empirical reference: below it, fewer than 20% of simulations fall inside the observation's window.
- A very low `Pnear` at one epoch is a warning. It is not, by itself, a rejection.
