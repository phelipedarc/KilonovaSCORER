# Introduction

## The problem

A kilonova fades within days. During a gravitational-wave follow-up you may have tens to hundreds of candidates, and each one has only a handful of measurements. With so little data, fitting a full model to every candidate gives weak, prior-dominated answers. You still have to decide *now* which candidates get telescope time.

## The idea

Simulate many kilonovae first. Then, for each observation, ask: **"Is this a plausible thing for my simulations to produce?"**

This is a *prior predictive check*. The data are never used to tune the simulations, so the method is conservative and does not over-fit.

## How it works

1. **Simulate** a large grid of kilonova light curves in absolute magnitude ({doc}`simulations`).
2. **Convert** each observed apparent magnitude to absolute magnitude, using the distance and its uncertainty.
3. **Compare** each observation with the simulations at the same time and band:
   - `Ptail` — is the observation in the extreme tail? ({doc}`ptail`)
   - `Pnear` — how many simulations lie close to it? ({doc}`pnear`)
4. **Combine** the `Ptail` values over time into one cumulative score with an error bar.
5. **Check the history** with the ABC diagnostic: does *any* simulation explain all epochs together? If none does, the score is set to zero ({doc}`abc`).

| Output | Range | Question it answers |
|---|---|---|
| `Ptail` | 0–1 | Is this single point typical of the model? |
| `Pnear` | 0–1 | How crowded is the model around this point? |
| Cumulative score | 0–1 (± error) | How consistent is the whole light curve so far? |
| Survivors $\lvert S_t\rvert$ | integer | How many simulations still fit *all* epochs? |

## Conventions

- Magnitudes are AB. Everything is compared in **absolute magnitude**.
- Time is in days since the merger (or trigger) time you provide.
- Bands are the LSST $g, r, i, z$ bands. Other filters must be mapped to these (see {doc}`quickstart`).
- Observer-frame and rest-frame bands are treated as equal. This is fine for nearby sources, and the paper reports it is a reasonable approximation for redshift $\lesssim 0.16$.

## What the paper showed

In [Darc & Kilpatrick (2026)](https://arxiv.org/abs/2604.22994):

- **AT 2017gfo** (a real kilonova) keeps a high score and the survival count never reaches zero.
- **SN 2025ulz** (a Type IIb supernova that looked like a kilonova early on) scores well at first, then the survival count hits zero when it re-brightens at about 5–6 days.
- In Rubin/LSST ToO simulations, median supernova scores fall to zero by about 3–4 days, while kilonova medians stay at about 0.4 or higher.

```{note}
Scores depend on the simulation grid you use. A transient whose physics is outside the grid can be mis-ranked, and with very few epochs a supernova can still score high.
```
