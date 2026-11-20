# Kilonova simulations

The scorer compares your data to a table of simulated kilonova light curves. You can download the table we used, or make your own.

## What is in the grid

- **Model:** the two-component ("blue" + "red") kilonova model of Villar et al. (2017), run through [Redback](https://redback.readthedocs.io).
- **Bands:** LSST `g`, `r`, `i`, `z`.
- **Time:** 0–10 days after merger, 1000 time steps (0.01 d apart).
- **Output:** absolute magnitudes, which are compared with your data.
- **Priors:** deliberately broad, so real kilonovae are not rejected early.

| Parameter | Prior (per component) |
|---|---|
| Ejecta mass $M_\mathrm{ej}$ | $U(10^{-4},\,0.1)\ M_\odot$ |
| Ejecta velocity $v_\mathrm{ej}$ | $U(0.01,\,0.7)\ c$ |
| Opacity $\kappa_1$ (blue) | $U(0.01,\,0.5)\ \mathrm{cm^2\,g^{-1}}$ |
| Opacity $\kappa_2$ (red) | $U(1,\,30)\ \mathrm{cm^2\,g^{-1}}$ |
| Temperature floor $T_\mathrm{floor}$ | $\log U(100,\,6000)\ \mathrm{K}$ |

<!-- CHECK BEFORE PUBLISHING: src/KilonovaScorer/simulation.py currently draws kappa_1 ~ U(0.1, 0.5), not U(0.01, 0.5). -->

Eight free parameters in total. The two components are added linearly, with no interaction between them.

## Option 1 — Download (recommended)

The simulations are on Zenodo: <https://zenodo.org/records/19824266> (CC BY 4.0).

| File | Size | Use it for |
|---|---|---|
| `mednumber_simulation_sample_1e3.csv` | 96 MB | Learning, tests, quick scoring (the name indicates 10³ light curves) |
| `simulations_IR_two_component_kilonova_model.csv` | 21.8 GB | Full grid. Does not fit in memory on most laptops. |

```{tip}
Start with the 96 MB file. We tested grids from $10^2$ to $10^5$ light curves and found that $10^3$ gives scores within the typical score uncertainty ($|\Delta| \lesssim 0.1$), about 100× faster than $10^5$.
```

Load it like this:

```python
import pandas as pd

sims = pd.read_csv("mednumber_simulation_sample_1e3.csv")
sims.columns
```

Each row is one light curve at one time in one band. Expect at least these columns:

| Column | Meaning |
|---|---|
| `sample_id` | Which simulated light curve (one parameter draw) |
| `band` | `lsstg`, `lsstr`, `lssti`, `lsstz` |
| `time` | Days since merger |
| `magnitude` | Apparent magnitude |
| `absolute_magnitude` | Absolute magnitude (used for scoring) |
| `mej_1`, `vej_1`, `kappa_1`, `temperature_floor_1`, … `_2` | Model parameters |

Then add the `filter_mapped` column that the scorer uses (see {doc}`quickstart`, step 1).

For the 21.8 GB file, read only what you need:

```python
cols = ["sample_id", "band", "time", "absolute_magnitude"]
chunks = pd.read_csv("simulations_IR_two_component_kilonova_model.csv",
                     usecols=cols, chunksize=5_000_000)
```

## Option 2 — Make your own (optional, needs Redback)

Install the optional tools first ({doc}`installation`). Then:

```python
from KilonovaScorer.simulation import simulate_kilonova

sims = simulate_kilonova(N_SIM=1000,
                         MODEL_NAME="two_component_kilonova_model",
                         SAVE_CSV=True)      # writes simulations_<model>.csv
```

Or from the command line, using the script in the repository:

```bash
python simulating_kne_pop.py --nsim 1000 --save_csv
```

```{warning}
Without `--save_csv` the script runs but saves nothing.
```

Things to know:

- Size grows fast. $N$ light curves × 4 bands × 1000 times gives $4000\,N$ rows (4×10⁸ rows for $N=10^5$).
- Distances are fixed in the code: absolute magnitudes assume $D_L = 259$ Mpc (the distance of S250818k).
