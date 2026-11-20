# Installation

## Install the package

```bash
pip install git+https://github.com/phelipedarc/KilonovaSCORER.git
```

Or from source, if you want to edit the code:

```bash
git clone https://github.com/phelipedarc/KilonovaSCORER
cd KilonovaSCORER
pip install -e .
```

Requires Python ≥ 3.8. Dependencies installed automatically: `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`.

## Check it works

```python
import KilonovaScorer as ks
print(ks.__version__)
```


## Optional: tools to make your own simulations

You do **not** need these to score candidates. The scorer only needs a table of simulated light curves, and you can [download one](simulations.md).

Install them only if you want to generate your own grid:

```bash
pip install redback astropy
```

`redback` pulls in several heavy dependencies and can take a while to install, so use a fresh virtual environment.

## Use a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install git+https://github.com/phelipedarc/KilonovaSCORER.git
```
