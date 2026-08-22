"""
core.py — KilonovaScorer core pipeline.

Implements:
  - JSON / CSV photometry loading with absolute magnitude computation
  - LSST-like cadence downsampling
  - P_tail_KNe and P_near_KNe scoring on the noise-convolved PPD
    (predictive_tail_kde), by exact error-function sums or by the original
    KDE Monte Carlo, selected with ``p_tail_method``
  - Optional conditioning of the reference on detectability, selected with
    ``M_lim``
  - ABC sequential survival diagnostic (overlap_chain), skippable with
    ``abc_compute``
  - Cumulative scoring by Stouffer or Brown p-value combination, or by the
    original logit-space inverse-variance weighted mean, selected with
    ``method`` (binned_stats_cumulative_ptail)

Every change made to the scoring since the paper is SELECTABLE rather than
substituted: the defaults are the new behaviour, and the previous behaviour is
reproduced exactly by

    kilonovascorer_v3(..., p_tail_method="montecarlo", M_lim=None)
    binned_stats_cumulative_ptail(..., method="ivw")

Scoring in FLUX rather than absolute magnitude is designed and measured in
REPORT.md Parts VIII-IX but is deliberately NOT implemented here; it belongs on
its own branch.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
from .utils import *  # noqa: F401,F403  (decorators and helpers)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
#
# EXPLICIT, because ``KilonovaScorer/__init__.py`` does ``from .core2 import *``
# and several names here collide with older ones in ``core.py``.  Without an
# ``__all__`` the star-import re-exported every public name in this module's
# namespace — including everything pulled in by ``from .utils import *`` — and
# which definition of ``binned_stats_cumulative_ptail`` the package exposed
# depended on nothing more than the ORDER of the two import lines in
# ``__init__.py``.  Reordering them would have silently reverted the whole
# pipeline from Stouffer p-value combination back to the legacy IVW mean.
#
# The ``utils`` helpers are listed here deliberately: they were already part of
# the package's public surface via the transitive star-import, so omitting them
# would be a silent API removal rather than a tidy-up.
__all__ = [
    # --- core2's own API ---
    "P_TAIL_METHODS",
    "arcade_progress_bar",
    "parse_json_photometry",
    "load_observations",
    "preprocess_lsst_like",
    "predictive_tail_kde",
    "compute_consistent_ids_anyhit",
    "overlap_chain",
    "binned_stats_cumulative_ptail",
    "estimate_rho",
    "combined_score_marginalised",
    "kilonovascorer_v3",
    # --- re-exported from .utils (previously transitive; kept explicit) ---
    "compute_abs_mag_samples",
    "ivw_stats_logit",
    "stouffer_combine",
    "brown_combine",
    "stouffer_stats",
    "calculate_sequential_score_stouffer",
    "calculate_sequential_score_logit",
    "timer_warp",
    "time_plot",
]

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def arcade_progress_bar(current: int, total: int, bar_length: int = 30) -> None:
    """Print an arcade-style progress bar to stdout."""
    percent = current / total
    filled = int(bar_length * percent)
    bar = "█" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\r[ {bar} ] {percent * 100:6.2f}% ⬛")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Photometry loading
# ---------------------------------------------------------------------------

def parse_json_photometry(file_path: Path, merger_mjd: float) -> pd.DataFrame:
    """
    Extract photometry from a JSON file following the standard schema.

    Returns a DataFrame with raw band names ready for FILTER_LOOKUP mapping.
    Pre-merger timestamps and upper limits are excluded.

    Parameters
    ----------
    file_path : Path
        Path to the JSON photometry file.
    merger_mjd : float
        MJD of the GW merger event; observations before this are discarded.

    Returns
    -------
    pd.DataFrame
        Columns: time, time_after_gw, magnitude, e_magnitude, band,
        instrument, telescope.  Empty DataFrame on parse failure.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from %s", file_path)
        return pd.DataFrame()

    if "photometry" not in data:
        logger.warning("No 'photometry' key found in %s", file_path)
        return pd.DataFrame()

    records = []
    for entry in data["photometry"]:
        # 1. Validate timestamp
        t = entry.get("timestamp")
        if t is None or t < merger_mjd:
            continue

        # 2. Extract nested magnitude / filter
        val = entry.get("value", {})
        app_mag = val.get("magnitude")
        app_err = val.get("error", 0)
        raw_filter = val.get("filter")

        # 3. Quality control — skip upper limits and missing data
        if app_mag is None or raw_filter is None or val.get("upper_limit", False):
            continue

        # 4. Append standardised record
        # "band" kept as raw string (e.g. 'ztfg') for downstream FILTER_LOOKUP.
        records.append({
            "time": t,
            "time_after_gw": t - merger_mjd,
            "magnitude": float(app_mag),
            "e_magnitude": float(app_err),
            "band": str(raw_filter).lower().strip(),
            "instrument": entry.get("instrument", "unknown"),
            "telescope": entry.get("telescope", "unknown"),
        })

    return pd.DataFrame(records)


def load_observations(
    file_path,
    merger_mjd: float,
    dist_mpc: float,
    dist_err_mpc: float,
) -> pd.DataFrame:
    """
    Load and standardise photometric observations, then compute absolute magnitudes.

    Supports .csv and .json input files.  Absolute magnitudes are derived via
    ``compute_abs_mag_samples`` (from utils), which is expected to accept
    array inputs for vectorised computation.

    Parameters
    ----------
    file_path : str or Path
        Path to the photometry file (.csv or .json).
    merger_mjd : float
        MJD of the GW merger event.
    dist_mpc : float
        Luminosity distance in Mpc.
    dist_err_mpc : float
        Uncertainty on the luminosity distance in Mpc.

    Returns
    -------
    pd.DataFrame
        Standardised DataFrame including ``absolute_magnitude`` and
        ``absolute_magnitude_error`` columns.
    """
    path = Path(file_path)
    logger.info("Loading observations from %s", path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        df["time_after_gw"] = df["time"] - merger_mjd
    elif suffix == ".json":
        df = parse_json_photometry(path, merger_mjd)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if df.empty:
        logger.warning("No valid observations loaded from %s", path)
        return df

    # Vectorised absolute-magnitude computation — compute_abs_mag_samples must
    # accept 1-D arrays and return (abs_mag_array, abs_err_array).
    abs_mag, abs_err, abs_err_phot, sigma_mu = compute_abs_mag_samples(  # noqa: F821
        df["magnitude"].to_numpy(),
        df["e_magnitude"].to_numpy(),
        dist_mpc=dist_mpc,
        dist_err_mpc=dist_err_mpc,
        return_components=True,
    )
    df["absolute_magnitude"] = abs_mag
    df["absolute_magnitude_error"] = abs_err
    # The two terms behind that total, kept separate because they behave
    # differently: the photometric one is independent per row, the distance one
    # is a single draw shared by every row of this candidate.  Nothing reads
    # these by default -- `absolute_magnitude_error` is unchanged and is still
    # what the scorer uses -- but they are what `sigma_col=` and `estimate_rho`
    # need in order to treat the systematic as systematic.  See REPORT.md Part X.
    df["absolute_magnitude_error_phot"] = abs_err_phot
    df["distance_modulus_error"] = sigma_mu

    return df


# ---------------------------------------------------------------------------
# LSST-like cadence downsampling
# ---------------------------------------------------------------------------

def preprocess_lsst_like(
    data_obs: pd.DataFrame,
    bands: Tuple[str, ...] = ("g-band", "z-band"),
    time_col: str = "time_after_gw",
    band_col: str = "filter_mapped",
    strategy: str = "earliest",
) -> pd.DataFrame:
    """
    Downsample high-cadence data to a standard LSST-like survey cadence.

    Retains at most one observation per (night, band) pair, making
    over-sampled events (e.g. AT2017gfo) comparable to typical KN candidates.

    Parameters
    ----------
    data_obs : pd.DataFrame
        Raw observational data with time and filter columns.
    bands : tuple of str
        Filters to retain.
    time_col : str
        Column name for time since merger (days).
    band_col : str
        Column name for the mapped photometric band.
    strategy : {'earliest', 'snr', 'random'}
        Selection rule when multiple observations fall on the same night:
        - ``'earliest'``: smallest timestamp.
        - ``'snr'``: highest signal-to-noise ratio (1 / e_magnitude).
        - ``'random'``: random draw (seed 42 for reproducibility).

    Returns
    -------
    pd.DataFrame
        Downsampled observations sorted by ``time_col``.
    """
    df = data_obs[data_obs[band_col].isin(bands)].copy()
    df["day"] = np.floor(df[time_col]).astype(int)
    df = df.sort_values(time_col)

    if strategy == "earliest":
        df_out = df.groupby(["day", band_col], as_index=False).first()

    elif strategy == "snr":
        df["snr"] = 1.0 / df["e_magnitude"]
        df_out = (
            df.sort_values("snr", ascending=False)
            .groupby(["day", band_col], as_index=False)
            .first()
        )
        df_out = df_out.drop(columns="snr")

    elif strategy == "random":
        df_out = df.groupby(["day", band_col], as_index=False).sample(
            n=1, random_state=42
        )

    else:
        raise ValueError("strategy must be 'earliest', 'snr', or 'random'")

    return df_out.sort_values(time_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core scoring: P_tail_KNe and P_near_KNe
# ---------------------------------------------------------------------------

#: Estimators available for the P_tail / P_near integrals.
#:
#: ``closed_form`` evaluates both as exact sums of error functions over the
#: simulations.  ``montecarlo`` is the ORIGINAL implementation -- fit a Gaussian
#: KDE, resample it, add observational noise, and count -- retained verbatim so
#: that results published against it remain reproducible and so that the two can
#: be compared on the same data.  See :func:`predictive_tail_kde`.
P_TAIL_METHODS = ("closed_form", "montecarlo")

#: Historical / convenience spellings.  ``kde`` is accepted because that is what
#: this function is named after.
_P_TAIL_ALIASES = {
    "closed-form": "closed_form",
    "closedform": "closed_form",
    "exact": "closed_form",
    "mc": "montecarlo",
    "monte_carlo": "montecarlo",
    "monte-carlo": "montecarlo",
    "kde": "montecarlo",
    "legacy": "montecarlo",
}


def _resolve_p_tail_method(p_tail_method: str) -> str:
    """Normalise and validate a ``p_tail_method`` argument."""
    key = str(p_tail_method).strip().lower()
    key = _P_TAIL_ALIASES.get(key, key)
    if key not in P_TAIL_METHODS:
        raise ValueError(
            "p_tail_method must be one of %r (aliases: %r); got %r."
            % (P_TAIL_METHODS, sorted(_P_TAIL_ALIASES), p_tail_method)
        )
    return key


def predictive_tail_kde(
    sim_values: np.ndarray,
    M_obs: float,
    sigma_obs: float,
    k: float = 1.5,
    n_sim: int = 50000,
    n_obs: int = 100,
    kde: Optional[gaussian_kde] = None,
    M_lim: Optional[float] = None,
    min_n_eff: float = 20.0,
    p_tail_method: str = "closed_form",
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute P_tail_KNe and P_near_KNe from the noise-convolved prior predictive
    distribution (PPD).

    Implements the two-sided tail-area probability (paper eq. 2)::

        F(M_obs) = Pr(M_rep <= M_obs)
        P_tail_KNe = 2 * min(F(M_obs), 1 - F(M_obs))

    and the ROPE-based local consistency score (paper eq. 4)::

        P_near_KNe = Pr(M_rep in [M_obs - k*sigma_obs, M_obs + k*sigma_obs])

    TWO ESTIMATORS, SELECTED BY ``p_tail_method``
    ---------------------------------------------
    Both target the same integrals.  Neither is removed; the default changed.

    ``p_tail_method="closed_form"`` (default).  The noise-convolved PPD is a
    finite Gaussian mixture with one component per simulation, so its CDF is the
    mixture of the component CDFs and both integrals are EXACT sums::

        F(M_obs)   = (1/N) sum_i Phi((M_obs - m_i) / sigma_obs)
        P_near_KNe = (1/N) sum_i [ Phi((M_obs + k*sigma_obs - m_i) / sigma_obs)
                                 - Phi((M_obs - k*sigma_obs - m_i) / sigma_obs) ]

    No KDE is fitted, no samples are drawn, the result is deterministic, and
    ``sigma_obs`` enters once.

    ``p_tail_method="montecarlo"``.  The original implementation, unchanged::

        X* ~ KDE(sim_values);  Y = X* + eps,  eps ~ N(0, sigma_obs^2)
        F_hat = fraction of the n_sim draws Y with Y <= M_obs

    with ``p_tail_mean`` / ``p_tail_std`` taken as the mean and standard
    deviation of P_tail over ``n_obs`` realisations of M_obs drawn from
    ``N(M_obs, sigma_obs^2)``.  Two consequences are worth knowing before
    selecting it, since they are why it is no longer the default:

    1. It applies ``sigma_obs`` TWICE.  The jitter loop perturbs M_obs by the
       same sigma already convolved into the reference, so the comparison is
       effectively against a population of width ``sqrt(2) * sigma_obs``.
    2. It is stochastic.  Two calls on identical inputs return different
       numbers unless ``random_state`` is set.

    It is retained because published results were produced with it, because it
    is the only way to reproduce them, and because it is the reference the
    closed form is checked against.

    ``p_tail_std`` MEANS DIFFERENT THINGS UNDER THE TWO
    ---------------------------------------------------
    Under ``closed_form`` it is the finite-grid standard error of F_hat -- the
    only quantity here that is genuinely uncertain.  It falls with the EFFECTIVE
    number of simulations, those surviving the ``M_lim`` detection cut, via the
    ``sum(w)`` denominator of the ratio-estimator form.  A shallow epoch
    therefore carries a larger error than a deep one at the same raw grid size.

    Under ``montecarlo`` it is the spread of P_tail over jittered M_obs, which
    is a function of where the observation sits on the PPD rather than of how
    well the PPD is resolved, and does not fall as the grid grows.

    Under the default Stouffer combiner ``p_tail_std`` is a REPORTED DIAGNOSTIC
    ONLY and is not read by the scoring path -- see ``stouffer_combine`` in
    utils.py.  It sets the weights solely on the legacy ``method='ivw'`` branch,
    where it was measurably harmful.

    M_lim CONDITIONING APPLIES TO BOTH
    ----------------------------------
    Supplying ``M_lim`` conditions the reference on being detectable, and is
    independent of the estimator choice.  Under ``closed_form`` that is the
    ratio ``sum_i phi_i / sum_i w_i`` with ``w_i = Phi((M_lim - m_i)/sigma_obs)``
    the probability that simulation ``i`` would be detected; under
    ``montecarlo`` it is the same estimator evaluated by keeping only those
    draws Y with ``Y <= M_lim``.  ``M_lim=None`` leaves the reference
    unconditioned under either.

    The numerator is ``phi_i`` rather than ``Pr(event AND detected)`` because in
    MAGNITUDE the event implies detection: a source brighter than a detected
    ``M_obs`` is brighter than ``M_lim``, so the intersection is the event
    itself.  That identity is specific to magnitude and does NOT survive a change
    to flux, where "fainter than observed" includes the whole undetectable tail
    -- see REPORT.md Part IX section 44 before porting this.

    P_near_KNe is a *local*, per-observation score and is intentionally not
    aggregated across bands or epochs (paper Section 2).  Only P_tail_KNe feeds
    into the cumulative score.  It is computed on the UNCONDITIONED PPD under
    both estimators.

    Parameters
    ----------
    sim_values : np.ndarray
        Simulated absolute magnitudes from the PPD for the relevant time bin.
    M_obs : float
        Observed absolute magnitude (paper notation: M_obs).
    sigma_obs : float
        Observational uncertainty on M_obs (paper notation: sigma_obs).
    k : float
        ROPE half-width factor.  Paper fiducial value: 1.5.
    n_sim : int
        Number of Monte Carlo draws for the noise-convolved PPD.  Used only when
        ``p_tail_method="montecarlo"``; ignored by the closed form, which is
        exact.
    n_obs : int
        Number of M_obs realisations for the P_tail_KNe uncertainty estimate.
        Paper value: N_obs = 100.  Used only when
        ``p_tail_method="montecarlo"``.
    kde : gaussian_kde or None
        Pre-fitted KDE, to avoid redundant fitting when several observations
        share a simulation time bin.  Used only when
        ``p_tail_method="montecarlo"``; the closed form needs no fitted KDE.
    M_lim : float or None
        Limiting ABSOLUTE magnitude.  ``None`` (default) leaves the reference
        unconditioned.
    min_n_eff : float
        Minimum effective reference size below which the observation is
        reported as unscoreable.  Applied only when ``M_lim`` is supplied.
    p_tail_method : {"closed_form", "montecarlo"}
        Estimator, as above.  ``"kde"`` and ``"legacy"`` are accepted aliases
        for ``"montecarlo"``.
    random_state : int or None
        Seed for the Monte Carlo path.  ``None`` (default) uses NumPy's global
        random state, reproducing the original behaviour exactly.  Ignored by
        the closed form, which is already deterministic.

    Returns
    -------
    dict with keys:
        F_hat        - CDF F(M_obs) under the noise-convolved PPD; exact under
                       the closed form, empirical under Monte Carlo.
        p_tail_KNe   - two-sided tail probability at M_obs.
        p_tail_mean  - closed form: identical to p_tail_KNe.  Monte Carlo: mean
                       of P_tail over the n_obs M_obs realisations.
        p_tail_std   - closed form: finite-grid standard error of p_tail_KNe,
                       2*se(F_hat) with se(F_hat) = sqrt(sum (phi - F w)^2) /
                       sum(w), floored at 1/n_eff.  Monte Carlo: standard
                       deviation of P_tail over the n_obs realisations.
        p_near_KNe   - ROPE-based local consistency score P_near_KNe.
        n_eff        - effective size of the reference actually used: N when
                       unconditioned, and the Kish effective size of the
                       detection weights when an M_lim is in force.  Computed
                       identically under both estimators.
        scoreable    - False when an M_lim was requested but left too little
                       reference to score against; the p_tail keys are NaN.
        p_tail_method- the estimator actually used, normalised.

    Raises
    ------
    ValueError
        If ``sim_values`` is empty, ``sigma_obs`` is non-positive, or
        ``p_tail_method`` is not recognised.
    """
    sim_values = np.asarray(sim_values)
    if sim_values.size == 0:
        raise ValueError("sim_values cannot be empty.")
    if sigma_obs <= 0:
        raise ValueError("sigma_obs must be positive.")

    p_tail_method = _resolve_p_tail_method(p_tail_method)

    m = np.asarray(sim_values, dtype=float)
    s = float(sigma_obs)

    # Selection conditioning is orthogonal to the estimator: both branches
    # honour it, and both fall back to the unconditioned form when M_lim is
    # absent or leaves nothing behind.
    use_limit = M_lim is not None and np.isfinite(M_lim)
    limit_requested = use_limit
    limit_degenerate = False

    if p_tail_method == "closed_form":
        phi = ndtr((M_obs - m) / s)          # per-simulation component CDFs

        n_eff = float(m.size)
        if use_limit:
            w = ndtr((float(M_lim) - m) / s)      # Pr(simulation i is detectable)
            w_sum = float(w.sum())
            if w_sum > 0.0:
                F_hat = float(np.clip(phi.sum() / w_sum, 0.0, 1.0))
                n_eff = float(w_sum ** 2 / np.sum(w ** 2))
            else:
                limit_degenerate = True
                n_eff = 0.0
                use_limit = False
        if not use_limit:
            F_hat = float(phi.mean())

        # P_tail_KNe - two-sided tail probability at M_obs (paper eq. 7)
        p_tail_KNe = 2.0 * min(F_hat, 1.0 - F_hat)
        p_tail_mean = p_tail_KNe

        # --- finite-grid standard error of F_hat ----------------------------
        # Spelled in the ratio-estimator form  sqrt(sum (phi - F w)^2) / sum(w),
        # which makes the EFFECTIVE-N scaling explicit.  The denominator is
        # sum(w), the detection-weighted grid size, so a shallow epoch whose
        # M_lim cut leaves few detectable simulations gets a correspondingly
        # larger error.
        #
        # This is the SAME estimator as the previous spelling,
        #     sd((phi - F w) / mean(w), ddof=1) / sqrt(n_grid),
        # not a change of definition: the 1/mean(w) inside the residual and the
        # 1/sqrt(n_grid) outside combine to exactly 1/sum(w), and the two agree
        # to the Bessel factor sqrt(n/(n-1)) alone -- 1.7e-2 at n = 30, falling
        # as 1/n.  It is rewritten because the old form divided by the RAW grid
        # count and so read as though the selection cut had been ignored.  It
        # had not been: measured on 210 real scoreable epoch-band rows,
        # p_tail_std already tracked n_eff, with the per-epoch median rising
        # 0.0248 -> 0.0309 -> 0.0396 -> 0.0400 as median n_eff fell 937 -> 610
        # -> 251 -> 145, and all 210 values distinct.  Dividing by sqrt(n_eff)
        # INSTEAD of sqrt(n_grid) would double-count the correction, inflating
        # the error by sqrt(n_grid/n_eff) (1.7x at M_lim = -14.5).
        n_grid = m.size
        if n_grid > 1:
            if use_limit:
                R_raw = float(phi.sum() / w.sum())
                se_F = float(np.sqrt(np.sum((phi - R_raw * w) ** 2)) / w.sum())
            else:
                se_F = float(np.sqrt(np.sum((phi - F_hat) ** 2)) / n_grid)
        else:
            se_F = 0.0

        # Resolution floor.  When F_hat saturates at 0 or 1 every residual
        # vanishes and the sample formula returns EXACTLY zero -- an assertion
        # of infinite precision at the one point where the estimate is least
        # trustworthy.  A reference of n_eff effective draws cannot resolve a
        # probability finer than 1/n_eff, so that is the floor.  It binds only
        # near saturation; elsewhere the sampling term is an order of magnitude
        # larger.
        if n_eff > 0.0:
            se_F = max(se_F, 1.0 / float(n_eff))

        # P_tail is a probability; its standard error cannot exceed the range.
        p_tail_std = float(min(2.0 * se_F, 1.0))

        # P_near_KNe - ROPE mass, likewise a difference of two mixture CDFs
        # (paper eq. 4; k=1.5 fiducial).  Not aggregated across epochs.
        half = k * s
        p_near_KNe = float(
            (ndtr((M_obs + half - m) / s) - ndtr((M_obs - half - m) / s)).mean()
        )

    else:  # p_tail_method == "montecarlo" - the original estimator, unchanged
        # NumPy's global random state when unseeded, which is what the original
        # used; a dedicated Generator when a seed is given.
        rng = np.random if random_state is None else np.random.default_rng(random_state)

        # 1. Noise-convolved PPD:  Y = X* + eps
        if kde is None:
            kde = gaussian_kde(m)
        n_draw = int(n_sim)
        x_star = kde.resample(n_draw, seed=random_state)[0]
        y_dist = x_star + rng.normal(0.0, s, size=n_draw)

        # 2. Detection conditioning, if requested.  The Monte Carlo counterpart
        #    of the sum(w) denominator: keep only draws that would have been
        #    detected, then take the tail fraction among those.
        #
        #    n_eff, however, is computed from the SAME Kish formula as the
        #    closed form, not from the survivor count.  Effective sample size
        #    is a property of the selection, not of how the integral is
        #    evaluated, and the survivor count estimates sum(w) rather than
        #    Kish's (sum w)^2 / sum w^2 -- a different quantity whenever the
        #    weights are diffuse (1311 against 1519 on a 4000-simulation grid
        #    at M_lim = -15.8).  Deriving it the same way both ways is what
        #    makes a single min_n_eff threshold mean one thing.
        y_ref = y_dist
        n_eff = float(m.size)
        if use_limit:
            w = ndtr((float(M_lim) - m) / s)   # Pr(simulation i is detectable)
            w_sum = float(w.sum())
            keep = y_dist <= float(M_lim)
            n_keep = int(np.count_nonzero(keep))
            if w_sum > 0.0 and n_keep > 0:
                y_ref = y_dist[keep]
                n_eff = float(w_sum ** 2 / np.sum(w ** 2))
            else:
                limit_degenerate = True
                n_eff = 0.0
                use_limit = False

        # 3. F_hat and the two-sided tail (paper eq. 2)
        F_hat = float(np.mean(y_ref <= M_obs))
        p_tail_KNe = 2.0 * min(F_hat, 1.0 - F_hat)

        # 4. P_near_KNe - ROPE mass on the unconditioned draws (paper eq. 4)
        p_near_KNe = float(np.mean(np.abs(y_dist - M_obs) <= k * s))

        # 5. Uncertainty by resampling M_obs itself.  This is the second
        #    application of sigma_obs noted above; it is preserved because it is
        #    what the original did.
        M_obs_samples = rng.normal(M_obs, s, size=int(n_obs))
        F_hat_samples = (y_ref <= M_obs_samples[:, np.newaxis]).mean(axis=1)
        p_tail_samples = 2.0 * np.minimum(F_hat_samples, 1.0 - F_hat_samples)
        p_tail_mean = float(np.mean(p_tail_samples))
        p_tail_std = float(np.std(p_tail_samples))

    # --- scoreability, shared by both estimators ----------------------------
    # Only gated when a limit was actually requested: without one the reference
    # is the whole grid and there is nothing for min_n_eff to protect against.
    scoreable = True
    if limit_requested:
        scoreable = (not limit_degenerate) and n_eff >= float(min_n_eff)
    if not scoreable:
        p_tail_KNe = float("nan")
        p_tail_mean = float("nan")
        p_tail_std = float("nan")

    return {
        "F_hat": F_hat,
        "p_tail_KNe": p_tail_KNe,
        # Under the closed form this is identical to p_tail_KNe; under Monte
        # Carlo it is the jitter mean.  Kept as a separate key so the downstream
        # schema is the same either way.  Only the legacy ivw_stats_logit path
        # reads p_tail_std; the default Stouffer combiner reads p_tail_mean
        # alone.
        "p_tail_mean": p_tail_mean,
        "p_tail_std": p_tail_std,
        "p_near_KNe": p_near_KNe,
        # Effective size of the reference actually used.  Equals N when
        # unconditioned; the Kish n_eff of the detection weights when a limit
        # is supplied, under EITHER estimator.  Use this, not a raw row count,
        # to gate scorability.
        "n_eff": n_eff,
        "scoreable": bool(scoreable),
        # Which estimator produced the numbers above.
        "p_tail_method": p_tail_method,
    }


# ---------------------------------------------------------------------------
# ABC diagnostic helpers
# ---------------------------------------------------------------------------

def compute_consistent_ids_anyhit(
    sim_band: pd.DataFrame,
    bin_idx: int,
    M_obs: float,
    sigma_obs: float,
    overlap_k: float = 2.0,
    sim_bin: Optional[pd.DataFrame] = None,
    count_only: bool = False,
) -> List:
    """
    Return simulation IDs whose predicted magnitude falls within the ROPE at
    the given time bin (conservative "any-hit" criterion).

    The ROPE acceptance kernel is:
        |M_rep - M_obs| <= overlap_k * sigma_obs

    Parameters
    ----------
    sim_band : pd.DataFrame
        Simulation data for a single photometric band, with ``time_bin``,
        ``sample_id``, and ``absolute_magnitude`` columns.
    bin_idx : int
        Time-bin index to filter on.
    M_obs : float
        Observed absolute magnitude.
    sigma_obs : float
        Observational uncertainty.
    overlap_k : float
        ROPE half-width multiplier (sigma units).
    sim_bin : pd.DataFrame or None
        The rows of ``sim_band`` already restricted to ``bin_idx``.  Supply it
        when the caller has it: ``sim_band`` holds every time bin for the band,
        so selecting on ``time_bin == bin_idx`` here rescans the whole band on
        every observation -- 200,000 rows to reach 10,000 of them, once per epoch
        -- and then takes two columns out of the result.  ``kilonovascorer_v3``
        already builds exactly this frame to score against and now passes it
        through, which measured 2.4x on the diagnostic at a 10,000-sample grid
        and 1.7x at 25,000.  ``None`` (default) keeps the original
        self-contained behaviour, so existing callers are unaffected.
    count_only : bool
        Return ``len(ids)`` instead of the id list, skipping the unique/tolist
        materialisation.  Used by ``abc_return_ids=False``.

    Returns
    -------
    list
        Unique sample IDs consistent with the ROPE at this epoch.  An ``int``
        count instead when ``count_only`` is set.
    """
    if sim_bin is None:
        sim_bin = sim_band.loc[
            sim_band["time_bin"] == bin_idx, ["sample_id", "absolute_magnitude"]
        ]
    if sim_bin.empty:
        return 0 if count_only else []

    rope_half_width = overlap_k * sigma_obs
    inside = np.abs(sim_bin["absolute_magnitude"].to_numpy() - M_obs) <= rope_half_width
    ids = sim_bin["sample_id"].to_numpy()[inside]
    ids = pd.unique(ids[pd.notna(ids)])
    return len(ids) if count_only else ids.tolist()


def overlap_chain(
    ids_lists: List[List],
    times: List[float],
    return_ids: bool = True,
) -> Dict[str, Any]:
    """
    Compute the sequential ABC survival diagnostic across observations.

    For a sequence of per-observation consistent-ID sets S_1, S_2, ..., S_N,
    this function computes:

    - pairwise overlaps: S_i ∩ S_{i+1}
    - running survivors: ⋂_{j<=i} S_j  (the set S_t from the paper)

    The survival count |S_t| is monotonically non-increasing by construction.

    Parameters
    ----------
    ids_lists : list of lists
        Per-observation lists of consistent simulation IDs.
    times : list of float
        Observation timestamps (days after merger), same order as ids_lists.
    return_ids : bool
        Include the id lists in the result.  ``True`` (default) is the original
        behaviour.  ``False`` reports every COUNT and leaves the lists empty,
        which skips a ``sorted()`` over the survivor set at every epoch and, more
        importantly, stops the caller materialising them into DataFrame cells --
        one candidate at a 10,000-sample grid otherwise carries ~320,000 ids.

    Returns
    -------
    dict with keys:
        times               – sorted observation times.
        pairwise            – list of dicts with pairwise overlap info.
        survivors_over_time – list of dicts with cumulative survivors per epoch.
        final_survivors     – sorted IDs surviving all epochs.
        final_n_survivors   – count of final survivors.
    """
    order = np.argsort(times)
    times_sorted = np.asarray(times)[order]
    sets = [set(ids_lists[i]) for i in order]

    if not sets:
        return {
            "times": [],
            "pairwise": [],
            "survivors_over_time": [],
            "final_survivors": [],
            "final_n_survivors": 0,
        }

    # Initialise running intersection from first observation
    survivors = sets[0].copy()
    _ids = (lambda s: sorted(s)) if return_ids else (lambda s: [])
    survivors_over_time = [{
        "t": float(times_sorted[0]),
        "n_survivors": len(survivors),
        "survivor_ids": _ids(survivors),
    }]

    pairwise = []
    for i in range(len(sets) - 1):
        # Pairwise: S_i ∩ S_{i+1}
        inter = sets[i] & sets[i + 1]
        pairwise.append({
            "t_left": float(times_sorted[i]),
            "t_right": float(times_sorted[i + 1]),
            "n_overlap": len(inter),
            "overlap_ids": _ids(inter),
        })

        # Cumulative: S_t = S_{t-1} ∩ S_t
        survivors &= sets[i + 1]
        survivors_over_time.append({
            "t": float(times_sorted[i + 1]),
            "n_survivors": len(survivors),
            "survivor_ids": _ids(survivors),
        })

    return {
        "times": times_sorted.tolist(),
        "pairwise": pairwise,
        "survivors_over_time": survivors_over_time,
        "final_survivors": _ids(survivors),
        "final_n_survivors": len(survivors),
    }


# ---------------------------------------------------------------------------
# Logit-space cumulative P_tail_KNe scoring
# ---------------------------------------------------------------------------

def binned_stats_cumulative_ptail(
    metric_df: pd.DataFrame,
    bin_size: float = 0.2,
    method: str = "stouffer",
    weight_col: Optional[str] = None,
    rho: float = 0.0,
) -> pd.DataFrame:
    """
    Aggregate per-observation P_tail_KNe scores into time-binned cumulative scores.

    Scores are combined within each time bin, then updated sequentially across
    bins to produce a running cumulative score.  THREE COMBINERS ARE AVAILABLE
    and none has been removed; the default changed from ``"ivw"`` to
    ``"stouffer"``.

    ``method="stouffer"`` (default).  Each epoch's P_tail is treated as what it
    is under the null — a p-value, uniform on (0, 1) — and combined as a
    standardised SUM of normal scores, ``Z = sum(w z) / sqrt(sum(w^2))`` with
    ``z = Phi^-1(1 - p)``.  Z is exactly standard normal under the null for any
    fixed positive weights, so the combined score is a calibrated p-value.

    ``method="brown"``.  Moment-matched scaled chi-square (correlation-corrected
    Fisher).  Slightly better power, slightly worse calibration, and it needs
    ``rho`` — with ``rho=0`` it degenerates exactly to Fisher, which is the worst
    option under correlated epochs.  A warning is emitted in that case.

    ``method="ivw"``.  The ORIGINAL combiner: inverse-variance weighted mean in
    logit space, weights ``1/sigma_z^2`` from ``p_tail_std`` via the delta method
    (paper Section 2).  Logit-space aggregation prevents extreme scores with
    small absolute uncertainties from dominating the weighted mean — a known
    pathology of direct probability-space averaging near the [0, 1] boundaries.
    Retained so that results published against it remain reproducible; see the
    notes above ``stouffer_combine`` in utils.py for why it is no longer the
    default.

    Parameters
    ----------
    metric_df : pd.DataFrame
        Output of ``kilonovascorer``.  Must contain ``obs_time`` and
        ``p_tail_mean``; ``p_tail_std`` is required only by ``method="ivw"``.
    bin_size : float
        Width of time bins in days.  Should match the scorer's
        ``time_bin_width`` (default 0.2 d).
    method : {"stouffer", "brown", "ivw"}
        Combiner, as above.
    weight_col : str or None
        Column of per-epoch weights for the p-value combiners.  ``None``
        (default) means equal weights, which is what the power test selected.
        Any column named here must be ANCILLARY — a function of the observing
        conditions only, never of ``p_tail_mean`` or ``p_tail_std``.  Ignored by
        ``method="ivw"``, which sets its own weights.
    rho : float
        Mean inter-epoch correlation of the normal scores, used by the p-value
        combiners to avoid treating correlated epochs as independent.  ``0.0``
        (default) assumes independence.  Ignored by ``method="ivw"``.

    Returns
    -------
    pd.DataFrame
        One row per time bin with columns:
        ``time_bin``, ``time_mid``, ``mean``, ``std``,
        ``running_mean``, ``running_std``.
    """
  #modify to match the bin edges of kilonovaScorer_V3 + bin_size / 2,
  #modidy back to +  bin_size
    if method not in ("stouffer", "brown", "ivw"):
        raise ValueError("method must be 'stouffer', 'brown' or 'ivw'.")
    if method == "brown" and not rho:
        logger.warning(
            "method='brown' with rho=0 reduces exactly to Fisher, which is the "
            "worst-calibrated option under correlated epochs. Supply rho."
        )

    bin_edges = np.arange(
        metric_df["obs_time"].min() - bin_size / 2,
        metric_df["obs_time"].max() + bin_size ,
        bin_size,
    )
    metric_df = metric_df.copy()
    metric_df["time_bin"] = pd.cut(metric_df["obs_time"], bins=bin_edges)

    if method == "ivw":
        # Retained for comparison and backwards compatibility only.  See the
        # notes above stouffer_combine in utils.py for why it is not the default.
        binned_stats = (
            metric_df.groupby("time_bin", observed=True)
            .apply(ivw_stats_logit)  # noqa: F821 (from utils.*)
            .reset_index()
        )
        binned_stats["time_mid"] = binned_stats["time_bin"].apply(lambda x: x.mid)
        binned_stats = binned_stats.dropna()

        running_mean, running_err = calculate_sequential_score_logit(  # noqa: F821
            binned_stats["mean"].values,
            binned_stats["std"].values,
        )
        binned_stats["running_mean"] = running_mean
        binned_stats["running_std"] = running_err
        return binned_stats

    # p-value combination.  Per-bin first, then a cumulative combination that
    # goes back to the raw per-epoch p-values rather than re-combining bin
    # scores.
    combiner = brown_combine if method == "brown" else stouffer_combine  # noqa: F821
    binned_stats = (
        metric_df.groupby("time_bin", observed=True)
        .apply(lambda g: stouffer_stats(  # noqa: F821 (from utils.*)
            g, weight_col=weight_col, rho=rho, combiner=combiner))
        .reset_index()
    )
    binned_stats["time_mid"] = binned_stats["time_bin"].apply(lambda x: x.mid)
    binned_stats = binned_stats.dropna(subset=["mean"])

    if binned_stats.empty:
        binned_stats["running_mean"] = []
        binned_stats["running_std"] = []
        return binned_stats

    # Chronological order, and the raw epochs behind each surviving bin.
    binned_stats = binned_stats.sort_values("time_mid").reset_index(drop=True)
    by_bin = {k: v for k, v in metric_df.groupby("time_bin", observed=True)}
    p_by_bin, w_by_bin = [], []
    for tb in binned_stats["time_bin"]:
        g = by_bin[tb]
        p_by_bin.append(g["p_tail_mean"].to_numpy(dtype=float))
        w_by_bin.append(
            g[weight_col].to_numpy(dtype=float)
            if weight_col is not None and weight_col in g else None
        )
    if all(w is None for w in w_by_bin):
        w_by_bin = None

    running_mean, running_err = calculate_sequential_score_stouffer(  # noqa: F821
        p_by_bin, weights_by_bin=w_by_bin, rho=rho, combiner=combiner,
    )
    binned_stats["running_mean"] = running_mean
    binned_stats["running_std"] = running_err

    return binned_stats





# ---------------------------------------------------------------------------
# The distance systematic: estimating rho, and marginalising it properly
# ---------------------------------------------------------------------------
#
# `absolute_magnitude_error` is the correct MARGINAL uncertainty on a single
# observation -- Var = sigma_phot^2 + sigma_mu^2 -- so each P_tail on its own is
# a calibrated p-value and nothing per epoch needs changing.  What it cannot
# express is that sigma_mu is the SAME DRAW at every epoch of one candidate.
# On AT2017gfo that is 97.7% of the variance, entirely shared, and the combiner
# is never told.  The result is calibrated epochs and a combined score
# miscalibrated by 4-6x (REPORT.md Part X).
#
# The three functions below are the two fixes measured there:
#
#   estimate_rho(..., sigma_mu=...)     the cheap one.  Same inflated sigma, but
#                                       rho is measured from the grid WITH the
#                                       shared draw injected, so the combiner
#                                       sees the correlation the systematic
#                                       induces.  No change to ranking.
#   combined_score_marginalised(...)    the correct one.  Score with sigma_phot
#                                       alone and calibrate the combined
#                                       statistic against an empirical null
#                                       simulated with one shared delta per
#                                       draw -- the systematic handled once per
#                                       candidate, where it belongs.
#
# NEITHER RECOVERS POWER, and it is worth being explicit because it is tempting
# to assume otherwise.  AUC and false-positive rate at fixed completeness are
# rank-based and identical to three decimals under all three treatments.  What a
# distance error destroys is irreducible: unknown to 30% in D is unknown to
# 0.65 mag in absolute magnitude, and no combiner invents that back.  These fix
# what the score MEANS, not how well it discriminates.


def _tail_from_grid(obs, grid_col, sigma, M_lim=None):
    """Vectorised closed-form P_tail of many observations against one epoch's
    reference column.  Mirrors ``predictive_tail_kde``'s closed form exactly,
    including the conditioned numerator, so rho is estimated on the same
    distribution the scorer actually produces."""
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    g = np.asarray(grid_col, dtype=float)
    s = float(sigma)
    phi = ndtr((obs[:, None] - g[None, :]) / s)
    if M_lim is not None and np.isfinite(M_lim):
        w = ndtr((float(M_lim) - g) / s)
        w_sum = float(w.sum())
        if w_sum > 0.0:
            F = np.clip(phi.sum(axis=1) / w_sum, 0.0, 1.0)
        else:
            F = phi.mean(axis=1)
    else:
        F = phi.mean(axis=1)
    return 2.0 * np.minimum(F, 1.0 - F)


def _epoch_grid(metric_df, data_sim, time_bin_width=0.2, sigma_col=None):
    """Line the candidate's scored epochs up against the grid.

    Returns ``(G, sig, lim, rows)`` with ``G`` of shape (n_sample, n_epoch) --
    one column per scored epoch-band, one row per simulation, indexed by a common
    set of ``sample_id`` so that a row is ONE simulated object seen at every
    epoch.  That is what carries the population's own inter-epoch correlation.
    """
    rows = (metric_df.dropna(subset=["p_tail_mean"])
            .sort_values("obs_time").reset_index(drop=True))
    if rows.empty:
        return None, None, None, rows

    cols, sig, lim, keep = [], [], [], []
    for r in rows.itertuples(index=False):
        band = data_sim[data_sim["filter_mapped"] == r.band]
        if band.empty:
            continue
        lo = getattr(r, "time_bin_low", r.obs_time - time_bin_width / 2)
        hi = getattr(r, "time_bin_high", r.obs_time + time_bin_width / 2)
        sub = band[(band["time"] > lo) & (band["time"] <= hi)]
        if sub.empty:
            sub = band.iloc[(band["time"] - r.obs_time).abs().argsort()[:0]]
        if sub.empty:
            continue
        s = sub.groupby("sample_id")["absolute_magnitude"].mean()
        cols.append(s)
        sigma = getattr(r, sigma_col) if sigma_col else r.observed_mag_err
        sig.append(float(sigma))
        M_lim = getattr(r, "M_lim", np.nan)
        lim.append(float(M_lim) if np.isfinite(M_lim) else None)
        keep.append(True)

    if not cols:
        return None, None, None, rows.iloc[:0]
    G = pd.concat(cols, axis=1).dropna()
    return G.to_numpy(), np.asarray(sig), lim, rows.iloc[:len(cols)]


def _simulate_epoch_p(G, sig, lim, sigma_mu, n_draws, rng, sigma_score=None):
    """Draw simulated candidates from the grid, observe them, and score them.

    One ``delta`` per DRAW -- not per epoch -- which is the whole point: the
    distance systematic is a single realisation shared by every epoch of a
    candidate, and that is what makes the epochs correlated.
    """
    n_grid, n_ep = G.shape
    idx = rng.integers(0, n_grid, n_draws)
    truth = G[idx, :]
    sig_score = sig if sigma_score is None else np.asarray(sigma_score, dtype=float)
    eps = rng.normal(0.0, 1.0, (n_draws, n_ep)) * sig_score[None, :]
    delta = rng.normal(0.0, float(sigma_mu), (n_draws, 1)) if sigma_mu else 0.0
    obs = truth + eps - delta
    return np.column_stack([
        _tail_from_grid(obs[:, j], G[:, j], sig[j], lim[j])
        for j in range(n_ep)])


def estimate_rho(
    metric_df: pd.DataFrame,
    data_sim: pd.DataFrame,
    sigma_mu: float = 0.0,
    n_draws: int = 4000,
    time_bin_width: float = 0.2,
    sigma_col: Optional[str] = None,
    eps: float = 1e-4,
    random_state: Optional[int] = None,
    return_matrix: bool = False,
):
    """Mean inter-epoch correlation of the normal scores, measured on the grid at
    THIS candidate's cadence.

    Implements COMBINING_REPLACEMENT.md section 3 -- the grid is the null, so the
    correlation to feed ``binned_stats_cumulative_ptail(rho=...)`` is read off
    the grid rather than guessed -- and adds the term that estimator was missing.

    ``sigma_mu`` injects the shared distance-modulus draw.  Pass the
    ``distance_modulus_error`` column that ``load_observations`` now emits.
    Leaving it at 0 reproduces the previous, grid-only estimate.

    It matters.  Measured on 8 epochs with a population correlation of 0.75, the
    grid-only estimator returns ~0.40 whatever the distance error, while the
    truth climbs with it::

        dD/D     grid alone    grid + distance
          0%        0.4323           0.4285
         18%        0.4253           0.5587
         30%        0.4057           0.6688
         50%        0.3843           0.7926

    and feeding the corrected value recovers most of the calibration loss:
    KS 0.0929 -> 0.0583 at dD/D = 30%.

    Note the correlation wanted here is of ``z = Phi^-1(1 - p)``, NOT of the
    underlying magnitude deviates.  ``P_tail`` is two-sided, so ``p`` depends on
    ``|deviate|`` and the sign is destroyed; the z-correlation is a folded
    quantity and is much smaller than the deviate correlation.  There is no
    closed form worth trusting -- measure it, which is what this does.

    Parameters
    ----------
    metric_df : pd.DataFrame
        Output of ``kilonovascorer_v3`` for one candidate.
    data_sim : pd.DataFrame
        The same simulation grid the candidate was scored against.
    sigma_mu : float
        Distance-modulus uncertainty, in magnitudes, shared across all epochs.
    n_draws : int
        Simulated candidates.  The standard error on rho is ~1/sqrt(n_draws).
    sigma_col : str or None
        Column of ``metric_df`` holding the per-epoch sigma to score with.
        ``None`` uses ``observed_mag_err``, i.e. whatever the scorer used.
    return_matrix : bool
        Also return the full (n_epoch, n_epoch) correlation matrix, whose mean
        off-diagonal is the scalar returned.  The combiners accept only the
        scalar today; the matrix is the open item in Part V section 31.

    Returns
    -------
    float, or (float, np.ndarray) when ``return_matrix``.
        NaN if fewer than two epochs are scoreable.
    """
    rng = np.random.default_rng(random_state)
    G, sig, lim, rows = _epoch_grid(metric_df, data_sim, time_bin_width, sigma_col)
    if G is None or G.shape[1] < 2 or G.shape[0] < 10:
        return (float("nan"), None) if return_matrix else float("nan")

    P = _simulate_epoch_p(G, sig, lim, sigma_mu, n_draws, rng)
    Z = ndtri(1.0 - np.clip(P, eps, 1.0 - eps))
    C = np.corrcoef(Z, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    rho = float(np.nanmean(C[iu]))
    return (rho, C) if return_matrix else rho


def combined_score_marginalised(
    metric_df: pd.DataFrame,
    data_sim: pd.DataFrame,
    sigma_mu: float,
    n_draws: int = 4000,
    time_bin_width: float = 0.2,
    sigma_col: str = "p_tail_sigma_phot",
    eps: float = 1e-4,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Cumulative score with the distance systematic marginalised ONCE per
    candidate instead of once per epoch.

    This is treatment C of REPORT.md Part X, and the statistically correct one:
    ``delta`` is a single shared nuisance parameter with a known prior (the
    distance posterior), so the right move is to marginalise it in the NULL
    DISTRIBUTION of the combined statistic, not to inflate every epoch's
    marginal and then combine as though they were independent.

    Procedure.  Score each epoch with the PHOTOMETRIC sigma alone, take
    ``S_k = sum_{j<=k} Phi^-1(1 - p_j)`` in chronological order, and read its
    p-value off an empirical null simulated from the grid with one shared
    ``delta`` per draw.  Because the null is simulated, no Gaussian
    equicorrelation assumption is needed and no ``rho`` is required -- the
    dependence, including the two-sided fold, is inherited rather than modelled.

    Measured on 5,000 held-out draws at 8 epochs, KS against U(0,1)
    (critical 0.0192)::

        dD/D     current   rho+distance   marginalised
          0%      0.0949         0.0952         0.0155
          5%      0.0847         0.0834         0.0126
         18%      0.0877         0.0695         0.0216
         30%      0.0929         0.0583         0.0205
         50%      0.1158         0.0510         0.0221

    Again: this does NOT improve ranking.  AUC is identical to three decimals
    across all three treatments.

    ``metric_df`` must have been produced by ``kilonovascorer_v3`` with
    ``sigma_col="absolute_magnitude_error_phot"`` so that its ``p_tail_mean``
    is the photometric-only p-value; ``sigma_col`` here names the matching
    column of per-epoch sigmas in ``metric_df`` (default
    ``observed_mag_err``, which is what that run records).

    Returns
    -------
    pd.DataFrame
        One row per epoch in chronological order: ``obs_time``, ``band``,
        ``z``, ``z_cumsum``, ``running_mean`` (the marginalised p-value), and
        ``n_null`` (draws behind it).
    """
    rng = np.random.default_rng(random_state)
    col = sigma_col if sigma_col in metric_df.columns else None
    G, sig, lim, rows = _epoch_grid(metric_df, data_sim, time_bin_width, col)
    if G is None or G.shape[1] < 1:
        return pd.DataFrame(columns=["obs_time", "band", "z", "z_cumsum",
                                     "running_mean", "n_null"])

    p_obs = rows["p_tail_mean"].to_numpy(dtype=float)[:G.shape[1]]
    z_obs = ndtri(1.0 - np.clip(p_obs, eps, 1.0 - eps))
    S_obs = np.cumsum(z_obs)

    # One null simulation serves every prefix: cumulative-sum the same draws.
    P_null = _simulate_epoch_p(G, sig, lim, sigma_mu, n_draws, rng)
    S_null = np.cumsum(ndtri(1.0 - np.clip(P_null, eps, 1.0 - eps)), axis=1)

    running = np.empty(len(S_obs))
    for k in range(len(S_obs)):
        col_k = np.sort(S_null[:, k])
        lo = np.searchsorted(col_k, S_obs[k], side="left")
        hi = np.searchsorted(col_k, S_obs[k], side="right")
        # mid-p: split the tie mass, as the pattern statistic does in
        # NON_DETECTIONS.md section 3.
        running[k] = (n_draws - 0.5 * (lo + hi)) / n_draws

    return pd.DataFrame({
        "obs_time": rows["obs_time"].to_numpy()[:len(S_obs)],
        "band": rows["band"].to_numpy()[:len(S_obs)],
        "z": z_obs,
        "z_cumsum": S_obs,
        "running_mean": running,
        "n_null": n_draws,
    })


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def kilonovascorer_v3(
    data_obs: pd.DataFrame,
    data_sim: pd.DataFrame,
    candidate_name: str,
    time_bin_width: float = 0.2,
    band_list: Tuple[str, ...] = ("g-band", "r-band", "i-band", "z-band"),
    k_near: float = 1.5,
    n_kde_sim: int = 50000,
    min_sim_points: int = 20,
    overlap_k: float = 2.0,
    M_lim: Optional[float] = None,
    M_lim_col: str = "limiting_absolute_magnitude",
    min_n_eff: float = 20.0,
    p_tail_method: str = "closed_form",
    n_obs: int = 100,
    random_state: Optional[int] = None,
    abc_compute: bool = True,
    abc_return_ids: bool = True,
    sigma_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score a kilonova candidate against a simulation grid.

    For each photometric band and observation, computes:

    - **P_tail_KNe** — two-sided tail probability of M_obs under the
      noise-convolved PPD (with uncertainty via observation sampling).
    - **P_near_KNe** — ROPE-based local consistency score.
    - **ABC survival diagnostic** — sequential intersection of consistent
      simulation IDs across epochs (|S_t| from paper Section 3).

    Parameters
    ----------
    data_obs : pd.DataFrame
        Observational data.  Required columns: ``filter_mapped``,
        ``time_after_gw``, ``absolute_magnitude``,
        ``absolute_magnitude_error``.
    data_sim : pd.DataFrame
        Simulation grid.  Required columns: ``filter_mapped``, ``time``,
        ``absolute_magnitude``, ``sample_id``.
    candidate_name : str
        Human-readable identifier for the transient candidate.
    time_bin_width : float
        Width of time bins used to match observations to simulations (days).
    band_list : tuple of str
        Photometric bands to score.
    k_near : float
        ROPE half-width factor for P_near_KNe (paper fiducial: 1.5).
    n_kde_sim : int
        Number of Monte Carlo draws for the noise-convolved PPD.  Used only
        when ``p_tail_method="montecarlo"``; the closed form is exact and
        ignores it.
    min_sim_points : int
        Minimum number of simulations required in a bin to attempt scoring.
    overlap_k : float
        ROPE half-width factor for the ABC diagnostic (sigma units).
    M_lim : float or None
        Limiting ABSOLUTE magnitude, ``M_lim = m_lim - mu``, used to condition
        the reference population on detectability (IMPROVEMENTS.md 19).  A
        scalar applies to every observation.  ``None`` (default) leaves the
        reference unconditioned, reproducing the previous behaviour exactly.

        Depth is a property of the exposure, so per-observation values are
        strongly preferred over a scalar — supply them via ``M_lim_col``.
        Deriving ``m_lim`` from each detection's own S/N is deliberately left to
        the caller: it needs facility metadata (the pipeline's sigma threshold,
        an S/N cap) that this package does not have, and reading a 5-sigma depth
        as 3-sigma is a 0.555 mag systematic error per facility.
    M_lim_col : str
        Column of ``data_obs`` holding a per-observation limiting absolute
        magnitude.  Used when present; falls back to the ``M_lim`` scalar, then
        to no conditioning.  Non-finite entries fall back the same way.
    min_n_eff : float
        Minimum Kish effective sample size for the conditioned reference.  Below
        it the observation is reported with NaN scores rather than a number the
        grid cannot support — at 400 Mpc a 10,000-sample grid can leave ~130
        usable simulations, and at 300 Mpc past 3 d, none at all.  Applied only
        when an ``M_lim`` is in force.
    p_tail_method : {"closed_form", "montecarlo"}
        Which estimator computes P_tail_KNe and P_near_KNe.

        ``"closed_form"`` (default) evaluates both as exact sums of error
        functions over the simulations in the bin: deterministic, ~20x faster,
        and it applies ``sigma_obs`` once.

        ``"montecarlo"`` restores the ORIGINAL estimator exactly — fit a
        Gaussian KDE per time bin, resample it ``n_kde_sim`` times, add
        observational noise, and count — including its ``p_tail_std``, taken
        as the spread of P_tail over ``n_obs`` jittered realisations of M_obs.
        Choose it to reproduce results published before the change, or to
        compare the two on the same photometry.  It is stochastic unless
        ``random_state`` is set, and it applies ``sigma_obs`` twice.

        See :func:`predictive_tail_kde` for the full comparison.
    n_obs : int
        Number of M_obs realisations behind the Monte Carlo ``p_tail_std``.
        Paper value: N_obs = 100.  Used only when
        ``p_tail_method="montecarlo"``.
    random_state : int or None
        Seed for the Monte Carlo path, making it reproducible.  ``None``
        (default) uses NumPy's global random state, which is what the original
        did.  Ignored by the closed form.
    abc_compute : bool
        Compute the ABC survival diagnostic.  ``True`` (default) preserves the
        existing output.  ``False`` skips it and leaves the ABC columns empty.
        Measured on warm runs at a 10,000-sample grid the diagnostic is 16% of
        the run and at 25,000 samples 10% -- worth having off when only the
        P_tail score is wanted, but not the dominant cost.  (An earlier note here
        claimed 50%; that came from timing the with-ABC arm as a cold first call
        against a warm without-ABC arm, and is withdrawn.)
    abc_return_ids : bool
        Include the per-epoch simulation id lists (``consistent_ids``,
        ``overlap_with_next_ids``, ``running_survivors_ids``).  ``True``
        (default) preserves the existing output.  ``False`` keeps every COUNT and
        empties the lists.

        This does NOT speed up an in-memory run -- the chain still needs the sets
        to intersect, so only the per-epoch ``sorted()`` is saved, and the
        measured difference is within noise.  Its purpose is OUTPUT SIZE: one
        candidate at a 10,000-sample grid otherwise materialises ~320,000 ids
        into 40 DataFrame cells, which most callers immediately reduce back to a
        length, and which dominate the cost of writing the metrics table.
    sigma_col : str or None
        Column of ``data_obs`` to use as the per-epoch uncertainty instead of
        ``absolute_magnitude_error``.  ``None`` (default) is the existing
        behaviour.

        The reason this exists is ``sigma_col="absolute_magnitude_error_phot"``.
        ``absolute_magnitude_error`` is the correct MARGINAL uncertainty --
        photometry and distance in quadrature -- so each P_tail computed from it
        is a calibrated per-epoch p-value.  But the distance term is ONE draw
        shared by every epoch (97.7% of the variance on AT2017gfo), and feeding
        it in per epoch hides that from the combiner, which then treats strongly
        correlated epochs as independent.  Scoring with the photometric term
        alone and marginalising the systematic once per candidate --
        ``combined_score_marginalised`` -- is 4-6x better calibrated.

        Used ON ITS OWN this column is NOT a calibrated p-value: it understates
        the per-epoch uncertainty by design, and the combined statistic must then
        be calibrated against a null that puts the systematic back.  See
        REPORT.md Part X.

    Returns
    -------
    results_df : pd.DataFrame
        Per-observation metrics including P_tail_KNe, P_near_KNe, and ABC
        diagnostics.
    summary_df : pd.DataFrame
        Per-band overlap chain summary.
    """
    # Validate once, up front, rather than on every observation.
    p_tail_method = _resolve_p_tail_method(p_tail_method)

    results: List[Dict[str, Any]] = []
    overlap_summary_by_band: Dict[str, Any] = {}

    # Split by band ONCE.  The previous form re-scanned the full simulation table
    # per band with `data_sim["filter_mapped"] == band`, which on a string column
    # is an elementwise comparison over every row -- 800,000 rows x 4 bands, and
    # the single largest cost in the profile, larger than the ABC diagnostic and
    # an order of magnitude larger than predictive_tail_kde.  groupby partitions
    # in one pass: measured 2.0x at a 10,000-sample grid, 1.9x at 25,000.
    sim_by_band = {k: v for k, v in data_sim.groupby("filter_mapped", observed=True)}
    obs_by_band = {k: v for k, v in data_obs.groupby("filter_mapped", observed=True)}

    for band in band_list:
        # 1. Filter data for this band
        sim_band = sim_by_band.get(band)
        obs_band = obs_by_band.get(band)

        if sim_band is None or obs_band is None or sim_band.empty or obs_band.empty:
            logger.debug("No data for band %s — skipping.", band)
            continue
        sim_band = sim_band.copy()
        obs_band = obs_band.copy()

        # 2. Assign simulation time bins (computed once per band).
        #    Bin edges are chosen so that the first and last observations both
        #    land at the centre of their respective bins:
        #      left edge  = t_first - bin_width/2
        #      right edge = t_last  + bin_width/2
        #    An extra bin_width is added to t_end so np.arange includes the
        #    final right edge.
        t_first = obs_band["time_after_gw"].min()
        t_last  = obs_band["time_after_gw"].max()
        t_start = t_first - time_bin_width / 2
        t_end   = t_last  + time_bin_width  
        bins = np.arange(t_start, t_end, time_bin_width)

        # assert bins[0] < t_first < bins[1],  "First observation not centred in first bin."
        # assert bins[-2] < t_last < bins[-1], "Last observation not centred in last bin."

        sim_band["time_bin"] = np.digitize(sim_band["time"], bins)

        # Pre-group simulations by time bin for O(1) lookup per observation
        sim_groups: Dict[int, pd.DataFrame] = {
            k: v for k, v in sim_band.groupby("time_bin")
        }

        # Per-band tracking for ABC overlap chain
        band_times: List[float] = []
        band_ids_lists: List[List] = []
        band_row_indices: List[int] = []

        # KDE cache: fitted once per bin_idx and reused across the observations
        # sharing that bin.  Only the Monte Carlo estimator needs it; the closed
        # form never fits a KDE, so the cache stays empty and costs nothing.
        kde_cache: Dict[int, gaussian_kde] = {}


        # 3. Process observations in chronological order
        obs_band = obs_band.sort_values("time_after_gw")
        total_obs = len(obs_band)

        for count, obs_row in enumerate(obs_band.itertuples(index=False), start=1):
            t_obs = float(obs_row.time_after_gw)
            M_obs = float(obs_row.absolute_magnitude)
            sigma_obs = float(obs_row.absolute_magnitude_error)

            # Depth is a property of the exposure, so it is per-observation.
            M_lim_row = getattr(obs_row, M_lim_col, None)
            if M_lim_row is None or not np.isfinite(M_lim_row):
                M_lim_row = M_lim
            if M_lim_row is not None and not np.isfinite(M_lim_row):
                M_lim_row = None

            # The per-epoch uncertainty actually scored with.  Normally the
            # column as given; `sigma_col` overrides it, which is how the
            # photometric-only path of Part X is selected.
            if sigma_col is not None:
                s_row = getattr(obs_row, sigma_col, None)
                if s_row is not None and np.isfinite(s_row) and s_row > 0:
                    sigma_obs = float(s_row)

            # Skip degenerate observations
            if not (np.isfinite(M_obs) and np.isfinite(sigma_obs) and sigma_obs > 0):
                logger.debug("Skipping invalid observation at t=%.3f d.", t_obs)
                continue

            bin_idx = int(np.digitize(t_obs, bins))
            sim_bin = sim_groups.get(bin_idx, pd.DataFrame())

            if len(sim_bin) < min_sim_points:
                logger.debug(
                    "Bin %d has %d simulations (< %d) — skipping.",
                    bin_idx, len(sim_bin), min_sim_points,
                )
                continue

            # 3a. Compute P_tail_KNe and P_near_KNe (paper eqs. 6-7 and 4).
            #     Under the closed form these are exact sums over the
            #     simulations in the bin, so there is no KDE to fit; under
            #     Monte Carlo one is fitted per bin and reused.
            cached_kde = None
            if p_tail_method == "montecarlo":
                if bin_idx not in kde_cache:
                    kde_cache[bin_idx] = gaussian_kde(
                        sim_bin["absolute_magnitude"].to_numpy()
                    )
                cached_kde = kde_cache[bin_idx]

            metric = predictive_tail_kde(
                sim_bin["absolute_magnitude"].to_numpy(),
                M_obs=M_obs,
                sigma_obs=sigma_obs,
                k=k_near,
                n_sim=n_kde_sim,
                n_obs=n_obs,
                kde=cached_kde,
                M_lim=M_lim_row,
                min_n_eff=min_n_eff,
                p_tail_method=p_tail_method,
                random_state=random_state,
            )

            # 3b. ABC diagnostic — consistent simulation IDs at this epoch.
            #     `sim_bin` is passed through: it is exactly the rows
            #     compute_consistent_ids_anyhit would otherwise select for itself,
            #     and the scorer already holds it.  Without it the function
            #     rescans the whole band once per observation -- 200,000 rows to
            #     reach 10,000 of them -- which measured 1.6x (10,000 samples) to
            #     1.8x (25,000 samples) of the entire diagnostic's cost.
            consistent_ids: List = []
            if abc_compute:
                consistent_ids = compute_consistent_ids_anyhit(
                    sim_band=sim_band,
                    bin_idx=bin_idx,
                    M_obs=M_obs,
                    sigma_obs=sigma_obs,
                    overlap_k=overlap_k,
                    sim_bin=sim_bin,
                )

            # 3c. Safe time-bin edge lookup
            bin_low = float(bins[bin_idx - 1] if bin_idx > 0 else bins[0])
            bin_high = float(bins[bin_idx] if bin_idx < len(bins) else bins[-1])

            row: Dict[str, Any] = {
                "candidate_name": candidate_name,
                "band": band,
                "obs_time": t_obs,
                "time_bin_low": bin_low,
                "time_bin_high": bin_high,
                "observed_mag": M_obs,
                "observed_mag_err": sigma_obs,
                "p_tail_KNe": metric["p_tail_KNe"],
                "p_tail_mean": metric["p_tail_mean"],
                "p_tail_std": metric["p_tail_std"],
                "p_near_KNe": metric["p_near_KNe"],
                "M_lim": float(M_lim_row) if M_lim_row is not None else np.nan,
                "n_eff": metric["n_eff"],
                "scoreable": metric["scoreable"],
                "p_tail_method": metric["p_tail_method"],
                "n_sim_bin": len(sim_bin),
                "n_consistent_lcs": len(consistent_ids),
                "consistent_ids": consistent_ids if abc_return_ids else [],
                # ABC overlap fields — populated in post-processing step 4
                "overlap_with_next_n": np.nan,
                "overlap_with_next_ids": [],
                "running_survivors_n": np.nan,
                "running_survivors_ids": [],
            }

            results.append(row)
            if abc_compute:
                band_times.append(t_obs)
                band_ids_lists.append(consistent_ids)
            band_row_indices.append(len(results) - 1)
            arcade_progress_bar(count, total_obs, bar_length=50)

        # 4. Post-processing: compute ABC overlap chain for this band
        if band_ids_lists:
            chain = overlap_chain(
                band_ids_lists, band_times, return_ids=abc_return_ids)
            overlap_summary_by_band[band] = chain

            # Map running survivors back to per-observation rows
            for j, surv in enumerate(chain["survivors_over_time"]):
                idx = band_row_indices[j]
                results[idx]["running_survivors_n"] = int(surv["n_survivors"])
                results[idx]["running_survivors_ids"] = surv["survivor_ids"]

            # Map pairwise overlaps to the left-hand observation of each pair
            for j, pw in enumerate(chain.get("pairwise", [])):
                idx_left = band_row_indices[j]
                results[idx_left]["overlap_with_next_n"] = int(pw["n_overlap"])
                results[idx_left]["overlap_with_next_ids"] = pw["overlap_ids"]

    return pd.DataFrame(results), pd.DataFrame(overlap_summary_by_band)
