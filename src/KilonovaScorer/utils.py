import time
from functools import wraps
from scipy.stats import gaussian_kde

"""
utils.py — KilonovaScorer utility functions.

Implements:
  - Apparent-to-absolute magnitude conversion via Monte Carlo sampling,
    supporting both scalar and array inputs.
  - Logit-space inverse-variance weighted aggregation (ivw_stats_logit).
  - Sequential logit-space score updating (calculate_sequential_score_logit).
"""

import logging

import numpy as np
import pandas as pd
from typing import List, Optional

from scipy.special import expit, logit, ndtr, ndtri

logger = logging.getLogger(__name__)

P_TAIL_STD_FLOOR = None


# ---------------------------------------------------------------------------
# Flux-space conversions and non-detection scoring
# ---------------------------------------------------------------------------

def flux_of(M, zp=0.0):
    """Absolute magnitude -> flux: ``F = 10**(-0.4*(M - zp))``. Monotone
    decreasing, so brighter = larger F, opposite of magnitude. ``zp`` cancels
    out of every probability computed from F; it only keeps values near
    order-unity."""
    return 10.0 ** (-0.4 * (np.asarray(M, dtype=float) - zp))


def flux_sigma_of(M, sigma_mag, zp=0.0):
    """Delta-method flux uncertainty from a magnitude uncertainty:
    ``sigma_F = F * ln(10)/2.5 * sigma_mag``. Only valid for a real detection
    -- a non-detection has no magnitude to propagate from, see
    :func:`flux_sigma_of_limit`."""
    F = flux_of(M, zp)
    return F * (np.log(10.0) / 2.5) * np.asarray(sigma_mag, dtype=float)


def flux_sigma_of_limit(M_lim, n_sigma=5.0, zp=0.0):
    """Flux uncertainty implied by a quoted N-sigma non-detection depth:
    ``sigma_F = flux_of(m_lim) / n_sigma``."""
    return flux_of(M_lim, zp) / float(n_sigma)


def _flux_score_axis(M, zp=0.0):
    """Negated flux, for internal use by the scorer only.

    The P_tail machinery assumes smaller = brighter (true of magnitude,
    false of flux). Feeding it raw flux would flip the sign of F_hat and
    invert the M_lim detectability weight's direction. Negating both m and
    M_lim restores the magnitude-like convention; F_hat/P_tail/p_tail_mean/
    P_near are all invariant to a simultaneous sign flip, so nothing else
    needs to change.
    """
    return -flux_of(M, zp)


def nondetection_tail(sim_values, M_lim, n_sigma_limit=5.0, flux_zp=0.0):
    """P_tail for a non-detection -- a different test from P_tail_KNe, not
    the same machinery fed an imputed measurement.

    A non-detection is censored ("flux below threshold"), not a point
    measurement at zero -- treating it as one saturates the ordinary PIT
    (real fluxes sit many sigma above zero, so every simulation's term
    collapses to 1 regardless of brightness; see
    ``sim/flux_space_validate.py``). Instead this uses the standard
    censored-data likelihood for an upper limit::

        P_detect = mean_sim( Phi((F_sim - F_thresh) / sigma_phot) )
        p_tail   = 1 - P_detect

    No two-sided fold: the observed outcome (non-detection) is fixed, so
    this is a one-sided p-value, not a PIT. It is also a BINARY-outcome
    p-value, so it is conservative (>= Uniform(0,1)) rather than exactly
    uniform under the null -- expected, not a bug.

    Parameters
    ----------
    sim_values : array-like
        Simulated absolute magnitudes for the relevant time bin.
    M_lim : float
        This observation's own quoted limiting absolute magnitude.
    n_sigma_limit : float
        Significance the depth was quoted at.
    flux_zp : float
        Flux zeropoint; cancels out of every probability.

    Returns
    -------
    dict with ``F_hat`` (= P_detect), ``p_tail_KNe``, ``p_tail_mean``
    (identical -- no jitter average here), ``p_tail_std`` (NaN),
    ``p_near_KNe`` (NaN), ``n_eff``, ``scoreable`` (always True),
    ``p_tail_method`` (``"nondetection"``).
    """
    F_sim = flux_of(np.asarray(sim_values, dtype=float), flux_zp)
    F_thresh = flux_of(M_lim, flux_zp)
    sigma_phot = flux_sigma_of_limit(M_lim, n_sigma_limit, flux_zp)
    P_detect = float(ndtr((F_sim - F_thresh) / sigma_phot).mean())
    p_tail = 1.0 - P_detect
    return {
        "F_hat": P_detect,
        "p_tail_KNe": p_tail,
        "p_tail_mean": p_tail,
        "p_tail_std": float("nan"),
        "p_near_KNe": float("nan"),
        "n_eff": float(np.asarray(sim_values).size),
        "scoreable": True,
        "p_tail_method": "nondetection",
    }


# ---------------------------------------------------------------------------
# Apparent → absolute magnitude conversion
# ---------------------------------------------------------------------------

def compute_abs_mag_samples(
    app_mag,
    app_mag_err,
    dist_mpc: float,
    dist_err_mpc: float,
    n_samples: int = 5000,
    return_components: bool = False,
):
    """
    Convert apparent magnitude(s) to absolute magnitude via Monte Carlo sampling.

    Supports both scalar and array inputs for ``app_mag`` / ``app_mag_err``,
    so it can be called once on an entire DataFrame column (vectorised) or on
    a single observation.

    The distance modulus is sampled from N(dist_mpc, dist_err_mpc); apparent
    magnitudes are sampled from N(app_mag, app_mag_err).  Non-physical
    (negative) distance draws are rejected before computation.

    Parameters
    ----------
    app_mag : float or array-like
        Apparent magnitude(s).
    app_mag_err : float or array-like
        Uncertainty on apparent magnitude(s).  Non-finite values are treated
        as zero (no photometric uncertainty).
    dist_mpc : float
        Luminosity distance in Mpc.
    dist_err_mpc : float
        Uncertainty on the luminosity distance in Mpc.
    n_samples : int
        Number of Monte Carlo draws per observation.
    return_components : bool
        Also return the two terms the total is built from.  ``False`` (default)
        preserves the original two-value return exactly.

    Returns
    -------
    abs_mag_mean : float or np.ndarray
        Mean absolute magnitude(s).  np.nan for invalid inputs.
    abs_mag_std : float or np.ndarray
        Standard deviation of absolute magnitude(s).  np.nan for invalid inputs.
    abs_mag_std_phot : float or np.ndarray
        Returned only when ``return_components``.  The PHOTOMETRIC term alone,
        i.e. ``app_mag_err``: independent per observation.
    sigma_mu : float
        Returned only when ``return_components``.  The distance-modulus
        uncertainty: ONE number for the whole candidate, identical at every
        epoch.
    """
    scalar_input = np.ndim(app_mag) == 0

    app_mag = np.atleast_1d(np.asarray(app_mag, dtype=float))
    app_mag_err = np.atleast_1d(np.asarray(app_mag_err, dtype=float))

    # Treat non-finite errors as zero (conservative: no photometric noise)
    app_mag_err = np.where(np.isfinite(app_mag_err), app_mag_err, 0.0)
    app_mag_err = np.clip(app_mag_err, 0.0, None)

    n_obs = len(app_mag)
    abs_mag_mean = np.full(n_obs, np.nan)
    abs_mag_std  = np.full(n_obs, np.nan)

    # Global distance validation
    def _out(mean, std, phot, sig_mu):
        if scalar_input:
            mean, std = float(mean[0]), float(std[0])
            phot = float(phot[0])
        if return_components:
            return mean, std, phot, float(sig_mu)
        return mean, std

    # The photometric term needs no Monte Carlo: the distance modulus is
    # additive, so it shifts the absolute magnitude without touching its
    # photometric spread.  sigma_phot in absolute magnitude IS app_mag_err.
    abs_mag_std_phot = np.where(np.isfinite(app_mag), app_mag_err, np.nan)

    if not np.isfinite(dist_mpc) or dist_mpc <= 0:
        logger.warning("Invalid distance dist_mpc=%.3f — returning NaN.", dist_mpc)
        return _out(abs_mag_mean, abs_mag_std, abs_mag_std_phot, np.nan)

    # Sample distance modulus (shared across all observations — distance is a
    # global systematic, not an independent draw per row)
    D_samples = np.random.normal(dist_mpc, dist_err_mpc, n_samples) * 1e6  # parsecs
    D_samples = D_samples[D_samples > 0]

    if len(D_samples) < 10:
        logger.warning(
            "Fewer than 10 valid distance samples drawn — dist_mpc=%.3f, "
            "dist_err_mpc=%.3f.  Returning NaN.",
            dist_mpc, dist_err_mpc,
        )
        return _out(abs_mag_mean, abs_mag_std, abs_mag_std_phot, np.nan)

    n_valid = len(D_samples)
    mu_samples = 5.0 * np.log10(D_samples) - 5.0  # shape (n_valid,)
    # The shared systematic, as one number.  Measured from the same draws the
    # totals are built from, so it is consistent with them by construction.
    sigma_mu = float(np.std(mu_samples))

    for i in range(n_obs):
        if not np.isfinite(app_mag[i]):
            continue  # leave as NaN

        # Independent apparent-magnitude noise per observation
        app_samples = np.random.normal(app_mag[i], app_mag_err[i], n_valid)
        abs_samples = app_samples - mu_samples

        abs_mag_mean[i] = np.mean(abs_samples)
        abs_mag_std[i]  = np.std(abs_samples)

    return _out(abs_mag_mean, abs_mag_std, abs_mag_std_phot, sigma_mu)


# ---------------------------------------------------------------------------
# Logit-space inverse-variance weighted aggregation
# ---------------------------------------------------------------------------

def ivw_stats_logit(
    group: pd.DataFrame,
    eps: float = 1e-4,
    s_floor: float = None,
) -> pd.Series:
    """
    Inverse-variance weighted mean and uncertainty in logit space.

    Statistically appropriate for bounded P_tail_KNe scores in (0, 1):
    aggregating directly in probability space biases the mean toward extreme
    values with small absolute uncertainties.  Operating in logit space
    stabilises variances near the boundaries (see paper Section 2).

    The uncertainty on each logit-transformed score is propagated via the
    delta method::

        sigma_z = sigma_p / (p * (1 - p))

    and the result is transformed back via the inverse logit (expit).

    Parameters
    ----------
    group : pd.DataFrame
        Subset of the metrics DataFrame for a single time bin.  Must contain
        ``p_tail_mean`` and ``p_tail_std`` columns.
    eps : float
        Clamping value to keep scores away from 0 and 1 before logit
        transform (prevents infinite logit values).  Rows with
        ``p_tail_mean == 0`` are kept and clamped to ``eps``; they are not
        filtered out, because a categorically-rejected epoch carries the
        strongest evidence in the bin.
    s_floor : float or None
        Lower bound applied to ``p_tail_std`` before the delta method, so that
        an epoch with zero spread gets a large but finite weight instead of a
        division by zero.  ``None`` falls back to :data:`P_TAIL_STD_FLOOR`, and
        thence to ``eps``.  This is a scientific lever, not a numerical
        detail — see the constant.

    Returns
    -------
    pd.Series
        ``mean``  – inverse-variance weighted mean (probability space).
        ``std``   – propagated uncertainty (probability space).
        ``count`` – number of valid scores used.

        All return paths carry all three keys.  The previous early return
        emitted only ``{mean, std}``, which under ``groupby(...).apply(...)``
        produced ``count = NaN`` — and the bare ``.dropna()`` in
        ``binned_stats_cumulative_ptail`` then deleted the entire bin on the
        strength of that missing field alone.
    """
    p = group["p_tail_mean"].to_numpy(dtype=float)
    s = group["p_tail_std"].to_numpy(dtype=float)

    mask = np.isfinite(p) & np.isfinite(s) & (s >= 0.0)
    p, s = p[mask], s[mask]

    if len(p) == 0:
        return pd.Series({"mean": np.nan, "std": np.nan, "count": 0})

    # Floor s == 0 rather than dropping it
    floor = s_floor if s_floor is not None else P_TAIL_STD_FLOOR
    if floor is None:
        floor = eps
    p_clipped = np.clip(p, eps, 1.0 - eps)
    s_floored = np.maximum(s, floor)
    z     = logit(p_clipped)
    z_std = s_floored / (p_clipped * (1.0 - p_clipped))    # delta method
    weights = 1.0 / z_std ** 2

    z_mean     = np.sum(weights * z) / np.sum(weights)
    z_std_comb = np.sqrt(1.0 / np.sum(weights))

    mean = float(expit(z_mean))
    std  = float(mean * (1.0 - mean) * z_std_comb)  # delta method back-transform

    return pd.Series({"mean": mean, "std": std, "count": len(p)})

def stouffer_combine(
    p,
    eps: float = 1e-4,
    rho: float = 0.0,
):
    """
    Combine p-values by the weighted Stouffer method, with Strube's
    correlation correction.

    ``rho = 0`` is Stouffer (1949); ``rho > 0`` is **Strube's method** (Strube
    1985, *Psychological Bulletin* 97, 334-341), the generalisation of Stouffer
    to non-independent tests.  Both are the same function because they differ
    only in the normaliser -- see ``rho`` below.  The name stays
    ``stouffer_combine`` because that is what it is at the default, and because
    a correlation argument on Stouffer is the standard interface (R's ``poolr``
    spells it ``stouffer(..., adjust=)``).

    Parameters
    ----------
    p : array-like
        Per-epoch p-values (``p_tail_mean``).  Non-finite entries are dropped.
        Zeros are KEPT and clamped to ``eps``: a categorically-rejected epoch is
        the strongest evidence available, not missing data.
    Weights
    -------
    There are none, and that is deliberate rather than a default.  Equal
    weighting is a MEASURED result: seven ancillary schemes were tested on
    held-out kilonovae plus three contaminant classes (``WEIGHTS.md``) and none
    beat it beyond noise, nor did the oracle weight ``w ~ mu_i`` that is
    provably optimal by Cauchy-Schwarz.  The ceiling on any weighting gain is
    ``sqrt(1 + CV^2)`` with CV the spread of per-epoch informativeness,
    measured at 0.060 -> a 0.18% gain, two orders of magnitude under the AUC
    noise floor.  Epoch informativeness is set by the grid's own population
    spread ``tau`` (0.21-0.56 mag), which dominates ``sigma_obs`` and barely
    varies across epochs, so there is little for weights to exploit.

    Weighting also costs something exact.  Stouffer accumulates like
    ``sqrt(n_eff)`` with ``n_eff = (sum w)^2 / sum(w^2)`` (Kish), and
    ``n_eff <= n`` for every non-uniform weighting, so unequal weights always
    spend effective epochs for a power gain that has to be real to pay for it.

    And the ``rho`` correction below stops being exact.  The scalar form
    assumes EXCHANGEABLE correlation, under which the mean off-diagonal is the
    off-diagonal.  Equal weights preserve that; unequal weights do not, so a
    weighted Z would carry a normaliser that is wrong by an unknown amount.
    Inverse-variance weights inside Stouffer were tested for exactly this and
    rejected on that ground -- if per-epoch variances are what you want to
    weight by, use :func:`ivw_stats_logit`, which is built for it.

    eps : float
        Clamp keeping p away from 0 and 1, where ``Phi^-1`` is infinite.
    rho : float
        Exchangeable inter-epoch correlation of the normal scores.  ``0.0``
        (the default) assumes independence; any positive value applies
        **Strube's correction** to the normaliser::

            Var(sum w_i z_i) = sum_i sum_j w_i w_j rho_ij
                             = sum(w^2) + rho * [ (sum w)^2 - sum(w^2) ]

        which for equal weights reduces to Strube's published form
        ``Z = sum(z) / sqrt(n + rho*n*(n-1))``.

        Epochs of one candidate ARE correlated -- same object, one smooth light
        curve, one shared distance draw -- with a mean of **0.284** measured on
        the grid.  Measure it for a given candidate with
        :func:`KilonovaScorer.core2.estimate_rho`, which evaluates the
        correlation at that candidate's own cadence rather than borrowing a
        number.

        Positive correlation makes the true variance larger than ``sum(w^2)``,
        so ``rho = 0`` is **overconfident by a known direction**.  Measured on
        20,000 exchangeably-correlated nulls of six epochs, KS against U(0,1)
        (5% critical value 0.0096):

        =========  =================  ==================
        true rho   Stouffer (rho=0)   Strube (rho=true)
        =========  =================  ==================
        0.000                 0.0090              0.0090
        0.100                 0.0535              0.0069
        0.284                 0.1111              0.0065
        0.500                 0.1495              0.0059
        0.800                 0.1874              0.0053
        =========  =================  ==================

        Strube holds calibration at every correlation; uncorrected Stouffer is
        outside the critical value as soon as rho exceeds ~0.05.

        rho is estimated, so misspecification matters.  At a true rho of 0.284,
        passing 0.0 gives KS 0.1036 and 0.2 gives 0.0232, while 0.4 gives 0.0283
        and 0.9 gives 0.1009.  **Erring high is safe and erring low is not** --
        too large a rho is merely conservative (the score rises), too small
        leaves the overconfidence it was meant to remove.

        It costs no power.  At fixed ``n`` the correction divides Z by a
        constant, so it is a monotone transform: the candidate ordering and the
        AUC are unchanged to machine precision.  What it changes is the
        comparison ACROSS candidates with different epoch counts, where
        uncorrected Stouffer keeps crediting every extra correlated epoch as
        independent evidence -- at 48 epochs it is optimistic by 796x relative
        to Strube, against 1.1x at two epochs.

    Returns
    -------
    dict with keys ``p_combined``, ``Z``, ``count``.
        ``p_combined`` is itself a p-value on (0, 1) — small means inconsistent
        with the kilonova hypothesis, the same orientation as ``P_tail``.
    """
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]

    n = p.size
    if n == 0:
        return {"p_combined": np.nan, "Z": np.nan, "count": 0}

    p_clipped = np.clip(p, eps, 1.0 - eps)
    z = ndtri(1.0 - p_clipped)          # uniform in -> standard normal out

    num = float(np.sum(z))
    # Strube 1985 eq. for equal weights: Var(sum z) = n + rho*n*(n-1).
    var = float(n) + float(rho) * float(n) * float(n - 1)
    if not np.isfinite(var) or var <= 0:
        return {"p_combined": np.nan, "Z": np.nan, "count": int(n)}

    Z = num / np.sqrt(var)
    return {
        "p_combined": float(ndtr(-Z)),   # 1 - Phi(Z), computed stably
        "Z": float(Z),
        "count": int(n),
    }


def stouffer_stats(
    group: pd.DataFrame,
    eps: float = 1e-4,
    rho: float = 0.0,
) -> pd.Series:
    """
    Per-time-bin Stouffer combination, shaped as a drop-in for
    :func:`ivw_stats_logit`.

    Parameters
    ----------
    group : pd.DataFrame
        Subset of the metrics DataFrame for a single time bin.  Must contain
        ``p_tail_mean``.  ``p_tail_std`` is neither required nor read — that is
        the point of the change.
    eps, rho : float
        See :func:`stouffer_combine`.

    Returns
    -------
    pd.Series
        ``mean``  – the combined p-value for this bin.
        ``std``   – standard error of the epoch spread about the mean, i.e. how
                    much the epochs in this bin disagree.  This is NOT a
                    propagated measurement error and is 0.0 for a single epoch;
                    nothing downstream weights by it.
        ``count`` – number of epochs combined.

        All return paths carry all three keys, so ``groupby(...).apply(...)``
        cannot introduce a spurious NaN column for ``dropna`` to act on.
    """
    p = group["p_tail_mean"].to_numpy(dtype=float)
    res = stouffer_combine(p, eps=eps, rho=rho)
    if res["count"] == 0:
        return pd.Series({"mean": np.nan, "std": np.nan, "count": 0})

    finite = p[np.isfinite(p)]
    spread = (float(np.std(finite, ddof=1) / np.sqrt(finite.size))
              if finite.size > 1 else 0.0)
    return pd.Series({
        "mean": res["p_combined"],
        "std": spread,
        "count": res["count"],
    })


def calculate_sequential_score_stouffer(
    p_by_bin,
    eps: float = 1e-4,
    rho: float = 0.0,
):
    """
    Cumulative score after each time bin, by Stouffer combination.

    The running score at bin ``k`` combines every epoch from bins ``0..k``
    directly from their per-epoch p-values, rather than re-combining
    already-combined bin scores.  Combining p-values is associative only if the
    inputs stay uniform, and a combined p-value is uniform but no longer
    independent of its own components, so going back to the raw epochs is both
    simpler and exactly calibrated at every ``k``.

    Parameters
    ----------
    p_by_bin : sequence of array-like
        Per-epoch p-values, grouped by time bin, in chronological order.
    eps, rho : float
        See :func:`stouffer_combine`.

    Returns
    -------
    running_score : np.ndarray
        Cumulative combined p-value after each bin.
    running_err : np.ndarray
        Standard error of the epoch spread accumulated so far.  Reported for
        schema compatibility; it is a dispersion, not a propagated error, and
        nothing consumes it as a weight.
    """
    n = len(p_by_bin)
    running_score = np.full(n, np.nan)
    running_err = np.full(n, np.nan)

    acc_p: List[float] = []
    for i in range(n):
        acc_p.extend(np.asarray(p_by_bin[i], dtype=float).ravel().tolist())
        res = stouffer_combine(acc_p, eps=eps, rho=rho)
        running_score[i] = res["p_combined"]

        finite = np.asarray(acc_p, dtype=float)
        finite = finite[np.isfinite(finite)]
        running_err[i] = (float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                          if finite.size > 1 else 0.0)

    return running_score, running_err

# ---------------------------------------------------------------------------
# Sequential logit-space cumulative score update
# ---------------------------------------------------------------------------

def calculate_sequential_score_logit(
    means: np.ndarray,
    stds: np.ndarray,
    eps: float = 1e-4,
):
    """
    Sequentially update the cumulative P_tail_KNe score in logit space.

    Implements the sequential inverse-variance weighted update from the paper::

        z_new = (z_prev / sigma_prev^2 + z_i / sigma_i^2)
                / (1/sigma_prev^2 + 1/sigma_i^2)

        sigma_new^2 = (1/sigma_prev^2 + 1/sigma_i^2)^{-1}

    all in logit space, with results transformed back via expit.  NaN bins
    are carried forward from the previous step without updating, so a missing
    time bin does not reset or corrupt the running score.

    Parameters
    ----------
    means : np.ndarray
        Per-bin inverse-variance weighted P_tail_KNe means (probability space).
    stds : np.ndarray
        Per-bin inverse-variance weighted P_tail_KNe standard deviations.
    eps : float
        Clamping value before logit transform.

    Returns
    -------
    running_score : np.ndarray
        Cumulative P_tail_KNe score at each time bin (probability space).
    running_error : np.ndarray
        Propagated uncertainty on the cumulative score (probability space).
    """
    means = np.asarray(means, dtype=float)
    stds  = np.asarray(stds, dtype=float)

    n = len(means)
    running_score = np.full(n, np.nan)
    running_error = np.full(n, np.nan)

    if n == 0:
        return running_score, running_error

    means_clipped = np.clip(means, eps, 1.0 - eps)
    z     = logit(means_clipped)
    z_std = stds / (means_clipped * (1.0 - means_clipped))  # delta method

    # A bin is usable only if its precision 1/z_std**2 is finite AND positive.
    # Testing np.isfinite(z_std) alone is not enough: 0.0 is finite, so a
    # zero-variance bin passed the old check and produced infinite precision.
    # The update (finite*finite + z*inf) / inf then evaluates to NaN, and every
    # subsequent bin inherits it.
    valid = np.isfinite(z) & np.isfinite(z_std) & (z_std > 0.0)

    # Initialise from the first VALID bin, not unconditionally from bin 0.
    # Bin 0 was never checked, so a single bad first bin set current_prec to inf
    # and NaN-poisoned the entire running score.  Bins before the first valid
    # one are reported NaN rather than given a fabricated value.
    first = int(np.argmax(valid)) if valid.any() else -1
    if first < 0:
        # No usable bin.  Return all-NaN rather than indexing z[0], which used
        # to raise IndexError on an empty frame.
        return running_score, running_error

    current_z    = z[first]
    current_prec = 1.0 / z_std[first] ** 2
    running_score[first] = float(means_clipped[first])
    running_error[first] = float(stds[first])

    for i in range(first + 1, n):
        if not valid[i]:
            # Carry forward without update
            running_score[i] = running_score[i - 1]
            running_error[i] = running_error[i - 1]
            continue

        new_prec     = 1.0 / z_std[i] ** 2
        updated_prec = current_prec + new_prec
        updated_z    = (current_z * current_prec + z[i] * new_prec) / updated_prec

        current_z    = updated_z
        current_prec = updated_prec

        score_i = float(expit(updated_z))
        running_score[i] = score_i
        running_error[i] = score_i * (1.0 - score_i) * np.sqrt(1.0 / updated_prec)

    return running_score, running_error

def timer_warp(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        # This executes your actual function
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"DEBUG: '{func.__name__}' executed in {duration:.4f} seconds")
        
        return result
    return wrapper

def time_plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"DEBUG: '{func.__name__}' rendered in {end - start:.3f}s")
        return result
    return wrapper


# #OLD VERSION - BETA ####################################################################################################
# def compute_abs_mag_samples(app_mag, app_mag_err, dist_mpc, dist_err_mpc, n_samples=5000):
#     """
#     Convert apparent magnitude to absolute magnitude via Monte Carlo sampling.
#     Pass dist_mpc and dist_err_mpc explicitly.
    
#     """
#     if (not np.isfinite(dist_mpc)) or dist_mpc <= 0:
#         return np.nan, np.nan

#     if np.any(~np.isfinite(app_mag)):
#         return np.nan, np.nan
    
#     app_mag_err = max(0, app_mag_err) if np.isfinite(app_mag_err) else 0.0

#     # 2. Sampling
#     # D_samples in parsecs
#     D_samples = np.random.normal(dist_mpc, dist_err_mpc, n_samples) * 1e6
    
#     # Filter non-physical distances
#     D_samples = D_samples[D_samples > 0]
#     if len(D_samples) < 10:
#         return np.nan, np.nan

#     # 3. Calculation
#     mu_samples = 5 * np.log10(D_samples) - 5
#     app_samples = np.random.normal(app_mag, app_mag_err, len(D_samples))
    
#     abs_mag_samples = app_samples - mu_samples

#     #print('MEAN abs samples:',np.mean(abs_mag_samples), np.std(abs_mag_samples))
    
#     return np.mean(abs_mag_samples), np.std(abs_mag_samples)


# def ivw_stats_logit(group, eps=1e-4):
#     """
#     Inverse-variance weighted aggregation in logit space.
#     Statistically appropriate for bounded scores in (0,1).
#     """
#     from scipy.special import logit, expit

#     p = group['p_tail_mean'].values
#     s = group['p_tail_std'].values

#     mask = np.isfinite(p) & np.isfinite(s) & (s > 0)
#     p, s = p[mask], s[mask]

#     if len(p) == 0:
#         return pd.Series({'mean': np.nan, 'std': np.nan, 'count': 0})

#     # Clamp to avoid infinities
#     p_clipped = np.clip(p, eps, 1.0 - eps)

#     # Transform to logit space
#     z = logit(p_clipped)

#     # Delta-method uncertainty propagation
#     z_std = s / (p_clipped * (1.0 - p_clipped))

#     weights = 1.0 / z_std**2
#     z_mean = np.sum(weights * z) / np.sum(weights)
#     z_std_comb = np.sqrt(1.0 / np.sum(weights))

#     # Transform back
#     mean = expit(z_mean)
#     std = mean * (1.0 - mean) * z_std_comb

#     return pd.Series({
#         'mean': mean,
#         'std': std,
#         'count': len(p)
#     })

# def calculate_sequential_score_logit(means, stds, eps=1e-4):
#     """
#     Sequential update performed in logit space.
#     """
#     from scipy.special import logit, expit

#     n = len(means)
#     running_score = np.zeros(n)
#     running_error = np.zeros(n)

#     # Clamp scores away from 0 and 1
#     means_clipped = np.clip(means, eps, 1 - eps)

#     # Transform to logit space
#     z = logit(means_clipped)

#     # Delta-method uncertainty propagation
#     z_std = stds / (means_clipped * (1 - means_clipped))

#     current_mean = z[0]
#     current_prec = 1.0 / z_std[0]**2

#     running_score[0] = means_clipped[0]
#     running_error[0] = stds[0]

#     for i in range(1, n):
#         if np.isnan(z[i]):
#             running_score[i] = running_score[i-1]
#             running_error[i] = running_error[i-1]
#             continue

#         new_prec = 1.0 / z_std[i]**2
#         updated_prec = current_prec + new_prec
#         updated_mean = (current_mean * current_prec + z[i] * new_prec) / updated_prec

#         current_mean = updated_mean
#         current_prec = updated_prec

#         # Transform back
#         running_score[i] = expit(updated_mean)
#         running_error[i] = running_score[i] * (1 - running_score[i]) * np.sqrt(1.0 / updated_prec)

#     return running_score, running_error


