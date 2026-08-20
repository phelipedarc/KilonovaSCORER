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
from scipy.stats import chi2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Apparent → absolute magnitude conversion
# ---------------------------------------------------------------------------

def compute_abs_mag_samples(
    app_mag,
    app_mag_err,
    dist_mpc: float,
    dist_err_mpc: float,
    n_samples: int = 5000,
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

    Returns
    -------
    abs_mag_mean : float or np.ndarray
        Mean absolute magnitude(s).  np.nan for invalid inputs.
    abs_mag_std : float or np.ndarray
        Standard deviation of absolute magnitude(s).  np.nan for invalid inputs.

    Notes
    -----
    When called with arrays, one independent set of ``n_samples`` distance
    draws is shared across all rows (same distance realisation), while
    apparent-magnitude noise is drawn independently per row.  This correctly
    reflects that the distance uncertainty is a global systematic.
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
    if not np.isfinite(dist_mpc) or dist_mpc <= 0:
        logger.warning("Invalid distance dist_mpc=%.3f — returning NaN.", dist_mpc)
        return (float("nan"), float("nan")) if scalar_input else (abs_mag_mean, abs_mag_std)

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
        return (float("nan"), float("nan")) if scalar_input else (abs_mag_mean, abs_mag_std)

    n_valid = len(D_samples)
    mu_samples = 5.0 * np.log10(D_samples) - 5.0  # shape (n_valid,)

    for i in range(n_obs):
        if not np.isfinite(app_mag[i]):
            continue  # leave as NaN

        # Independent apparent-magnitude noise per observation
        app_samples = np.random.normal(app_mag[i], app_mag_err[i], n_valid)
        abs_samples = app_samples - mu_samples

        abs_mag_mean[i] = np.mean(abs_samples)
        abs_mag_std[i]  = np.std(abs_samples)

    if scalar_input:
        return float(abs_mag_mean[0]), float(abs_mag_std[0])
    return abs_mag_mean, abs_mag_std


# ---------------------------------------------------------------------------
# Logit-space inverse-variance weighted aggregation
# ---------------------------------------------------------------------------

def ivw_stats_logit(group: pd.DataFrame, eps: float = 1e-4) -> pd.Series:
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
        transform (prevents infinite logit values).

    Returns
    -------
    pd.Series
        ``mean``  – inverse-variance weighted mean (probability space).
        ``std``   – propagated uncertainty (probability space).
        ``count`` – number of valid scores used.
    """
    #Handling the zero scores:
    # inside ivw_stats_logit, before computing weights:
    valid = (group["p_tail_std"] > 0) & (group["p_tail_mean"] > 0)
    group = group[valid]
    if group.empty:
        return pd.Series({"mean": 0.0, "std": 0.0})
      
    p = group["p_tail_mean"].to_numpy()
    s = group["p_tail_std"].to_numpy()

    mask = np.isfinite(p) & np.isfinite(s) & (s > 0)
    p, s = p[mask], s[mask]

    if len(p) == 0:
        return pd.Series({"mean": np.nan, "std": np.nan, "count": 0})

    p_clipped = np.clip(p, eps, 1.0 - eps)
    z     = logit(p_clipped)
    z_std = s / (p_clipped * (1.0 - p_clipped))    # delta method
    weights = 1.0 / z_std ** 2

    z_mean     = np.sum(weights * z) / np.sum(weights)
    z_std_comb = np.sqrt(1.0 / np.sum(weights))

    mean = float(expit(z_mean))
    std  = float(mean * (1.0 - mean) * z_std_comb)  # delta method back-transform

    return pd.Series({"mean": mean, "std": std, "count": len(p)})


# ---------------------------------------------------------------------------
# Stouffer combination of per-epoch p-values
# ---------------------------------------------------------------------------
#
# WHY NOT INVERSE-VARIANCE WEIGHTING
# ----------------------------------
# IVW assumes several noisy measurements of ONE SHARED QUANTITY, each with a
# known variance; then sum(w_i z_i)/sum(w_i) with w_i = 1/sigma_i^2 is the
# minimum-variance unbiased combination.  That is not the situation here.
#
# Under the null each P_tail_i is a P-VALUE, uniform on (0, 1) — this is the
# probability integral transform, measured holding at KS 0.018.  Six uniforms
# are not six noisy estimates of a common mean.  There is no shared parameter to
# average toward, so IVW's optimality theorem simply does not apply, and the
# framework has no slot for a per-observation uncertainty.  That is why every
# candidate definition of ``p_tail_std`` felt arbitrary: they are three answers
# to a question that should not be asked.  Under the null, a badly-measured
# epoch's p-value is not "noisier" — it is exactly as uniform as a well-measured
# one.
#
# It is not merely unmotivated, it is harmful.  Measured on held-out kilonovae,
# where the null is true by construction and a correct combiner must return a
# uniform score:
#
#     per-epoch P_tail (the input)         KS 0.018
#     IVW logit, p_tail_std weights        KS 0.204   <- production
#     Fisher (assumes independence)        KS 0.159
#     IVW logit, EQUAL weights             KS 0.137
#     Stouffer Z                           KS 0.115
#     Brown (correlation-corrected Fisher) KS 0.050
#
# The combiner was an order of magnitude worse than its own input, and deleting
# the weights entirely improved it.  Two further defects compound this: the delta
# method sigma_z = sigma_p/(p(1-p)) is a first-order linearisation requiring
# sigma_p << p(1-p), which fails outright at small P_tail (at P_tail = 0.0055 the
# measured sigma_p = 0.0094 EXCEEDS the probability); and with sigma_obs
# identical at every row the weight still varied 67x with where the observation
# sat relative to the population — backwards, so that the point most strongly
# rejecting the hypothesis got weight 0.34 while a mildly informative one got
# 22.9.
#
# WHAT STOUFFER CHANGES
# ---------------------
# Two changes, one cosmetic and one that is the whole point.
#
#   now       z_i = logit(p_i)          Z = sum(w_i z_i) / sum(w_i)      AVERAGE
#   proposed  z_i = Phi^-1(1 - p_i)     Z = sum(w_i z_i) / sqrt(sum(w_i^2))  SUM
#
# The link function logit -> Phi^-1 is cosmetic; the two agree to a scale factor
# (logit(p) ~ 1.70 * Phi^-1(p)) across the range that matters.  It is swapped
# because Phi^-1 is the inverse of the NORMAL CDF, which is what makes the
# normaliser exact.
#
# The normaliser sum(w) -> sqrt(sum(w^2)) is the real change.  Dividing by
# sum(w) produces a weighted average — a number with no known null distribution.
# Dividing by sqrt(sum(w^2)) produces a STANDARDISED TEST STATISTIC.  Under the
# null each p_i is uniform, so z_i = Phi^-1(1 - p_i) is exactly N(0, 1).  A
# weighted sum of independent normals is normal with mean sum(w_i)*0 = 0 and
# variance sum(w_i^2)*1, so
#
#     Z ~ N(0, 1)   EXACTLY, for ANY fixed positive weights.
#
# The weights appear in the numerator and the normaliser and cancel.  Under IVW
# the weights WERE the calibration, and getting them wrong miscalibrated the
# output — which is exactly what was measured.  Under Stouffer the weights
# cannot break calibration; they affect only POWER, i.e. how well real kilonovae
# separate from contaminants.  That converts an unanswerable question ("what is
# the uncertainty on a p-value?") into a measurable one ("which weights best
# separate kilonovae from supernovae?").
#
# WHY NOT BROWN, WHICH MEASURED BEST
# ----------------------------------
# Brown's method reaches KS 0.050 by moment-matching a scaled chi-square to the
# observed covariance of -2 ln p.  It needs that covariance estimated across the
# ENSEMBLE of candidates, which a per-candidate function cannot see, and the
# better-founded alternative — deriving the correlation from the simulation grid
# at the candidate's own cadence — is unimplemented and untested.  Stouffer with
# equal weights is the step that is safe and provable today; ``rho`` below is the
# opt-in hook for the correlation correction when it is ready.


def stouffer_combine(
    p,
    weights=None,
    eps: float = 1e-4,
    rho: float = 0.0,
):
    """
    Combine p-values by the weighted Stouffer method.

    Parameters
    ----------
    p : array-like
        Per-epoch p-values (``p_tail_mean``).  Non-finite entries are dropped.
        Zeros are KEPT and clamped to ``eps``: a categorically-rejected epoch is
        the strongest evidence available, not missing data.
    weights : array-like or None
        Per-epoch weights.  ``None`` (default) means equal weights.  Any fixed
        positive weights leave the null distribution of Z exactly standard
        normal, so this cannot affect calibration — only power.
    eps : float
        Clamp keeping p away from 0 and 1, where ``Phi^-1`` is infinite.
    rho : float
        Exchangeable inter-epoch correlation, used to correct the normaliser::

            Var(sum w_i z_i) = sum_i sum_j w_i w_j rho_ij

        Epochs of one candidate are genuinely correlated — same object, smooth
        light curve, one shared distance — with a measured mean of 0.284.
        Positive correlation makes the true variance larger than ``sum(w^2)``,
        so the default ``rho = 0`` is OVERCONFIDENT by a known direction.  It is
        left at 0 because the well-founded estimate of rho (from the grid, at
        this candidate's own cadence) is not implemented yet, and borrowing one
        number from the literature would be a guess wearing a measurement's
        clothes.

    Returns
    -------
    dict with keys ``p_combined``, ``Z``, ``count``.
        ``p_combined`` is itself a p-value on (0, 1) — small means inconsistent
        with the kilonova hypothesis, the same orientation as ``P_tail``.
    """
    p = np.asarray(p, dtype=float)
    finite = np.isfinite(p)
    p = p[finite]

    if weights is None:
        w = np.ones(p.size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)[finite]
        bad = ~np.isfinite(w) | (w <= 0)
        w = np.where(bad, 0.0, w)

    keep = w > 0
    p, w = p[keep], w[keep]

    if p.size == 0:
        return {"p_combined": np.nan, "Z": np.nan, "count": 0}

    p_clipped = np.clip(p, eps, 1.0 - eps)
    z = ndtri(1.0 - p_clipped)          # uniform in -> standard normal out

    num = float(np.sum(w * z))
    var = float(np.sum(w ** 2))
    if rho:
        # cross-terms: sum_{i != j} w_i w_j = (sum w)^2 - sum w^2
        cross = float(np.sum(w) ** 2 - np.sum(w ** 2))
        var = var + float(rho) * cross
    if not np.isfinite(var) or var <= 0:
        return {"p_combined": np.nan, "Z": np.nan, "count": int(p.size)}

    Z = num / np.sqrt(var)
    return {
        "p_combined": float(ndtr(-Z)),   # 1 - Phi(Z), computed stably
        "Z": float(Z),
        "count": int(p.size),
    }


def brown_combine(
    p,
    weights=None,
    eps: float = 1e-4,
    rho: float = 0.0,
):
    """
    Combine p-values by Brown's method — Fisher, moment-matched for correlation.

    Fisher's statistic ``X2 = -2 sum ln(p_i)`` is chi2(2k) under independence.
    Brown (1975) rescales it for correlated p-values by matching the first two
    moments of a scaled chi-square::

        X2 / c ~ chi2(f),   c = Var(X2) / (2 E[X2]),   f = 2 E[X2]^2 / Var(X2)

    with ``E[X2] = 2k`` and ``Var(X2) = 4k + sum_{i!=j} cov(-2 ln p_i, -2 ln p_j)``.
    The covariance uses the Kost & McDermott (2002) polynomial in the underlying
    correlation, which refines Brown's original fit::

        cov ~ 3.263 rho + 0.710 rho^2 + 0.027 rho^3

    ``f`` is the EFFECTIVE NUMBER OF INDEPENDENT EPOCHS, which is the useful
    diagnostic this method produces for free.

    WHEN TO PREFER THIS OVER STOUFFER.  Measured on correlated epochs with both
    methods given the same rho, Brown is the better calibrated of the two —
    KS 0.0210 vs 0.0259 at rho = 0.06, and 0.0617 vs 0.0695 at rho = 0.26.  If
    calibration is the only criterion, use this.

    THREE REASONS IT IS NOT THE DEFAULT.

    1. It needs ``rho``, and with ``rho = 0`` it reduces EXACTLY to Fisher —
       which is the worst option of all under real correlation (KS 0.1547 at
       rho = 0.26, against 0.1141 for Stouffer with no correction at all).  So
       running this without a correlation estimate is worse than not using it.
       Most of Brown's published advantage is the correlation correction, not
       Fisher's combination rule.
    2. It is far more sensitive to a single very small p-value.  With one
       corrupted photometric point among six good epochs, the median score fell
       by 7.1x under Brown against 2.7x under correlated Stouffer.  That
       sensitivity is a virtue against a genuine outlier epoch and a liability
       against a bad subtraction, which a real-time stream produces routinely.
    3. There is no natural slot for weights.  Stouffer takes arbitrary positive
       weights with its null distribution exactly preserved, which is where the
       per-epoch informativeness implied by ``M_lim`` has to go.  ``weights`` is
       accepted here and IGNORED, for signature compatibility only.

    Returns
    -------
    dict with keys ``p_combined``, ``X2``, ``f`` (effective dof), ``count``.
    """
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return {"p_combined": np.nan, "X2": np.nan, "f": np.nan, "count": 0}

    k = int(p.size)
    X2 = float(-2.0 * np.log(np.clip(p, eps, 1.0)).sum())
    E = 2.0 * k
    r = float(rho)
    cov = 3.263 * r + 0.710 * r ** 2 + 0.027 * r ** 3
    Var = 4.0 * k + k * (k - 1) * cov
    if not np.isfinite(Var) or Var <= 0:
        Var = 4.0 * k                       # fall back to independence
    c = Var / (2.0 * E)
    f = 2.0 * E ** 2 / Var
    return {
        "p_combined": float(chi2.sf(X2 / c, f)),
        "X2": X2,
        "f": float(f),
        "count": k,
    }


def stouffer_stats(
    group: pd.DataFrame,
    eps: float = 1e-4,
    weight_col: Optional[str] = None,
    rho: float = 0.0,
    combiner=None,
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
    weight_col : str or None
        Column holding per-epoch weights.  ``None`` means equal weights, which
        is the only choice justified so far; see the module notes above.

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
    w = (group[weight_col].to_numpy(dtype=float)
         if weight_col is not None and weight_col in group else None)

    combine = stouffer_combine if combiner is None else combiner
    res = combine(p, weights=w, eps=eps, rho=rho)
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
    weights_by_bin=None,
    eps: float = 1e-4,
    rho: float = 0.0,
    combiner=None,
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
    weights_by_bin : sequence of array-like or None
        Matching per-epoch weights.  ``None`` means equal weights throughout.
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

    combine = stouffer_combine if combiner is None else combiner
    acc_p: List[float] = []
    acc_w: List[float] = []
    for i in range(n):
        acc_p.extend(np.asarray(p_by_bin[i], dtype=float).ravel().tolist())
        if weights_by_bin is None:
            acc_w.extend([1.0] * np.asarray(p_by_bin[i]).size)
        else:
            acc_w.extend(np.asarray(weights_by_bin[i], dtype=float).ravel().tolist())

        res = combine(acc_p, weights=acc_w, eps=eps, rho=rho)
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
    n = len(means)
    running_score = np.zeros(n)
    running_error = np.zeros(n)

    means_clipped = np.clip(means, eps, 1.0 - eps)
    z     = logit(means_clipped)
    z_std = stds / (means_clipped * (1.0 - means_clipped))  # delta method

    # Initialise from first bin
    current_z    = z[0]
    current_prec = 1.0 / z_std[0] ** 2
    running_score[0] = float(means_clipped[0])
    running_error[0] = float(stds[0])

    for i in range(1, n):
        if not np.isfinite(z[i]) or not np.isfinite(z_std[i]):
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


