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
        # Must carry the SAME three keys as every other return path.  This
        # branch previously returned a two-key Series {"mean", "std"}, and
        # pandas' groupby.apply only assembles a DataFrame when the Series it
        # collects share an index -- so as soon as one bin was fully filtered
        # out and another was not, the result came back as a Series of Series
        # and the caller died on `binned_stats["mean"]` with KeyError: 'mean'.
        # That fired on exactly the objects this combiner should reject most
        # confidently (every epoch p_tail_mean == 0), 44 of 250 simulated
        # SN IIn/Ibn at four nights.  Returning NaN rather than 0.0 matches the
        # `len(p) == 0` branch below and lets the caller's dropna() drop the
        # bin, which is what the filter above already intends; it changes no
        # number this function has ever successfully produced.
        #
        # The filter itself -- dropping p_tail_mean == 0 epochs that do carry
        # rejecting power -- is a separate defect, fixed on branch
        # `zero-epochs-handling`, not here.
        return pd.Series({"mean": np.nan, "std": np.nan, "count": 0})
      
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
        Per-epoch weights.  ``None`` (default) means equal weights, and that
        default is now a MEASURED result rather than a placeholder — see
        ``trove_tests/docs/WEIGHTS.md``.  Seven schemes were tested on held-out
        kilonovae plus three contaminant classes; none beat equal weighting
        beyond noise, and neither did the oracle weight ``w ~ mu_i`` that is
        provably optimal by Cauchy-Schwarz.  The ceiling on any weighting gain
        is ``sqrt(1 + CV^2)`` with CV the spread of per-epoch informativeness,
        measured at 0.060 -> a 0.18% gain, two orders of magnitude below the
        AUC noise floor.  Epoch informativeness is set by the grid's own
        population spread ``tau`` (0.21-0.56 mag), which dominates ``sigma_obs``
        and varies little across epochs.

        Weights must be ANCILLARY: fixed with respect to z, i.e. functions of
        the observing conditions only.  Any fixed positive weights leave the
        null distribution of Z exactly standard normal, so they cannot affect
        calibration — only power.  Weights that peek at the p-value being
        weighted are NOT fixed and do break it: on 40000 synthetic draws,
        ``w = 1/(p+0.01)`` gives KS 0.4316 against 0.0045 for equal weights,
        while a fixed but extreme ``w = (1,1,1,1,1,1000)`` gives 0.0032.  In the
        real pipeline ``w ~ 1/p_tail_std`` halves the median score of genuine
        kilonovae (0.517 -> 0.313) and more than doubles KS (0.097 -> 0.234).
        Do not weight by ``p_tail_std`` or anything else derived from
        ``p_tail_mean``.
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


def stouffer_stats(
    group: pd.DataFrame,
    eps: float = 1e-4,
    weight_col: Optional[str] = None,
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
    weight_col : str or None
        Column holding per-epoch weights.  ``None`` means equal weights, which
        is what the power test selected: no ancillary scheme beat it beyond
        noise and neither did the provably-optimal oracle weight.  See
        ``trove_tests/docs/WEIGHTS.md`` and the ``weights`` note on
        :func:`stouffer_combine`.  Any column named here must be ANCILLARY --
        derived from observing conditions, never from ``p_tail_mean`` or
        ``p_tail_std``.

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

    res = stouffer_combine(p, weights=w, eps=eps, rho=rho)
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

    acc_p: List[float] = []
    acc_w: List[float] = []
    for i in range(n):
        acc_p.extend(np.asarray(p_by_bin[i], dtype=float).ravel().tolist())
        if weights_by_bin is None:
            acc_w.extend([1.0] * np.asarray(p_by_bin[i]).size)
        else:
            acc_w.extend(np.asarray(weights_by_bin[i], dtype=float).ravel().tolist())

        res = stouffer_combine(acc_p, weights=acc_w, eps=eps, rho=rho)
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


