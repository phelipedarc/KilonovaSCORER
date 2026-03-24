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
from scipy.special import expit, logit

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


