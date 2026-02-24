import time
from functools import wraps
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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



def compute_abs_mag_samples(app_mag, app_mag_err, dist_mpc, dist_err_mpc, n_samples=5000):
    """
    Convert apparent magnitude to absolute magnitude via Monte Carlo sampling.
    Pass dist_mpc and dist_err_mpc explicitly.
    
    """
    if (not np.isfinite(dist_mpc)) or dist_mpc <= 0:
        return np.nan, np.nan

    if np.any(~np.isfinite(app_mag)):
        return np.nan, np.nan
    
    app_mag_err = max(0, app_mag_err) if np.isfinite(app_mag_err) else 0.0

    # 2. Sampling
    # D_samples in parsecs
    D_samples = np.random.normal(dist_mpc, dist_err_mpc, n_samples) * 1e6
    
    # Filter non-physical distances
    D_samples = D_samples[D_samples > 0]
    if len(D_samples) < 10:
        return np.nan, np.nan

    # 3. Calculation
    mu_samples = 5 * np.log10(D_samples) - 5
    app_samples = np.random.normal(app_mag, app_mag_err, len(D_samples))
    
    abs_mag_samples = app_samples - mu_samples

    #print('MEAN abs samples:',np.mean(abs_mag_samples), np.std(abs_mag_samples))
    
    return np.mean(abs_mag_samples), np.std(abs_mag_samples)


def ivw_stats_logit(group, eps=1e-4):
    """
    Inverse-variance weighted aggregation in logit space.
    Statistically appropriate for bounded scores in (0,1).
    """
    from scipy.special import logit, expit

    p = group['p_tail_mean'].values
    s = group['p_tail_std'].values

    mask = np.isfinite(p) & np.isfinite(s) & (s > 0)
    p, s = p[mask], s[mask]

    if len(p) == 0:
        return pd.Series({'mean': np.nan, 'std': np.nan, 'count': 0})

    # Clamp to avoid infinities
    p_clipped = np.clip(p, eps, 1.0 - eps)

    # Transform to logit space
    z = logit(p_clipped)

    # Delta-method uncertainty propagation
    z_std = s / (p_clipped * (1.0 - p_clipped))

    weights = 1.0 / z_std**2
    z_mean = np.sum(weights * z) / np.sum(weights)
    z_std_comb = np.sqrt(1.0 / np.sum(weights))

    # Transform back
    mean = expit(z_mean)
    std = mean * (1.0 - mean) * z_std_comb

    return pd.Series({
        'mean': mean,
        'std': std,
        'count': len(p)
    })

def calculate_sequential_score_logit(means, stds, eps=1e-4):
    """
    Sequential update performed in logit space.
    """
    from scipy.special import logit, expit

    n = len(means)
    running_score = np.zeros(n)
    running_error = np.zeros(n)

    # Clamp scores away from 0 and 1
    means_clipped = np.clip(means, eps, 1 - eps)

    # Transform to logit space
    z = logit(means_clipped)

    # Delta-method uncertainty propagation
    z_std = stds / (means_clipped * (1 - means_clipped))

    current_mean = z[0]
    current_prec = 1.0 / z_std[0]**2

    running_score[0] = means_clipped[0]
    running_error[0] = stds[0]

    for i in range(1, n):
        if np.isnan(z[i]):
            running_score[i] = running_score[i-1]
            running_error[i] = running_error[i-1]
            continue

        new_prec = 1.0 / z_std[i]**2
        updated_prec = current_prec + new_prec
        updated_mean = (current_mean * current_prec + z[i] * new_prec) / updated_prec

        current_mean = updated_mean
        current_prec = updated_prec

        # Transform back
        running_score[i] = expit(updated_mean)
        running_error[i] = running_score[i] * (1 - running_score[i]) * np.sqrt(1.0 / updated_prec)

    return running_score, running_error


