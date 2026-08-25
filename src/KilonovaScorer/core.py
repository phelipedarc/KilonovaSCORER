import numpy as np
from scipy.stats import gaussian_kde
from typing import Dict, Union
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import gaussian_kde

import sys
import time
def arcade_progress_bar(current, total, bar_length=30):
    """
    Prints an arcade-style progress bar to the console.
    """
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r[ {bar} ] {percent*100:6.2f}% ⬛')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')

# Internal imports from your own package
from .utils import *  # if you use the decorator here

def parse_json_photometry(file_path: Path, merger_mjd: float):
    """
    Extracts photometry from the specific JSON schema.
    Returns a DataFrame with raw band names ready for the FILTER_LOOKUP.
    """
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            #logger.error(f"Failed to decode JSON from {file_path}")
            return pd.DataFrame()

    results = []
    
    # Check for top-level photometry key
    if "photometry" not in data:
        #logger.warning(f"No photometry key found in {file_path}")
        return pd.DataFrame()

    for entry in data["photometry"]:
        # 1. Get and Validate Time
        t = entry.get("timestamp")
        if t is None or t < merger_mjd:
            continue

        # 2. Extract nested magnitude and filter data
        val = entry.get("value", {})
        app_mag = val.get("magnitude")
        app_err = val.get("error", 0)
        raw_filter = val.get("filter")

        # 3. Validation & Quality Control
        # Ignore non-detections (upper limits) or missing data
        is_upper_limit = val.get("upper_limit", False)
        
        if app_mag is None or raw_filter is None or is_upper_limit:
            continue

        # 4. Append Standardized Dictionary
        # We keep "band" as the raw string (e.g., 'ztfg') so the 
        # FILTER_LOOKUP in the pipeline can handle the mapping.
        results.append({
            "time": t,
            "time_after_gw": t - merger_mjd,
            "magnitude": float(app_mag),
            "e_magnitude": float(app_err),
            "band": str(raw_filter).lower().strip(),
            "instrument": entry.get("instrument", "unknown"),
            "telescope": entry.get("telescope", "unknown")
        })

    # Create DataFrame
    df = pd.DataFrame(results)
    return df


def load_observations(file_path, merger_mjd, dist_mpc, dist_err_mpc):
    """
    Primary Entry Point: Detects file type and loads data into a 
    standardized DataFrame, then computes absolute magnitudes.
    """
    from pathlib import Path

    path = Path(file_path)
    print('Loading Data: ',path)
    
    # 1. Load Raw Data
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        # Note: Ensure your CSV has a column named 'time' and 'band'
        df['time_after_gw'] = df['time'] - merger_mjd
    
    elif path.suffix.lower() == '.json':
        # We pass merger_mjd to the json parser to filter pre-merger data
        df = parse_json_photometry(file_path, merger_mjd)
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if df.empty:
        return df

    # 2. Compute Absolute Magnitudes (Monte Carlo)

    results = df.apply(
        lambda row: compute_abs_mag_samples(
            row['magnitude'], 
            row['e_magnitude'],
            dist_mpc=dist_mpc,
            dist_err_mpc=dist_err_mpc
        ), 
        axis=1
    )
    
    df['absolute_magnitude'] = [res[0] for res in results]
    df['absolute_magnitude_error'] = [res[1] for res in results]
    
    return df

def preprocess_lsst_like(
    data_obs,
    bands=("g-band", "z-band"),
    time_col="time_after_gw",
    band_col="filter_mapped",
    strategy="earliest", 
    ):
    """
    Downsamples high-cadence observational data to mimic a standard LSST-like 
    survey cadence (typically one observation per band per night).

    This is primarily used to make over-sampled events (like AT2017gfo) 
    comparable to standard kilonova candidates by reducing data density.

    Parameters:
    -----------
    data_obs : pandas.DataFrame
        The raw observational data containing timestamps and filter bands.
    bands : tuple of str, default=("g-band", "z-band")
        The specific filters to retain in the processed dataset.
    time_col : str, default="time_after_gw"
        The column name representing the time since the merger event.
    band_col : str, default="filter_mapped"
        The column name identifying the filter/band for each observation.
    strategy : {'earliest', 'snr', 'random'}, default="earliest"
        The logic used to select a single point if multiple observations 
        exist within the same time bin:
        - "earliest": Pick the observation with the smallest timestamp.
        - "snr": Pick the observation with the highest Signal-to-Noise Ratio.
        - "random": Randomly select one observation.

    Returns:
    --------
    pandas.DataFrame
        A downsampled version of the input data conforming to the chosen strategy.
    """
    df = data_obs.copy()

    # 1. Keep only desired bands
    df = df[df[band_col].isin(bands)].copy()

    # 2. Define LSST-style observing day
    df["day"] = np.floor(df[time_col]).astype(int)

    # 3. Sort for deterministic selection
    df = df.sort_values(time_col)

    # 4. Select at most one obs per (day, band)
    if strategy == "earliest":
        df_lsst = (
            df.groupby(["day", band_col], as_index=False)
              .first()
        )

    elif strategy == "snr":
        df["snr"] = 1.0 / df["e_magnitude"]
        df_lsst = (
            df.sort_values("snr", ascending=False)
              .groupby(["day", band_col], as_index=False)
              .first()
        )
        df_lsst = df_lsst.drop(columns="snr")

    elif strategy == "random":
        df_lsst = (
            df.groupby(["day", band_col], as_index=False)
              .sample(n=1, random_state=42)
        )

    else:
        raise ValueError("strategy must be 'earliest', 'snr', or 'random'")

    # 5. Final cleanup
    df_lsst = df_lsst.sort_values(time_col).reset_index(drop=True)

    return df_lsst




def predictive_tail_kde_python(
    sim_values: np.ndarray, 
    x0: float, 
    sigma: float, 
    k: float = 1.0, 
    n_sim: int = 50000):
    """
    Calculate the posterior predictive tail probability and Region of Practical 
    Equivalence (ROPE) metrics using Kernel Density Estimation (KDE).

    This implementation uses a vectorized version of 'Charlie's Method' to 
    estimate uncertainty in the tail probability by sampling the observational 
    uncertainty space.

    Args:
        sim_values (np.ndarray): Array of simulated model values (e.g., magnitudes).
        x0 (float): The observed value to compare against the simulations.
        sigma (float): The observational uncertainty (standard deviation) of x0.
        k (float, optional): The scaling factor for the 'prob_near' calculation 
            (ROPE width). Defaults to 1.0.
        n_sim (int, optional): Number of Monte Carlo samples to draw from the 
            KDE and noise distributions. Defaults to 50000.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "F_hat": The cumulative distribution function value at x0.
            - "p_tail": The two-sided tail probability.
            - "p_tail_mean": Mean p-tail value across sampled observations.
            - "p_tail_std": Standard deviation (uncertainty) of the p-tail.
            - "prob_near": Probability that a simulation falls within k*sigma of x0.

    Raises:
        ValueError: If sim_values is empty or sigma is non-positive.
    """
    # 1. Input Validation
    sim_values = np.asarray(sim_values)
    if sim_values.size == 0:
        raise ValueError("sim_values array cannot be empty.")
    if sigma <= 0:
        raise ValueError("sigma (uncertainty) must be a positive value.")

    # 2. KDE Generation and Resampling
    kde = gaussian_kde(sim_values)
    
    # Population samples + observational noise (X* ~ KDE, Y = X* + epsilon)
    x_star = kde.resample(n_sim)[0]
    y_dist = x_star + np.random.normal(0, sigma, size=n_sim)

    # 3. Standard Metrics Calculation
    f_hat = np.mean(y_dist <= x0)
    p_tail = 2 * min(f_hat, 1 - f_hat)
    prob_near = np.mean(np.abs(y_dist - x0) <= k * sigma)

    # 4. Uncertainty Estimation (Vectorized Charlie's Method)
    # Generate samples of the observation x0 based on its uncertainty sigma
    x0_samples = np.random.normal(x0, sigma, 100)
    
    # Broadcast comparison: (100, 1) vs (n_sim,) -> (100, n_sim)
    f_hat_samples = (y_dist <= x0_samples[:, np.newaxis]).mean(axis=1)
    
    # Calculate two-sided p-tail for all samples
    p_tail_samples = 2 * np.minimum(f_hat_samples, 1 - f_hat_samples)

    return {
        "F_hat": float(f_hat),
        "p_tail": float(p_tail),
        "p_tail_mean": float(np.mean(p_tail_samples)),
        "p_tail_std": float(np.std(p_tail_samples)),
        "prob_near": float(prob_near)
    }

def compute_consistent_ids_anyhit(sim_band, bin_idx, M_obs, sigma_obs, overlap_k=2.0):
    """
    Conservative ROPE:
    keep sample_id if ANY point in the bin falls within ROPE
    
    """
    sim_bin = sim_band.loc[
        sim_band["time_bin"] == bin_idx, ["sample_id", "absolute_magnitude"]
    ]
    if sim_bin.empty:
        return []

    rope = overlap_k * sigma_obs
    inside = np.abs(sim_bin["absolute_magnitude"].to_numpy() - M_obs) <= rope
    return sim_bin.loc[inside, "sample_id"].dropna().unique().tolist()

def overlap_chain(ids_lists, times):
    """
    Given per-observation ID lists in time order, compute:
      - pairwise overlaps (S_i ∩ S_{i+1})
      - running survivors (⋂_{j<=i} S_j)
    Returns a dict with per-step diagnostics + final survivors.
    """
    # Ensure ordering by time
    order = np.argsort(times)
    times = np.asarray(times)[order]
    sets = [set(ids_lists[i]) for i in order]

    if len(sets) == 0:
        return {
            "times": [],
            "pairwise": [],
            "survivors_over_time": [],
            "final_survivors": [],
            "final_n_survivors": 0,
        }

    # Running intersection survivors
    survivors = sets[0].copy()
    survivors_over_time = [{
        "t": float(times[0]),
        "n_survivors": len(survivors),
        "survivor_ids": sorted(survivors),
    }]

    # Pairwise overlaps
    pairwise = []
    for i in range(len(sets) - 1):
        inter = sets[i].intersection(sets[i + 1])
        pairwise.append({
            "t_left": float(times[i]),
            "t_right": float(times[i + 1]),
            "n_overlap": len(inter),
            "overlap_ids": sorted(inter),
        })

        # Update running survivors after seeing next observation
        survivors = survivors.intersection(sets[i + 1])
        survivors_over_time.append({
            "t": float(times[i + 1]),
            "n_survivors": len(survivors),
            "survivor_ids": sorted(survivors),
        })

    return {
        "times": times.tolist(),
        "pairwise": pairwise,
        "survivors_over_time": survivors_over_time,
        "final_survivors": sorted(survivors),
        "final_n_survivors": len(survivors),
    }

    
def binned_stats_cumulative_ptail(metric_df,bin_size=0.2):
    #this function can be improved by saving time_bin during the New_obsmetric func
    # --- 4. Prepare Sequential Data ---
    bin_edges = np.arange(metric_df['obs_time'].min()-bin_size/2, 
                          metric_df['obs_time'].max() + bin_size, bin_size)
    metric_df['time_bin'] = pd.cut(metric_df['obs_time'], bins=bin_edges)
    binned_stats = metric_df.groupby('time_bin', observed=True).apply(ivw_stats_logit).reset_index()
    binned_stats['time_mid'] = binned_stats['time_bin'].apply(lambda x: x.mid)
    binned_stats= binned_stats.dropna()
    running_mean, running_err = calculate_sequential_score_logit(
        binned_stats['mean'].values, 
        binned_stats['std'].values
    )
    binned_stats['running_mean'] = running_mean
    binned_stats['running_std'] = running_err

    return binned_stats

######################################################################################################

# def kilonovascorer_v1(
#     data_obs,
#     data_sim,
#     candidate_name,
#     time_bin_width=0.2,
#     band_list=("g-band", "r-band", "i-band", "z-band"),
#     k_near=1.0,
#     n_kde_sim=50000,
#     min_sim_points=20,
#     overlap_k=2.0,
# ):
#     """
#     - Stores per-observation: consistent_ids (list) + n_consistent_lcs.
#     - For each band: computes pairwise overlaps and running survivors (intersection).
#     - Adds overlap diagnostics back into the per-observation rows.
#     - Returns:
#         (results_df, overlap_summary_by_band)
#     """
#     results = []
#     overlap_summary_by_band = {}

#     for band in band_list:
#         sim_band = data_sim[data_sim["filter_mapped"] == band]
#         obs_band = data_obs[data_obs["filter_mapped"] == band]

#         if sim_band.empty or obs_band.empty:
#             continue

#         # bins from observations -- Standard method (bin using the observations starting value and last value) 
#         #note: This can be speed up by loading the simulation already binned.
#         t_start = obs_band["time_after_gw"].min() - time_bin_width / 2
#         t_end = obs_band["time_after_gw"].max() + time_bin_width
#         bins = np.arange(t_start, t_end, time_bin_width)

#         sim_band = sim_band.copy()
#         sim_band["time_bin"] = np.digitize(sim_band["time"], bins)

#         # per-band storage for overlap
#         band_times = []
#         band_ids_lists = []
#         band_row_indices = []  # indices into `results` list

#         # loop over observations
#         for _, obs_row in obs_band.iterrows():
#             t_obs = float(obs_row["time_after_gw"])
#             M_obs = float(obs_row["absolute_magnitude"])
#             sigma_obs = float(obs_row["absolute_magnitude_error"])

#             if (not np.isfinite(M_obs)) or (not np.isfinite(sigma_obs)) or (sigma_obs <= 0):
#                 continue

#             bin_idx = int(np.digitize(t_obs, bins))
#             sim_bin = sim_band[sim_band["time_bin"] == bin_idx]

#             # time-bin edges
#             t_low = bins[bin_idx - 1] if bin_idx > 0 else bins[0]
#             t_high = bins[bin_idx] if bin_idx < len(bins) else bins[-1]

#             if len(sim_bin) < min_sim_points:
#                 continue

#             # KDE-based metrics
#             metric = predictive_tail_kde_python(
#                 sim_bin["absolute_magnitude"].values,
#                 M_obs,
#                 sigma_obs,
#                 k=k_near,
#                 n_sim=n_kde_sim,
#             )

#             # Consistent IDs (any-hit)
#             consistent_ids = compute_consistent_ids_anyhit(
#                 sim_band=sim_band,
#                 bin_idx=bin_idx,
#                 M_obs=M_obs,
#                 sigma_obs=sigma_obs,
#                 overlap_k=overlap_k,
#             )

#             row = {
#                 "candidate_name": candidate_name,
#                 "band": band,
#                 "obs_time": t_obs,
#                 "time_bin_low": float(t_low),
#                 "time_bin_high": float(t_high),
#                 "observed_mag": M_obs,
#                 "observed_mag_err": sigma_obs,
#                 "p_tail": metric["p_tail"],
#                 "p_tail_mean": metric["p_tail_mean"],
#                 "p_tail_std": metric["p_tail_std"],
#                 "prob_near": metric["prob_near"],
#                 "n_sim_bin": int(len(sim_bin)),
#                 "n_consistent_lcs": int(len(consistent_ids)),
#                 "consistent_ids": consistent_ids,   # <-- stored list of IDs
#                 # placeholders to be filled after overlap chain
#                 "overlap_with_next_n": np.nan,
#                 "overlap_with_next_ids": [],
#                 "running_survivors_n": np.nan,
#                 "running_survivors_ids": [],
#             }

#             results.append(row)

#             band_times.append(t_obs)
#             band_ids_lists.append(consistent_ids)
#             #print(consistent_ids)
#             band_row_indices.append(len(results) - 1)

#         # build overlap chain for this band
#         chain = overlap_chain(band_ids_lists, band_times)
#         overlap_summary_by_band[band] = chain

#         # attach pairwise overlaps to the LEFT observation row (i -> i+1)
#         # and running survivors to each observation row
#         # Need the band rows in time order:
#         if len(band_row_indices) > 0:
#             order = np.argsort(band_times)
#             ordered_row_idx = [band_row_indices[i] for i in order]

#             # running survivors per time
#             for j, surv in enumerate(chain["survivors_over_time"]):
#                 ridx = ordered_row_idx[j]
#                 results[ridx]["running_survivors_n"] = int(surv["n_survivors"])
#                 results[ridx]["running_survivors_ids"] = surv["survivor_ids"]

#             # pairwise overlap per transition
#             for j, pw in enumerate(chain["pairwise"]):
#                 ridx_left = ordered_row_idx[j]
#                 results[ridx_left]["overlap_with_next_n"] = int(pw["n_overlap"])
#                 results[ridx_left]["overlap_with_next_ids"] = pw["overlap_ids"]

#     results_df = pd.DataFrame(results)
#     return results_df, pd.DataFrame(overlap_summary_by_band)

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional



def kilonovascorer_v1(
    data_obs: pd.DataFrame,
    data_sim: pd.DataFrame,
    candidate_name: str,
    time_bin_width: float = 0.2,
    band_list: Tuple[str, ...] = ("g-band", "r-band", "i-band", "z-band"),
    k_near: float = 1.0,
    n_kde_sim: int = 50000,
    min_sim_points: int = 20,
    overlap_k: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function for kilonova lightcurve scoring. 

    Args:
        data_obs: Observational data (must contain 'filter_mapped', 'time_after_gw', etc.)
        data_sim: Simulation grid data.
        candidate_name: Identifier for the transient candidate.
        time_bin_width: Width of the time window for grouping simulations.
        band_list: List of filters/bands to process.
        k_near: Scaling factor for the 'prob_near' metric.
        n_kde_sim: Number of Monte Carlo samples for the KDE.
        min_sim_points: Minimum simulations required in a bin to perform scoring.
        overlap_k: Sigma threshold for ROPE acceptance Metric [ABC-diagnostic].

    Returns:
        Tuple containing:
            - results_df: Detailed metrics for every observation.
            - summary_df: Summary of the overlap chain logic per band.
    """
    results: List[Dict[str, Any]] = []
    overlap_summary_by_band: Dict[str, Any] = {}

    for band in band_list:
        # 1. Filter and Validate Data for the specific band
        sim_band = data_sim[data_sim["filter_mapped"] == band].copy()
        obs_band = data_obs[data_obs["filter_mapped"] == band].copy()

        if sim_band.empty or obs_band.empty:
            continue

        # 2. Establish Time Binning
        t_start = obs_band["time_after_gw"].min() - (time_bin_width / 2)
        t_end = obs_band["time_after_gw"].max() + time_bin_width
        bins = np.arange(t_start, t_end, time_bin_width)
        
        sim_band["time_bin"] = np.digitize(sim_band["time"], bins)

        # Tracking for the overlap chain
        band_times: List[float] = []
        band_ids_lists: List[List[int]] = []
        band_row_indices: List[int] = []

        # 3. Process Observations sequentially
        # Sorting by time ensures the results list follows chronological order
        obs_band = obs_band.sort_values("time_after_gw")

        #setup progress bar
        count = 0
        total_count = len(obs_band)

        for _, obs_row in obs_band.iterrows():
            t_obs = float(obs_row["time_after_gw"])
            m_obs = float(obs_row["absolute_magnitude"])
            sigma_obs = float(obs_row["absolute_magnitude_error"])

            # Skip invalid data
            if not (np.isfinite(m_obs) and np.isfinite(sigma_obs) and sigma_obs > 0):
                continue

            bin_idx = int(np.digitize(t_obs, bins))
            sim_bin = sim_band[sim_band["time_bin"] == bin_idx]

            if len(sim_bin) < min_sim_points:
                continue

            # Calculate KDE Metrics
            metric = predictive_tail_kde_python(
                sim_bin["absolute_magnitude"].values,
                m_obs,
                sigma_obs,
                k=k_near,
                n_sim=n_kde_sim,
            )

            # Compute ABC diagnostic with ROPE criteria
            consistent_ids = compute_consistent_ids_anyhit(
                sim_band=sim_band,
                bin_idx=bin_idx,
                M_obs=m_obs,
                sigma_obs=sigma_obs,
                overlap_k=overlap_k,
            )

            row = {
                "candidate_name": candidate_name,
                "band": band,
                "obs_time": t_obs,
                "time_bin_low": float(bins[bin_idx - 1] if bin_idx > 0 else bins[0]),
                "time_bin_high": float(bins[bin_idx] if bin_idx < len(bins) else bins[-1]),
                "observed_mag": m_obs,
                "observed_mag_err": sigma_obs,
                "p_tail": metric["p_tail"],
                "p_tail_mean": metric["p_tail_mean"],
                "p_tail_std": metric["p_tail_std"],
                "prob_near": metric["prob_near"],
                "n_sim_bin": int(len(sim_bin)),
                "n_consistent_lcs": int(len(consistent_ids)),
                "consistent_ids": list(consistent_ids),
                "overlap_with_next_n": np.nan,
                "overlap_with_next_ids": [],
                "running_survivors_n": np.nan,
                "running_survivors_ids": [],
            }

            results.append(row)
            band_times.append(t_obs)
            band_ids_lists.append(consistent_ids)
            band_row_indices.append(len(results) - 1)
            arcade_progress_bar(count, total_count, bar_length=50)
            count = count+1
            

        # 4. Post-processing: Compute the Overlap Chain (ABC-Diagnostic)
        if band_ids_lists:
            chain = overlap_chain(band_ids_lists, band_times)
            overlap_summary_by_band[band] = chain

            # Map chain results back to the individual observation rows
            for j, surv in enumerate(chain["survivors_over_time"]):
                idx = band_row_indices[j]
                results[idx]["running_survivors_n"] = int(surv["n_survivors"])
                results[idx]["running_survivors_ids"] = surv["survivor_ids"]

            for j, pw in enumerate(chain.get("pairwise", [])):
                idx_left = band_row_indices[j]
                results[idx_left]["overlap_with_next_n"] = int(pw["n_overlap"])
                results[idx_left]["overlap_with_next_ids"] = pw["overlap_ids"]

    return pd.DataFrame(results), pd.DataFrame(overlap_summary_by_band)
