"""
core.py — KilonovaScorer core pipeline.

Implements:
  - JSON / CSV photometry loading with absolute magnitude computation
  - LSST-like cadence downsampling
  - P_tail_KNe and P_near_KNe scoring via noise-convolved KDE (predictive_tail_kde)
  - ABC sequential survival diagnostic (overlap_chain)
  - Logit-space inverse-variance weighted cumulative scoring
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
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
from .utils import *  # noqa: F401,F403  (decorators and helpers)

logger = logging.getLogger(__name__)

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
    abs_mag, abs_err = compute_abs_mag_samples(  # noqa: F821 (from utils.*)
        df["magnitude"].to_numpy(),
        df["e_magnitude"].to_numpy(),
        dist_mpc=dist_mpc,
        dist_err_mpc=dist_err_mpc,
    )
    df["absolute_magnitude"] = abs_mag
    df["absolute_magnitude_error"] = abs_err

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

def predictive_tail_kde(
    sim_values: np.ndarray,
    M_obs: float,
    sigma_obs: float,
    k: float = 1.0,
    n_sim: int = 50000,
    kde: Optional[gaussian_kde] = None,
) -> Dict[str, float]:
    """
    Compute P_tail_KNe and P_near_KNe from the noise-convolved prior predictive
    distribution (PPD) via KDE.

    Implements the two-sided tail-area probability::

        P_tail_KNe = 2 * min(F_hat, 1 - F_hat)

    and the ROPE-based local consistency score::

        P_near_KNe = Pr(M_rep in [M_obs - k*sigma_obs, M_obs + k*sigma_obs])

    Uncertainty on P_tail_KNe is estimated by sampling N_obs = 100 realisations
    of M_obs from N(M_obs, sigma_obs) ("Charlie's Method", vectorised).

    A pre-fitted KDE can be supplied via ``kde`` to avoid redundant fitting when
    multiple observations share the same simulation time bin.

    Parameters
    ----------
    sim_values : np.ndarray
        Simulated absolute magnitudes from the PPD for the relevant time bin.
    M_obs : float
        Observed absolute magnitude (M_obs in paper notation).
    sigma_obs : float
        Observational uncertainty on M_obs.
    k : float
        ROPE half-width factor (k * sigma_obs).  Paper fiducial: 1.5.
    n_sim : int
        Number of Monte Carlo draws for the noise-convolved PPD.
    kde : gaussian_kde or None
        Pre-fitted KDE object.  If None, a new KDE is fitted to ``sim_values``.

    Returns
    -------
    dict with keys:
        F_hat        – empirical CDF evaluated at M_obs.
        p_tail_KNe   – two-sided tail-area probability.
        p_tail_mean  – mean of P_tail_KNe over M_obs uncertainty samples.
        p_tail_std   – std  of P_tail_KNe over M_obs uncertainty samples.
        p_near_KNe   – ROPE-based local consistency score (P_near_KNe).

    Raises
    ------
    ValueError
        If ``sim_values`` is empty or ``sigma_obs`` is non-positive.
    """
    sim_values = np.asarray(sim_values)
    if sim_values.size == 0:
        raise ValueError("sim_values cannot be empty.")
    if sigma_obs <= 0:
        raise ValueError("sigma_obs must be positive.")

    # 1. Build noise-convolved PPD: Y = X* + epsilon, X* ~ KDE, epsilon ~ N(0, sigma_obs)
    if kde is None:
        kde = gaussian_kde(sim_values)
    x_star = kde.resample(n_sim)[0]
    y_dist = x_star + np.random.normal(0, sigma_obs, size=n_sim)

    # 2. P_tail_KNe — two-sided tail probability at M_obs
    F_hat = float(np.mean(y_dist <= M_obs))
    p_tail_KNe = 2.0 * min(F_hat, 1.0 - F_hat)

    # 3. P_near_KNe — fraction of PPD draws within the ROPE
    p_near_KNe = float(np.mean(np.abs(y_dist - M_obs) <= k * sigma_obs))

    # 4. Uncertainty on P_tail_KNe via vectorised sampling of M_obs
    #    N_obs = 100 realisations; broadcast (100,1) vs (n_sim,) -> (100, n_sim)
    M_obs_samples = np.random.normal(M_obs, sigma_obs, 100)
    F_hat_samples = (y_dist <= M_obs_samples[:, np.newaxis]).mean(axis=1)
    p_tail_samples = 2.0 * np.minimum(F_hat_samples, 1.0 - F_hat_samples)

    return {
        "F_hat": F_hat,
        "p_tail_KNe": p_tail_KNe,
        "p_tail_mean": float(np.mean(p_tail_samples)),
        "p_tail_std": float(np.std(p_tail_samples)),
        "p_near_KNe": p_near_KNe,
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

    Returns
    -------
    list
        Unique sample IDs consistent with the ROPE at this epoch.
    """
    sim_bin = sim_band.loc[
        sim_band["time_bin"] == bin_idx, ["sample_id", "absolute_magnitude"]
    ]
    if sim_bin.empty:
        return []

    rope_half_width = overlap_k * sigma_obs
    inside = np.abs(sim_bin["absolute_magnitude"].to_numpy() - M_obs) <= rope_half_width
    return sim_bin.loc[inside, "sample_id"].dropna().unique().tolist()


def overlap_chain(ids_lists: List[List], times: List[float]) -> Dict[str, Any]:
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
    survivors_over_time = [{
        "t": float(times_sorted[0]),
        "n_survivors": len(survivors),
        "survivor_ids": sorted(survivors),
    }]

    pairwise = []
    for i in range(len(sets) - 1):
        # Pairwise: S_i ∩ S_{i+1}
        inter = sets[i] & sets[i + 1]
        pairwise.append({
            "t_left": float(times_sorted[i]),
            "t_right": float(times_sorted[i + 1]),
            "n_overlap": len(inter),
            "overlap_ids": sorted(inter),
        })

        # Cumulative: S_t = S_{t-1} ∩ S_t
        survivors &= sets[i + 1]
        survivors_over_time.append({
            "t": float(times_sorted[i + 1]),
            "n_survivors": len(survivors),
            "survivor_ids": sorted(survivors),
        })

    return {
        "times": times_sorted.tolist(),
        "pairwise": pairwise,
        "survivors_over_time": survivors_over_time,
        "final_survivors": sorted(survivors),
        "final_n_survivors": len(survivors),
    }


# ---------------------------------------------------------------------------
# Logit-space cumulative P_tail_KNe scoring
# ---------------------------------------------------------------------------

def binned_stats_cumulative_ptail(
    metric_df: pd.DataFrame,
    bin_size: float = 0.2,
) -> pd.DataFrame:
    """
    Aggregate per-observation P_tail_KNe scores into time-binned cumulative scores.

    Within each time bin, individual scores are combined using an
    inverse-variance weighted mean in logit space (see paper Section 2).
    The result is then updated sequentially across bins to produce a running
    cumulative score, also in logit space.

    Logit-space aggregation prevents extreme scores with small absolute
    uncertainties from dominating the weighted mean — a known pathology of
    direct probability-space averaging near the [0, 1] boundaries.

    Parameters
    ----------
    metric_df : pd.DataFrame
        Output of ``kilonovascorer``.  Must contain ``obs_time``,
        ``p_tail_mean``, and ``p_tail_std`` columns.
    bin_size : float
        Width of time bins in days.  Should match the scorer's
        ``time_bin_width`` (default 0.2 d).

    Returns
    -------
    pd.DataFrame
        One row per time bin with columns:
        ``time_bin``, ``time_mid``, ``mean``, ``std``,
        ``running_mean``, ``running_std``.
    """
    bin_edges = np.arange(
        metric_df["obs_time"].min() - bin_size / 2,
        metric_df["obs_time"].max() + bin_size,
        bin_size,
    )
    metric_df = metric_df.copy()
    metric_df["time_bin"] = pd.cut(metric_df["obs_time"], bins=bin_edges)

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
        Monte Carlo samples for the noise-convolved KDE.
    min_sim_points : int
        Minimum number of simulations required in a bin to attempt scoring.
    overlap_k : float
        ROPE half-width factor for the ABC diagnostic (sigma units).

    Returns
    -------
    results_df : pd.DataFrame
        Per-observation metrics including P_tail_KNe, P_near_KNe, and ABC
        diagnostics.
    summary_df : pd.DataFrame
        Per-band overlap chain summary.
    """
    results: List[Dict[str, Any]] = []
    overlap_summary_by_band: Dict[str, Any] = {}

    for band in band_list:
        # 1. Filter data for this band
        sim_band = data_sim[data_sim["filter_mapped"] == band].copy()
        obs_band = data_obs[data_obs["filter_mapped"] == band].copy()

        if sim_band.empty or obs_band.empty:
            logger.debug("No data for band %s — skipping.", band)
            continue

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
        t_end   = t_last  + time_bin_width / 2 + time_bin_width
        bins = np.arange(t_start, t_end, time_bin_width)

        assert bins[0] < t_first < bins[1],  "First observation not centred in first bin."
        assert bins[-2] < t_last < bins[-1], "Last observation not centred in last bin."

        sim_band["time_bin"] = np.digitize(sim_band["time"], bins)

        # Pre-group simulations by time bin for O(1) lookup per observation
        sim_groups: Dict[int, pd.DataFrame] = {
            k: v for k, v in sim_band.groupby("time_bin")
        }

        # Per-band tracking for ABC overlap chain
        band_times: List[float] = []
        band_ids_lists: List[List] = []
        band_row_indices: List[int] = []

        # 3. Process observations in chronological order
        obs_band = obs_band.sort_values("time_after_gw")
        total_obs = len(obs_band)

        for count, obs_row in enumerate(obs_band.itertuples(index=False), start=1):
            t_obs = float(obs_row.time_after_gw)
            M_obs = float(obs_row.absolute_magnitude)
            sigma_obs = float(obs_row.absolute_magnitude_error)

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

            # 3a. Fit KDE once per unique (band, bin_idx); reuse across observations
            #     in the same bin to avoid redundant 50 k-sample draws.
            if not hasattr(sim_groups, "_kde_cache"):
                sim_groups._kde_cache = {}  # type: ignore[attr-defined]
            if bin_idx not in sim_groups._kde_cache:  # type: ignore[attr-defined]
                sim_groups._kde_cache[bin_idx] = gaussian_kde(  # type: ignore[attr-defined]
                    sim_bin["absolute_magnitude"].to_numpy()
                )
            cached_kde = sim_groups._kde_cache[bin_idx]  # type: ignore[attr-defined]

            # 3b. Compute P_tail_KNe and P_near_KNe
            metric = predictive_tail_kde(
                sim_bin["absolute_magnitude"].to_numpy(),
                M_obs=M_obs,
                sigma_obs=sigma_obs,
                k=k_near,
                n_sim=n_kde_sim,
                kde=cached_kde,
            )

            # 3c. ABC diagnostic — consistent simulation IDs at this epoch
            consistent_ids = compute_consistent_ids_anyhit(
                sim_band=sim_band,
                bin_idx=bin_idx,
                M_obs=M_obs,
                sigma_obs=sigma_obs,
                overlap_k=overlap_k,
            )

            # 3d. Safe time-bin edge lookup
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
                "n_sim_bin": len(sim_bin),
                "n_consistent_lcs": len(consistent_ids),
                "consistent_ids": consistent_ids,
                # ABC overlap fields — populated in post-processing step 4
                "overlap_with_next_n": np.nan,
                "overlap_with_next_ids": [],
                "running_survivors_n": np.nan,
                "running_survivors_ids": [],
            }

            results.append(row)
            band_times.append(t_obs)
            band_ids_lists.append(consistent_ids)
            band_row_indices.append(len(results) - 1)
            arcade_progress_bar(count, total_obs, bar_length=50)

        # 4. Post-processing: compute ABC overlap chain for this band
        if band_ids_lists:
            chain = overlap_chain(band_ids_lists, band_times)
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
