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
# `import *` skips underscore-prefixed names, so the star import above does NOT
# bring `_flux_score_axis` across even though it sits beside the other flux
# helpers in utils.  Without this line every `space="flux"` DETECTION raises
# `NameError` at the three call sites below, while magnitude space and
# non-detections (which convert to flux internally) keep working -- so the
# breakage stays invisible until someone asks for flux space.
from .utils import _flux_score_axis  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "P_TAIL_METHODS",
    "SPACE_METHODS",
    "flux_of",
    "flux_sigma_of",
    "flux_sigma_of_limit",
    "nondetection_tail",
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
    "compute_abs_mag_samples",
    "ivw_stats_logit",
    "stouffer_combine",
    "stouffer_stats",
    "calculate_sequential_score_stouffer",
    "calculate_sequential_score_logit",
    "timer_warp",
    "time_plot",
]

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

#: Set once the stream is found unable to encode the block characters, so the
#: encoding is probed once rather than once per observation.
# TODO: Is this really necessary?
_ASCII_BAR = False

def arcade_progress_bar(current: int, total: int, bar_length: int = 30) -> None:
    "Print an arcade-style progress bar to stdout."
    global _ASCII_BAR
    percent = current / total
    filled = int(bar_length * percent)

    if not _ASCII_BAR:
        bar = "█" * filled + "-" * (bar_length - filled)
        try:
            sys.stdout.write(f"\r[ {bar} ] {percent * 100:6.2f}% ⬛")
            sys.stdout.flush()
            if current == total:
                sys.stdout.write("\n")
            return
        except UnicodeEncodeError:
            _ASCII_BAR = True

    bar = "#" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\r[ {bar} ] {percent * 100:6.2f}%")
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

P_TAIL_METHODS = ("closed_form", "montecarlo")


SPACE_METHODS = ("magnitude", "flux")


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
    sigma_model: float = 0.0,
    random_state: Optional[int] = None,
    p_near_compute: bool = True,
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
    Neither is removed; the default changed.  They do NOT target quite the same
    integral, and the difference is not negligible: the closed form integrates
    against the EMPIRICAL grid, while ``montecarlo`` integrates against a
    KDE-SMOOTHED version of it, so the Monte Carlo reference is wider by the
    kernel bandwidth h. Feeding the closed form
    ``sqrt(sigma_obs^2 + h^2)`` reproduces the Monte Carlo result to within its
    own Monte Carlo error.

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
        Estimator, as above.  Exactly one of ``P_TAIL_METHODS``; anything else
        raises.
    random_state : int or None
        Seed for the Monte Carlo path.  ``None`` (default) uses NumPy's global
        random state, reproducing the original behaviour exactly.  Ignored by
        the closed form, which is already deterministic.
    sigma_model : float
        Model-inadequacy allowance in magnitudes, added IN QUADRATURE to
        ``sigma_obs`` when convolving the reference: ``sqrt(sigma_obs^2 +
        sigma_model^2)``.  It enters ONCE, here.  It is a statement about how
        far the model family is trusted -- a property of the MODELS, not of the
        telescope, which is why it is additive rather than a multiple of
        ``sigma_obs``: one ``k`` would grant a different allowance to every
        object.  ``0.0`` (default) reproduces previous behaviour exactly.
        ``docs/SIGMA_MODEL.md`` measures ~0.70 mag on the local grid, chosen by
        calibration drift on held-out kilonovae rather than by the AUC peak.

        It does NOT widen the detection weight, the ROPE, or the measurement
        jitter -- those are properties of the observation and keep ``sigma_obs``.
    p_near_compute : bool
        If True (default), compute P_near_KNe.  If False the ROPE evaluation is
        skipped entirely and ``p_near_KNe`` comes back NaN.  P_near_KNe is a
        purely diagnostic, per-observation quantity that never feeds the
        cumulative score, so disabling it cannot affect P_tail_KNe.


    Returns
    -------
    dict with keys:
        F_hat        - CDF F(M_obs) under the noise-convolved PPD; exact under
                       the closed form, empirical under Monte Carlo.
        p_tail_KNe   - two-sided tail probability at the POINT measurement
                       (paper eq. 7).  Reported for reference; not scored.
        p_tail_mean  - THE SCORED QUANTITY (paper eq. 8): mean of P_tail over
                       ``n_obs`` realisations of M_obs jittered by sigma_obs.
                       Lower than p_tail_KNe for epochs near the population
                       median, because the two-sided fold is concave there.
        p_tail_std   - sd of P_tail over those same realisations.  Free from
                       the samples already drawn.  Read only by
                       ``method="ivw"``; the Stouffer/Strube default reads no
                       per-epoch uncertainty.
        p_near_KNe   - ROPE-based local consistency score P_near_KNe, or
                       NaN when ``p_near_compute`` is False.
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

    if not (sigma_obs > 0):
        raise ValueError(
            "sigma_obs must be positive and finite; got %r." % (sigma_obs,)
        )

    if p_tail_method not in P_TAIL_METHODS:
        raise ValueError(
            "p_tail_method must be one of %r; got %r."
            % (P_TAIL_METHODS, p_tail_method)
        )

    if not np.isfinite(sigma_model) or sigma_model < 0.0:
        raise ValueError(
            "sigma_model must be finite and non-negative; got %r."
            % (sigma_model,)
        )

    m = np.asarray(sim_values, dtype=float)

    # TWO widths, and they stop being the same thing once sigma_model > 0.
    #   s      -- what the reference POPULATION is convolved with; carries the
    #             model-inadequacy allowance, a statement about the MODELS.
    #   s_obs  -- what the DETECTOR did; governs detectability, the ROPE, and
    #             the size of the measurement jitter.
    # At sigma_model = 0 they coincide and every number is unchanged.
    s_obs = float(sigma_obs)
    s = float(np.hypot(s_obs, float(sigma_model)))

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
            # s_obs, NOT s: detectability is what the telescope could see.
            # Widening it with sigma_model would make faint simulations
            # look detectable, which a model allowance does not mean.
            w = ndtr((float(M_lim) - m) / s_obs)
            w_sum = float(w.sum())
            w_sq = float(np.sum(w ** 2))
            if w_sum > 0.0 and w_sq > 0.0:
                F_hat = float(np.clip(phi.sum() / w_sum, 0.0, 1.0))
                n_eff = float(w_sum ** 2 / w_sq)
            else:
                limit_degenerate = True
                n_eff = 0.0
                use_limit = False
        if not use_limit:
            F_hat = float(phi.mean())

        # P_tail_KNe - two-sided tail probability at M_obs (paper eq. 7),
        # evaluated at the point measurement.  Reported, not scored.
        p_tail_KNe = 2.0 * min(F_hat, 1.0 - F_hat)

        # p_tail_mean - THE SCORED QUANTITY.  P_tail averaged over n_obs
        # realisations of the measurement, M_obs + eta, eta ~ N(0, sigma_obs^2):
        # the paper's N_obs jitter loop, and what the combiner reads.
        #
        # It does not equal p_tail_KNe.  E_eta[F] is F against a reference
        # widened to sqrt(s^2 + sigma_obs^2), and 2*min(F, 1-F) is concave at
        # F = 0.5, so the fold pulls the average below the point value for any
        # epoch near the population median.  Both effects are intended: the
        # score is deliberately the more conservative of the two.
        #
        # p_tail_std falls out of the same samples for the cost of one np.std
        # and is read only by method="ivw"; the Stouffer/Strube default reads
        # no per-epoch uncertainty at all.
        rng_j = (np.random if random_state is None
                 else np.random.default_rng(random_state))
        # sigma_obs, NOT s: the jitter is what the DETECTOR did.  Using s would
        # spend the model-inadequacy allowance a second time, on the
        # measurement side, where it does not belong.
        x_j = M_obs + rng_j.normal(0.0, s_obs, int(n_obs))
        phi_j = ndtr((x_j[:, None] - m[None, :]) / s)
        if use_limit:
            F_j = np.clip(phi_j.sum(axis=1) / w_sum, 0.0, 1.0)
        else:
            F_j = phi_j.mean(axis=1)
        p_j = 2.0 * np.minimum(F_j, 1.0 - F_j)
        p_tail_mean = float(p_j.mean())
        p_tail_std = float(p_j.std())

        # P_near_KNe - ROPE mass, likewise a difference of two mixture CDFs
        # (paper eq. 4; k=1.5 fiducial).  Not aggregated across epochs.
        # P_near_KNe is a per-observation DIAGNOSTIC that is never aggregated,
        # so it can be skipped outright.  (origin/trove, `p-near-optional`)
        p_near_KNe = float("nan")
        if p_near_compute:
            half = k * s_obs   # observational tolerance, not model
            p_near_KNe = float(
                (ndtr((M_obs + half - m) / s)
                 - ndtr((M_obs - half - m) / s)).mean()
            )

    else:  # p_tail_method == "montecarlo" - the original estimator, unchanged
        # Fixed seed so scores are deterministic
        rng = np.random if random_state is None else np.random.default_rng(random_state)

        if kde is None:
            kde = gaussian_kde(m)
        n_draw = int(n_sim)
        x_star = kde.resample(n_draw, seed=random_state)[0]
        y_dist = x_star + rng.normal(0.0, s, size=n_draw)

        y_ref = y_dist
        n_eff = float(m.size)
        if use_limit:
            w = ndtr((float(M_lim) - m) / s_obs)   # detectability: s_obs
            w_sum = float(w.sum())
            w_sq = float(np.sum(w ** 2))       # see the closed form on why
            keep = y_dist <= float(M_lim)
            n_keep = int(np.count_nonzero(keep))
            if w_sum > 0.0 and w_sq > 0.0 and n_keep > 0:
                y_ref = y_dist[keep]
                n_eff = float(w_sum ** 2 / w_sq)
            else:
                limit_degenerate = True
                n_eff = 0.0
                use_limit = False

        F_hat = float(np.mean(y_ref <= M_obs))
        p_tail_KNe = 2.0 * min(F_hat, 1.0 - F_hat)

        p_near_KNe = (float(np.mean(np.abs(y_dist - M_obs) <= k * s))
                      if p_near_compute else float("nan"))

        # s_obs, not s -- see the closed form.
        M_obs_samples = rng.normal(M_obs, s_obs, size=int(n_obs))
        F_hat_samples = (y_ref <= M_obs_samples[:, np.newaxis]).mean(axis=1)
        p_tail_samples = 2.0 * np.minimum(F_hat_samples, 1.0 - F_hat_samples)
        p_tail_mean = float(np.mean(p_tail_samples))
        p_tail_std = float(np.std(p_tail_samples))

    scoreable = True
    if not np.isfinite(M_obs):
        scoreable = False
    if limit_requested:
        scoreable = scoreable and (not limit_degenerate) and n_eff >= float(min_n_eff)
    if not scoreable:
        p_tail_KNe = float("nan")
        p_tail_mean = float("nan")
        p_tail_std = float("nan")

    return {
        "F_hat": F_hat,
        "p_tail_KNe": p_tail_KNe,
        "p_tail_mean": p_tail_mean,
        "p_tail_std": p_tail_std,
        "p_near_KNe": p_near_KNe,
        "n_eff": n_eff,
        "scoreable": bool(scoreable),
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

    return _anyhit_from_arrays(
        sim_bin["absolute_magnitude"].to_numpy(),
        sim_bin["sample_id"].to_numpy(),
        M_obs, sigma_obs, overlap_k, count_only,
    )


def _anyhit_from_arrays(
    mag: np.ndarray,
    sid: np.ndarray,
    M_obs: float,
    sigma_obs: float,
    overlap_k: float = 2.0,
    count_only: bool = False,
):
    """The ROPE any-hit test itself, on two aligned arrays.

    Shared by both partition paths so that the DataFrame form and the array-view
    form cannot drift apart: whichever way the rows were selected, this is the
    code that decides which simulations are consistent.
    """
    if mag.size == 0:
        return 0 if count_only else []

    rope_half_width = overlap_k * sigma_obs
    inside = np.abs(mag - M_obs) <= rope_half_width
    ids = sid[inside]
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
    rho: float = 0.0,
) -> pd.DataFrame:
    """
    Aggregate per-observation P_tail_KNe scores into time-binned cumulative scores.

    Scores are combined within each time bin, then updated sequentially across
    bins to produce a running cumulative score.  BOTH COMBINERS ARE AVAILABLE;
    the default changed from ``"ivw"`` to ``"stouffer"``.

    ``method="stouffer"`` (default).  Each epoch's P_tail is treated as what it
    is under the null — a p-value, uniform on (0, 1) — and combined as a
    standardised SUM of normal scores, ``Z = sum(z) / sqrt(n + rho*n*(n-1))``
    with ``z = Phi^-1(1 - p)``.  Z is exactly standard normal under the null,
    so the combined score is a calibrated p-value.

    EQUAL WEIGHTS, and the option to pass others is gone rather than merely
    defaulted off.  Two reasons, one empirical and one structural.  The power
    test (``trove_tests/docs/WEIGHTS.md``) found no ancillary scheme that beat
    equal weighting beyond noise, nor did the provably-optimal oracle weight.
    And Strube's normaliser needs the FULL correlation matrix once the weights
    are unequal: the scalar rho is exact only under exchangeability, which
    equal weights preserve and unequal weights break.  Weighted Stouffer with
    a scalar rho is therefore miscalibrated by an unknown amount, which is a
    bad trade for a power gain measured at zero.

    ``method="ivw"``.  The ORIGINAL combiner: inverse-variance weighted mean in
    logit space, weights ``1/sigma_z^2`` from ``p_tail_std`` via the delta method
    (paper Section 2).  Logit-space aggregation prevents extreme scores with
    small absolute uncertainties from dominating the weighted mean — a known
    pathology of direct probability-space averaging near the [0, 1] boundaries.
    Retained so that results published against it remain reproducible; see the
    notes above ``stouffer_combine`` in utils.py for why it is no longer the
    default.

    Brown's method (moment-matched scaled chi-square) was implemented and
    measured head to head, then REMOVED because it is not being used.  It won on
    power and lost on robustness -- ~7x more swayed by a single bad epoch -- and
    with ``rho=0`` it degenerates exactly to Fisher.  The full comparison is
    kept in REPORT.md section 16.6.

    Parameters
    ----------
    metric_df : pd.DataFrame
        Output of ``kilonovascorer``.  Must contain ``obs_time`` and
        ``p_tail_mean``; ``p_tail_std`` is required only by ``method="ivw"``.
    bin_size : float
        Width of time bins in days.  Should match the scorer's
        ``time_bin_width`` (default 0.2 d).
    method : {"stouffer", "ivw"}
        Combiner, as above.
    rho : float
        Mean inter-epoch correlation of the normal scores.  ``0.0`` (default)
        assumes independence and gives plain Stouffer; a positive value applies
        **Strube's method** (Strube 1985), the correlation-corrected Stouffer --
        see ``stouffer_combine`` in utils.py for the formula and the measured
        calibration.  Obtain the value from :func:`estimate_rho` rather than
        guessing; the grid-measured mean is 0.284, at which uncorrected
        Stouffer is measurably overconfident (KS 0.1111 against Strube's
        0.0065).  Ignored by ``method="ivw"``, which has no correlation
        correction available at all.

    Returns
    -------
    pd.DataFrame
        One row per time bin with columns:
        ``time_bin``, ``time_mid``, ``mean``, ``std``,
        ``running_mean``, ``running_std``.
    """
  #modify to match the bin edges of kilonovaScorer_V3 + bin_size / 2,
  #modidy back to +  bin_size
    if method not in ("stouffer", "ivw"):
        raise ValueError("method must be 'stouffer' or 'ivw'.")

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
        # Same narrowing as the stouffer path below, and this is the path it
        # was written for: a bare .dropna() deletes a bin if ANY column is NaN,
        # which is how all-zero bins used to vanish before ivw_stats_logit
        # returned `count` on every path.  (origin/trove, `zero-epochs-handling`)
        binned_stats = binned_stats.dropna(subset=["mean", "std"])

        if binned_stats.empty:
            binned_stats["running_mean"] = []
            binned_stats["running_std"] = []
            return binned_stats

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
    binned_stats = (
        metric_df.groupby("time_bin", observed=True)
        .apply(lambda g: stouffer_stats(g, rho=rho))  # noqa: F821
        .reset_index()
    )
    binned_stats["time_mid"] = binned_stats["time_bin"].apply(lambda x: x.mid)
    # Narrowed from a bare .dropna(): with how="any" a single NaN in ANY
    # column deleted the whole bin.  That is how all-zero bins used to vanish —
    # ivw_stats_logit's old two-key early return left count = NaN under
    # groupby.apply, and the bin was dropped on the strength of that missing
    # field alone, never because its score was unusable.  Restricting the
    # subset to the two columns the running score actually consumes means a
    # future schema addition cannot silently start deleting bins again.
    binned_stats = binned_stats.dropna(subset=["mean", "std"])


    if binned_stats.empty:
        binned_stats["running_mean"] = []
        binned_stats["running_std"] = []
        return binned_stats

    # Chronological order, and the raw epochs behind each surviving bin.
    binned_stats = binned_stats.sort_values("time_mid").reset_index(drop=True)
    by_bin = {k: v for k, v in metric_df.groupby("time_bin", observed=True)}
    p_by_bin = [by_bin[tb]["p_tail_mean"].to_numpy(dtype=float)
                for tb in binned_stats["time_bin"]]

    running_mean, running_err = calculate_sequential_score_stouffer(  # noqa: F821
        p_by_bin, rho=rho,
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
    for i, r in enumerate(rows.itertuples(index=False)):
        band = data_sim[data_sim["filter_mapped"] == r.band]
        if band.empty:
            continue
        lo = getattr(r, "time_bin_low", r.obs_time - time_bin_width / 2)
        hi = getattr(r, "time_bin_high", r.obs_time + time_bin_width / 2)
        window = band[(band["time"] > lo) & (band["time"] <= hi)]
        if window.empty:
            # No grid point inside this epoch's bin, so the epoch cannot be
            # simulated and is dropped.  A nearest-neighbour fallback used to be
            # attempted here as `band.iloc[...argsort()[:0]]`, which selects
            # ZERO rows and so was never anything but this `continue`.
            continue
        cols.append(window.groupby("sample_id")["absolute_magnitude"].mean())

        # getattr with a default: a sigma_col naming a column this frame does
        # not have used to raise AttributeError from inside the loop.
        sigma = getattr(r, sigma_col, None) if sigma_col else None
        sig.append(float(r.observed_mag_err if sigma is None else sigma))

        # None is not NaN: np.isfinite(None) raises, and a hand-built metric_df
        # can carry None in an object-dtype M_lim column.
        M_lim = getattr(r, "M_lim", np.nan)
        M_lim = np.nan if M_lim is None else float(M_lim)
        lim.append(M_lim if np.isfinite(M_lim) else None)

        keep.append(i)          # the ROW that produced this column

    if not cols:
        return None, None, None, rows.iloc[:0]
    G = pd.concat(cols, axis=1).dropna()
    # `rows.iloc[keep]`, NOT `rows.iloc[:len(cols)]`.  Skipping one epoch shifts
    # every later row, so slicing the head pairs column j with a different
    # epoch than the one that built it -- feeding combined_score_marginalised
    # the wrong p-value and labelling it with the wrong time and band.
    return (G.to_numpy(), np.asarray(sig), lim,
            rows.iloc[keep].reset_index(drop=True))


def _simulate_epoch_p(G, sig, lim, sigma_mu, n_draws, rng, sigma_score=None,
                      is_limit=None, depth=None, n_sigma_limit=5.0,
                      nondetection_gate=0.5):
    """Draw simulated candidates from the grid, observe them, and score them.

    One ``delta`` per DRAW -- not per epoch -- which is the whole point: the
    distance systematic is a single realisation shared by every epoch of a
    candidate, and that is what makes the epochs correlated.
    """
    n_grid, n_ep = G.shape
    idx = rng.integers(0, n_grid, n_draws)
    truth = G[idx, :]
    sig_score = sig if sigma_score is None else np.asarray(sigma_score, dtype=float)
    delta = rng.normal(0.0, float(sigma_mu), (n_draws, 1)) if sigma_mu else 0.0

    if is_limit is None:
        eps = rng.normal(0.0, 1.0, (n_draws, n_ep)) * sig_score[None, :]
        obs = truth + eps - delta
        return np.column_stack([
            _tail_from_grid(obs[:, j], G[:, j], sig[j], lim[j])
            for j in range(n_ep)])

    # Survey-realised path.  A limit epoch's score cannot be simulated the way
    # a detection's is: `nondetection_tail` is a function of the DEPTH and the
    # GRID alone, so redrawing the candidate leaves it unchanged and the
    # column has zero variance -- its correlation with anything is 0/0, which
    # is why the old path returned all-NaN for those columns and `np.nanmean`
    # silently reduced rho to a detections-only quantity (REPORT.md Part XV
    # section 71).
    #
    # Stop conditioning on the epoch type instead.  Under the null, whether a
    # given epoch of a given simulated candidate is a detection or a
    # non-detection is ITSELF part of the draw -- bright objects are detected
    # at every epoch, faint ones at none -- and that indicator is what carries
    # the correlation.  So threshold each draw the way the survey did.
    is_limit = np.asarray(is_limit, dtype=bool)
    depth = np.asarray(depth, dtype=float)
    truth_mu = truth - delta          # distance systematic, shared by epochs
    P = np.full((n_draws, n_ep), np.nan)

    for j in range(n_ep):
        if not is_limit[j]:
            e = rng.normal(0.0, 1.0, n_draws) * sig_score[j]
            P[:, j] = _tail_from_grid(truth_mu[:, j] + e, G[:, j], sig[j], lim[j])
            continue

        M_lim_j = depth[j]
        if not np.isfinite(M_lim_j):
            continue                  # no quoted depth: column stays NaN

        F_thresh = flux_of(M_lim_j)
        s_phot = flux_sigma_of_limit(M_lim_j, n_sigma_limit)
        P_detect = float(ndtr((flux_of(G[:, j]) - F_thresh) / s_phot).mean())
        if P_detect < nondetection_gate:
            # The scorer excludes this epoch from combining outright, so it is
            # not part of the correlation the combiner needs either.
            continue

        F_meas = flux_of(truth_mu[:, j]) + rng.normal(0.0, s_phot, n_draws)
        detected = F_meas >= F_thresh
        # non-detection: the constant score the scorer reports, capped as it is
        P[~detected, j] = min(1.0 - P_detect, 0.5)
        if detected.any():
            F_d = np.clip(F_meas[detected], 1e-300, None)
            m_d = -2.5 * np.log10(F_d)
            s_d = (2.5 / np.log(10.0)) * s_phot / F_d
            # `_tail_from_grid` takes ONE sigma per call and these differ per
            # draw, so group them by sigma and use each group's median.
            out = np.empty(m_d.size)
            order = np.argsort(s_d)
            for c in np.array_split(order, min(20, max(order.size, 1))):
                if c.size:
                    out[c] = _tail_from_grid(m_d[c], G[:, j],
                                             float(np.median(s_d[c])), lim[j])
            P[detected, j] = out
    return P


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
    limit_mode: str = "survey",
    n_sigma_limit: float = 5.0,
    nondetection_gate: float = 0.5,
):
    """Mean inter-epoch correlation of the normal scores, measured on the grid at
    THIS candidate's cadence.

    This is the ``rho`` that ``binned_stats_cumulative_ptail(rho=...)``
    and ``stouffer_combine(rho=...)`` need in order to apply **Strube's
    method** (Strube 1985) instead of assuming independent epochs.
    Without it the combined score is overconfident by a known direction:
    at the grid-measured rho of 0.284, uncorrected Stouffer lands at KS
    0.1111 against U(0,1) where Strube reaches 0.0065.

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
    if sigma_col is not None and sigma_col not in metric_df.columns:
        logger.warning(
            "estimate_rho: sigma_col=%r is not a column of metric_df; "
            "falling back to observed_mag_err.", sigma_col,
        )
        sigma_col = None
    if limit_mode not in ("survey", "drop"):
        raise ValueError("limit_mode must be 'survey' or 'drop'; got %r."
                         % (limit_mode,))
    G, sig, lim, rows = _epoch_grid(metric_df, data_sim, time_bin_width, sigma_col)
    if G is None or G.shape[1] < 2 or G.shape[0] < 10:
        return (float("nan"), None) if return_matrix else float("nan")

    # Limit epochs need the survey-realised simulation; see _simulate_epoch_p.
    # `limit_mode="drop"` restores the previous behaviour, which is not "drop"
    # by intent -- it kept the columns and lost them to NaN -- but is what the
    # old code did, kept so the change is measurable rather than asserted.
    is_lim = (rows["is_limit"].to_numpy(dtype=bool)
              if "is_limit" in rows.columns
              else np.zeros(len(rows), dtype=bool))
    if limit_mode == "survey" and is_lim.any():
        P = _simulate_epoch_p(
            G, sig, lim, sigma_mu, n_draws, rng,
            is_limit=is_lim, depth=rows["observed_mag"].to_numpy(dtype=float),
            n_sigma_limit=n_sigma_limit, nondetection_gate=nondetection_gate)
    else:
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
    if col is None:
        logger.warning(
            "combined_score_marginalised: sigma_col=%r is not a column of "
            "metric_df; falling back to observed_mag_err.  If metric_df was NOT "
            "produced with the photometric-only sigma, the null this calibrates "
            "against is the wrong one.", sigma_col,
        )
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

#: Returned for a bin that holds no simulations, so that the per-observation
#: branch always yields arrays and never None.
_EMPTY_F = np.empty(0, dtype=float)


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
    sigma_model: float = 0.0,
    n_obs: int = 100,
    random_state: Optional[int] = None,
    abc_compute: bool = True,
    abc_return_ids: bool = True,
    sigma_col: Optional[str] = None,
    band_split: bool = True,
    abc_reuse_bin: bool = True,
    array_bins: bool = True,
    p_near_compute: bool = True,
    space: str = "magnitude",
    score_limits: bool = False,
    is_limit_col: str = "is_limit",
    n_sigma_limit: float = 5.0,
    flux_zp: float = 0.0,
    nondetection_gate: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score a kilonova candidate against a simulation grid.

    For each photometric band and observation, computes:

    - **P_tail_KNe** — two-sided tail probability of M_obs under the
      noise-convolved PPD (with uncertainty via observation sampling).
    - **P_near_KNe** — ROPE-based local consistency score (optional; see
      ``p_near_compute``).
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
        Number of M_obs realisations behind ``p_tail_mean`` and
        ``p_tail_std``.  Paper value: 100.
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

    band_split : bool
        Partition ``data_sim`` and ``data_obs`` by filter ONCE with a groupby
        (``True``, default) instead of re-scanning the full table per band with
        ``data_sim["filter_mapped"] == band``.  On an object-dtype string column
        that comparison is an elementwise Python-string test over every row, run
        once per band, and it was the single largest line in the profile --
        larger than the ABC diagnostic and an order of magnitude larger than
        ``predictive_tail_kde``.  Measured 2.0x at a 10,000-sample grid and 1.9x
        at 25,000.

        ``False`` restores the original per-band comparison.  The two select the
        same rows in the same order, so this is a speed switch only; it exists to
        make that claim checkable rather than asserted.

    abc_reuse_bin : bool
        Hand the already-selected time-bin slice to the ABC diagnostic
        (``True``, default) instead of letting it re-select the slice for itself.
        ``kilonovascorer_v3`` groups the band by ``time_bin`` up front and holds
        the current bin in order to score against it;
        ``compute_consistent_ids_anyhit`` was then rebuilding exactly those rows
        with a boolean mask over the WHOLE band, once per observation -- 200,000
        rows scanned to reach 10,000 of them at a 10,000-sample grid.  The waste
        is a factor of the number of time bins.  Measured 3.3x on the diagnostic
        at 25,000 samples.

        ``False`` restores the re-selection, which is the code the diagnostic
        still runs whenever it is called directly without ``sim_bin=``.  The ROPE
        test, the any-hit rule and the returned ids are untouched either way, so
        the two agree exactly.

    array_bins : bool
        Partition each band's grid into per-bin ARRAY VIEWS (``True``, default)
        rather than into a DataFrame per bin.  ``sim_band.groupby("time_bin")``
        materialises one small DataFrame per bin -- 80 of them for four bands and
        twenty bins -- and every use site then immediately calls ``.to_numpy()``
        on two of their columns.  A stable argsort on the bin codes plus
        ``np.diff`` gives the same partition as slice boundaries, and a slice of
        a numpy array is a view: no take, no copy, no DataFrame.  After the other
        two optimisations this was 72% of the run and the top three entries in
        the profile; measured 2.3x on the partition.

        ``False`` restores the groupby.  Equivalence rests on ``groupby``
        preserving within-group row order and ``np.argsort(kind="stable")`` doing
        the same, so group k and slice k hold the same rows in the same ORDER --
        which is what makes the ABC id lists match element for element and not
        merely as sets.

        The ROPE test is shared: both paths call ``_anyhit_from_arrays`` on the
        same two arrays, so they cannot drift apart.  With
        ``abc_reuse_bin=False`` the diagnostic re-selects from ``sim_band``
        exactly as before, under either partition.

    All three flags are speed switches with no effect on any output.  Turning
    them off reproduces the pre-optimisation scorer::

        kilonovascorer_v3(..., band_split=False, abc_reuse_bin=False,
                          array_bins=False)
    p_near_compute : bool
        If True (default), compute P_near_KNe for each observation.  If False,
        the ROPE evaluation is skipped and the ``p_near_KNe`` column is filled
        with NaN.  P_near_KNe is a per-observation diagnostic that is never
        aggregated, so disabling it leaves P_tail_KNe, the cumulative score and
        the ABC diagnostics unchanged.
    space : {"magnitude", "flux"}
        Space a real DETECTION is compared in.  ``"magnitude"`` (default)
        is unchanged from before this option existed.  ``"flux"`` runs the
        same P_tail machinery on exponentiated values instead -- a
        legitimate but numerically DIFFERENT test, not interchangeable with
        magnitude-space scores for the same epoch.
    score_limits : bool
        If True, rows flagged by ``is_limit_col`` are scored via
        ``nondetection_tail`` instead of being dropped.  Default False
        reproduces every prior result unchanged.  See ``nondetection_gate``
        below for how a limit's score is allowed to affect combining.
    is_limit_col : str
        Column flagging a non-detection row.  Its ``absolute_magnitude`` is
        read as that row's own quoted limiting magnitude.
    n_sigma_limit : float
        Significance the quoted depth was reported at (paper-specific --
        get it from the source, don't guess).
    flux_zp : float
        Flux zeropoint; cancels out of every probability, kept only to keep
        numbers near order-unity.
    nondetection_gate : float
        A non-detection's ``p_tail_mean`` only enters combining
        (Stouffer/Strube or IVW) when its ``F_hat`` (= P_detect, the
        model's own detection probability) is at least this value;
        otherwise it is excluded (NaN), the same as being dropped. One that
        does pass is additionally capped at ``min(p_tail_mean, 0.5)``. Both
        exist for the same reason: Stouffer maps ``p_tail_mean -> 1`` to
        strong evidence FOR the model, but most non-detections are simply
        uninformative, so an uncapped, ungated non-detection can inflate a
        candidate's score toward 1 regardless of what its detections say.
        The gate excludes epochs that carry no information at all; the cap
        ensures the ones that remain can only argue tension (push the score
        down), never confirmation. Default 0.5 is a starting point, not a
        swept value -- see ``docs/FLUX_SPACE.md``.

    Returns
    -------
    results_df : pd.DataFrame
        Per-observation metrics including P_tail_KNe, P_near_KNe (NaN when
        ``p_near_compute`` is False), and ABC diagnostics.
    summary_df : pd.DataFrame
        Per-band overlap chain summary.
    """
    # Validate once, up front, rather than on every observation.
    if p_tail_method not in P_TAIL_METHODS:
        raise ValueError(
            "p_tail_method must be one of %r; got %r."
            % (P_TAIL_METHODS, p_tail_method)
        )
    if space not in SPACE_METHODS:
        raise ValueError(
            "space must be one of %r; got %r." % (SPACE_METHODS, space)
        )
    if not (np.isfinite(n_sigma_limit) and n_sigma_limit > 0):
        raise ValueError(
            "n_sigma_limit must be finite and positive; got %r." % (n_sigma_limit,)
        )
    if not (np.isfinite(nondetection_gate) and 0.0 <= nondetection_gate <= 1.0):
        raise ValueError(
            "nondetection_gate must be in [0, 1]; got %r." % (nondetection_gate,)
        )

    results: List[Dict[str, Any]] = []
    overlap_summary_by_band: Dict[str, Any] = {}

    # Split by band ONCE, unless the caller asked for the original behaviour.
    # The previous form re-scanned the full simulation table per band with
    # `data_sim["filter_mapped"] == band`, which on a string column is an
    # elementwise comparison over every row -- 800,000 rows x 4 bands, and the
    # single largest cost in the profile, larger than the ABC diagnostic and an
    # order of magnitude larger than predictive_tail_kde.  groupby partitions in
    # one pass: measured 2.0x at a 10,000-sample grid, 1.9x at 25,000.
    #
    # The two select the same rows in the same order -- groupby preserves
    # within-group order, and a NaN key is dropped by both (dropna=True there,
    # `NaN == band` being False here) -- so this is a speed switch and nothing
    # more.
    sim_by_band: Optional[Dict[Any, pd.DataFrame]] = None
    obs_by_band: Optional[Dict[Any, pd.DataFrame]] = None
    if band_split:
        sim_by_band = {k: v for k, v in data_sim.groupby("filter_mapped", observed=True)}
        obs_by_band = {k: v for k, v in data_obs.groupby("filter_mapped", observed=True)}

    for band in band_list:
        # 1. Filter data for this band
        if band_split:
            sim_band = sim_by_band.get(band)
            obs_band = obs_by_band.get(band)
            if sim_band is None or obs_band is None:
                logger.debug("No data for band %s — skipping.", band)
                continue
        else:
            # The original: one elementwise comparison over the whole table.
            sim_band = data_sim[data_sim["filter_mapped"] == band]
            obs_band = data_obs[data_obs["filter_mapped"] == band]

        if sim_band.empty or obs_band.empty:
            logger.debug("No data for band %s — skipping.", band)
            continue

        # sim_band is copied only when a `time_bin` COLUMN has to be written to
        # it, which is exactly when the legacy ABC path might re-select on that
        # column.  Under array_bins with abc_reuse_bin the bin codes live in a
        # standalone array and the frame is only ever read, so the copy --
        # 500,000 rows x 4 columns per band at a 25,000-sample grid -- buys
        # nothing.
        _needs_bin_col = not (array_bins and abc_reuse_bin)
        if _needs_bin_col:
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

        _bin_codes = np.digitize(sim_band["time"].to_numpy(), bins)
        if _needs_bin_col:
            sim_band["time_bin"] = _bin_codes

        # Pre-partition simulations by time bin for O(1) lookup per observation.
        sim_groups: Optional[Dict[int, pd.DataFrame]] = None
        sim_arrays: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
        if array_bins:
            # A stable sort on the bin codes puts each bin's rows in one
            # contiguous run, IN THEIR ORIGINAL ORDER, so the runs are exactly
            # groupby's groups.  np.diff finds the run boundaries; the slices
            # are views, not copies.
            _tb = _bin_codes
            _order = np.argsort(_tb, kind="stable")
            _tb_sorted = _tb[_order]
            _mag = sim_band["absolute_magnitude"].to_numpy()[_order]
            _sid = sim_band["sample_id"].to_numpy()[_order]
            _cuts = np.flatnonzero(np.diff(_tb_sorted)) + 1
            _starts = np.concatenate(([0], _cuts))
            _stops = np.concatenate((_cuts, [_tb_sorted.size]))
            sim_arrays = {
                int(_tb_sorted[a]): (_mag[a:b], _sid[a:b])
                for a, b in zip(_starts, _stops)
            }
        else:
            sim_groups = {k: v for k, v in sim_band.groupby("time_bin")}

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

            # Non-detections score via nondetection_tail regardless of
            # `space` (which only picks how detections are compared).
            is_limit_row = (score_limits
                            and bool(getattr(obs_row, is_limit_col, False)))
            if is_limit_row:
                sigma_obs = float("nan")   # nothing to propagate for a limit

            # Depth is a property of the exposure, per-observation, and
            # orthogonal to is_limit_row -- conditions the reference
            # population's detectability for every row.
            M_lim_row = getattr(obs_row, M_lim_col, None)
            if M_lim_row is None or not np.isfinite(M_lim_row):
                M_lim_row = M_lim
            if M_lim_row is not None and not np.isfinite(M_lim_row):
                M_lim_row = None

            # The per-epoch uncertainty actually scored with.  Normally the
            # column as given; `sigma_col` overrides it, which is how the
            # photometric-only path of Part X is selected.
            if sigma_col is not None and not is_limit_row:
                s_row = getattr(obs_row, sigma_col, None)
                if s_row is not None and np.isfinite(s_row) and s_row > 0:
                    sigma_obs = float(s_row)

            # Skip degenerate observations.  A limit row supplies its depth
            # through M_obs and needs no sigma_obs of its own -- see below.
            if is_limit_row:
                if not np.isfinite(M_obs):
                    logger.debug("Skipping limit with no quoted depth at "
                                "t=%.3f d.", t_obs)
                    continue
            elif not (np.isfinite(M_obs) and np.isfinite(sigma_obs)
                     and sigma_obs > 0):
                logger.debug("Skipping invalid observation at t=%.3f d.", t_obs)
                continue

            bin_idx = int(np.digitize(t_obs, bins))
            # One branch, yielding the same three things either way: how many
            # simulations the bin holds, their magnitudes, and their ids.
            if array_bins:
                mag_bin, sid_bin = sim_arrays.get(bin_idx, (_EMPTY_F, _EMPTY_F))
                sim_bin = None
                n_bin = int(mag_bin.size)
            else:
                sim_bin = sim_groups.get(bin_idx, pd.DataFrame())
                n_bin = len(sim_bin)
                mag_bin = (sim_bin["absolute_magnitude"].to_numpy()
                           if n_bin else _EMPTY_F)
                sid_bin = sim_bin["sample_id"].to_numpy() if n_bin else _EMPTY_F

            if n_bin < min_sim_points:
                logger.debug(
                    "Bin %d has %d simulations (< %d) — skipping.",
                    bin_idx, n_bin, min_sim_points,
                )
                continue

            # 3a. Compute P_tail_KNe and P_near_KNe (paper eqs. 6-7 and 4).
            #     Non-detections use nondetection_tail (censored, not a
            #     point measurement); detections use the one PIT-based
            #     estimator regardless of space, since it only needs
            #     converted inputs.
            if is_limit_row:
                metric = nondetection_tail(
                    mag_bin, M_lim=M_obs,
                    n_sigma_limit=n_sigma_limit, flux_zp=flux_zp,
                )
                # Stouffer maps p_tail_mean -> 1 to strong evidence FOR the
                # model, but most non-detections are simply uninformative
                # (nothing was expected to be seen). Gate: exclude those
                # entirely rather than let them count as confirmation. Cap:
                # even an informative one can only say "tension" (z >= 0),
                # never "confirms" (z < 0) -- see docs/FLUX_SPACE.md.
                metric["p_tail_nondetect_raw"] = metric["p_tail_mean"]
                if metric["F_hat"] < nondetection_gate:
                    metric["p_tail_KNe"] = float("nan")
                    metric["p_tail_mean"] = float("nan")
                else:
                    metric["p_tail_KNe"] = min(metric["p_tail_KNe"], 0.5)
                    metric["p_tail_mean"] = min(metric["p_tail_mean"], 0.5)
            else:
                score_bin, score_M_obs, score_sigma_obs = mag_bin, M_obs, sigma_obs
                score_M_lim = M_lim_row
                if space == "flux":
                    score_bin = _flux_score_axis(mag_bin, flux_zp)
                    score_M_obs = float(_flux_score_axis(M_obs, flux_zp))
                    score_sigma_obs = float(
                        flux_sigma_of(M_obs, sigma_obs, flux_zp))
                    if M_lim_row is not None:
                        score_M_lim = float(_flux_score_axis(M_lim_row, flux_zp))

                #     Under the closed form these are exact sums over the
                #     simulations in the bin, so there is no KDE to fit;
                #     under Monte Carlo one is fitted per bin and reused.
                cached_kde = None
                if p_tail_method == "montecarlo":
                    if bin_idx not in kde_cache:
                        kde_cache[bin_idx] = gaussian_kde(score_bin)
                    cached_kde = kde_cache[bin_idx]

                metric = predictive_tail_kde(
                    score_bin,
                    M_obs=score_M_obs,
                    sigma_obs=score_sigma_obs,
                    k=k_near,
                    n_sim=n_kde_sim,
                    n_obs=n_obs,
                    kde=cached_kde,
                    M_lim=score_M_lim,
                    min_n_eff=min_n_eff,
                    p_tail_method=p_tail_method,
                    sigma_model=sigma_model,
                    random_state=random_state,
                    p_near_compute=p_near_compute,
                )

            # 3b. ABC diagnostic — consistent simulation IDs at this epoch.
            #     `sim_bin` is passed through when abc_reuse_bin: it is exactly
            #     the rows compute_consistent_ids_anyhit would otherwise select
            #     for itself, and the scorer already holds it.  Without it the
            #     function rescans the whole band once per observation -- 200,000
            #     rows to reach 10,000 of them at a 10,000-sample grid, a waste
            #     of a factor of the number of time bins -- which measured 3.3x
            #     of the diagnostic's cost at 25,000 samples.
            #
            #     Passing None is not a re-implementation of the old path: it IS
            #     the old path, still the default inside the diagnostic and still
            #     what runs for any caller invoking it directly.
            consistent_ids: List = []
            # ABC's ROPE test is defined in magnitude space; a limit row has
            # no magnitude-space sigma_obs to give it (that is the whole
            # reason it needed flux space to score at all), so it is skipped
            # for that row rather than fed a manufactured number.
            if abc_compute and not is_limit_row:
                if abc_reuse_bin and array_bins:
                    # The rows are already in hand as two arrays; go straight to
                    # the shared ROPE test.
                    consistent_ids = _anyhit_from_arrays(
                        mag_bin, sid_bin, M_obs, sigma_obs, overlap_k,
                    )
                else:
                    consistent_ids = compute_consistent_ids_anyhit(
                        sim_band=sim_band,
                        bin_idx=bin_idx,
                        M_obs=M_obs,
                        sigma_obs=sigma_obs,
                        overlap_k=overlap_k,
                        sim_bin=sim_bin if abc_reuse_bin else None,
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
                # F_hat is the ONLY field carrying the DIRECTION of the
                # deviation.  P_tail = 2*min(F, 1-F) folds the two sides
                # together, so from P_tail alone a candidate that is too bright
                # and one that is too faint are indistinguishable.
                #
                # Mind the sign.  F_hat = Pr(M_rep <= M_obs), and magnitudes run
                # backwards, so it is the probability that a MODEL IS BRIGHTER
                # THAN THE OBSERVATION:
                #     F -> 0   the observation is BRIGHTER than the grid
                #     F ~ 0.5  the observation sits mid-population
                #     F -> 1   the observation is FAINTER than the grid
                "F_hat": metric["F_hat"],
                "p_tail_KNe": metric["p_tail_KNe"],
                "p_tail_mean": metric["p_tail_mean"],
                "p_tail_std": metric["p_tail_std"],
                "p_near_KNe": metric["p_near_KNe"],
                "M_lim": float(M_lim_row) if M_lim_row is not None else np.nan,
                "n_eff": metric["n_eff"],
                "scoreable": metric["scoreable"],
                "p_tail_method": metric["p_tail_method"],
                "space": space,
                "is_limit": bool(is_limit_row),
                "p_tail_nondetect_raw": metric.get("p_tail_nondetect_raw", np.nan),
                "n_sim_bin": n_bin,
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
