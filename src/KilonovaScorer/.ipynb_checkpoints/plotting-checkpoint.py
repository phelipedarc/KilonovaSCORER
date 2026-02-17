
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde


def set_plot_style():
    """Applies professional publication standards to matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })



def plot_simulations_LCS(data_sim_all,BIN_WIDTH = 0.2):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # df columns: time, band, absolute_magnitude
  
    
    # -----------------------------
    # 1) Fast mean/std per (band, 0.2-day bin)
    # -----------------------------
    t = data_sim_all["time"].to_numpy(copy=False)
    m = data_sim_all["absolute_magnitude"].to_numpy(copy=False)
    band_arr = data_sim_all["band"].to_numpy(copy=False)
    
    # valid rows only
    mask = np.isfinite(t) & np.isfinite(m) & pd.notna(band_arr)
    t = t[mask]
    m = m[mask]
    band_arr = band_arr[mask]
    
    # bin index for 0.2-day bins
    tmin = t.min()
    time_idx = np.floor((t - tmin) / BIN_WIDTH).astype(np.int32)
    n_time = int(time_idx.max()) + 1
    
    # factorize bands
    band_codes, band_levels = pd.factorize(band_arr, sort=True)
    band_codes = band_codes.astype(np.int32)
    n_bands = len(band_levels)
    
    # 1D key for bincount
    key = band_codes * n_time + time_idx
    size = n_bands * n_time
    
    # use float64 accumulators for numerical stability on huge N
    m64 = m.astype(np.float64, copy=False)
    
    counts = np.bincount(key, minlength=size).astype(np.int64)
    sum_m = np.bincount(key, weights=m64, minlength=size)
    sum_m2 = np.bincount(key, weights=m64 * m64, minlength=size)
    
    # mean and (population) std; if you want sample std, see note below
    mean = np.full(size, np.nan, dtype=np.float64)
    std = np.full(size, np.nan, dtype=np.float64)
    
    nz = counts > 0
    mean[nz] = sum_m[nz] / counts[nz]
    var = (sum_m2[nz] / counts[nz]) - mean[nz] ** 2
    var = np.maximum(var, 0.0)  # numerical guard
    std[nz] = np.sqrt(var)
    
    mean = mean.reshape(n_bands, n_time)
    std = std.reshape(n_bands, n_time)
    counts = counts.reshape(n_bands, n_time)
    
    # time bin centers
    time_centers = tmin + (np.arange(n_time, dtype=np.float64) + 0.5) * BIN_WIDTH
    
    # -----------------------------
    # 2) Plot mean ± std as fill_between
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    
    for b_idx, band in enumerate(band_levels):
        mu = mean[b_idx]
        sd = std[b_idx]
        ok = np.isfinite(mu) & np.isfinite(sd) & (counts[b_idx] > 0)
    
        if not np.any(ok):
            continue
    
        x = time_centers[ok]
        y = mu[ok]
        ylo = y - sd[ok]
        yhi = y + sd[ok]
    
        ax.fill_between(x, ylo, yhi, alpha=0.25, linewidth=0, label=str(band))
        ax.plot(x, y, linewidth=1.4)
    
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Simulated Absolute magnitude [AB]")
    ax.invert_yaxis()  # magnitudes: brighter = smaller
    ax.set_ylim(0,-20)
    ax.set_xlim(0,10)
    ax.grid(alpha=0.3)
    ax.legend(title="Band", ncols=2, fontsize=9)
    plt.tight_layout()
    plt.show()

@timer_warp
def plot_observational_data_Apparent(data_obs):
    # 1. Define your marker and color mapping
    # Matches the 'g-band', 'r-band' format from your pipeline
    marker_map = {
        'g-band': 's',  # square
        'r-band': '^',  # triangle_up
        'i-band': 'D',  # diamond
        'z-band': 'v',  # triangle_down
        'y-band': 'P'   # plus (filled)
    }
    
    color_map = {
    'g-band': '#E69F00',  # golden orange
    'r-band': '#D55E00',  # burnt orange
    'i-band': '#CC79A7',  # muted magenta
    'z-band': '#8E44AD',  # medium purple
    'y-band': '#5B2C6F'   # deep purple
    }


    plt.figure(figsize=(10, 6))

    # This automatically creates the grouped legend
    for band in np.unique(data_obs['filter_mapped']):
        mask = data_obs['filter_mapped'] == band
        
        plt.errorbar(
            data_obs.loc[mask, 'time_after_gw'],
            data_obs.loc[mask, 'magnitude'],
            yerr=data_obs.loc[mask, 'e_magnitude'],
            fmt=marker_map.get(band, 'o'), 
            color=color_map.get(band, 'black'),
            label=band,
            markersize=8,
            capsize=3,
            linestyle='none',
            markeredgecolor='black',
            alpha=0.8
        )

    # 3. Formatting
    plt.gca().invert_yaxis()
    plt.xlabel('Time after GW merger [days]', fontsize=12)
    plt.ylabel('Apparent Magnitude', fontsize=12)
    plt.title('Multi-band Photometry of Kilonova Candidate', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Filters", frameon=True)
    plt.xlim(0,10)
    plt.tight_layout()
    plt.show()

def plot_survivor_relative(metric_df, candidate_name=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    run_df = metric_df

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    t = run_df["obs_time"].to_numpy(dtype=float)
    n_run = run_df["n_running_survivors"].to_numpy(dtype=float)
    n_acc = run_df["n_consistent_lcs"].to_numpy(dtype=float)

    y_rel = n_run / n_acc

    # ---- Main line ----
    ax.plot(
        t,
        y_rel,
        lw=2.5,
        color="indigo",
        zorder=2,
        label="Global running survivors / N_accepted"
    )

    # ---- Color-encoded markers (log scaling for low-count emphasis) ----
    n_for_color = np.clip(n_run, 1, None)  # avoid log(0)
    norm = LogNorm(vmin=1, vmax=max(n_for_color))

    sc = ax.scatter(
        t,
        y_rel,
        c=n_for_color,
        norm=norm,
        cmap='Spectral',
        s=75,
        edgecolors='black',
        zorder=3
    )

    # ---- Horizontal colorbar inset ----
    cax = inset_axes(
        ax,
        width="55%",   # fraction of parent axes width
        height="4%",   # fraction of parent axes height
        loc="upper right",
        borderpad=1.2
    )
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Number of Accepted Simulations over time", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Optional: emphasize your threshold visually (N≈10)
    cbar.ax.axvline(norm(10), lw=2, color='k')

    # ---- Axes formatting ----
    ax.set_xlabel("Time since merger [days]")
    ax.set_ylabel("Cumulative Acceptance / N_accepted(t,band)")
    ax.grid(alpha=0.3)
    #ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=0, color="darkred", lw=1.2)
    ax.axhline(y=0.1, color="darkred", lw=1.2,alpha=0.4)

    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.legend(frameon=False, ncol=2)
    # Use a symmetric log scale
    ax.set_yscale("symlog", linthresh=0.05)  # linear region below ~0.05, log-like above
    ax.set_ylim(-0.01, 1.05)


    # ---- First zero-survivor event annotation (kept) ----
    hit0 = np.where(n_run == 0)[0]
    if len(hit0) > 0:
        i0 = int(hit0[0])
        t0 = float(run_df.loc[i0, "obs_time"])
        band0 = str(run_df.loc[i0, "band"])
        m0 = float(run_df.loc[i0, "observed_mag"]) if "observed_mag" in run_df.columns else np.nan
        e0 = float(run_df.loc[i0, "observed_mag_err"]) if "observed_mag_err" in run_df.columns else np.nan
        tl = run_df.loc[i0, "time_bin_low"]
        th = run_df.loc[i0, "time_bin_high"]

        ax.scatter([t0], [0.0], s=90, color="darkred", zorder=5)

        msg = (
            "Running survivors → 0\n"
            f"t = {t0:.4f} d, band = {band0}\n"
            f"M = {m0:.3f} ± {e0:.3f}\n"
            f"bin = [{tl:.3f}, {th:.3f}] d"
        )

        ax.annotate(
            msg,
            xy=(t0, 0.0),
            xytext=(15, 25),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="darkred", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
            zorder=6
        )

        ax.axvline(t0, color="darkred", linestyle=":", lw=1.2, alpha=0.8, zorder=1)

    plt.tight_layout()
    plt.show()


def plot_survivor_param_kde_grid(
    metric_df,
    data_sim_all,
    params=("mej", "vej", "kappa", "beta"),
    epochs=None,              # list of row indices in run_df; None => evenly sample up to max_epochs
    max_epochs=10,
    n_grid=300,               # KDE evaluation points
    bw_method=None,           # passed to gaussian_kde (None uses Scott)
    min_n=30,                 # minimum survivors to draw KDE
    ):
    """
    KDE grid plot:
      - Rows = parameters
      - Columns = epochs (time increases to the right)
      - X-axis for each parameter is fixed across epochs (global min/max across all epochs)
      - NO log scaling (raw parameter values)

    Requires:
      - run_df['running_survivors_ids'] list of sample_id per epoch
      - param_df columns: sample_id + params (one row per sample_id)
    """
    HALLOWEEN_COLORS = {
        "mej_1":   "#FF8C00",  # pumpkin orange
        "vej_1":   "#7B1FA2",  # deep purple
        "kappa_1": "#2E2E2E",  # near-black
        "temperature_floor_1":  "darkred",  # dark burnt orange
        "mej_2":   "#FF8C00",  # pumpkin orange
        "vej_2":   "#7B1FA2",  # deep purple
        "kappa_2": "#2E2E2E",  # near-black
        "temperature_floor_2":  "darkred",  # dark burnt orange
    }
    metric_df = (metric_df.sort_values("obs_time").reset_index(drop=True))

    param_df = data_sim_all[["sample_id", "mej_1", "vej_1", "kappa_1", 
     "temperature_floor_1","mej_2", "vej_2", "kappa_2", "temperature_floor_2"]].drop_duplicates("sample_id")

    
    if "running_survivors_ids" not in metric_df.columns:
        raise ValueError("metric_df must contain 'running_survivors_ids'.")

    # --- param lookup (one row per sample_id) ---
    p = (
        param_df[["sample_id"] + list(params)]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )

    # --- choose epochs (columns) ---
    n = len(metric_df)
    if epochs is None:
        if n <= max_epochs:
            epochs = list(range(n))
        else:
            epochs = np.unique(np.linspace(0, n - 1, max_epochs).round().astype(int)).tolist()

    # --- global x-range per parameter based on survivors across chosen epochs ---
    global_ranges = {}
    for par in params:
        all_vals = []
        for idx in epochs:
            ids = metric_df.iloc[idx]["running_survivors_ids"]
            if not ids:
                continue
            sub = p.loc[p.index.intersection(ids)]
            v = sub[par].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                all_vals.append(v)
        if len(all_vals) == 0:
            raise ValueError(f"No finite values found for parameter '{par}' in selected epochs.")
        vv = np.concatenate(all_vals)
        vmin, vmax = np.min(vv), np.max(vv)
        if np.isclose(vmin, vmax):
            # expand a tiny bit to make plotting possible
            pad = 1e-12 if vmin == 0 else 0.01 * abs(vmin)
            vmin, vmax = vmin - pad, vmax + pad
        global_ranges[par] = (vmin, vmax)

    # --- figure: rows=params, cols=epochs (time to the right) ---
    nrows = len(params)
    ncols = len(epochs)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.1 * ncols, 2.3 * nrows),
        dpi=150,
        sharex="row",
        sharey="row",
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # --- precompute x grids per parameter ---
    xgrid = {par: np.linspace(*global_ranges[par], n_grid) for par in params}

    # --- plot ---
    for j, idx in enumerate(epochs):
        r = metric_df.iloc[idx]
        t = float(r["obs_time"]) if "obs_time" in metric_df.columns else np.nan
        band = str(r["band"]) if "band" in metric_df.columns else ""
        ids = r["running_survivors_ids"]

        # column header (time)
        axes[0, j].set_title(f"t={t:.3f} d\n{band}", fontsize=9)

        # extract survivor rows once per epoch
        sub = p.loc[p.index.intersection(ids)] if ids else None

        for i, par in enumerate(params):
            ax = axes[i, j]
            color = HALLOWEEN_COLORS.get(par, f"C{i}")

            # empty survivors
            if sub is None or sub.empty:
                ax.text(0.5, 0.5, "∅", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12)
                ax.set_xlim(global_ranges[par])
                ax.grid(alpha=0.2)
                continue

            v = sub[par].to_numpy(dtype=float)
            v = v[np.isfinite(v)]

            ax.set_xlim(global_ranges[par])
            ax.grid(alpha=0.2)

            if len(v) < min_n or np.allclose(np.std(v), 0):
                # too few points or degenerate: show rug as fallback
                # (still keeps x-axis comparable)
                y = np.zeros_like(v)
                ax.plot(v, y, "|", color=color, alpha=0.8)
                continue

            kde = gaussian_kde(v, bw_method=bw_method)
            y = kde(xgrid[par])
            ax.plot(xgrid[par], y, color=color, lw=2.0)
            ax.fill_between(xgrid[par], 0, y, color=color, alpha=0.25)

            # annotate N on top-right of each panel (optional but helpful)
            ax.text(0.98, 0.92, f"N={len(v)}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)

            # row labels (parameter names) only on leftmost column
            if j == 0:
                ax.set_ylabel(par)

            # x-labels only on bottom row
            if i == nrows - 1:
                ax.set_xlabel(par)

    plt.tight_layout()
    plt.show()
    
def plot_lc_probnear_global_survivors_and_survivor_models(
    metric_df,
    data_sim,
    candidate_name,
    cmap=plt.cm.magma,
    max_models=100,
    alpha_models=0.10,
    lw_models=1.0,
    seed=12345,
    band_col_obs="band",
    band_col_sim="filter_mapped",
    time_col_obs="obs_time",
    time_col_sim="time",
    mag_col_obs="observed_mag",
    err_col_obs="observed_mag_err",
    mag_col_sim="absolute_magnitude",
    id_col_sim="sample_id",
):
    # --------- Filter Candidate + Sort ----------
    df = metric_df.copy()
    if "candidate_name" in df.columns:
        df = df[df["candidate_name"] == candidate_name]
    if df.empty:
        raise ValueError(f"No rows for candidate '{candidate_name}' in metric_df.")

    required = [time_col_obs, band_col_obs, mag_col_obs, err_col_obs, "prob_near", "consistent_ids", "n_consistent_lcs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"metric_df missing required columns: {missing}")

    df = df.sort_values([time_col_obs, band_col_obs]).reset_index(drop=True)

    # --------- Compute Running Survivors ----------
    sets = [set(x) if isinstance(x, (list, tuple, np.ndarray)) else set()
            for x in df["consistent_ids"].values]
    
    running = sets[0].copy()
    run_n = [len(running)]
    run_ids = [sorted(running)]
    for i in range(1, len(sets)):
        running &= sets[i]
        run_n.append(len(running))
        run_ids.append(sorted(running))

    run_df = df[[time_col_obs, band_col_obs, "n_consistent_lcs"]].copy()
    run_df["n_running_survivors"] = np.array(run_n, dtype=int)
    run_df["running_survivors_ids"] = run_ids

    # --------- Identify Survivors Right Before Zero ----------
    hit0 = np.where(run_df["n_running_survivors"].to_numpy() == 0)[0]
    
    if len(hit0) > 0:
        i0 = int(hit0[0])
        t_collapse = float(run_df.loc[i0, time_col_obs])
        # Epoch immediately before going to zero
        i_use = max(i0 - 1, 0)
    else:
        t_collapse = None
        # Use last available epoch if it never hits zero
        i_use = len(run_df) - 1

    survivor_ids = run_df.loc[i_use, "running_survivors_ids"]
    survivor_ids_plot = []
    if survivor_ids:
        if len(survivor_ids) > max_models:
            rng = np.random.default_rng(seed)
            survivor_ids_plot = rng.choice(survivor_ids, size=max_models, replace=False).tolist()
        else:
            survivor_ids_plot = list(survivor_ids)

    # --------- Prep Plotting ----------
    t = df[time_col_obs].to_numpy(float)
    mag = df[mag_col_obs].to_numpy(float)
    mag_err = df[err_col_obs].to_numpy(float)
    pnear = df["prob_near"].to_numpy(float)
    band = df[band_col_obs].astype(str).to_numpy()

    marker_map = {"u-band": "o", "g-band": "s", "r-band": "^", "i-band": "D", "z-band": "v", "y-band": "P"}
    band_colors = {"u-band": "#9467bd", "g-band": "#2ca02c", "r-band": "#d62728",
                   "i-band": "#ff7f0e", "z-band": "#1f77b4", "y-band": "#8c564b"}

    norm_p = Normalize(vmin=0.0, vmax=1.0)
    sm_p = ScalarMappable(norm=norm_p, cmap=cmap)

    fig, (ax_lc, ax_p, ax_run) = plt.subplots(
        3, 1, figsize=(11, 12), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]}, dpi=150
    )

    # (1) Panel 1: Light curve + Models right before collapse
    sim_legend_handles = []
    if len(survivor_ids_plot) > 0:
        sim_sub = data_sim[data_sim[id_col_sim].isin(survivor_ids_plot)]
        for b_str in np.unique(band):
            sim_b = sim_sub[sim_sub[band_col_sim].astype(str) == str(b_str)]
            color = band_colors.get(str(b_str), "grey")
            for _, g in sim_b.groupby(id_col_sim):
                g = g.sort_values(time_col_sim)
                ax_lc.plot(g[time_col_sim], g[mag_col_sim], lw=lw_models, 
                           alpha=alpha_models, color=color, zorder=1)
            
            line_handle, = ax_lc.plot([], [], color=color, lw=2, label=f"Sim {b_str}")
            sim_legend_handles.append(line_handle)

    obs_legend_handles = []
    for b_str in np.unique(band):
        m = band == b_str
        mk = marker_map.get(b_str, "o")
        for ti, mi, ei, pi in zip(t[m], mag[m], mag_err[m], pnear[m]):
            ax_lc.errorbar(ti, mi, yerr=ei, fmt=mk, markersize=8, markeredgecolor="black",
                           color=cmap(norm_p(pi)), alpha=0.9, zorder=3)
        obs_handle, = ax_lc.plot([], [], marker=mk, linestyle="none", color="gray", 
                                 markeredgecolor="black", label=f"Obs {b_str}")
        obs_legend_handles.append(obs_handle)

    ax_lc.invert_yaxis()
    ax_lc.set_ylim(-9, -17)
    ax_lc.set_ylabel("Abs Mag (AB)")
    leg1 = ax_lc.legend(handles=obs_legend_handles, loc='upper center', fontsize=9, frameon=True, title="Obs Bands", ncol=2)
    ax_lc.add_artist(leg1)
    ax_lc.legend(handles=sim_legend_handles, loc='upper right', fontsize=9, frameon=True, title="Simulation Models", ncol=2)
    ax_lc.grid(True, alpha=0.3)

    # (2) Panel 2: prob_near vs time
    for b_str in np.unique(band):
        m = band == b_str
        ax_p.scatter(t[m], pnear[m], marker=marker_map.get(b_str, "o"), s=60, 
                     edgecolor="black", c=pnear[m], cmap=cmap, norm=norm_p, alpha=0.8)
    ax_p.set_ylabel(r"$P_{\mathrm{near,KNe}}$")
    ax_p.axhline(y=0.2, color="darkred", lw=1.2, alpha=0.5, linestyle="--")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.grid(True, alpha=0.3)

    # (3) Panel 3: Global running survivors (RELATIVE)
    y_rel = run_df["n_running_survivors"].to_numpy() / run_df["n_consistent_lcs"].to_numpy()
    ax_run.plot(run_df[time_col_obs], y_rel, lw=2.5, color="indigo", zorder=2)
    
    n_for_color = np.clip(run_df["n_running_survivors"], 1, None)
    norm_log = LogNorm(vmin=1, vmax=max(n_for_color) if max(n_for_color) > 1 else 10)
    sc = ax_run.scatter(run_df[time_col_obs], y_rel, c=n_for_color, norm=norm_log, cmap='Spectral', 
                        s=75, edgecolors='black', zorder=3)

    ax_run.set_yscale("symlog", linthresh=0.05)
    ax_run.set_ylim(-0.01, 1.5)
    ax_run.set_ylabel("Rel. Acceptance")
    ax_run.set_xlabel("Time since merger [days]")
    ax_run.grid(True, alpha=0.3)
    ax_run.axhline(y=0.1, color="darkred", lw=1.2, alpha=0.4, linestyle="--")
    ax_run.axhline(y=0, color="darkred", lw=1.2, alpha=0.9, linestyle="-")

    cax_in = inset_axes(ax_run, width="30%", height="5%", loc="upper right", borderpad=1.5)
    fig.colorbar(sc, cax=cax_in, orientation="horizontal").set_label("Accepted Sims", fontsize=9)

    # --------- Global Vertical Lines (Collapse) ----------
    if t_collapse is not None:
        for ax in (ax_lc, ax_p, ax_run):
            ax.axvline(t_collapse, color="darkred", linestyle=":", lw=2.5, alpha=0.8)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(sm_p, cax=cbar_ax).set_label(r"$P_{\mathrm{near,KNe}}$")

    plt.show()