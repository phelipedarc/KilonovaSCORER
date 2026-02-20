import sys
import time
import multiprocessing
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
import redback
from redback.model_library import all_models_dict
from bilby.core.prior import Uniform

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

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

# Worker function must be top-level for multiprocessing
def simulate_single_sample(sample_id, MODEL_NAME, TIME, FILTER_BANDS, z, mu, RANDOM_SEED=42):
    import numpy as np
    import pandas as pd
    import redback
    from bilby.core.prior import Uniform
    from redback.model_library import all_models_dict

    prior = redback.priors.get_priors(model=MODEL_NAME)
    if MODEL_NAME == 'two_component_kilonova_model':
        prior['mej_1'] = Uniform(minimum=1e-4, maximum=0.1,  name='mej_1', latex_label='$M_{\\mathrm{ej}~1}~(M_\\odot)$', unit=None,  boundary=None)
        prior['mej_2'] = Uniform(minimum=1e-4, maximum=0.1,  name='mej_2', latex_label='$M_{\\mathrm{ej}~2}~(M_\\odot)$', unit=None, boundary=None)
        #Ejecta velocity:
        prior['vej_1'] = Uniform(minimum=0.01, maximum=0.7, name='vej_1', latex_label='$v_{\\mathrm{ej}~1}~(c)$', unit=None, boundary=None)
        prior['vej_2'] = Uniform(minimum=0.01, maximum=0.7, name='vej_2', latex_label='$v_{\\mathrm{ej}~1}~(c)$', unit=None, boundary=None)
        #Kappa- opacity Blue + Red:
        prior['kappa_1'] = Uniform(minimum=0.1, maximum=0.5,name='kappa_1', latex_label='$\\kappa_{1}~(\\mathrm{cm}^{2}/\\mathrm{g})$', unit=None, boundary=None)
        prior['kappa_2'] = Uniform(minimum=1, maximum=30,name='kappa_2', latex_label='$\\kappa_{2}~(\\mathrm{cm}^{2}/\\mathrm{g})$', unit=None, boundary=None)
    params = prior.sample()
    params['redshift'] = z

    rows = []
    for band in FILTER_BANDS:
        mag = all_models_dict[MODEL_NAME](TIME, **params, output_format='magnitude', bands=[band])
        abs_mag = mag - mu
        row_dict = {'sample_id': sample_id, 'band': band, 'time': TIME, 
                    'magnitude': mag, 'absolute_magnitude': abs_mag}
        for k, v in params.items():
            row_dict[k] = v
        rows.append(pd.DataFrame(row_dict))
    return pd.concat(rows, ignore_index=True)

# Main simulation function
def simulate_kilonova(N_SIM=100, MODEL_NAME='two_component_kilonova_model', SAVE_CSV=True):
    TIME = np.linspace(0, 10, 1000)
    FILTER_BANDS = ['lsstg','lsstr','lssti','lsstz']
    DL_Mpc = 259
    z = z_at_value(cosmo.luminosity_distance, DL_Mpc*u.Mpc).value
    mu = 5 * np.log10(DL_Mpc*1e6) - 5

    # Use 4 cores max
    ncores = min(6, multiprocessing.cpu_count() - 1)
    print(f"🕹 Starting simulations on {ncores} cores...\n")

    all_dfs = []
    with multiprocessing.Pool(ncores) as pool:
        for i, df in enumerate(pool.starmap(simulate_single_sample,
                                            [(i, MODEL_NAME, TIME, FILTER_BANDS, z, mu) for i in range(N_SIM)]),
                               start=1):
            all_dfs.append(df)
            arcade_progress_bar(i, N_SIM, bar_length=40) 

    final_df = pd.concat(all_dfs, ignore_index=True)

    if SAVE_CSV:
        filename = f"simulations_{MODEL_NAME}.csv"
        final_df.to_csv(filename, index=False)
        print(f"💾 Simulations saved as {filename}")

    print("✅ Simulation complete!")
    return final_df


# -------------------------------------------------------------------------
# 3. EXECUTION BLOCK
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# 3. EXECUTION BLOCK
# -------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    from redback.model_library import all_models_dict

    # ------------------------
    # Parse command-line arguments
    # ------------------------
    parser = argparse.ArgumentParser(description="Simulate kilonova light curves using REDBACK.")
    parser.add_argument('--model', type=str, default='two_component_kilonova_model',
                        help='Name of the REDBACK model to use')
    parser.add_argument('--nsim', type=int, default=int(1e4),
                        help='Number of simulated light curves')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save output DataFrame as CSV')

    args = parser.parse_args()

    # ------------------------
    # Verify model exists
    # ------------------------
    if args.model not in all_models_dict:
        print(f"❌ Error: Model '{args.model}' not found in REDBACK model library.")
    else:
        print(f"✅ Using model: {args.model}")
        print(f"Number of simulations: {args.nsim}")
        print(f"Save CSV: {args.save_csv}")

        # ------------------------
        # Run simulation
        # ------------------------
        final_dataframe = simulate_kilonova(
            N_SIM=args.nsim,
            MODEL_NAME=args.model,
            SAVE_CSV=args.save_csv
        )
