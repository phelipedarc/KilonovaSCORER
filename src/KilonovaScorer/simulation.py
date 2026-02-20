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






# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# from astropy.cosmology import Planck18 as cosmo
# from astropy.cosmology import z_at_value
# from astropy import units as u
# import redback
# from redback.model_library import all_models_dict
# from bilby.core.prior import Uniform



# def simulate_single_sample(sample_id, MODEL_NAME, TIME, FILTER_BANDS, z, mu, RANDOM_SEED=42):
#     """
#     Simulate a single kilonova light curve sample for all bands.
#     """
#     import numpy as np
#     import pandas as pd
#     import redback
#     from redback.model_library import all_models_dict
#     from bilby.core.prior import Uniform

#     # ------------------------
#     # Build priors
#     # ------------------------
#     prior = redback.priors.get_priors(model=MODEL_NAME)
#     if MODEL_NAME == 'metzger_kilonova_model':
#         prior['mej'] = Uniform(1e-4, 0.1)
#         prior['vej'] = Uniform(0.01, 0.5)
#         prior['kappa'] = Uniform(0.1, 30)
#     elif MODEL_NAME == 'two_component_kilonova_model':
#         #ejecta Mass:
#         prior['mej_1'] = Uniform(minimum=1e-4, maximum=0.1,  name='mej_1', latex_label='$M_{\\mathrm{ej}~1}~(M_\\odot)$', unit=None,  oundary=None)
#         prior['mej_2'] = Uniform(minimum=1e-4, maximum=0.1,  name='mej_2', latex_label='$M_{\\mathrm{ej}~2}~(M_\\odot)$', unit=None, boundary=None)
#         #Ejecta velocity:
#         prior['vej_1'] = Uniform(minimum=0.01, maximum=0.7, name='vej_1', latex_label='$v_{\\mathrm{ej}~1}~(c)$', unit=None, boundary=None)
#         prior['vej_2'] = Uniform(minimum=0.01, maximum=0.7, name='vej_2', latex_label='$v_{\\mathrm{ej}~1}~(c)$', unit=None, boundary=None)
#         #Kappa- opacity Blue + Red:
#         prior['kappa_1'] = Uniform(minimum=0.1, maximum=0.5,name='kappa_1', latex_label='$\\kappa_{1}~(\\mathrm{cm}^{2}/\\mathrm{g})$', unit=None, boundary=None)
#         prior['kappa_2'] = Uniform(minimum=1, maximum=30,name='kappa_2', latex_label='$\\kappa_{2}~(\\mathrm{cm}^{2}/\\mathrm{g})$', unit=None, boundary=None)

#     # ------------------------
#     # Sample parameters
#     # ------------------------
#     np.random.seed(RANDOM_SEED + sample_id)
#     params = prior.sample()
#     params['redshift'] = z

#     model_func = all_models_dict[MODEL_NAME]
#     rows = []

#     # ------------------------
#     # Simulate all bands
#     # ------------------------
#     for band in FILTER_BANDS:
#         mag = model_func(TIME, **params, output_format='magnitude', bands=[band])
#         abs_mag = mag - mu
#         row_dict = {
#             'sample_id': sample_id,
#             'band': band,
#             'time': TIME,
#             'magnitude': mag,
#             'absolute_magnitude': abs_mag
#         }
#         # Add all sampled parameters
#         for k, v in params.items():
#             row_dict[k] = v
#         rows.append(pd.DataFrame(row_dict))

#     return pd.concat(rows, ignore_index=True)


# # -----------------------------------
# # Main simulation function
# # -----------------------------------
# def simulate_kilonova(N_SIM=100000, MODEL_NAME='two_component_kilonova_model', SAVE_CSV=True):
#     """
#     Simulate kilonova light curves using REDBACK models.

#     Parameters
#     ----------
#     N_SIM : int
#         Number of light curve samples to simulate.
#     MODEL_NAME : str
#         Name of the REDBACK model to use.
#     SAVE_CSV : bool
#         If True, saves the simulated DataFrame to CSV.

#     Returns
#     -------
#     pd.DataFrame
#         Concatenated DataFrame with all simulated samples.
#     """
#     # ------------------------
#     # Check model availability
#     # ------------------------
#     if MODEL_NAME not in all_models_dict:
#         raise ValueError(f"Model '{MODEL_NAME}' not available in REDBACK.")

#     # ------------------------
#     # Simulation configuration
#     # ------------------------
#     TIME = np.linspace(0, 10, 1000)
#     FILTER_BANDS = ['lsstg', 'lsstr', 'lssti', 'lsstz']

#     # ------------------------
#     # Cosmology
#     # ------------------------
#     DL_Mpc = 259
#     z = z_at_value(cosmo.luminosity_distance, DL_Mpc * u.Mpc, zmin=0.001, zmax=5).value
#     mu = 5 * np.log10(DL_Mpc * 1e6) - 5
#     print(f"Using redshift z={z:.4f}, distance modulus mu={mu:.2f}")

#     # ------------------------
#     # Parallel simulation
#     # ------------------------
#     ncores = max(1, min(cpu_count() - 1, 8))  # up to 8 cores safely
#     print(f"Using {ncores} cores for {N_SIM} simulations")

#     args_list = [(i, MODEL_NAME, TIME, FILTER_BANDS, z, mu) for i in range(N_SIM)]

#     with Pool(ncores) as pool:
#         all_samples = list(tqdm(pool.starmap(simulate_single_sample, args_list),
#                                 total=N_SIM, desc="Simulating"))

#     # Flatten and concatenate all DataFrames
#     df = pd.concat(all_samples, ignore_index=True)

#     # ------------------------
#     # Save CSV if requested
#     # ------------------------
#     if SAVE_CSV:
#         filename = f"simulations_{MODEL_NAME}.csv"
#         df.to_csv(filename, index=False)
#         print(f"✅ Simulations saved to {filename}")

#     return df
