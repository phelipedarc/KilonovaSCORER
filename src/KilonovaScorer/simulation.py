import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
import redback
from redback.model_library import all_models_dict
from bilby.core.prior import Uniform

def simulate_kilonova(N_SIM=100000, MODEL_NAME='two_component_kilonova_model', SAVE_CSV=True):
    """
    Simulate kilonova light curves using REDBACK models.

    Parameters
    ----------
    N_SIM : int
        Number of light curve samples to simulate.
    MODEL_NAME : str
        Name of the REDBACK model to use.
    SAVE_CSV : bool
        If True, saves the simulated DataFrame to CSV.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with all simulated samples.
    """

    # ------------------------
    # Check model availability
    # ------------------------
    if MODEL_NAME not in all_models_dict:
        raise ValueError(f"Model '{MODEL_NAME}' not available in REDBACK.")

    # ------------------------
    # Simulation configuration
    # ------------------------
    TIME = np.linspace(0, 10, 1000)
    FILTER_BANDS = ['lsstg', 'lsstr', 'lssti', 'lsstz']
    RANDOM_SEED = 42

    # ------------------------
    # Cosmology
    # ------------------------
    DL_Mpc = 259
    z = z_at_value(cosmo.luminosity_distance, DL_Mpc * u.Mpc, zmin=0.001, zmax=5).value
    mu = 5 * np.log10(DL_Mpc * 1e6) - 5
    print(f"Using redshift z={z:.4f}, distance modulus mu={mu:.2f}")

    # ------------------------
    # Build priors once
    # ------------------------
    prior = redback.priors.get_priors(model=MODEL_NAME)
    if MODEL_NAME == 'metzger_kilonova_model':
        prior['mej'] = Uniform(1e-4, 0.1)
        prior['vej'] = Uniform(0.01, 0.5)
        prior['kappa'] = Uniform(0.1, 30)
    elif MODEL_NAME == 'two_component_kilonova_model':
        prior['mej_1'] = Uniform(1e-4, 0.1)
        prior['mej_2'] = Uniform(1e-4, 0.1)
        prior['vej_1'] = Uniform(0.01, 0.7)
        prior['vej_2'] = Uniform(0.01, 0.7)
        prior['kappa_1'] = Uniform(0.1, 0.5)
        prior['kappa_2'] = Uniform(1, 30)

    # ------------------------
    # Worker function
    # ------------------------
    def simulate_single_sample(sample_id):
        np.random.seed(RANDOM_SEED + sample_id)
        params = prior.sample()
        params['redshift'] = z
        model_func = all_models_dict[MODEL_NAME]

        rows = []
        for band in FILTER_BANDS:
            mag = model_func(TIME, **params, output_format='magnitude', bands=[band])
            abs_mag = mag - mu
            row_dict = {
                'sample_id': sample_id,
                'band': band,
                'time': TIME,
                'magnitude': mag,
                'absolute_magnitude': abs_mag
            }
            # Add all sampled parameters dynamically
            for k, v in params.items():
                row_dict[k] = v
            rows.append(pd.DataFrame(row_dict))
        return pd.concat(rows, ignore_index=True)

    # ------------------------
    # Parallel execution
    # ------------------------
    ncores = max(1, min(cpu_count() - 1, 8))
    print(f"Simulating {N_SIM} samples using {ncores} cores...")

    with Pool(ncores) as pool:
        all_samples = list(tqdm(pool.imap(simulate_single_sample, range(N_SIM)),
                                total=N_SIM, desc="Simulating"))

    df = pd.concat(all_samples, ignore_index=True)

    # ------------------------
    # Save CSV if requested
    # ------------------------
    if SAVE_CSV:
        filename = f"simulations_{MODEL_NAME}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Simulations saved to {filename}")

    return df
