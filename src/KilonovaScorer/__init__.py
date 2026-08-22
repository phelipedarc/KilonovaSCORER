# ``kilonovascorer_v1`` is the ONLY name core.py still supplies that core2.py
# does not.  Everything else it used to export here — load_observations,
# preprocess_lsst_like, overlap_chain, binned_stats_cumulative_ptail — is also
# defined in core2 and was already being shadowed by the star-import below.
# Importing them from both modules made the package's behaviour depend on the
# order of these two lines: with them reversed, binned_stats_cumulative_ptail
# would silently revert from Stouffer combination to the legacy IVW mean.
# core2 now declares an explicit __all__, and this import lists only what is
# genuinely unique to core.
from .core import kilonovascorer_v1
from .core2 import *
from .plotting import plot_final_all_metrics, plot_simulations_LCS

#from .simulation import simulate_kilonova

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib as mpl

mpl.rcParams["text.usetex"] = False

__version__ = "0.1.1"
