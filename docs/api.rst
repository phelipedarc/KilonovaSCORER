API reference
=============

Main functions
--------------

.. currentmodule:: KilonovaScorer

.. autosummary::

   load_observations
   preprocess_lsst_like
   core.kilonovascorer_v1
   core2.kilonovascorer_v3
   binned_stats_cumulative_ptail
   overlap_chain
   plot_final_all_metrics
   plot_simulations_LCS

Scoring (``core``)
------------------

.. automodule:: KilonovaScorer.core
   :members: kilonovascorer_v1, predictive_tail_kde_python

Scoring, newer version (``core2``)
----------------------------------

.. automodule:: KilonovaScorer.core2
   :members: kilonovascorer_v3, predictive_tail_kde, load_observations, preprocess_lsst_like, binned_stats_cumulative_ptail, overlap_chain, compute_consistent_ids_anyhit, abc_collapse_time

Utilities (``utils``)
---------------------

.. automodule:: KilonovaScorer.utils
   :members: compute_abs_mag_samples, ivw_stats_logit, calculate_sequential_score_logit

Plotting (``plotting``)
-----------------------

.. automodule:: KilonovaScorer.plotting
   :members: plot_final_all_metrics, plot_simulations_LCS, plot_survivor_param_kde_grid, plot_survivor_relative

Simulation (``simulation``, optional)
-------------------------------------

Needs ``redback`` and ``astropy``. See :doc:`simulations`.

.. automodule:: KilonovaScorer.simulation
   :members: simulate_kilonova
