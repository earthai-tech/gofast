
.. _api_ref:

===============
API Reference
===============

The :code:`GoFast` library provides a range of utilities designed to accelerate machine learning workflows. This API reference provides detailed documentation for all the modules, classes, and functions within :code:`GoFast`.

.. toctree::
   :maxdepth: 2

   gofast.analysis
   gofast.bases
   gofast.datasets
   gofast.geo
   gofast.models
   gofast.metrics
   gofast.utils
   gofast.tools
   gofast.transformers 
   gofast.plot 
   gofast.stats 


:mod:`gofast.analysis`: Analyses 
====================================

The `gofast.analysis` module includes utilities for dimensional reduction, decomposition and factor analyses used in machine learning.
 
.. automodule:: gofast.analysis
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`analysis <analysis>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
	analysis.compare_pca_fa_scores
	analysis.get_eigen_components
	analysis.get_pca_fa_scores
	analysis.get_most_variance_component
	analysis.get_total_variance_ratio
	analysis.iPCA
	analysis.kPCA
	analysis.ledoit_wolf_score
	analysis.linear_discriminant_analysis
	analysis.LLE
	analysis.make_scedastic_data
	analysis.nPCA
    analysis.oblimin_rotation 
	analysis.principal_axis_factoring
	analysis.plot_decision_regions
	analysis.project_ndim_vs_explained_variance
	analysis.promax_rotation
	analysis.rotated_factor
	analysis.samples_hotellings_t_square
	analysis.spectral_fa	
	analysis.transform_to_principal_components
	analysis.varimax_rotation  


:mod:`gofast.base`: Base Data Operations 
==========================================

The `gofast.base` module includes data classes for loading and manipulations single or many dataframes at the same time.
 
.. automodule:: gofast.base
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`base <base>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: class.rst
   
   base.Data
   base.FrameOperations
   base.MergeableSeries
   base.MergeableFrames
   base.Missing


:mod:`gofast.datasets`: Datasets 
==================================

The `gofast.datasets` module includes utilities for loading and fetching popular datasets used in machine learning.
 
.. automodule:: gofast.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets <datasets>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   datasets.fetch_data
   datasets.load_bagoue
   datasets.load_iris
   datasets.load_hlogs
   datasets.load_nlogs 
   datasets.load_mxs 
   datasets.make_sounding 
   datasets.make_african_demo 
   datasets.make_agronomy
   datasets.make_cc_factors
   datasets.make_elogging 
   datasets.make_erp 
   datasets.make_ert 
   datasets.make_gadget_sales 
   datasets.make_medical_diagnostic 
   datasets.make_mining
   datasets.make_retail_store 
   datasets.make_tem 
   datasets.make_water_demand
   datasets.make_well_logging 

:mod:`gofast.geo`: Geosciences
================================

The `gofast.geo` module is composed of geosciences utilities for handling 
geology drilling and boreholes used in machine learning. To use the :ref:`geosciences <geo>` 
module, ``pyproj`` library or ``osgeo`` needs to be installed. If ``osgeo`` is preferred ,
`GDAL <https://opensourceoptions.com/how-to-install-gdal/>`_ needs to be installed and configure. 
 
.. automodule:: gofast.geo
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`geosciences <geo>` section for further details.


Classes
~~~~~~~~~~~~

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: class.rst

   geo.Profile
   geo.Location
   
Functions
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   geo.make_coords 
   geo.refine_locations 
   geo.get_azimuth
   geo.get_bearing
   geo.get_stratum_thickness
   geo.smart_thickness_ranker 
   geo.build_random_thickness 
   geo.plot_stratalog
   geo.select_base_stratum 
   geo.get_aquifer_section 
   geo.get_aquifer_sections
   geo.get_unique_section
   geo.get_compressed_vector 
   geo.get_hole_partitions
   geo.reduce_samples
   geo.get_sections_from_depth
   geo.make_mxs_labels
   geo.predict_nga_labels
   geo.find_aquifer_groups
   geo.find_similar_labels
   geo.classify_k 
   geo.label_importance


gofast.preprocessing
=====================

This module provides a set of common utilities for preprocessing data.

.. automodule:: gofast.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:


gofast.models
=============

The `gofast.models` module contains various machine learning models optimized for speed and efficiency.

.. automodule:: gofast.models
   :members:
   :undoc-members:
   :show-inheritance:


gofast.metrics
==============

`gofast.metrics` includes performance metrics for evaluating models.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.precision_recall_tradeoff
   metrics.roc_curve_
   metrics.confusion_matrix_ 
   metrics.get_eval_scores
   metrics.mean_squared_log_error
   metrics.balanced_accuracy
   metrics.information_value
   metrics.mean_absolute_error
   metrics.mean_squared_error 
   metrics.root_mean_squared_error
   metrics.r_squared
   metrics.mean_absolute_percentage_error 
   metrics.explained_variance_score
   metrics.median_absolute_error
   metrics.max_error
   metrics.mean_squared_log_error
   metrics.mean_poisson_deviance
   metrics.mean_gamma_deviance
   metrics.mean_absolute_deviation
   metrics.dice_similarity_coeff
   metrics.gini_coeff
   metrics.hamming_loss
   metrics.fowlkes_mallows_index
   metrics.rmse_log_error
   metrics.mean_percentage_error
   metrics.percentage_bias
   metrics.spearmans_rank_correlation
   metrics.precision_at_k 
   metrics.ndcg_at_k 
   metrics.mean_reciprocal_rank
   metrics.average_precision
   metrics.jaccard_similarity_coeff

gofast.utils
============

Utility functions supporting various operations within :code:`GoFast`.

.. automodule:: gofast.utils
   :members:
   :undoc-members:
   :show-inheritance:

