
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


gofast.datasets
===============

The `gofast.datasets` module includes utilities for loading and fetching popular datasets used in machine learning.
 
.. automodule:: gofast.analysis
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`analysis <analysis>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
	analysis.LLE
	analysis.pcavsfa
	analysis.compute_scores
	analysis.decision_region
	analysis.extract_pca
	analysis.feature_transformation
	analysis.find_features_importances
	analysis.get_component_with_most_variance
	analysis.iPCA
	analysis.kPCA
	analysis.linear_discriminant_analysis
	analysis.LW_score
	analysis.make_scedastic_data
	analysis.nPCA
	analysis.plot_projection
	analysis.shrunk_cov_score
	analysis.total_variance_ratio   

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

.. automodule:: gofast.metrics
   :members:
   :undoc-members:
   :show-inheritance:


gofast.utils
============

Utility functions supporting various operations within :code:`GoFast`.

.. automodule:: gofast.utils
   :members:
   :undoc-members:
   :show-inheritance:

