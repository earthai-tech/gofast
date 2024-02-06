"""
Gofast Dataset Subpackage
=========================

This subpackage is designed to facilitate data fetching from the local machine
for various datasets. It also includes utilities to generate synthetic datasets
for different use cases. For remote data loading capabilities, refer to
:mod:`~gofast.datasets.rload`.

Available Functions
-------------------
- Data Loading: Functions to load various real-world datasets.
- Data Generation: Functions to create synthetic datasets for testing and development.

"""

# Data loading functions
from ._data_loader import (
    load_bagoue, load_iris, load_hlogs, load_nlogs, load_mxs,
    fetch_data, load_forensic, load_jrs_bet, load_dyspnea, 
)

# Data generation functions
from ._create import (
    make_elogging, make_erp, make_ert, make_gadget_sales,
    make_medical_diagnosis, make_mining_ops, make_retail_store,
    make_tem, make_well_logging, make_sounding, make_african_demo,
    make_agronomy_feedback, make_social_media_comments, make_cc_factors,
    make_water_demand, make_regression, make_classification
)

__all__ = [
    "load_bagoue", "load_iris", "load_hlogs", "load_nlogs", "load_forensic",
    "load_jrs_bet", "load_dyspnea", "load_mxs", "fetch_data", "make_elogging", 
    "make_erp","make_ert", "make_gadget_sales", "make_medical_diagnosis", 
    "make_mining_ops","make_retail_store", "make_tem", "make_well_logging", 
    "make_sounding","make_african_demo", "make_cc_factors", "make_agronomy_feedback",
    "make_social_media_comments", "make_water_demand", "make_regression",
    "make_classification"
]
