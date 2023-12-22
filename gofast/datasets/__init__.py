"""
Dataset subpackage is used to fetch data from the local machine. 
If the data does not exist or deleted, the remote searching 
(repository or zenodo record ) triggers via the module 
:mod:`~gofast.datasets.rload`
"""
from .sets import ( 
    load_bagoue, 
    load_iris, 
    load_hlogs,
    load_nlogs, 
    load_mxs, 
    fetch_data, 
    DATASET
    )
from ._create import ( 
    make_elogging, 
    make_erp, 
    make_ert, 
    make_gadget_sales, 
    make_medical_diagnostic, 
    make_mining, 
    make_retail_store, 
    make_tem, 
    make_well_logging, 
    make_sounding, 
    make_african_demo, 
    make_agronomy,
    make_cc_factors, 
    make_water_demand, 
    )
__all__=[ 
         "load_bagoue",
         "load_iris",
         "load_hlogs",
         "load_nlogs", 
         "load_mxs", 
         "fetch_data",
         "make_elogging", 
         "make_erp", 
         "make_ert", 
         "make_gadget_sales", 
         "make_medical_diagnostic", 
         "make_mining", 
         "make_retail_store", 
         "make_tem", 
         "make_well_logging", 
         "make_sounding", 
         "make_african_demo", 
         "make_cc_factors",
         "make_agronomy",
         "make_water_demand", 
         "DATASET", 
         ]