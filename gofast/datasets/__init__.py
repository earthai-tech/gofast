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
__all__=[ 
         "load_bagoue",
         "load_iris",
         "load_hlogs",
         "load_nlogs", 
         "load_mxs", 
         "fetch_data",
         "DATASET"
         ]