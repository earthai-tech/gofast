
"""
Analysis sub-package is used for basic feature extraction, transformation and 
matrices covariance computations (:mod:`~gofast.analysis.decomposition`). 
It also includes some dimensional reduction (:mod:`~gofast.analysis.dimensionality`) 
and factor analysis from :mod:`~gofast.analysis.factors`. 
"""
from .dimensionality import (   
    nPCA, kPCA, LLE, iPCA, 
    get_most_variance_component,
    project_ndim_vs_explained_variance, 
    )
from .decomposition import ( 
    get_eigen_components, 
    plot_decision_regions, 
    transform_to_principal_components, 
    get_total_variance_ratio , 
    linear_discriminant_analysis
    )
from .factors import ( 
    ledoit_wolf_score, 
    compare_pca_fa_scores, 
    make_scedastic_data, 
    rotated_factor, 
    principal_axis_factoring, 
    varimax_rotation, 
    oblimin_rotation, 
    get_pca_fa_scores, 
    samples_hotellings_t_square, 
    promax_rotation, 
    spectral_fa, 
   )

__all__= [ 
    "nPCA", "kPCA", "LLE", "iPCA", 
    "get_most_variance_component",
    "project_ndim_vs_explained_variance", 
    "get_eigen_components", 
    "plot_decision_regions", 
    "transform_to_principal_components", 
    "get_total_variance_ratio" , 
    "linear_discriminant_analysis", 
    "ledoit_wolf_score",  
    "compare_pca_fa_scores", 
    "make_scedastic_data", 
    "rotated_factor", 
    "principal_axis_factoring", 
    "varimax_rotation", 
    "oblimin_rotation", 
    "get_pca_fa_scores", 
    "samples_hotellings_t_square", 
    "promax_rotation", 
    "spectral_fa", 
    ]

