"""
Utils sub-package offers several tools for data handling, parameters computation 
models estimation and evalution, and graphs visualization. The extension of the 
mathematical concepts, and the core of program are performed via the modules 
:mod:`~gofast.utils.exmath`. Whereas the machine learning utilities and 
additional functionalities are performed with :mod:`~gofast.utils.mlutils` and 
:mod:`~gofast.utils.funcutils` respectively. 
"""

from .baseutils import ( 
    read_data,
    move_file_to_directory, 
    get_remote_data, 
    array2hdf5, 
    save_or_load, 
    request_data, 
    download_file2, 
    )
from .mathex import ( 
    interpolate1d, 
    interpolate2d,
    scaley, 
    get_bearing, 
    moving_average, 
    linkage_matrix, 
    get_distance,
    smooth1d, 
    smoothing, 
    qc, 
    adaptive_moving_average, 
    )
from .funcutils import ( 
    reshape, 
    to_numeric_dtypes, 
    smart_label_classifier, 
    remove_outliers,
    normalizer, 
    cleaner, 
    savejob, 
    random_selector, 
    interpolate_grid, 
    twinning, 
    random_sampling, 
    replace_data, 
    storeOrwritehdf5
    )

from .mlutils import ( 
    selectfeatures, 
    getGlobalScore, 
    split_train_test, 
    correlatedfeatures, 
    findCatandNumFeatures,
    evalModel, 
    cattarget, 
    labels_validator, 
    projection_validator, 
    rename_labels_in , 
    naive_imputer, 
    naive_scaler, 
    select_feature_importances, 
    make_naive_pipe, 
    bi_selector, 
    get_target, 
    resampling, 
    bin_counting, 
    )

__all__=[
        'read_data',
        'move_file_to_directory', 
        'array2hdf5', 
        'save_or_load', 
        'request_data', 
        'get_remote_data', 
        'download_file2', 
        'interpolate1d', 
        'interpolate2d',
        'scaley', 
        'selectfeatures', 
        'getGlobalScore',  
        'split_train_test', 
        'correlatedfeatures', 
        'findCatandNumFeatures',
        'evalModel',
        'moving_average', 
        'linkage_matrix',
        'reshape', 
        'to_numeric_dtypes' , 
        'smart_label_classifier', 
        'cattarget', 
        'labels_validator', 
        'projection_validator', 
        'rename_labels_in', 
        'read_data', 
        'naive_imputer', 
        'naive_scaler', 
        'select_feature_importances',
        'make_naive_pipe',
        'bi_selector',
        'classify_k',
        'label_importance', 
        'remove_outliers', 
        'normalizer',
        'get_target', 
        'get_distance',
        'get_bearing', 
        'qc', 
        'cleaner', 
        'savejob', 
        'random_selector', 
        'interpolate_grid',
        'smooth1d', 
        'smoothing', 
        'twinning', 
        'random_sampling', 
        'plot_voronoi', 
        'plot_roc_curves', 
        'replace_data', 
        'storeOrwritehdf5', 
        "resampling", 
        "bin_counting",
        "adaptive_moving_average", 
        "butterworth_filter",
        "plot_l_curve"
        ]



