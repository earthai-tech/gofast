# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""List of `gofast` exceptions for warning users."""

class ArgumentError(Exception):
    """
    Exception raised for errors in the passed argument values.

    This exception is deprecated and should be replaced by `TypeError` or
    `ValueError` in the next release.
    """
    pass

class SiteError(Exception):
    """
    Exception raised for errors related to site or location information.

    For example, this exception is raised when coordinates values provided are
    inappropriate.
    """
    pass

class DatasetError(Exception):
    """
    Exception raised for inconsistencies in the dataset provided.

    This exception is raised when multiple datasets passed as arguments have
    mismatching shapes, columns, or sizes, or when items in the data are not
    valid according to predefined criteria.
    """
    pass

class HeaderError(Exception):
    """
    Exception raised when headers or required columns are missing in the data.
    """
    pass

class ConfigError(Exception):
    """
    Exception raised for errors in configuration setup or execution.
    """
    pass

class FileHandlingError(Exception):
    """
    Exception raised for errors encountered during file manipulation.

    This exception occurs if there are file permission issues or other problems
    encountered when opening, reading, or writing files.
    """
    pass

class TipError(Exception):
    """
    Exception raised for inappropriate tips proposed for plot visualization
    shortcuts.
    """
    pass

class PlotError(Exception):
    """
    Exception raised when a plot cannot be generated successfully.
    """
    pass

class ParameterNumberError(Exception):
    """
    Exception raised when the number of parameters provided is incorrect.

    This exception is raised when the parameters given do not match the expected
    count for proper computation.
    """
    pass

class ProcessingError(Exception):
    """
    Exception raised for failures in the data processing pipeline.
    """
    pass

class ProfileError(Exception):
    """
    Exception raised for mismatches in arguments passed to a profile object.
    """
    pass

class FeatureError(Exception):
    """
    Exception raised for errors in feature processing or handling.
    """
    pass

class EstimatorError(Exception):
    """
    Exception raised when an incorrect estimator or assessor is provided.
    """
    pass

class GeoPropertyError(Exception):
    """
    Exception raised when there is an attempt to externally modify geological
    property objects.
    """
    pass

class GeoArgumentError(Exception):
    """
    Exception raised for inappropriate arguments passed to geology modules.
    """
    pass

class HintError(Exception):
    """
    Exception raised for inappropriate hints proposed for processing shortcuts.
    """
    pass

class SQLError(Exception):
    """
    Exception raised for errors in SQL queries or database interactions.
    """
    pass

class StrataError(Exception):
    """
    Exception raised for incorrect stratum values or missing 'sname' in
    hydro-logging datasets, where 'sname' is the column name for strata.
    """
    pass

class SQLManagerError(Exception):
    """
    Exception raised for failures in SQL request transfers or executions.
    """
    pass

class GeoDatabaseError(Exception):
    """
    Exception raised when a geospatial database fails to respond or process
    requests.
    """
    pass

class ModelError(Exception):
    """
    Exception raised for errors in geospatial or other model constructions.
    """
    pass

class ERPError(Exception):
    """
    Exception raised for invalid electrical resistivity profiling data.

    'station' and 'resistivity' columns must be present in the ERP dataset.
    """
    pass

class ExtractionError(Exception):
    """
    Exception raised for failures in data extraction from path-like objects or
    file formats like JSON or YAML.
    """
    pass

class CoordinateError(Exception):
    """
    Exception raised for issues with coordinate values or computations.
    """
    pass

class TopModuleError(Exception):
    """
    Exception raised when a key dependency package fails to install or load.

    'scikit-learn' is an example of a key dependency for the `gofast` package.
    """
    pass

class NotFittedError(Exception):
    """
    Exception raised when a 'fit' method is called on an unfitted object.

    This is a common exception for classes in `gofast` that implement a 'fit'
    method for parameter initialization.
    """
    pass

class ScikitLearnImportError(Exception):
    """
    Exception raised when importing scikit-learn fails.

    Refer to `gofast` documentation for more information on scikit-learn
    dependencies.
    """
    pass

class GISError(Exception):
    """
    Exception raised for failures in GIS parameter calculations.
    """
    pass


class LearningError(Exception): 
    """
    Raises an Exception for issues during the learning inspection phase 
    of training.
    """
    pass 

class kError (Exception):
    """
    Raises an exception if the permeability coefficient array is missing 
    or if 'kname' is not specified for permeability in Hydro-log data.
    """
    pass

class DepthError (Exception):
    """
    Raises an exception for depth line issues in a multidimensional array. 
    Depth should be one-dimensional and labeled 'z' in pandas dataframes 
    or series.
    """
    pass 

class AquiferGroupError (Exception):
    """
    Raises an exception for issues with aquifer data, which should be 
    one-dimensional and categorical, representing layer/rock names.
    """
    pass 
