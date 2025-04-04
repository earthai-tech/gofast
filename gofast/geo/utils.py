# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Set of utilities that deal with geological rocks, strata and 
stratigraphic details for log construction. 
"""
from __future__ import annotations 
import copy 
import warnings

import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from ..api.types import List, Tuple,  Any, Optional, Union 
from ..api.types import  _F, ArrayLike, DataFrame 
from ..exceptions import  DepthError 
from ..api.property import GeoscienceProperties 
from ..core.checks import find_closest 
from ..core.utils import ellipsis2false  
from ..utils.mathext import convert_value_in 
from ..utils.validator import  assert_xy_in  
from .gis_utils import utm_to_ll, project_point_utm2ll # HAS_GDAL 

from .._gofastlog import gofastlog 
_logger = gofastlog().get_gofast_logger(__name__ )

__all__=["correct_data_location", "make_coords", "compute_azimuth", 
         "calculate_bearing","enriched_landslide_samples" ]


def enriched_landslide_samples(
    landslide_events: Union[List[dict], pd.DataFrame],
    station_data: Union[List[dict], pd.DataFrame],
    n_stations_per_event: int = 3,
    rainfall_days: int = 5,
    base_name: Optional[str] = None,
    station_id_name: str = 'station_id',
    rainfall_col: str = 'rainfall',
    seed: Optional[int] = None,
    verbose: int = 0
) -> pd.DataFrame:
    """Augments landslide event data with rainfall from multiple nearby stations.

    This function creates an "enriched" dataset suitable for machine
    learning by simulating multiple perspectives on each landslide
    event. For every landslide event provided, it randomly selects a
    specified number (`n_stations_per_event`) of rainfall stations
    from the `station_data`. It then generates a new data sample (row)
    for each unique pairing of a landslide event and one of its
    selected stations. This new sample combines the original event's
    features with the rainfall sequence from the paired station. The
    rainfall sequence is unpacked into distinct columns, one for each
    day specified by `rainfall_days`.

    This process effectively multiplies the number of samples, allowing
    models to learn from different potential rainfall scenarios associated
    with each landslide. The selection of stations is performed randomly
    without replacement using ``pandas.DataFrame.sample``, seeded by the
    `seed` parameter for reproducibility.

    Parameters
    ----------
    landslide_events : list of dict or pd.DataFrame
        A collection of landslide events. Each event must contain at
        least an identifier, location, and date, though other features
        are preserved. If a list of dictionaries, each dictionary
        represents one event. If a DataFrame, each row represents an
        event. Mandatory conceptual fields: 'id', 'lat', 'lon', 'date'.

    station_data : list of dict or pd.DataFrame
        A collection of rainfall station records. Each record must
        include a unique station identifier (specified by
        `station_id_name`) and the associated rainfall data (specified
        by `rainfall_col`). The rainfall data must be a list or NumPy
        array of numerical values with a length equal to or greater
        than `rainfall_days`.

    n_stations_per_event : int, default=3
        The target number of distinct rainfall stations to associate
        with *each* landslide event. The function will randomly select
        up to this many stations from `station_data` for each event,
        without replacement. If `station_data` contains fewer stations
        than this number, all available stations will be used for each
        event.

    rainfall_days : int, default=5
        The number of consecutive rainfall days to extract from each
        selected station's rainfall sequence (`rainfall_col`). The
        rainfall sequence in `station_data` must contain at least this
        many values. These values will be placed into new columns in
        the output DataFrame.

    base_name : str, optional
        A prefix used to name the new rainfall columns generated in the
        output DataFrame. The columns will be named following the
        pattern: ``f"{base_name}_day_{i}"`` (where `i` ranges from 1
        to `rainfall_days`). If ``None`` (the default), the value of
        the `rainfall_col` parameter is used as the base name. For
        example, if `base_name` is ``'precip'`` and `rainfall_days`
        is ``3``, the new columns will be ``'precip_day_1'``,
        ``'precip_day_2'``, ``'precip_day_3'``.

    station_id_name : str, default='station_id'
        The exact column name in `station_data` that holds the unique
        identifier for each rainfall station. This identifier will be
        added to the enriched samples. Example: ``'station_code'``.

    rainfall_col : str, default='rainfall'
        The exact column name in `station_data` that contains the list
        or array of rainfall measurements for each station. The length
        of this list/array must be >= `rainfall_days`. Example: ``'rain'``.

    seed : int, optional
        A seed for the random number generator (used by ``np.random.seed``
        and subsequently by ``pd.DataFrame.sample``). Providing an
        integer ensures that the random selection of stations is
        reproducible across runs. If ``None``, the selection will be
        non-deterministic.

    verbose : int, default=0
        Controls the level of messages printed during execution:
        - ``0``: Silent operation. No messages are printed.
        - ``1``: Prints a summary message indicating the total number
          of enriched samples generated.
        - ``2`` or higher: Prints the summary message (from level 1)
          and also displays the ``.head()`` of the newly created
          rainfall columns and the station ID column in the resulting
          DataFrame for quick inspection.

    Returns
    -------
    pd.DataFrame
        An enriched DataFrame where each original landslide event is
        potentially represented multiple times (up to
        `n_stations_per_event` times), each time paired with rainfall
        data from a different, randomly selected station. The DataFrame
        includes all original columns from `landslide_events`, the
        `station_id_name` column, and `rainfall_days` new columns
        containing the unpacked rainfall data.

    Raises
    ------
    ValueError
        - If `station_data` does not contain the required columns
          specified by `station_id_name` or `rainfall_col`.
        - If the rainfall data found in `station_data` under
          `rainfall_col` is not a list or NumPy array for any station.
        - If the rainfall sequence for any selected station has fewer
          than `rainfall_days` elements.
    TypeError
        - If `landslide_events` or `station_data` are not list/dict or
          DataFrame formats. (Implicitly raised by pandas if conversion fails).

    See Also
    --------
    pandas.DataFrame.sample : Method used for random station selection.
    numpy.random.seed : Used to control reproducibility.
    # (Potentially add links to functions that find nearby stations if available)
    # e.g., find_nearest_stations, calculate_distance

    Notes
    -----
    - The selection of stations for each event is random and performed
      *without* considering the geographical proximity between the event
      and the stations unless the `station_data` provided has already
      been pre-filtered based on proximity.
    - The function assumes the rainfall data in `station_data` is aligned
      correctly (e.g., the list represents rainfall for the relevant
      days leading up to potential events). No date matching is performed
      internally between `landslide_events` 'date' and the rainfall data.
    - The total number of rows in the output DataFrame will be exactly
      `len(landslide_events) * min(n_stations_per_event, len(station_data))`.
    
    Let :math:`E` be the set of input landslide events, and :math:`S` be
    the set of input rainfall stations. Each event :math:`e \in E` has
    attributes :math:`A_e`, and each station :math:`s \in S` has an ID
    :math:`id_s` (from `station_id_name`) and a rainfall sequence
    :math:`R_s` of length `rainfall_days` (from `rainfall_col`).

    For each event :math:`e \in E`:
      1. Randomly select a subset of stations :math:`S_e \subseteq S`
         such that :math:`|S_e| = k = \min(\text{n\_stations\_per\_event}, |S|)`.
         This selection is done without replacement.
         .. math::
             S_e = \text{RandomSample}(S, k, \\text{seed}=\\text{seed})

      2. For each station :math:`s \in S_e`:
         a. Create a new enriched sample :math:`d_{e,s}`.
         b. Copy attributes from event :math:`e`: :math:`d_{e,s}[a] = A_e[a]`
            for all attributes :math:`a` of :math:`e`.
         c. Add station ID: :math:`d_{e,s}[\text{station\_id\_name}] = id_s`.
         d. Unpack rainfall sequence :math:`R_s = [r_1, r_2, ..., r_m]`
            (where :math:`m=\text{rainfall\_days}`):
            .. math::
                d_{e,s}[\text{colname}_i] = r_i \quad \text{for } i = 1, \dots, m
            Where :math:`\text{colname}_i = \text{f"{base\_name}\_day\_{i}"}`.
            If `base_name` is ``None``, it defaults to the value of
            `rainfall_col`.

    The final output is a DataFrame containing all generated samples:
    :math:`D_{enriched} = \{ d_{e,s} \mid e \in E, s \in S_e \}`.
    The total number of rows will be :math:`|E| \times k`.

    The function internally uses ``pandas.DataFrame`` for data handling,
    ``.iterrows()`` for looping, and ``.sample()`` for station selection.


    References
    ----------
    .. [1] Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python",
           Journal of Machine Learning Research, 12, pp. 2825-2830, 2011.
           (While not directly used, scikit-learn principles often guide
           such data preparation techniques).

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.geo.utils import enriched_landslide_samples # Use actual path

    >>> # Define sample landslide events
    >>> events = [
    ...     {"id": "LS-001", "latitude": 24.8, "longitude": 113.5, "date": "2022-06-15"},
    ...     {"id": "LS-002", "latitude": 24.9, "longitude": 113.7, "date": "2022-07-20"}
    ... ]

    >>> # Define sample station data with rainfall lists
    >>> stations = [
    ...     {"station_code": "S1", "rain": [20, 30, 25, 10, 5]}, # 5 days data
    ...     {"station_code": "S2", "rain": [15, 18, 22, 16, 12]}, # 5 days data
    ...     {"station_code": "S3", "rain": [30, 35, 40, 20, 15]}  # 5 days data
    ... ]

    >>> # Enrich the landslide events
    >>> enriched_df = enriched_landslide_samples(
    ...     landslide_events=events,
    ...     station_data=stations,
    ...     n_stations_per_event=2,   # Link each event to 2 random stations
    ...     rainfall_days=5,          # Use 5 days of rainfall
    ...     base_name='precip',       # Name new columns 'precip_day_X'
    ...     station_id_name='station_code', # Station ID column is 'station_code'
    ...     rainfall_col='rain',      # Rainfall list column is 'rain'
    ...     seed=42,                  # For reproducible random selection
    ...     verbose=1                 # Print summary info
    ... )
    [INFO] Generated 4 enriched samples.

    >>> print("\\nEnriched DataFrame:")
    >>> print(enriched_df)

    Enriched DataFrame:
         id  latitude  longitude        date station_code  precip_day_1  \
    0  LS-001      24.8      113.5  2022-06-15           S3            30 
    1  LS-001      24.8      113.5  2022-06-15           S2            15 
    2  LS-002      24.9      113.7  2022-07-20           S2            15
    3  LS-002      24.9      113.7  2022-07-20           S1            20

         precip_day_2  precip_day_3  precip_day_4  precip_day_5
    0              35            40            20            15
    1              18            22            16            12
    2              18            22            16            12
    3              30            25            10             5
    
    >>> # Example using default base_name (derived from rainfall_col)
    >>> enriched_df_default = enriched_landslide_samples(
    ...     landslide_events=events[0], # Single event
    ...     station_data=stations,
    ...     n_stations_per_event=3,
    ...     rainfall_days=3,          # Only use 3 days
    ...     station_id_name='station_code',
    ...     rainfall_col='rain',      # Base name will be 'rain'
    ...     seed=123
    ... )
    >>> print("\\nEnriched DataFrame (Default base_name, 3 days):")
    >>> print(enriched_df_default)
 
    Enriched DataFrame (Default base_name, 3 days):
         id  latitude  longitude        date station_code  \
    0  LS-001      24.8      113.5  2022-06-15           S2          
    1  LS-001      24.8      113.5  2022-06-15           S3          
    2  LS-001      24.8      113.5  2022-06-15           S1          


           rain_day_1  rain_day_2  rain_day_3
    0            15          18          22
    1            30          35          40
    2            20          30          25
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # --- Input Normalization ---
    # Convert list of dicts to DataFrame if necessary
    if isinstance(landslide_events, dict):
        # Handle single dictionary case
        landslide_events = [landslide_events]
    if isinstance(landslide_events, list):
        try:
            landslide_events = pd.DataFrame(landslide_events)
        except Exception as e:
            raise TypeError(
                "Could not convert landslide_events list/dict to DataFrame. "
                f"Original error: {e}"
            ) from e
    elif not isinstance(landslide_events, pd.DataFrame):
        raise TypeError(
            "landslide_events must be a list of dicts or a pandas DataFrame."
            )

    if isinstance(station_data, dict):
        # Handle single dictionary case
        station_data = [station_data]
    if isinstance(station_data, list):
        try:
            station_data = pd.DataFrame(station_data)
        except Exception as e:
            raise TypeError(
                "Could not convert station_data list/dict to DataFrame. "
                f"Original error: {e}"
                ) from e
    elif not isinstance(station_data, pd.DataFrame):
         raise TypeError(
             "station_data must be a list of dicts or a pandas DataFrame."
             )

    # --- Input Validation ---
    # Check for required columns in station_data
    if station_id_name not in station_data.columns:
        raise KeyError(
            f"station_data DataFrame missing required station ID column: "
            f"'{station_id_name}'"
        )
    if rainfall_col not in station_data.columns:
         raise KeyError(
             f"station_data DataFrame missing required rainfall column: "
             f"'{rainfall_col}'"
         )
    if len(station_data) == 0:
        raise ValueError("station_data cannot be empty.")
    if len(landslide_events) == 0:
        raise ValueError("landslide_events cannot be empty.")

    # Determine the base name for new rainfall columns
    # Use rainfall_col if base_name is not provided or is an empty string
    effective_base_name = base_name if base_name else rainfall_col

    # --- Enrichment Process ---
    enriched_samples = [] # List to store the new enriched sample dicts

    # Iterate through each landslide event
    for _, event_row in landslide_events.iterrows():
        # Randomly select 'n_stations_per_event' stations for this event
        # Use min() to handle cases where fewer stations are available
        num_to_sample = min(n_stations_per_event, len(station_data))

        if num_to_sample > 0:
            selected_stations = station_data.sample(
                n=num_to_sample,
                replace=False, # Ensure unique stations per event
                # Use seeded generator if available
                random_state=np.random.default_rng(seed) if seed else None 
            )
        else:
            # Should not happen if len(station_data)>0, but handle defensively
            continue # Skip event if no stations can be sampled

        # Create a new sample for each selected station paired with the event
        for _, station_row in selected_stations.iterrows():
            # Get the rainfall sequence for the current station
            rain_values = station_row[rainfall_col]

            # Validate the rainfall data format and length
            if not isinstance(rain_values, (list, np.ndarray)):
                raise ValueError(
                    f"Rainfall data in column '{rainfall_col}' for station "
                    f"'{station_row[station_id_name]}' must be a list or "
                    f"numpy array. Found: {type(rain_values)}"
                )
            if len(rain_values) < rainfall_days:
                raise ValueError(
                    f"Station '{station_row[station_id_name]}' has only "
                    f"{len(rain_values)} rainfall days recorded in "
                    f"'{rainfall_col}', but {rainfall_days} days are required."
                )

            # Create the base sample dictionary from the event data
            # Using .to_dict() preserves original event features
            sample = event_row.to_dict()

            # Add the station identifier to the sample
            sample[station_id_name] = station_row[station_id_name]

            # Unpack the rainfall values into separate columns
            for i in range(rainfall_days):
                colname = f"{effective_base_name}_day_{i+1}"
                # Access the i-th rainfall value (0-indexed)
                sample[colname] = rain_values[i]

            # Add the completed enriched sample to our list
            enriched_samples.append(sample)

    # Convert the list of enriched sample dictionaries into a DataFrame
    if not enriched_samples:
        # Return an empty DataFrame with expected columns if no samples generated
        # Construct expected columns dynamically
        if verbose : 
            print(
                 "Warning: No enriched samples were generated."
                 " Returning empty DataFrame."
                )
        example_event_cols = list(landslide_events.columns)
        rainfall_cols = [
            f"{effective_base_name}_day_{i+1}" for i in range(rainfall_days)
            ]
        all_expected_cols = example_event_cols + [station_id_name] + rainfall_cols
        # Remove duplicates if station_id_name was already in event columns
        all_expected_cols = sorted(
            list(set(all_expected_cols)), key=all_expected_cols.index)
        
        return pd.DataFrame(columns=all_expected_cols)

    df_enriched = pd.DataFrame(enriched_samples)

    # --- Verbose Output ---
    if verbose >= 1:
        print(f"[INFO] Generated {len(df_enriched)} enriched samples from "
              f"{len(landslide_events)} events and {len(station_data)} stations.")
    if verbose >= 2:
        # Construct the list of new columns for concise head display
        new_cols_to_show = [station_id_name] + [
            f"{effective_base_name}_day_{i+1}" for i in range(rainfall_days)
        ]
        # Ensure columns exist before trying to display them
        cols_present = [col for col in new_cols_to_show if col in df_enriched.columns]
        if cols_present:
             print("[INFO] Head of new/key columns in the enriched DataFrame:")
             print(df_enriched[cols_present].head())
        else:
             print("[INFO] No new/key columns generated to display.")


    return df_enriched

def make_coords(
    reflong: Union[str, Tuple[float, float]],
    reflat: Union[str, Tuple[float, float]],
    nsites: int,
    *,
    r: float = 45.0,
    utm_zone: Optional[str] = None,
    step: Union[str, float] = '1km',
    order: str = '+',
    todms: bool = False,
    is_utm: bool = False,
    raise_warning: bool = True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a sequence of geographic coordinates (longitude and latitude). 
    
    Coordinates are generated by starting from a reference point, extending 
    over a specified number of sites with a defined step and direction.

    Parameters
    ----------
    reflong : Union[str, Tuple[float, float]]
        Reference longitude as a decimal degree or DMS string (DD:MM:SS), or a
        tuple indicating start and end longitudes for the coordinate generation.
    reflat : Union[str, Tuple[float, float]]
        Reference latitude as a decimal degree or DMS string (DD:MM:SS), or a
        tuple indicating start and end latitudes for the coordinate generation.
    nsites : int
        The number of sites (coordinates) to generate.
    r : float, optional
        Rotation angle in degrees from the north, used to define the direction
        of the generated line of sites. Default is 45 degrees.
    utm_zone : Optional[str], optional
        Specifies the UTM zone for conversion if coordinates are given in UTM format.
        Must be provided if is_utm is True.
    step : Union[str, float], optional
        Step size between sites, in meters ('m') or kilometers ('km').
        Default is '1km'.
    order : str, optional
        Order of the generated coordinates. '+' for ascending (default), 
        '-' for descending.
    todms : bool, optional
        If True, converts generated coordinates from decimal degrees to DMS 
        format. Default is False.
    is_utm : bool, optional
        If True, treats reflong and reflat as UTM coordinates (easting and northing).
        Default is False.
    raise_warning : bool, optional
        If True, raises a warning when necessary (
            e.g., GDAL not installed or improper coordinate system). Default is True.
    kwargs : dict
        Additional keyword arguments for internal functions 
        (e.g., conversion functions).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two numpy arrays containing the generated longitudes and latitudes,
        respectively, either in decimal degrees or DMS format.

    Examples
    --------
    Generate coordinates in decimal degrees with a step of 1km, 45-degree angle:
    >>> reflong, reflat = "110.485", "26.051"
    >>> nsites = 5
    >>> rlons, rlats = make_coords(reflong, reflat, nsites)
    >>> print(rlons)
    >>> print(rlats)

    Generate coordinates in DMS format with a step of 2km, 90-degree angle:
    >>> reflong, reflat = ("110:29:09", "26:03:05")
    >>> nsites = 3
    >>> rlons, rlats = make_coords(reflong, reflat, nsites, step='2km', r=90, todms=True)
    >>> print(rlons)
    >>> print(rlats)
    """
    step = _prepare_step(step)
    x0, y0 = (_convert_to_float(reflong), _convert_to_float(reflat)) if not is_utm else (reflong, reflat)
    if isinstance(reflong, Tuple):
        x0 = reflong[0]
    if isinstance(reflat, Tuple):
        y0 = reflat[0]

    x_end, y_end = _compute_endpoints(x0, y0, r, step, nsites)
    reflon_ar, reflat_ar = _generate_coordinates(x0, y0, x_end, y_end, nsites, order)

    # UTM conversion, warnings, and other processing can be added here
    if is_utm:
        reflon_ar, reflat_ar = _convert_utm_to_latlon(
            reflon_ar, reflat_ar, utm_zone, raise_warning, **kwargs)

    if todms:
        reflon_ar = np.array([_convert_dd_to_dms(lon) for lon in reflon_ar])
        reflat_ar = np.array([_convert_dd_to_dms(lat) for lat in reflat_ar])

    return reflon_ar, reflat_ar


def _convert_utm_to_latlon(
        reflon_ar: np.ndarray, reflat_ar: np.ndarray, utm_zone: str,
        raise_warning: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts UTM coordinates to latitude and longitude.

    Parameters
    ----------
    reflon_ar : np.ndarray
        Array of UTM easting values.
    reflat_ar : np.ndarray
        Array of UTM northing values.
    utm_zone : str
        UTM zone for the coordinates, e.g., '10S' or '03N'.
    raise_warning : bool, optional
        If True, issues warnings about potential inaccuracies without GDAL.
    kwargs : dict
        Additional keyword arguments for the conversion function.

    Returns
    -------
    lat_ar : np.ndarray
        Array of latitude values in decimal degrees.
    lon_ar : np.ndarray
        Array of longitude values in decimal degrees.
    """
    if utm_zone is None:
        raise TypeError("Please provide your UTM zone e.g., '10S' or '03N'!")

    lat_ar = np.zeros_like(reflon_ar)
    lon_ar = np.zeros_like(reflat_ar)

    for kk, (lo, la) in enumerate(zip(reflon_ar, reflat_ar)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lat_ar[kk], lon_ar[kk] = project_point_utm2ll(
                    easting=la, northing=lo, utm_zone=utm_zone, **kwargs)
        except Exception as e:
            # Fallback or error handling here
            try:lat_ar[kk], lon_ar[kk] = utm_to_ll(
                23, northing=lo, easting=la, zone=utm_zone)
            except: raise ValueError(f"Error converting UTM to lat/lon: {e}")

    if raise_warning:
        warnings.warn("Conversion from UTM to lat/lon performed. Accuracy"
                      " depends on the underlying conversion method.")

    return lat_ar, lon_ar

def _convert_to_float(coord: str) -> float:
    """Converts coordinate string to float."""
    try:
        return float(coord)
    except ValueError:
        if ':' not in coord:
            raise ValueError(f'Could not convert value to float: {coord!r}')
        else:
            return _convert_dms_to_dd(coord)

def _convert_dms_to_dd(dms: str) -> float:
    """Converts coordinates from DMS (Degrees:Minutes:Seconds) to Decimal Degrees."""
    degrees, minutes, seconds = [float(part) for part in dms.split(':')]
    return degrees + minutes / 60 + seconds / 3600

def _convert_dd_to_dms(dd: float) -> str:
    """
    Converts coordinates from Decimal Degrees (DD) to Degrees, Minutes,
    Seconds (DMS) format.

    Parameters
    ----------
    dd : float
        The coordinate in decimal degrees to be converted.

    Returns
    -------
    str
        The coordinate in DMS format, formatted as "D:M:S".

    Example
    -------
    >>> _convert_dd_to_dms(110.4875)
    '110:29:15'
    """
    # Separate the decimal degrees into degrees, minutes, and remaining decimal minutes
    degrees = int(dd)
    minutes_decimal = abs(dd - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = int((minutes_decimal - minutes) * 60)

    # Format the DMS string and return
    dms = f"{degrees}:{minutes:02d}:{seconds:02d}"
    return dms

def _prepare_step(step: Union[str, float]) -> float:
    """Prepares the step value, converting it to meters if necessary."""
    step = str(step).lower()
    if 'km' in step:  # Convert km to meters
        return float(step.replace('km', '')) * 1000
    elif 'm' in step:  # Assume meters if 'm' is specified
        return float(step.replace('m', ''))
    return float(step)

def _compute_endpoints(x0: float, y0: float, r: float, step: float, nsites: int
                       ) -> Tuple[float, float]:
    """Computes the endpoint coordinates based on the starting point, rotation 
    angle, step, and number of sites."""
    x_end = x0 + (np.sin(np.deg2rad(r)) * step * nsites) / (364000 * 0.3048)
    y_end = y0 + (np.cos(np.deg2rad(r)) * step * nsites) / (288200 * 0.3048)
    return x_end, y_end

def _generate_coordinates(x0: float, y0: float, x_end: float, y_end: float, 
                          nsites: int, order: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generates linearly spaced coordinates between start and end points."""
    reflon_ar = np.linspace(x0, x_end, nsites)
    reflat_ar = np.linspace(y0, y_end, nsites)
    if order == '-':
        return reflon_ar[::-1], reflat_ar[::-1]
    return reflon_ar, reflat_ar


def correct_data_location(
    ydata: ArrayLike,
    xdata: Optional[ArrayLike] = None,
    func: Optional[_F[[Any], Any]] = None,
    column_selector: Optional[int|str] = 0,
    show: bool = False,
    return_fit_values: bool = False,
    **kwargs
) -> Tuple[ArrayLike, ArrayLike] | Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Corrects the location or position of data based on a given model function
    and returns the corrected data along with model parameters and optionally
    the covariance of these parameters.

    Parameters
    ----------
    ydata : ArrayLike
        The dependent data, a length M array - nominally `f(xdata, ...)`.
    xdata : ArrayLike, optional
        The independent variable where the data is measured. It should usually
        be an M-length sequence or an (k, M)-shaped array for functions with
        k predictors, but can be any object. If None, `xdata` is generated
        using the length of `ydata`.
    func : callable, optional
        The model function, `f(x, ...)`, where `x` is the first argument and
        additional parameters are fitted. The default `func` is a linear
        function, `f(x) = ax + b`. Users are encouraged to define their own
        function for better fitting.
    column_selector : int or str, optional
        Specifies the column to use from `ydata` if it is a DataFrame. Can be
        an integer index or a column name.
    show : bool, optional
        If True, displays a quick visualization of the original and fitted
        data.
    return_fit_values : bool, optional
        If True, returns the optimized and covariance matrix of the fitted 
        parameters in addition to the corrected data and parameters.
    kwargs : dict
        Additional keyword arguments passed to `scipy.optimize.curve_fit`.

    Returns
    -------
    ydata_corrected : np.ndarray
        The corrected dependent data.
    popt : np.ndarray
        Optimal values for the parameters minimizing the squared residuals.
    pcov : np.ndarray, optional
        The estimated covariance of popt, returned if `return_cov` is True.

    Examples
    --------
    >>> from scipy.optimize import curve_fit
    >>> from gofast.geo.utils import correct_data_location
    >>> import pandas as pd
    >>> df = pd.read_excel('data/erp/l10_gbalo.xlsx')
    >>> # Correcting northing coordinates from easting data
    >>> northing_corrected, popt, pcov = correct_data_location(
            ydata=df['northing'],
            xdata=df['easting'],
            show=True
        )
    >>> print(len(df['northing']), len(northing_corrected))
    >>> print(popt)  # Should print the slope and intercept by default
    """

    def default_func(x, a, b):
        """Linear model function: f(x) = ax + b."""
        return a * x + b

    # Set default function if none provided or validate provided function
    if func is None or str(func).lower() in ('none', 'linear'):
        func = default_func
    elif not callable(func):
        raise TypeError(f'`func` argument must be callable, got {type(func).__name__!r}')

    if isinstance(ydata, pd.DataFrame):
        if isinstance(column_selector, str):
            if column_selector not in ydata.columns:
                raise ValueError(f'Column {column_selector!r} not found '
                                 'in DataFrame columns.')
            ydata = ydata[column_selector]
        else:
            ydata = ydata.iloc[:, column_selector]

    ydata = np.asarray(ydata)

    if xdata is None:
        xdata = np.arange(len(ydata))
    else:
        xdata = np.asarray(xdata)

    if len(xdata) != len(ydata):
        raise ValueError("xdata and ydata must have the same length.")

    popt, pcov = curve_fit(func, xdata, ydata, **kwargs)

    ydata_corrected = func(xdata, *popt)

    if show:
        plt.figure(figsize=(10, 6))
        plt.plot(xdata, ydata, 'b-', label='Original Data')
        plt.plot(xdata, ydata_corrected, 'r--',
                 label=f'Fitted: a={popt[0]:.3f}, b={popt[1]:.3f}')
        plt.title('Data Correction Visualization')
        plt.xlabel('X Data')
        plt.ylabel('Y Data')
        plt.legend()
        plt.show()

    return (ydata_corrected, popt, pcov) if return_fit_values else ydata_corrected 

def compute_azimuth(
    xlon: Union[str, ArrayLike], 
    ylat: Union[str, ArrayLike], 
    *, 
    data: Optional[DataFrame] = None, 
    utm_zone: Optional[str] = None, 
    projection: str = 'll', 
    isdeg: bool = True, 
    mode: str = 'soft', 
    extrapolate: bool = ...,
    view: bool = ...
) -> np.ndarray:
    """
    Computes azimuth between consecutive points given longitude and latitude 
    (or easting and northing) coordinates.

    Parameters
    ----------
    xlon : str or ArrayLike
        Longitudes or eastings of the points. If a string is provided, `data` must
        also be provided and should contain this column.
    ylat : str or ArrayLike
        Latitudes or northings of the points. If a string is provided, `data` must
        also be provided and should contain this column.
    data : DataFrame, optional
        DataFrame containing the `xlon` and `ylat` columns if `xlon` and `ylat`
        are provided as string names.
    utm_zone : str, optional
        UTM zone of the coordinates, necessary if projection is 'utm'. Should be 
        in the format '##N' or '##S'.
    projection : str, default 'll'
        Specifies the coordinate system of the input data ('utm' or 'll' for 
        longitude-latitude).
    isdeg : bool, default True
        If True, coordinates are assumed to be in degrees. Set to False if 
        coordinates are in radians.
    mode : str, default 'soft'
        If 'soft', projection type is automatically detected. If 'strict', 
        projection must be explicitly set.
    extrapolate : bool, default False
        If True, extrapolates the first azimuth to match the size of input 
        coordinates array.
    view : bool, default False
        If True, displays a plot of the computed azimuths.

    Returns
    -------
    azim : np.ndarray
        Array of computed azimuths between consecutive points.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.geo.utils import compute_azimuth
    >>> # Example DataFrame
    >>> data = pd.DataFrame({
    ...     'longitude': [30, 31, 32],
    ...     'latitude': [10, 11, 12]
    ... })
    >>> compute_azimuth('longitude', 'latitude', data=data)
    array([45., 45.])

    """
    from .site import Location 
    # Handle ellipsis for extrapolate and view if not explicitly passed
    extrapolate, view = ellipsis2false(extrapolate, view)

    # Validate and extract xlon and ylat from data if necessary
    xlon, ylat = assert_xy_in(xlon, ylat, data=data)

    # Auto-detect projection if mode is 'soft'
    if mode == 'soft' and projection == 'll' and (xlon.max() > 180. or ylat.max() > 90.):
        projection = 'utm'
        if not utm_zone:
            raise ValueError("UTM zone must be specified for UTM projection.")

    # Convert UTM to lat-lon if necessary
    if projection == 'utm':
        if utm_zone is None:
            raise ValueError("utm_zone cannot be None when projection is UTM.")
        ylat, xlon = Location.to_latlon(xlon, ylat, utm_zone=utm_zone)

    # Ensure there are at least two points to calculate azimuth
    if len(xlon) < 2 or len(ylat) < 2:
        raise ValueError("At least two points are required to compute azimuth.")

    # Convert degrees to radians if necessary
    if isdeg:
        xlon = np.deg2rad(xlon)
        ylat = np.deg2rad(ylat)

    # Compute azimuth
    azim = np.arctan2(np.sin(xlon[1:] - xlon[:-1]) * np.cos(ylat[1:]),
                      np.cos(ylat[:-1]) * np.sin(ylat[1:]) - 
                      np.sin(ylat[:-1]) * np.cos(ylat[1:]) * np.cos(xlon[1:] - xlon[:-1]))
    azim = np.rad2deg(azim)
    azim = np.mod(azim, 360)  # Normalize to 0-360 degrees

    if extrapolate:
        # Extrapolate first azimuth
        azim = np.insert(azim, 0, azim[0])

    if view:
        plt.plot(azim, label='Azimuth')
        plt.xlabel('Point Index')
        plt.ylabel('Azimuth (degrees)')
        plt.legend()
        plt.show()

    return azim
 
def calculate_bearing(
    latlon1: Tuple[float, float], 
    latlon2: Tuple[float, float], 
    to_deg: bool = True
    ) -> float:
    """
    Calculates the bearing from one geographic location to another.

    The bearing is the compass direction to go from the starting point to the
    destination. It is defined as the angle measured in degrees clockwise from
    the north direction.

    The formula for calculating the bearing (\(\beta\)) between two points is given by:

    .. math::
        \beta = \arctan2(\sin(\Delta\lambda) \cdot \cos(\phi_2),
                         \cos(\phi_1) \cdot \sin(\phi_2) - \sin(\phi_1) \cdot \cos(\phi_2) \cdot \cos(\Delta\lambda))

    where:
        - \(\phi_1, \lambda_1\) are the latitude and longitude of the first 
          point (in radians),
        - \(\phi_2, \lambda_2\) are the latitude and longitude of the second
          point (in radians),
        - \(\Delta\lambda = \lambda_2 - \lambda_1\).

    Parameters
    ----------
    latlon1 : Tuple[float, float]
        The latitude and longitude of the first point in degrees (lat1, lon1).
    latlon2 : Tuple[float, float]
        The latitude and longitude of the second point in degrees (lat2, lon2).
    to_deg : bool, default True
        If True, converts the bearing result from radians to degrees.

    Returns
    -------
    bearing : float
        The calculated bearing in degrees if `to_deg` is True, otherwise in radians.

    Examples
    --------
    >>> calculate_bearing((28.41196763902007, 109.3328724432221),
                          (28.38756530909265, 109.36931920880758))
    127.26739270447973

    """
    lat1, lon1 = np.deg2rad(latlon1)
    lat2, lon2 = np.deg2rad(latlon2)

    delta_lon = lon2 - lon1

    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

    bearing = np.arctan2(x, y)

    if to_deg:
        bearing = np.degrees(bearing)
        bearing = (bearing + 360) % 360  # Normalize to 0-360 degrees

    return bearing

def find_similar_structures(
        *resistivities,  return_values: bool=...):
    """
    Find similar geological structures from electrical rock properties 
    stored in configure data. 
    
    Parameters 
    ------------
    resistivities: float 
       Array of geological rocks resistivities 
       
    return_values: bool, default=False, 
       return the closest resistivities of the close structures with 
       similar resistivities. 
       
    Returns 
    --------
    structures , str_res: list 
      List of similar structures that fits each resistivities. 
      returns also the closest resistivities if `return_values` is set 
      to ``True``.
    
    Examples
    ---------
    >>> from watex.utils.geotools import find_similar_structures
    >>> find_similar_structures (2 , 10, 100 , return_values =True)
    Out[206]: 
    (['sedimentary rocks', 'metamorphic rocks', 'clay'], [1.0, 10.0, 100.0])
    """
    if return_values is ...: 
        return_values =False 
    
    def get_single_structure ( res): 
        """ get structure name and it correspoinding values """
        n, v = [], []
        for names, rows in zip ( stc_names, stc_values):
            # sort values 
            rs = sorted (rows)
            if rs[0] <= res and res <= rs[1]: 
            #if rows[0] <= res <= rows[1]: 
                n.append ( names ); v.append (rows )
                
        if len(n)==0: 
            return "*Struture not found", np.nan  
        
        v_values = np.array ( v , dtype = float ) 
        # find closest value 
        close_val  = find_closest ( v_values , res )
        close_index, _= np.where ( v_values ==close_val) 
        # take the first close values 
        close_index = close_index [0] if len(close_index ) >1 else close_index 
        close_name = str ( n[int (close_index )]) 
        # remove the / and take the first  structure 
        close_name = close_name.split("/")[0] if close_name.find(
            "/")>=0 else close_name 
        
        return close_name, close_val 
    
    #---------------------------
    # make array of names structures names and values 
    dict_conf =  GeoscienceProperties().geo_rocks_properties 
    stc_values = np.array (list( dict_conf.values()))
    stc_names = np.array ( list(dict_conf.keys() ))

    try : 
        np.array (resistivities) .astype (float)
    except : 
        raise TypeError("Resistivities array expects numeric values."
                        f" Got {np.array (resistivities).dtype.name!r}") 
        
    structures = [] ;  str_res =[]
    for res in resistivities: 
        struct, value = get_single_structure(res )
        structures.append ( struct) ; str_res.append (float(value) )

    return ( structures , str_res ) if  return_values else structures 

def build_random_thickness(
    depth, / , 
    n_layers=None, 
    h0= 1 , 
    shuffle = True , 
    dirichlet_dist=False, 
    random_state= None, 
    unit ='m'
): 
    """ Generate a random thickness value for number of layers 
    in deeper. 
    
    Parameters 
    -----------
    depth: ArrayLike, float 
       Depth data. If ``float`` the number of layers `n_layers` must 
       be specified. Otherwise an error occurs. 
    n_layers: int, Optional 
       Number of layers that fit the samples in depth. If depth is passed 
       as an ArrayLike, `n_layers` is ignored instead. 
    h0: int, default='1m' 
      Thickness of the first layer. 
      
    shuffle: bool, default=True 
      Shuffle the random generated thicknesses. 

    dirichlet_dis: bool, default=False 
      Draw samples from the Dirichlet distribution. A Dirichlet-distributed 
      random variable can be seen as a multivariate generalization of a 
      Beta distribution. The Dirichlet distribution is a conjugate prior 
      of a multinomial distribution in Bayesian inference.
      
    random_state: int, array-like, BitGenerator, np.random.RandomState, \
         np.random.Generator, optional
      If int, array-like, or BitGenerator, seed for random number generator. 
      If np.random.RandomState or np.random.Generator, use as given.
      
    unit: str, default='m' 
      The reference unit for generated layer thicknesses. Default is 
      ``meters``
      
    Return 
    ------ 
    thickness: Arraylike of shape (n_layers, )
      ArrayLike of shape equals to the number of layers.
      
    Examples
    ---------
    >>> from gofast.geo.utils import build_random_thickness 
    >>> build_random_thickness (7, 10, random_state =42  )
    array([0.41865079, 0.31785714, 1.0234127 , 1.12420635, 0.51944444,
           0.92261905, 0.6202381 , 0.8218254 , 0.72103175, 1.225     ])
    >>> build_random_thickness (7, 10, random_state =42 , dirichlet_dist=True )
    array([1.31628992, 0.83342521, 1.16073915, 1.03137592, 0.79986286,
           0.8967135 , 0.97709521, 1.34502617, 1.01632075, 0.62315132])
    """

    if hasattr (depth , '__array__'): 
        max_depth = max( depth )
        n_layers = len(depth )
        
    else: 
        try: 
            max_depth = float( depth )
        except: 
            raise DepthError("Depth must be a numeric or arraylike of float."
                             f" Got {type (depth).__name__!r}")

    if n_layers is None: 
        raise DepthError ("'n_layers' is needed when depth is not an arraylike.")

    layer0 = copy.deepcopy(h0)

    try: 
        h0= convert_value_in (h0 , unit=unit)
    except : 
        raise TypeError(f"Invalid thickness {layer0}. The thickness for each"
                        f" stratum should be numeric.Got {type(layer0).__name__!r}")

    thickness = np.linspace  (h0 , max_depth, n_layers) 
    thickness /= max_depth 
    # add remain data value to depth. 
    if  round ( max_depth - thickness.sum(), 2)!=0: 
        
        thickness +=  np.linspace (h0, abs (max_depth - thickness.sum()),
                                   n_layers )/thickness.sum()
    if dirichlet_dist: 
        if random_state: 
            np.random.seed (random_state )
        if n_layers < 32: 
            thickness= np.random.dirichlet (
                np.ones ( n_layers), size =n_layers) 
            thickness= np.sum (thickness, axis = 0 )
        else: 
            thickness= np.random.dirichlet (thickness) 
            thickness *= max_depth  
    
    if shuffle: 
        ix = np.random.permutation (
            np.arange ( len(thickness)))
        thickness= thickness[ix ]
  
    return thickness 



