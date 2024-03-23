# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import pandas as pd
import numpy as np

from ..tools.coreutils import is_iterable, unpack_list_of_dicts 
from ..tools.validator import validate_dates 
from .util import manage_data, validate_region 

from .util import find_mineral_distributions, find_countries_by_minerals
from .util import extract_minerals_from_regions, build_distributions_from
from .util import check_distributions, select_location_for_mineral
from .util import generate_ore_infos

def simulate_water_reserves(
    *, n_samples=300, 
    start_date="2024-01-01", 
    end_date="2024-01-31", 
    as_frame=False, 
    return_X_y=False, 
    target_names=None, 
    noise=None, 
    seed=None, 
  ):
    """
    Generate a simulated dataset of water reserve measurements across various
    locations over a specified time period. The dataset includes measurements
    such as total capacity, current volume, and water usage, along with 
    environmental factors affecting the reserves like rainfall and evaporation.

    Parameters
    ----------
    n_samples : int, optional
        Number of unique locations for which data will be generated. Each 
        location will have entries for each day within the specified date range,
        thus creating a comprehensive dataset across time and space. 
        Default is 300.
        
    start_date, end_date : str, optional
        The time range for the dataset, specified in "YYYY-MM-DD" format. 
        The function will generate data for each day within this range, 
        inclusive of both start and end dates. 
        Default range is from ``"2024-01-01"`` to ``"2024-01-31"``.
        
    as_frame : bool, optional
        Determines the format of the returned dataset. If True, the dataset
        is returned as a pandas DataFrame, which is useful for data analysis 
        and manipulation with pandas. If ``False``, the dataset is returned in
        a Bunch object or as arrays, based on the `return_X_y` parameter.
        Default is ``False``.
        
    return_X_y : bool, optional
        If True, the function returns a tuple `(X, y)` where `X` is the array
        of feature values and `y` is the array of targets. This is useful 
        for directly feeding data into machine learning models. If ``False``, 
        returns a Bunch object containing the dataset, targets, and 
        additional information. This parameter is ignored if `as_frame` is 
        ``True``. Default is False.
        
    target_names : list of str, optional
        Specifies the names of the target variables to be included in the 
        output. By default (`None`), the function uses ``"percentage_full"`` as 
        the target variable, which represents the percentage of the water 
        reserve's capacity that is currently filled.
        
    noise : float or None, optional
        Adds Gaussian noise with standard deviation equal to `noise` to the 
        numerical features of the dataset to simulate measurement errors or 
        environmental fluctuations. If None, no noise is added. This can be 
        useful for creating more realistic datasets for robustness testing of 
        models. Default is ``None``.
        
    seed : int or None, optional
        Seed for the random number generator to ensure reproducibility of 
        the dataset. If ``None``, the random number generator is initialized
        without a fixed seed. Setting a seed is useful when conducting 
        experiments that require consistent results. Default is ``None``.

    **kws : dict
        Additional keyword arguments not explicitly listed above. This allows
        for future expansion or customization of the dataset generation without
        modifying the function signature.

    Returns
    -------
    Depending on the combination of `as_frame` and `return_X_y` parameters, 
    the function can return a pandas DataFrame, a tuple of `(X, y)` arrays, or 
    a Bunch object encapsulating the  dataset, targets, and additional 
    metadata.

    Notes
    -----
    The simulated dataset is generated using pseudo-random numbers to model 
    the dynamics of water reserves, including environmental impacts and human
    usage. It's intended for testing, educational purposes, and as a placeholder
    in the development of data analysis pipelines.

    Examples
    --------
    Generating a DataFrame of simulated water reserve data for 10 locations 
    over January 2024:
    
    >>> simulate_water_reserves(n_samples=10, as_frame=True)
    
    Generating `(X, y)` arrays suitable for use with scikit-learn models:
    
    >>> X, y = simulate_water_reserves(return_X_y=True)

    See Also
    --------
    pandas.DataFrame : The primary data structure used when `as_frame=True`.
    sklearn.utils.Bunch : Used to package the dataset when arrays are returned.
    """
    from ._globals import WATER_RESERVES_LOC
    
    feature_descr= {
    "location_id": "A unique identifier for each location.",
    "location_name": "The name of the location (e.g., city, river, reservoir).",
    "date": "The date of the record.",
    "total_capacity_ml": "The total capacity of the water reserve in megaliters (ML).",
    "current_volume_ml": "The current volume of water in the reserve in megaliters (ML).",
    "percentage_full": "The percentage of the total capacity that is currently filled.",
    "rainfall_mm": "Rainfall in millimeters (mm) for the location on the date.",
    "evaporation_mm": "Estimated evaporation in millimeters (mm) on the date.",
    "inflow_ml": "Inflow of water into the reserve in megaliters (ML) on the date.",
    "outflow_ml": "Outflow of water from the reserve in megaliters (ML) on the date.",
    "usage_ml": "Water usage from the reserve in megaliters (ML) on the date."
    }
    
    np.random.seed(seed)
    start_date, end_date = validate_dates(
        start_date, end_date, return_as_date_str= True )
    dates = pd.date_range(start=start_date, end=end_date)
    data = []
    # Generate a unique location name for each location_id before generating data
    location_names = {i+1: np.random.choice(WATER_RESERVES_LOC) for i in range(n_samples)}

    for i in range(n_samples):
        for date in dates:
            total_capacity_ml = np.random.uniform(5000, 10000)
            current_volume_ml = np.random.uniform(1000, total_capacity_ml)
            percentage_full = (current_volume_ml / total_capacity_ml) * 100
            rainfall_mm = np.random.uniform(0, 50)
            evaporation_mm = np.random.uniform(0, 10)
            inflow_ml = np.random.uniform(0, 500)
            outflow_ml = np.random.uniform(0, 500)
            usage_ml = np.random.uniform(0, 300)

            data.append({
                "location_id": i + 1,
                "location_name": location_names[i + 1],  
                "date": date,
                "total_capacity_ml": total_capacity_ml,
                "current_volume_ml": current_volume_ml,
                "percentage_full": percentage_full,
                "rainfall_mm": rainfall_mm,
                "evaporation_mm": evaporation_mm,
                "inflow_ml": inflow_ml,
                "outflow_ml": outflow_ml,
                "usage_ml": usage_ml,
            })

    water_reserves_df = pd.DataFrame(data)
    if target_names is None:
        target_names = ["percentage_full"]

    return manage_data(
        data=water_reserves_df, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_names, noise=noise, seed=seed,
        DESCR="Simulated water reserves dataset",
        feature_descr=feature_descr,
    )

def simulate_world_mineral_reserves(
    *, n_samples=100, 
    as_frame=False, 
    return_X_y=False, 
    target_names=None, 
    noise=None, seed=None, 
    regions=None, 
    distributions=None, 
    mineral_types=None, 
    countries=None, 
    economic_impact_factor=0.05, 
    default_location='Global HQ',
):
    """
    Simulates a dataset of world mineral reserves, providing insights into global 
    mineral production. This function allows for the generation of data reflecting 
    mineral reserve quantities across different countries and regions, incorporating
    economic impact factors and statistical noise to mimic real-world variability.

    The simulation process involves selecting countries known for producing specified
    minerals, calculating reserve quantities, and optionally assigning a default location 
    for minerals with undetermined origins. The function offers flexibility in focusing 
    the simulation on specific minerals, regions, or countries, catering to diverse 
    geoscientific research needs.

    Parameters
    ----------
    n_samples : int, optional
        Number of data points to generate, representing individual mineral reserve 
        instances. Default is 300.
    as_frame : bool, optional
        If set to True, outputs the dataset as a pandas DataFrame, facilitating 
        data analysis and visualization. Default is False.
    return_X_y : bool, optional
        If True, separates the features and target variable into two entities, 
        supporting machine learning applications. Default is False.
    target_names : list of str, optional
        Specifies the names for the target variables, enhancing dataset 
        interpretability. Default is None.
    noise : float, optional
        Specifies the standard deviation of Gaussian noise added to reserve quantities,
        simulating measurement errors or estimation uncertainties. Default is None.
    seed : int, optional
        Seed for the random number generator, ensuring reproducibility of the simulated
        dataset. Default is None.
    regions : list of str, optional
        Filters the simulation to include countries within specified geographical regions,
        reflecting regional mineral production characteristics. Default is None.
    distributions : dict, optional
        Custom mapping of regions to mineral types for targeted simulation scenarios.
        Default is None.
    mineral_types : list of str, optional
        Filters the dataset to simulate reserves for specified minerals, aiding in 
        focused geoscientific studies. Default is None.
    countries : list of str, optional
        Specifies a list of countries to be included in the simulation, enabling 
        country-specific mineral production analysis. Default is None.
    economic_impact_factor : float, optional
        Adjusts the simulated quantity of reserves based on economic conditions, 
        introducing variability in mineral production estimates. Default is 0.05.
    default_location : str, optional
        Placeholder for the location when a mineral's producing country is 
        undetermined, typically a headquarters or primary research center. 
        Default is 'Global HQ'.

    Returns
    -------
    pandas.DataFrame or tuple
        The simulated dataset of mineral reserves. The structure of the returned
        data depends on the `as_frame` and `return_X_y` parameters.

    Raises
    ------
    ValueError
        Raised if required parameters are not provided or if the simulation 
        process encounters an error due to invalid parameter values.

    Examples
    --------
    Generate a simple dataset of mineral reserves:
    
    >>> from gofast.datasets.simulate import simulate_world_mineral_reserves
    >>> min_reserves = simulate_world_mineral_reserves().frame 
    
    Simulate reserves focusing on gold and diamonds in Africa and Asia:
    
    >>> df=simulate_world_mineral_reserves(regions=['Africa', 'Asia'],
    ...                                 mineral_types=['gold', 'diamond'],
    ...                                 n_samples=100, as_frame=True)
    >>> df.head()
    >>> b.frame 
        sample_id  ...      quantity
     0          1  ...    317.714995
     1          2  ...  10009.865385
     2          3  ...   6234.385021
     3          4  ...   1406.044839
     4          5  ...   7581.913884
    
     [5 rows x 6 columns]
    Handle undetermined production countries with a custom default location:
    >>> X, y= simulate_world_mineral_reserves(default_location='Research Center',
    ...                                 noise=0.1, seed=42, return_X_y=True )
    >>> len(y)
    100
    """
    # Initialize country dict 
    mineral_countries_map ={}
    # Set a seed for reproducibility of results
    np.random.seed(seed)

    # Normalize region input and validate regions if provided
    if regions:
        regions=is_iterable(regions, exclude_string=True, transform =True)
        regions = [validate_region(region, mode='soft') for region in regions] 

    # Normalize mineral_types to a list and find corresponding countries if specified
    if mineral_types:
        mineral_types= is_iterable(
            mineral_types, exclude_string=True, transform=True)
        countries, mineral_countries_map = find_countries_by_minerals(
            mineral_types, return_minerals_and_countries= True
        )
    # Extract minerals from regions if mineral_types are not explicitly provided
    if mineral_types is None and regions:  
        mineral_types = extract_minerals_from_regions(regions) 
    
    # Create distributions mapping if both regions and mineral types are available
    if regions and mineral_types: 
        distributions = build_distributions_from(regions, mineral_types)
        
    # Validate the structure of distributions, falling back to default if incorrect
    if distributions:
        if not check_distributions(distributions):
            # If distributions are invalid, revert to None and possibly warn the user
            distributions = None 

    # Use default distributions if none provided or after fallback
    if distributions is None:
        regions = ["Africa", "Asia", "Europe", "America", "Oceania"]
        distributions = unpack_list_of_dicts(
            [find_mineral_distributions(region) for region in regions])
        mineral_types = extract_minerals_from_regions(regions) 
    # Find countries based on mineral types if countries are not explicitly provided
    if not countries:
        countries, mineral_countries_map = find_countries_by_minerals(
            mineral_types, return_minerals_and_countries= True 
        )
    # Generate ore informations related to each country
    infos_dict = generate_ore_infos(countries, error="ignore")
    
    # Simulate mineral reserve data for each sample
    data = []
    for i in range(n_samples):
        selected_region = np.random.choice(list(distributions.keys()))
        available_minerals = distributions[selected_region]
        mineral_type = np.random.choice(available_minerals)
        base_quantity = np.random.uniform(100, 10000)
        economic_impact = 1 + (np.random.rand() * economic_impact_factor - (
            economic_impact_factor / 2)) # make sure to not have negative quantities
        quantity = max(0, base_quantity * economic_impact + (
            np.random.normal(0, noise) if noise else 0))
        # Select location 
        selected_region, location= select_location_for_mineral (
            mineral_type, mineral_countries_map,
            fallback_countries=countries, 
            selected_region =selected_region, 
            substitute_for_missing= default_location 
        )
        info = infos_dict.get(location, "No information available")
        data.append({
            'sample_id': i + 1,
            'region': selected_region,
            'location': location,
            'mineral_type': mineral_type,
            'info': info,
            # production soon 
            'quantity': quantity
        })
    # Convert simulated data into a DataFrame
    mineral_reserves_df = pd.DataFrame(data)
    
    # Handle data return format based on function parameters
    return manage_data(
        data=mineral_reserves_df, as_frame=as_frame, return_X_y=return_X_y,
        target_names=target_names if target_names else ['quantity'],
        DESCR="Simulated mineral reserves dataset", 
    )
