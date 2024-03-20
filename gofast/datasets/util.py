# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
import copy
import warnings 
from collections import defaultdict
import itertools
import numpy as np 
import pandas as pd 
from ..tools.coreutils import is_in_if, add_noises_to, is_iterable
from ..tools.coreutils import smart_format 
from ..compat.sklearn import  train_test_split 
from ..tools.box import Boxspace


def find_countries_by_region(
    region, 
    country_region_dict=None 
    ):
    """
    Find all countries in the given region. The function is case-insensitive and
    considers partial matches, allowing for flexible searches such as 
    'america' matching both 'North America' and 'South America', or 'Europe'
    matching 'Europe/Asia'.

    Parameters:
    ----------
    region : str
        The region to search for. The search is case-insensitive and supports
        partial matches. Expected regions include 'Africa', 'America', 'Asia',
        'Oceania', and 'Europe', but also recognizes combined regions 
        like 'Europe/Asia'.
        
    country_region_dict : dict
        A dictionary mapping country names (keys) to their respective 
        regions (values). Regions should accommodate combined entries like
        'Europe/Asia' for accurate matching.

    Returns:
    -------
    list
        A list of country names that are located within the specified region 
        or match the region criteria.

    Raises:
    ------
    ValueError
        If the specified region does not partially match any of the region
        entries in the country_region_dict values, indicating an unexpected
        or misspelled region name.

    Examples:
    --------
    >>> find_countries_by_region('America', country_region_dict)
    ['United States', 'Canada', 'Brazil', 'Chile', 'Peru', ...]

    >>> find_countries_by_region('europe', country_region_dict)
    ['Russia', 'Sweden', 'Ukraine', 'Poland', 'Norway', ...]

    Note:
    -----
    The function is designed to be flexible with region names, accommodating 
    scenarios where regions may overlap or be part of a combined region 
    (e.g., 'Europe/Asia'). This is particularly useful for geopolitical regions
    and territories that do not fit neatly into a single continent.
    """
    if country_region_dict is None: 
        from ._globals import COUNTRY_REGION 
        country_region_dict = copy.deepcopy(COUNTRY_REGION)
    
    # Simplifying region matching by only checking for substrings in region names,
    # without pre-validating against a list of normalized regions.
    region_lower = region.lower()
    
    # Compiling a list of unique region parts to avoid errors 
    # on combined or cardinal directions
    unique_region_parts = set(
        part for regions in country_region_dict.values() 
        for part in regions.replace('/', ' ').lower().split()
        )
    # Validate if the provided region or any part of it matches the 
    # known regions or cardinal points
    if not any(part in unique_region_parts for part in region_lower.split('/')):
        raise ValueError(
            f"Unexpected region '{region}'. Please check the region"
            " name for typos. Expected regions include: smart_format(regions)"
            )
    # Filter countries by region, accommodating partial and case-insensitive matches
    countries = [country for country, reg in country_region_dict.items()
                 if region_lower in reg.lower()]
    if len(countries) == 0:
         warnings.warn(f"No countries found for '{region}'. Falling back to"
                       " default: returning all countries.")
         countries = list(country_region_dict.keys())
        
    return countries

def extract_minerals_from_countries(
    mineral_prod_by_country=None
    ):
    """
    Extracts mineral reserves information from a provided dictionary mapping countries 
    to their mineral production and reserves information. The mineral reserves are 
    extracted from the textual descriptions within the dictionary values.

    Parameters
    ----------
    mineral_prod_by_country : dict, optional
        A dictionary where each key is a country name and each value is a list of 
        strings describing the mineral production and reserves of that country. The 
        mineral reserves are expected to be within square brackets in the first element 
        of the list. If None, attempts to import a default dictionary from a global module.

    Returns
    -------
    dict
        A new dictionary where each key is a country name and the corresponding value 
        is a string listing the mineral reserves for that country, extracted from the 
        provided descriptions.

    Examples
    --------
    >>> mineral_prod_by_country = {
    ...     "Australia": ["Vast reserves of bauxite, [iron ore, lithium]"],
    ...     "China": ["Large reserves of [coal, rare earth elements]"]
    ... }
    >>> extract_minerals_from_countries(mineral_prod_by_country)
    {'Australia': 'iron ore, lithium', 'China': 'coal, rare earth elements'}

    Notes
    -----
    This function is designed to process structured textual data where specific 
    information is enclosed in square brackets. It's particularly useful for quickly 
    extracting key data points from descriptive texts.
    """
    if mineral_prod_by_country is None:
        try:
            from ._globals import MINERAL_PROD_BY_COUNTRY
            mineral_prod_by_country = copy.deepcopy(MINERAL_PROD_BY_COUNTRY)
        except ImportError:
            raise ImportError("MINERAL_PROD_BY_COUNTRY not found. Please provide"
                              " a mineral production dictionary.")
    # Initialize an empty dictionary to store the country and its mineral reserves
    country_minerals_dict = {}
    
    for country, info_list in mineral_prod_by_country.items():
        # Extract the mineral reserves from the first element of the list
        # Assuming the reserves are always in square brackets in the first element
        minerals_text = info_list[0]# Get the first element
        start = minerals_text.find('[') + 1 # Find the start of the bracket
        end = minerals_text.find(']') # Find the end of the bracket
        minerals = minerals_text[start:end]# Extract the text within the brackets
        # Assign the extracted minerals to the country in the new dictionary
        country_minerals_dict[country] = minerals

    return country_minerals_dict


def find_mineral_distributions(
    region,
    mineral_prod_by_country=None, 
    detailed_by_country=False, 
    include_region_summary=False
    ):
    """
    Aggregates and presents mineral distribution data for specified geographical
    regions, with options for detailed data by country and a summary of
    minerals found in the region.

    This function first retrieves countries within the specified region, then 
    extracts and aggregates mineral reserves information from these countries. 
    It offers flexibility  in the output format, allowing for detailed 
    information by country, a summary of unique minerals in the region, or both.

    Parameters:
    ----------
    region : str
        The geographical region of interest. The function is case-insensitive and
        supports partial matches to accommodate broad searches.
    mineral_prod_by_country : dict, optional
        A dictionary mapping countries to their mineral production and reserves.
        If None, the function attempts to use a default dictionary defined 
        in the scope.
    detailed_by_country : bool, optional
        If set to True, the function returns a dictionary with countries in 
        the specified region and their corresponding minerals. Default is False.
    include_region_summary : bool, optional
        If True, the function appends a summary list of unique minerals found across
        the entire region to the output. This parameter requires `detailed_by_country`
        to be True to take effect. Default is False.

    Returns:
    -------
    dict
        Depending on the parameters, the function returns:
        - A dictionary with a single key-value pair, where the key is the region
          and the value
          is a list of unique minerals, if `detailed_by_country` is False.
        - A dictionary with a single key-value pair, where the key is the region
          and the value is a detailed mapping of each country to its minerals,
          if `detailed_by_country` is True.
        - If `include_region_summary` is also True, the value includes both 
          the detailed mapping and a list of unique minerals in the region.

    Examples:
    --------
    >>> from gofast.datasets.util import find_mineral_distributions
    >>> find_mineral_distributions('Europe')
    {'Europe': ['palladium', 'nickel', 'potash', 'uranium', ...]}
    
    >>> find_mineral_distributions('Europe', detailed_by_country=True)
    {'Europe': {'Russia': 'palladium, nickel', 'Sweden': 'iron ore, copper', ...}}
    
    >>> find_mineral_distributions('Europe', detailed_by_country=True, include_region_summary=True)
    {'Europe': [{'Russia': 'palladium, nickel', 'Sweden': 'iron ore, copper', ...},
                ['palladium', 'nickel', 'iron ore', 'copper', ...]]}

    Note:
    -----
    The function is designed to provide flexibility in accessing mineral 
    distribution data across different geographical regions. It can be used 
    for data analysis, educational purposes, or informing policy and investment
    decisions related to mineral resources.
    """
    # Implementation remains the same as provided earlier

    
    countries = find_countries_by_region(region)
    
    # If no custom dictionary is provided, use a default
    mineral_info = extract_minerals_from_countries(mineral_prod_by_country)
    
    # If detailed_by_country is True or include_region_summary is True,
    # prepare country-specific data
    if detailed_by_country or include_region_summary:
        country_minerals = {country: mineral_info.get(country, None) for country in countries}
    
    # Aggregate unique minerals found across the region
    region_minerals = set(mineral_info.get(country, "") for country in countries)
    
    # Assemble the return data based on the function parameters
    if include_region_summary:
        # Returns both country-specific data and a summary of minerals in the region
        return {region: [country_minerals, list(region_minerals)]}
    
    if detailed_by_country:
        # Returns a dictionary mapping each country in the region to its minerals
        return {region: country_minerals}
    
    # Default return: a list of unique minerals found in the region
    return {region: list(region_minerals)}


def find_countries_by_minerals(
    region, 
    minerals=None, 
    return_countries_only=False
    ):
    """
    Finds countries within a specified region that are associated with
    given minerals.

    Parameters:
    ----------
    region : str
        The geographical region to search within.
    minerals : str or list of str, optional
        A single mineral or a list of minerals to find countries for. 
        If None, the function returns all countries in the region without 
        filtering by mineral.
    return_countries_only : bool, optional
        If True, returns a list of unique countries associated with the minerals.
        If False, returns a dictionary mapping each mineral to a list of countries.

    Returns:
    -------
    dict or list
        Depending on `return_countries_only`, returns either:
        - A dictionary mapping each specified mineral to a list of countries in the region
          that have reserves of that mineral.
        - A list of unique countries in the region associated with the specified minerals.

    Example:
    --------
    >>> from gofast.datasets.util import find_countries_by_minerals
    >>> find_countries_by_minerals('Europe', minerals=['gold', 'nickel'], 
    ...                             return_countries_only=True)
    ['Russia', 'Sweden']

    >>> find_countries_by_minerals('Europe', minerals='gold')
    {'gold': ['Russia', 'Sweden']}
    """
    extracted_minerals = find_mineral_distributions(region, detailed_by_country=True)
    
    if minerals is None: 
        return list( extracted_minerals.get(region).keys() ) 
    
    # Ensure minerals is a list for iteration
    if isinstance(minerals, str):
        minerals = [minerals]
    
    # Using defaultdict to automatically initialize lists for new keys
    mineral_by_country = defaultdict(list)

    # Loop through each mineral and aggregate countries
    for reg, countries_minerals in extracted_minerals.items():
        for country, country_minerals in countries_minerals.items():
            for mineral in minerals:
                # Normalize case for comparison
                if mineral.lower() in country_minerals.lower():
                    mineral_by_country[mineral].append(country)
    
    if return_countries_only:
        # Combine all lists and remove duplicates
        all_countries = set(itertools.chain(*mineral_by_country.values()))
        return list(all_countries)
    
    return dict(mineral_by_country)
     
def manage_data(
    data, 
    as_frame= False, 
    return_X_y= False, 
    split_X_y= False, 
    target_names= None, 
    test_size= 0.3, 
    noise= None, 
    seed= None, 
    **kwargs
):

    """ Manage the data and setup into an Object 
    
    Parameters
    -----------
    data: Pd.DataFrame 
        The dataset to manage 

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bowlspace object. See
        below for more information about the `data` and `target` object.
        
    split_X_y: bool, default=False,
        If True, the data is splitted to hold the training set (X, y)  and the 
        testing set (Xt, yt) with the according to the test size ratio. 
        
    target_names: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively
        . 
    noise : float, Optional
        The percentage of values to be replaced with NaN in each column. 
        This must be a number between 0 and 1. Default is None.
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns 
    -------
    data : :class:`~gofast.tools.box.Boxspace` object
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} 
            The data matrix. If ``as_frame=True``, `data` will be a pandas DataFrame.
        target: {ndarray, Series} 
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame 
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

    data, target: tuple if `return_X_y` is ``True``
        A tuple of two ndarray. The first containing a 2D array of shape
        (n_samples, n_features) with each row representing one sample and
        each column representing the features. The second ndarray of shape
        (n_samples,) containing the target samples.

    X, Xt, y, yt: Tuple if `split_X_y` is ``True`` 
        A tuple of two ndarray (X, Xt). The first containing a 2D array of:
            
        .. math:: 
            
            \\text{shape}(X, y) =  1-  \\text{test_ratio} *\
                (n_{samples}, n_{features}) *100
            
            \\text{shape}(Xt, yt)= \\text{test_ratio} * \
                (n_{samples}, n_{features}) *100
        
        where each row representing one sample and each column representing the 
        features. The second ndarray of shape(n_samples,) containing the target 
        samples.
    
    """
    # Ensure the correct data types for the parameters
    as_frame, return_X_y, split_X_y = map(
        lambda x: bool(x), [as_frame, return_X_y, split_X_y]
    )
    test_size = float(test_size)
    if noise is not None:
        noise = float(noise)
    if seed is not None:
        seed = int(seed)
    
    frame = data.copy()
    feature_names = (
        is_in_if(list( frame.columns), target_names, return_diff =True )
        if target_names else list(frame.columns )
    )
    y = None
    if return_X_y:
        y = data [target_names].squeeze ()  
        data.drop( columns = target_names, inplace =True )
        
    # Noises only in the data not in target
    data = add_noises_to(data, noise=noise, seed=seed)

    if not as_frame:
        data = np.asarray(data)
        y = np.squeeze(np.asarray(y))
    
    if split_X_y:
        return train_test_split(data, y, test_size=test_size, random_state=seed)
    
    if return_X_y:
        return data, y
    
    frame[feature_names] = add_noises_to(frame[feature_names], noise=noise, seed=seed)
    
    if as_frame:
        return frame
    
    return Boxspace(
        data=data,
        target=frame[target_names].values if target_names else None,
        frame=frame,
        target_names=[target_names] if target_names else [],
        feature_names=feature_names,
        **kwargs
    )

def get_item_from ( spec , /,  default_items, default_number = 7 ): 
    """ Accept either interger or a list. 
    
    If integer is passed, number of items is randomly chosen. if 
    `spec` is given as a list of objects, then do anything.
    
    Parameters 
    -----------
    spec: int, list 
        number of specimens or speciments to fecth 
    default_items: list 
       Global list that contains the items. 
       
    default_number: int, 
        Randomly select the number of specimens if `spec` is ``None``. 
        If ``None`` then take the length of all default items in 
        `default_items`. 
    Return
    -------
    spec: list 
       List of items retrieved according to the `spec`. 
       
    """
    if default_number is None: 
        default_number= len(default_items )
        
    if spec is None: 
        spec =default_number 
        
    if isinstance ( spec, ( int, float)): 
        spec = np.random.choice (
            default_items, default_number if int(spec)==0 else int (spec) )
    
    spec = is_iterable ( spec, exclude_string= True, transform =True )
    
    return spec 


def generate_synthetic_values(
        samples, range_min, range_max, noise=None, seed=None):
    """
    Generate synthetic data within a given range, optionally adding noise.
    """
    np.random.seed(seed)
    values = np.random.uniform(range_min, range_max, samples)
    if noise:
        values += np.random.normal(0, noise, samples)
    return values
 
def generate_categorical_values(samples, categories, seed=None):
    """
    Generate synthetic categorical data based on specified categories.
    """
    np.random.seed(seed)
    values = np.random.choice(categories, size=samples)
    return values

def generate_regression_output(X, coef, bias, noise, regression_type):
    """
    Generates the regression output based on the specified regression type.
    """
    from ..tools.mathex import linear_regression, quadratic_regression
    from ..tools.mathex import exponential_regression,logarithmic_regression
    from ..tools.mathex import sinusoidal_regression, cubic_regression
    from ..tools.mathex import step_regression
    
    available_reg_types = [ 'linear', 'quadratic', 'cubic','exponential', 
                           'logarithmic', 'sinusoidal', 'step' ]
    regression_dict =  dict ( zip ( available_reg_types, [ 
        linear_regression, quadratic_regression, cubic_regression, 
        exponential_regression,logarithmic_regression, sinusoidal_regression,
        step_regression] ))
                          
    if regression_type not in available_reg_types: 
        raise ValueError(f"Invalid regression_type '{regression_type}'. Expected"
                         f" {smart_format(available_reg_types, 'or')}.")

    return regression_dict[regression_type](X, coef=coef , bias=bias, noise=noise )
        
def apply_scaling(X, y, method):
    """
    Applies the specified scaling method to the data.
    """
    from ..tools.mathex import standard_scaler, minmax_scaler, normalize
    
    scale_dict = {'standard':standard_scaler ,
    'minmax':minmax_scaler , 'normalize':normalize }
    if method not in  (scale_dict.keys()): 
        raise ValueError (f"Invalid scale method '{method}'. Expected"
                          f"{smart_format(scale_dict.keys(),'or')}")
    return scale_dict[method] ( X, y=y )

def rename_data_columns(data, new_columns=None):
    """
    Renames the columns of a pandas DataFrame or the name of a pandas Series.

    This function adjusts the column names of a DataFrame or the name of a Series
    to match the provided list of new column names. If the new column names list is
    shorter than the number of existing columns, the remaining columns are 
    left unchanged.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The DataFrame or Series whose columns or name are to be renamed.
    new_columns : list or iterable, optional
        The new column names or Series name. If None, no changes are made.

    Returns
    -------
    pd.DataFrame or pd.Series
        The DataFrame or Series with updated column names or name.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    >>> rename_data_columns(df, ['X', 'Y'])
       X  Y
    0  1  2
    1  3  4

    >>> series = pd.Series([1, 2, 3], name='A')
    >>> rename_data_columns(series, ['X'])
    Name: X, dtype: int64
    """
    if new_columns is not None and hasattr(data, 'columns'):
        # Ensure new_columns is a list and has the right length
        new_columns = list(new_columns)
        extra_columns = len(data.columns) - len(new_columns)
        if extra_columns > 0:
            # Extend new_columns with the remaining original columns if not 
            # enough new names are provided
            new_columns.extend(data.columns[-extra_columns:])
        data.columns = new_columns[:len(data.columns)]
    elif new_columns is not None and isinstance(data, pd.Series):
        data.name = new_columns[0]
    return data




