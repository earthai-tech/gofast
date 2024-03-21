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


def validate_region(region, mode="strict"):
    """
    Validates and normalizes a region name based on the specified mode.

    Parameters
    ----------
    region : str
        The name of the region to validate and normalize. The region can
        include cardinal points (North, South, East, West) and their variants
        (Northern, Southern, Eastern, Western), followed by the continent name.
    mode : str, optional
        The mode of operation, either 'strict' or 'soft'. In 'strict' mode,
        any cardinal points are removed, and the region is normalized to the
        continent name only. In 'soft' mode, cardinal points are normalized
        and retained in the output. The default is 'strict'.

    Returns
    -------
    str
        The normalized region name, with or without cardinal directions,
        depending on the mode.

    Raises
    ------
    ValueError
        If the region name is not valid based on the mode of operation or if an
        unknown mode is specified.

    Examples
    --------
    >>> from gofast.datasets.util import validate_region 
    
    >>> validate_region("North America", mode="strict")
    'America'
    
    >>> validate_region("Southern Africa", mode="soft")
    'South Africa'
    
    >>> validate_region("Northern America", mode="soft")
    'North America'
    
    >>> validate_region("Western Asia", mode="strict")
    'Asia'
    
    In 'strict' mode, invalid region names raise an error:

    >>> validate_region("Central Europe", mode="strict")
    ValueError: Invalid region name 'Central Europe'. Expected one of the continents.

    In 'soft' mode, unrecognized region parts are ignored, and an attempt is made
    to normalize and retain valid parts:

    >>> validate_region("Eastern Central Europe", mode="soft")
    'East Europe'
    """

    cardinal_mapping = {
        "northern": "north",
        "southern": "south",
        "western": "west",
        "eastern": "east"
    }
    continents = ["africa", "america", "asia", "europe", "oceania"]
    cardinal_points = ["north", "south", "east", "west"]

    # Normalize input
    region_lower = region.lower()
    for variant, cardinal in cardinal_mapping.items():
        region_lower = region_lower.replace(variant, cardinal)
    
    # Split region name to handle cardinal directions
    parts = region_lower.split()

    # Normalize based on mode
    if mode == "strict":
        # Remove cardinal directions if present
        normalized_region = next((part for part in parts if part in continents), None)
        if normalized_region is None:
            raise ValueError(f"Invalid region name '{region}'. Expected one of the continents.")
    elif mode == "soft":
        # Keep the cardinal direction but ensure it's the correct form
        normalized_region = " ".join(part.capitalize() for part in parts 
                                     if part in continents + cardinal_points)
        if not normalized_region:
            raise ValueError(f"Invalid region name '{region}'.")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'strict' or 'soft'.")

    return normalized_region

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

def extract_minerals_from_regions(
    regions, 
    detailed_by_region=False, 
    separate_regions_and_minerals=False
):
    """
    Extracts and compiles a list of minerals from specified global regions. 
    This function can return data in different formats based on the input 
    parameters, offering flexibility in how the extracted information is 
    presented.

    Parameters
    ----------
    regions : str or list of str
        A single region name or a list of region names for which to extract 
        mineral information. The function fetches mineral distribution data 
        for these regions.
    detailed_by_region : bool, optional
        If True, returns a detailed list of tuples, each containing a region 
        name and a list of corresponding minerals. Default is False.
    separate_regions_and_minerals : bool, optional
        If True, returns two separate lists: one for all the region names and 
        another for lists of minerals corresponding to each region. Only effective 
        if `detailed_by_region` is False. Default is False.

    Returns
    -------
    list or tuple
        Depending on the parameters, this function can return:
        - A list of all minerals across specified regions if both 
          `detailed_by_region` and `separate_regions_and_minerals` are False.
        - A list of tuples with detailed region-mineral information if 
          `detailed_by_region` is True.
        - A tuple containing two lists (regions and their minerals) if 
          `separate_regions_and_minerals` is True.

    Examples
    --------
    Extracting minerals without details:

    >>> from gofast.datasets.util import extract_minerals_from_regions
    >>> extract_minerals_from_regions(["Europe", "Asia"])
    ['palladium', 'nickel', 'potash', 'uranium', 'coal', 'iron', 'silver']

    Extracting detailed mineral information by region:

    >>> extract_minerals_from_regions(["Europe"], detailed_by_region=True)
    [('Europe', ['palladium', 'nickel', 'potash', 'uranium'])]

    Separating region names and minerals into separate lists:

    >>> regions, minerals = extract_minerals_from_regions(
    ...     ["Europe", "Asia"], separate_regions_and_minerals=True)
    >>> print(regions)
    ['Europe', 'Asia']
    >>> print(minerals)
    [['palladium', 'nickel', 'potash', 'uranium'], ['coal', 'iron', 'silver']]
    """
    if isinstance(regions, str):
        regions = [regions]
    # Fetching distributions for all regions
    mineral_list_dicts = [find_mineral_distributions(region, include_region_summary=True)
                          for region in regions]

    if detailed_by_region:
        detailed_info = [(region, minerals[1]) for mineral_info in mineral_list_dicts
                         for region, minerals in mineral_info.items()]
        return detailed_info
        
    elif separate_regions_and_minerals:
        regions_list = [list(mineral_info.keys())[0] for mineral_info in mineral_list_dicts] 
        minerals_lists = [mineral_info[region][1] for mineral_info, region in zip(
            mineral_list_dicts, regions_list)]
        return regions_list, minerals_lists

    else:
        # Extract all minerals and directly split by both ',' and ';' in one step
        split_minerals = itertools.chain.from_iterable(
            # Split each mineral string by ',' then by ';' to handle both cases
            itertools.chain.from_iterable(
                min_str.split(';') for min_str in mineral.split(',')
            ) for mineral in itertools.chain.from_iterable(
                mineral_info[region][1] for mineral_info in mineral_list_dicts 
                for region in mineral_info
            )
        )
        # Remove duplicates and return the list
        return list(set([mineral.strip() for mineral in  split_minerals]))

def check_distributions(distributions, raise_error='warn'):
    """
    Validates the structure of the distributions dictionary to ensure it
    adheres to the expected format: a dictionary with region names as keys
    and lists of minerals as values.

    Parameters
    ----------
    distributions : dict
        The distributions dictionary to validate.
    raise_error : str, optional
        Controls error handling behavior. Accepts 'raise' to raise exceptions,
        'warn' to issue a warning, or 'ignore' to do nothing on failure.
        Default is 'warn'.

    Returns
    -------
    bool
        True if the distributions dictionary fits the expected arrangement,
        otherwise False if 'raise_error' is set to 'ignore' or 'warn'.
    """

    error_message = ""

    if not isinstance(distributions, dict):
        error_message = ( 
            "Distributions must be a dictionary with regions"
            " as keys and lists of minerals as values."
        )

    else:
        for region, minerals in distributions.items():
            if not isinstance(region, str):
                error_message = ( 
                    "All keys in the distributions dictionary must be"
                    " strings representing region names."
                    )
                break
            if isinstance(minerals, str):
                minerals = [minerals]  # Normalize single string to list
            if not isinstance(minerals, list) or not all(
                    isinstance(mineral, str) for mineral in minerals):
                error_message = ( 
                    "Each value in the distributions dictionary must"
                    " be a list of strings representing minerals."
                    )
                break

    if error_message:
        if raise_error == 'raise':
            raise ValueError(error_message)
        elif raise_error == 'warn':
            warnings.warn(error_message)
            return False
        elif raise_error == 'ignore':
            return False
    
    return True
  
def build_distributions_from(regions, minerals, raise_error='warn'):
    """
    Constructs a dictionary mapping each region to its corresponding list of 
    minerals.
    This function ensures that each region is paired with a specific list of 
    minerals, facilitating the creation of detailed regional mineral 
    distributions.

    Parameters
    ----------
    regions : list or str
        A list of region names or a single region name. If a single region name is
        provided, it is automatically transformed into a list. Regions define the
        geographical areas for the mineral distribution mapping.
    minerals : list
        A list of lists, where each sublist contains minerals associated with
        the corresponding region in the `regions` list. If `regions` contains
        only one region, `minerals` can be a single list which will be automatically
        nested into a list of lists.
    raise_error : {'warn', 'ignore', 'raise'}, optional
        Controls the error handling behavior when the lengths of `regions` and
        `minerals` do not match. 'warn' issues a warning and returns `None`,
        'ignore' silently ignores the discrepancy and also returns `None`, and
        'raise' raises a `ValueError`. The default is 'warn'.

    Returns
    -------
    dict or None
        A dictionary where keys are region names and values are lists of minerals
        associated with each region. Returns `None` if there is a discrepancy
        between the lengths of `regions` and `minerals` and `raise_error` is
        set to 'warn' or 'ignore'.

    Raises
    ------
    ValueError
        If `raise_error` is set to 'raise' and the lengths of `regions` and
        `minerals` do not match, indicating an inability to map each region to
        a specific list of minerals.

    Examples
    --------
    Basic usage with matching lengths of `regions` and `minerals`:

    >>> from gofast.datasets.util import build_distributions_from
    >>> build_distributions_from(['Africa', 'Asia'], [['Gold', 'Diamond'], ['Coal']])
    {'Africa': ['Gold', 'Diamond'], 'Asia': ['Coal']}

    Handling a single region and its minerals:

    >>> build_distributions_from('Europe', ['Uranium', 'Nickel'])
    {'Europe': ['Uranium', 'Nickel']}

    Handling mismatched lengths with 'raise':

    >>> build_distributions_from(['Africa', 'Asia'], ['Gold'], raise_error='raise')
    ValueError: Mismatch in list lengths: expected regions and minerals to have the same length, ...

    Using 'warn' with mismatched lengths (will print a warning and return `None`):

    >>> build_distributions_from(['Africa', 'Asia'], ['Gold'], raise_error='warn')
    UserWarning: Mismatch in list lengths: expected regions and minerals to have the same length, ...
    """
    regions = is_iterable(regions, exclude_string=True, transform=True)
    minerals = is_iterable(minerals, exclude_string=True, transform=True)
    
    # Ensuring minerals list is nested to correspond to regions if necessary
    if len(regions) == 1 and not is_structure_nested(minerals, list):
        minerals = [minerals]
    
    # Error handling for mismatched lengths
    if len(regions) != len(minerals):
        message = (f"Mismatch in list lengths: expected regions and minerals "
                   f"to have the same length, but got {len(regions)} region(s)"
                   f" and {len(minerals)} mineral list(s). This discrepancy"
                   f" prevents accurate distribution mapping. Please ensure"
                   " each region is paired with a corresponding list of minerals."
                   )
        if raise_error == 'raise':
            raise ValueError(message)
        elif raise_error == 'warn':
            warnings.warn(message)
        elif raise_error == 'ignore':
            pass  # Explicitly handling 'ignore' for clarity
        return None

    # Building the distributions dictionary
    distributions = {region: mineral for region, mineral in zip(regions, minerals)}
    
    return distributions

def manage_nested_lists(elements, mode='unpack', pack_criteria=None):
    """
    A versatile function that either packs elements into nested lists based 
    on a given criterion or unpacks nested lists into a unique, flat list.

    Parameters:
    - elements (list): The list of elements to be either packed or unpacked.
    - mode (str): Operation mode, either 'pack' for creating nested lists or 
      'unpack'
      for flattening nested lists. Default is 'unpack'.
    - pack_criteria (function): A function used as a criterion for packing 
      elements into
      nested lists. Only relevant if mode is 'pack'. It should accept an 
      element as input and return a boolean indicating whether the element 
      belongs to the current nested list.

    Returns:
    - list: Depending on the mode, either a nested list structure or a flat 
      list of unique elements.

    Example Usage:
    >>> manage_nested_lists([1, 2, [3, 4], [5, [6, 7]], 8], mode='unpack')
    [1, 2, 3, 4, 5, 6, 7, 8]
    
    >>> manage_nested_lists([1, 2, 3, 4, 5], mode='pack',
    ...                     pack_criteria=lambda x: x % 2 == 0)
    [[1], [2], [3], [4], [5]]
    """
    elements = is_iterable(elements, exclude_string=True, transform=True )
    mode_lower = str(mode).lower() 
    if mode_lower == 'unpack':
        if not is_structure_nested(elements, check_for='list'):
            return elements 
        # Flatten the list and remove duplicates
        return list(set(itertools.chain.from_iterable(elements) if isinstance(
            elements, list) else [elements]
                       for elements in elements))
    elif mode_lower == 'pack':
        if not pack_criteria:
            raise ValueError(
                "pack_criteria function must be provided when mode is 'pack'.")

        packed_list = []
        for element in elements:
            if isinstance(element, list):
                # Recursively pack sublists
                packed_list.append(manage_nested_lists(
                    element, mode='pack', pack_criteria=pack_criteria))
            elif pack_criteria(element):
                packed_list.append([element])
            else:
                if not packed_list or not pack_criteria(packed_list[-1][-1]):
                    packed_list.append([])
                packed_list[-1].append(element)
        return packed_list
    else:
        raise ValueError("Invalid mode. Choose either 'pack' or 'unpack'.")
        
def is_structure_nested(input_structure, check_for=list):
    """
    Checks if the given structure (list or dict) is nested with specified 
    types.

    Parameters:
    - input_structure (list or dict): The structure to check for nestedness.
    - check_for (str, list, dict, or tuple): Specifies the types to check for
      nesting. Can be 'list', 'dict', the built-in list or dict types, or a 
      tuple containing these types.

    Returns:
    - bool: True if the structure is nested with the specified type(s), False otherwise.

    Example Usage:
    >>> is_structure_nested([1, 2, 3], check_for='list')
    False
    >>> is_structure_nested([1, [2, 3], 4], check_for=list)
    True
    >>> is_structure_nested({'key': {'nested_key': 'value'}}, check_for=dict)
    True
    >>> is_structure_nested([1, {'nested_dict': 2}], check_for='dict')
    True
    >>> is_structure_nested([1, 2, 3], check_for=('dict',))
    False
    """
    type_map = {'list': list, 'dict': dict}
    # Normalize the check_for parameter to handle strings or direct type references
    if not isinstance(check_for, (list, tuple)):
        check_for = [check_for]
    check_for_types = [type_map.get(cf, cf) for cf in check_for]

    def is_nested(element):
        return any(isinstance(element, cf_type) for cf_type in check_for_types)
    
    if isinstance(input_structure, dict):
        return any(is_nested(value) for value in input_structure.values())
    
    if isinstance(input_structure, list):
        return any(is_nested(element) for element in input_structure)
    
    return False


def extract_minerals_from_countries(
    mineral_prod_by_country=None,
    use_default=True,
    split_minerals=True,
):
    """
    Extracts mineral reserves information from a provided dictionary mapping
    countries to their mineral production and reserves information, or from a
    default global dictionary. The function can parse the mineral reserves
    from structured textual descriptions and optionally split the extracted
    minerals into a list.

    Parameters
    ----------
    mineral_prod_by_country : dict, optional
        A dictionary where each key is a country name and each value is a list
        of strings or a single string describing the mineral production and
        reserves of that country. Minerals are expected to be within square
        brackets in the first element of the list for the default dictionary.
        If None and `use_default` is True, attempts to import a default
        dictionary from a global module.
    use_default : bool, optional
        Indicates whether to use the default mineral production dictionary if
        `mineral_prod_by_country` is not provided. Default is True.
    split_minerals : bool, optional
        If True, splits the extracted mineral reserves string into a list based
        on commas or semicolons. Default is True.

    Returns
    -------
    dict
        A new dictionary where each key is a country name and the corresponding
        value is a string or list of the mineral reserves for that country,
        extracted and optionally split from the provided descriptions.

    Raises
    ------
    ImportError
        If `mineral_prod_by_country` is None, `use_default` is True, and the
        global dictionary `MINERAL_PROD_BY_COUNTRY` cannot be imported.

    Examples
    --------
    Using the default dictionary and extracting minerals as a list:

    >>> from gofast.datasets.util import extract_minerals_from_countries
    >>> extract_minerals_from_countries()
    {'Australia': ['iron ore', 'lithium'], 'China': ['coal', 'rare earth elements']}

    Providing a custom dictionary and extracting minerals as a string:

    >>> mineral_prod_by_country = {
    ...     "Australia": "Vast reserves of bauxite, iron ore, lithium",
    ...     "China": "Large reserves of coal; rare earth elements"
    ... }
    >>> extract_minerals_from_countries(mineral_prod_by_country, 
    ...                                  use_default=False, split_minerals=False)
    {'Australia': 'iron ore, lithium', 'China': 'coal, rare earth elements'}

    Providing a custom dictionary and splitting minerals:

    >>> extract_minerals_from_countries(mineral_prod_by_country, split_minerals=True)
    {'Australia': ['bauxite', 'iron ore', 'lithium'], 
     'China': ['coal', 'rare earth elements']}

    Notes
    -----
    This function is particularly useful for quickly extracting key data points
    from structured textual data where specific information is enclosed in
    square brackets or separated by commas/semicolons.
    """
    if mineral_prod_by_country is None and use_default:
        mineral_prod_by_country = get_mineral_production_by_country()
        
    country_minerals_dict = {}
    
    for country, info_list in mineral_prod_by_country.items():
        if use_default:
            minerals_text = info_list[0]
            minerals = minerals_text[minerals_text.find('[') + 1: minerals_text.find(']')]
        else:
            if isinstance(info_list, list):
                info_list = ', '.join(info_list)  # Join list elements into a single string
            minerals = info_list
        
        # If 'split_minerals' is True, split the minerals string into a list,
        # considering ',' or ';'
        if split_minerals:
            separators = (',', ';')
            for sep in separators:
                if sep in minerals:
                    minerals = [mineral.strip() for mineral in minerals.split(sep)]
                    break  # Break after the first separator is found and used

        country_minerals_dict[country] = minerals

    return country_minerals_dict

def find_countries_by_distributions(distributions, return_countries_only=False):
    """
    Aggregates countries based on their mineral distributions across different
    regions. The function can return either a list of countries per region
    or a detailed mapping of minerals to countries within each region,
    depending on the `return_countries_only` flag.

    Parameters
    ----------
    distributions : dict
        A dictionary where keys are region names and values are lists of 
        minerals significant to those regions. The structure should strictly 
        follow {'RegionName': ['mineral1', 'mineral2', ...]} format.
    return_countries_only : bool, optional
        Determines the format of the returned data. If True, the function 
        returns a list of countries for each region that produce any of the 
        specified minerals. If False, the function provides a detailed 
        mapping of each mineral to its producing countries within the region. 
        The default is False.

    Returns
    -------
    dict
        Depending on `return_countries_only`, returns:
        - If False (default): A dictionary with regions as keys and dictionaries
          as values, where each key-value pair within the inner dictionary 
          maps a mineral to a list of countries producing it in that region.
        - If True: A dictionary with regions as keys and lists of countries 
          as values, each list representing countries in that region producing
          any of the specified minerals.

    Raises
    ------
    ValueError
        If `distributions` is not a dictionary or if any of the values in the
        `distributions` dictionary are not lists, indicating an incorrect
        format of the input data.

    Examples
    --------
    >>> from gofast.datasets.util import find_countries_by_distributions
    >>> distributions = {
    ...     "Oceania": ["iron ore", "gold"],
    ...     "Africa": ["diamonds", "gold"]
    ... }
    >>> find_countries_by_distributions(distributions, return_countries_only=True)
    {'Oceania': ['Australia', 'Papua New Guinea', 'New Zealand'],
     'Africa': ['South Africa', 'Botswana', 'Ghana', 'Zimbabwe']}

    >>> find_countries_by_distributions(distributions, return_countries_only=False)
    {'Oceania': {'iron ore': ['Australia'], 
                 'gold': ['Australia', 'Papua New Guinea', 'New Zealand']},
     'Africa': {'diamonds': ['Botswana', 'South Africa'], 
                'gold': ['South Africa', 'Ghana', 'Zimbabwe']}}

    Notes
    -----
    The function is designed to process structured data where regions are
    mapped to their significant minerals. This information is then used to
    aggregate country-level data based on regional mineral distributions.
    The actual country and mineral data processing is handled by the
    `find_countries_and_minerals_by_region` function, which is not defined
    in this snippet.
    """
    if not isinstance(distributions, dict):
        raise ValueError("Distributions must be a dictionary with regions as keys"
                         " and lists of minerals as values.")

    region_countries = defaultdict(list)  # More Pythonic and clear initialization

    for region, minerals_list in distributions.items():
        # Ensure minerals_list is actually a list for consistency
        if not isinstance(minerals_list, list):
            raise ValueError(f"Minerals for region '{region}' must be provided as a list.")

        # Simplified logic by eliminating redundancy
        countries_or_mineral_countries = find_countries_and_minerals_by_region(
            region, minerals=minerals_list,
            return_countries_only=return_countries_only,
        )
        # Directly use the result without unnecessary list wrapping
        region_countries[region] = countries_or_mineral_countries
    # Convert defaultdict back to dict for the return value, if necessary
    return dict(region_countries)  
            
def find_countries_by_minerals1(
    minerals, 
    mineral_prod_by_country=None, 
    return_countries_only=False, 
    return_minerals_and_countries=False, 
    ):
    """
    Finds countries that produce specified minerals, either returning a list
    of countries or a mapping of minerals to countries based on the input
    parameters. It can accept a single mineral or a list of minerals.

    Parameters
    ----------
    minerals : str or list of str
        The name(s) of the mineral(s) to find the producing countries for.
        Can be a single mineral name as a string or a list of mineral names.
    mineral_prod_by_country : dict, optional
        A dictionary mapping country names to a string of minerals they produce,
        separated by commas. If not provided, the function attempts to use
        a default global dictionary. Default is None.
    return_countries_only : bool, optional
        If True, the function returns a list of unique countries that produce
        any of the specified minerals. If False, returns a dictionary mapping
        each specified mineral to a list of countries that produce it.
        Default is False.

    Returns
    -------
    list or dict
        If return_countries_only is False, returns a dictionary where each key
        is a mineral, and each value is a list of countries that produce that
        mineral. If return_countries_only is True, returns a list of unique
        countries that produce any of the specified minerals.

    Raises
    ------
    ImportError
        If `mineral_prod_by_country` is not provided and the global dictionary
        `MINERAL_PROD_BY_COUNTRY` cannot be imported.

    Examples
    --------
    >>> from gofast.datasets.util import find_countries_by_minerals
    >>> find_countries_by_minerals('Gold')
    {'gold': ['Australia', 'South Africa', 'Canada']}

    >>> find_countries_by_minerals(['Gold', 'Silver'], return_countries_only=True)
    ['Australia', 'South Africa', 'Canada', 'Mexico']

    Providing a dictionary directly and looking for multiple minerals:

    >>> minerals = ['coal', 'rare earth elements']
    >>> mineral_data = {'Australia': 'iron ore, lithium', 
    ...                 'China': 'coal, rare earth elements'}
    >>> find_countries_by_minerals(minerals, mineral_prod_by_country=mineral_data)
    {'coal': ['China'], 'rare earth elements': ['China']}
    """
    # Load the mineral production data if not provided
    if mineral_prod_by_country is None:
        mineral_prod_by_country = get_mineral_production_by_country()
    
    # Normalize minerals input to a set of lowercase mineral names
    normalized_minerals = {min.lower() for min in minerals} if isinstance(
        minerals, list) else {minerals.lower()}

    # Extract mineral production information from countries
    countries_minerals = extract_minerals_from_countries(
        mineral_prod_by_country, split_minerals=True)
    
    # Initialize containers for results
    countries = set()
    mineral_by_country = defaultdict(list)
    
    # Loop through the country-mineral data to populate results
    for country, country_minerals in countries_minerals.items():
        country_mineral_set = set(m.lower() for m in country_minerals)
        intersecting_minerals = normalized_minerals & country_mineral_set
        if intersecting_minerals:
            countries.add(country)
            for mineral in intersecting_minerals:
                mineral_by_country[mineral].append(country)
    
    # Error handling for no matches found
    if not countries:
        message = ( 
            "No countries found producing the specified minerals:"
            f" {', '.join(normalized_minerals)}. Consider providing your"
            " own mineral production dictionary."
         )
        raise ValueError(message)
    
    # Decide return value based on parameters
    if return_countries_only:
        return list(countries)
    elif return_minerals_and_countries:
        return list(countries), dict(mineral_by_country)
    else:
        return dict(mineral_by_country)
       
        
def find_mineral_by_country(country, mineral_prod_by_country=None):
    """
    Finds and returns the mineral production data for a given country, 
    allowing for case-insensitive matching of country names. If the country 
    is not found within the provided or default mineral production data, 
    an informative error is raised.

    Parameters
    ----------
    country : str
        The name of the country for which to find mineral production data.
    mineral_prod_by_country : dict, optional
        A dictionary mapping country names to their mineral production data. 
        If not provided, the function attempts to use a default global 
        dictionary. If the global dictionary is also unavailable, 
        an ImportError is raised.

    Returns
    -------
    dict
        The mineral production data for the specified country.

    Raises
    ------
    ImportError
        If `mineral_prod_by_country` is not provided and the global dictionary
        `MINERAL_PROD_BY_COUNTRY` cannot be imported.
    ValueError
        If the specified country is not found in the mineral production data.

    Examples
    --------
    Assuming a global dictionary `MINERAL_PROD_BY_COUNTRY` is available 
    and contains relevant data:

    >>> find_mineral_by_country('Canada')
    {'Gold': 100, 'Silver': 200}

    Providing a dictionary directly:

    >>> mineral_data = {'USA': {'Coal': 500}, 'Canada': {'Gold': 100, 'Silver': 200}}
    >>> find_mineral_by_country('canada', mineral_prod_by_country=mineral_data)
    {'Gold': 100, 'Silver': 200}

    Case where the country is not found, illustrating the error message:

    >>> find_mineral_by_country('Atlantis')
    ValueError: Country 'Atlantis' not found in the list of minerals. 
    Sample countries include: Canada, USA, France, Germany, Japan.

    Note: The function is case-insensitive and will normalize the country name.
    """
    if mineral_prod_by_country is None:
        mineral_prod_by_country = get_mineral_production_by_country()

    # Transform the country names in the dictionary to lowercase once
    country_minerals_dict = {k.lower(): v for k, v in extract_minerals_from_countries(
        mineral_prod_by_country, split_minerals=False).items()}
    
    # Perform a case-insensitive lookup by converting the input country to lowercase
    country_lower = country.lower()
    if country_lower not in country_minerals_dict:
        # Generate an informative error message
        # Just as an example, take the first 5 countries
        sample_countries = list(country_minerals_dict.keys())[:5]  
        sample_countries_str = ', '.join(sample_countries)
        raise ValueError(f"Country '{country}' not found in the list of minerals. "
                         f"Sample countries include: {sample_countries_str}.")
    
    return country_minerals_dict.get(country_lower)

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
    countries = find_countries_by_region(region)
    
    # If no custom dictionary is provided, use a default
    mineral_info = extract_minerals_from_countries(
        mineral_prod_by_country, split_minerals=False )
    
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


def find_countries_and_minerals_by_region (
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
    >>> from gofast.datasets.util import find_countries_and_minerals_by_region
    >>> find_countries_and_minerals_by_region('Europe', minerals=['gold', 'nickel'], 
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

def generate_ore_infos(countries=None, raise_error='warn'):
    """
    Generates informational summaries about ore reserves and production
    capabilities from a predefined or provided dictionary of mineral
    production by country. This function allows for case-insensitive
    country name comparison and offers flexible error handling.

    Parameters
    ----------
    countries : str or list of str, optional
        A country name or a list of country names for which to generate ore
        information. If None, information for all countries in the
        mineral production dictionary will be generated. Names are
        case-insensitive.
    raise_error : {'warn', 'raise', 'ignore'}, optional
        Specifies how to handle countries not found in the mineral production
        dictionary. 'warn' issues a warning, 'raise' raises a ValueError, and
        'ignore' does nothing. Default is 'warn'.

    Returns
    -------
    dict
        A dictionary where each key is a country name and each value is a
        string summarizing the last two provided information strings about
        mineral production and reserves for that country.

    Raises
    ------
    ImportError
        If the global mineral production dictionary cannot be imported and no
        alternative dictionary is provided.
    ValueError
        If `raise_error` is set to 'raise' and one or more specified countries
        are not found in the mineral production dictionary.

    Examples
    --------
    Assuming `MINERAL_PROD_BY_COUNTRY` is available and formatted correctly:

    >>> generate_ore_infos(['Australia', 'China'])
    {'Australia': 'High production capacity for iron ore, lithium and major\\
     exporter of lithium, iron ore.',
     'China': 'World's top producer of several minerals including rare earths,\\
         coal and significant exporter of rare earth elements, coal.'}

    Specifying a single country and using 'warn' for not found countries:

    >>> generate_ore_infos('Atlantis', raise_error='warn')
    Warning: Countries not found: Atlantis.
    {}

    Providing a list of countries with mixed case and requesting 'raise' for errors:

    >>> generate_ore_infos(['australia', 'Atlantis'], raise_error='raise')
    ValueError: Countries not found: Atlantis.
    """
    mineral_prod_by_country = get_mineral_production_by_country()
    # Normalize countries input to a list of lowercase names
    # for case-insensitive comparison
    if countries is not None:
        countries = [countries.lower()] if isinstance(countries, str) else [
            country.lower() for country in countries]
    else:
        countries = list(mineral_prod_by_country.keys())
        
    country_infos = defaultdict(str)
    for country, info_list in mineral_prod_by_country.items():
        # Skip countries not in the provided list, if applicable
        if countries and country.lower() not in countries:
            continue
        
        # Process the last two items from the info list
        if len(info_list) >= 2:
            infos = " and ".join(info_list[-2:])  # Join the last two descriptions
            infos = infos.replace("[", "").replace("]", "")  # Remove square brackets
            country_infos[country] = infos.capitalize()  # Capitalize the first letter
        else:
            # Handle countries with less than 2 info strings differently
            # This is just an example, adjust based on actual requirements
            country_infos[country] = info_list[0] if info_list else "No information available."

    # Check for any countries provided but not found in the dictionary
    not_found_countries = [country for country in countries if country.lower() 
                           not in [c.lower() for c in country_infos]]
    if not_found_countries and raise_error != 'ignore':
        message = f"Countries not found: {', '.join(not_found_countries)}."
        if raise_error == 'warn':
            print(f"Warning: {message}")
        elif raise_error == 'raise':
            raise ValueError(message)

    return dict(country_infos)

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

def get_mineral_production_by_country():
    """
    Attempts to import and return a copy of the MINERAL_PROD_BY_COUNTRY 
    dictionary from the ._globals module. If the import fails, it raises 
    an ImportError with a message indicating that the dictionary is not found 
    and suggesting to provide a mineral production dictionary.

    Returns
    -------
    dict
        A copy of the MINERAL_PROD_BY_COUNTRY dictionary.

    Raises
    ------
    ImportError
        If MINERAL_PROD_BY_COUNTRY cannot be found in the ._globals module.
    """
    try:
        from ._globals import MINERAL_PROD_BY_COUNTRY
        return MINERAL_PROD_BY_COUNTRY.copy()
    except ImportError:
        raise ImportError("MINERAL_PROD_BY_COUNTRY not found. Please provide"
                          " a mineral production dictionary.")




