# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio
"""
Created on Thu Dec 21 14:43:09 2023
@author: a.k.a Daniel
"""
from __future__ import annotations 

import pandas as pd
import numpy as np
import random
from datetime import timedelta
from sklearn.model_selection import train_test_split 
from ..tools.box import Boxspace 
from ..tools.funcutils import ( 
    ellipsis2false ,
    assert_ratio,
    is_iterable, 
    is_in_if , 
    _assert_all_types, 
    add_noises_to
    ) 

def make_african_demo(*, 
    start_year=1960,
    end_year=2020, 
    countries= None, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a dataset for African demography from 1960 to the present.

    This function generates a DataFrame with demographic data for specified 
    African countries from a start year to an end year. It randomly 
    simulates population size, birth rate, death rate, urbanization rate, 
    and GDP per capita for each country and year. The generated data should 
    be used for simulation or testing purposes only, as it does not represent 
    real demographic statistics
    
    Parameters
    ----------
    start_year : int
        The starting year for the dataset (e.g., 1960).

    end_year : int
        The ending year for the dataset.

    countries : int or list of str
        A list of African countries to include in the dataset.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated demographic dataset.
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
        
    Example
    -------
    >>> from gofast.datasets import make_african_demo
    >>> start_year = 1960
    >>> end_year = 2020
    >>> countries = ['Nigeria', 'Egypt', 'South Africa']
    >>> demography_data = make_african_demo(start_year, end_year, countries)
    >>> print(demography_data.head())

    """ 
    # Random seed for reproducibility
    np.random.seed(seed); data = []
    # check the given data 
    start_year = int (_assert_all_types(start_year, int, float, str, 
                      objname="'start_name' parameter "))
    end_year = int (_assert_all_types(end_year, int, float, str, 
                      objname="'start_name' parameter "))
    
    countries = _get_item_from ( countries, AFRICAN_COUNTRIES, 7  )
    for year in range(start_year, end_year + 1):
        for country in countries:
            population = np.random.randint(1e6, 2e8)  # Random population
            birth_rate = np.random.uniform(20, 50)  # Births per 1000 people
            death_rate = np.random.uniform(5, 20)  # Deaths per 1000 people
            urbanization_rate = np.random.uniform(10, 85)  # Percentage of urban population
            gdp_per_capita = np.random.uniform(500, 20000)  # USD

            data.append([country, year, population, birth_rate, death_rate,
                         urbanization_rate, gdp_per_capita])

    columns = ['Country',
               'Year', 
               'Population', 
               'BirthRate', 
               'DeathRate', 
               'UrbanizationRate', 
               'GDP_PerCapita'
               ]
    demo_data = pd.DataFrame(data, columns=columns)
  
    tnames = list( is_iterable(
        tnames or 'GDP_PerCapita', exclude_string= True, transform =True ) ) 
    
    demo_data = _manage_data(
        demo_data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size,
        noises= noises, 
        seed=seed
        ) 

    return demo_data

def make_agronomy_feedback(
    *, 
    samples=100, 
    num_years=5, 
    n_specimens:int =7, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates an agronomy dataset including information about 
    crop cultivation and pesticide usage.

    This function generates a DataFrame with data for multiple farms over 
    several years, including details about the type of crop grown, soil pH, 
    temperature, rainfall, types and amounts of pesticides used, and crop yield.

    The generated data should be used for simulation or testing purposes only. 
    In real-world agronomy studies, data collection would involve more 
    detailed and precise measurements, and the interaction between these 
    variables can be quite complex.
    Parameters
    ----------
    samples : int
        The number of farm entries in the dataset.

    num_years : int
        The number of years for which data is generated.

    n_specimens: int, 
       Number of differnt crop and pesticide types to include in the 
       dataset.
 
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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated agronomy dataset.
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
        
    Example
    -------
    >>> from gofast.datasets import make_agronomy_feedback
    >>> samples = 100
    >>> num_years = 5
    >>> agronomy_data = make_agronomy_feedback(samples, num_years, n_specimens=3)
    >>> print(agronomy_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    n_specimens = int(_assert_all_types(n_specimens, int, float,
                objname='The number of specimens (crop and pesticides)')
        )
    
    pesticide_types = random.choice(COMMON_PESTICIDES, n_specimens)
    crop_types = random.choice(COMMON_CROPS, n_specimens)
    data = []
    for entry_id in range(samples):
        for year in range(num_years):
            for crop in crop_types:
                # Soil pH value
                soil_ph = np.random.uniform(4.0, 9.0)  
                # Average annual temperature in Celsius
                temperature = np.random.uniform(10, 35) 
                # Annual rainfall in mm
                rainfall = np.random.uniform(200, 2000)  
                pesticide = random.choice(pesticide_types)
                # Pesticide amount in kg/hectare
                pesticide_amount = np.random.uniform(0.1, 10) 
                # Crop yield in kg/hectare
                crop_yield = np.random.uniform(100, 10000)  

                data.append([entry_id, 
                             year, 
                             crop, 
                             soil_ph, 
                             temperature, 
                             rainfall, 
                             pesticide, 
                             pesticide_amount, 
                             crop_yield]
                            )

    columns = ['FarmID', 
               'Year', 
               'Crop', 
               'SoilPH', 
               'Temperature_C', 
               'Rainfall_mm', 
               'PesticideType', 
               'PesticideAmount_kg_per_hectare', 
               'CropYield_kg_per_hectare'
               ]
    
    agronomy_dataset = pd.DataFrame(data, columns=columns)
    tnames = list( is_iterable(
        tnames or 'CropYield_kg_per_hectare',exclude_string= True,
        transform =True ) ) 
    agronomy_dataset = _manage_data(
        agronomy_dataset,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size, 
        noises= noises, 
        seed=seed
        ) 
    return agronomy_dataset

def make_mining_ops(
    *, 
    samples=1000, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a dataset for mining operations.
    
    This function generates a DataFrame with a mix of simulated data that 
    reflects various aspects of a mining operation. The features include 
    geospatial coordinates for drilling, types and concentrations of ore, 
    details of drilling and blasting operations, information about mining 
    equipment, and daily production figures.

    The generated data is random and should be used for simulation or 
    testing purposes only. In real-world mining operations, the data would 
    be much more complex and subject to various operational and environmental 
    factors.
    
    - Easting_m: Represents the eastward measurement (in meters) in a 
      geospatial coordinate system. It is often used in conjunction with 
      northing to pinpoint locations in a mining area.

    - Northing_m: Similar to easting, this is the northward measurement 
      (in meters) in the geospatial coordinate system. Easting and northing 
      together provide a precise location for drilling and other mining 
      activities.
    
    - Depth_m: The depth (in meters) at which the mining or drilling operation 
      is taking place. In a mining context, this could refer to the depth 
      of a drill hole or the depth at a mining site.
    
    - OreType: The type of ore being mined or surveyed. Different ore types 
      (e.g., Type1, Type2, Type3) might have different mineral compositions 
      and economic values.
    
    - OreConcentration_Percent: The percentage concentration of the ore in 
      a given sample. Higher concentrations often indicate more valuable 
      mining sites.
    
    - DrillDiameter_mm: The diameter of the drill bit used in drilling 
      operations, measured in millimeters. This affects the size of the 
      drill hole and is selected based on the mining requirements.
    
    - BlastHoleDepth_m: The depth of the blast holes used in blasting 
      operations, measured in meters. This depth is a crucial factor in 
      determining the effectiveness of blasting.
    
    - ExplosiveType: The type of explosive material used in blasting. 
      Different explosives (e.g., Explosive1, Explosive2, Explosive3) have 
      varying strengths and are chosen based on the blasting requirements.
    
    - ExplosiveAmount_kg: The amount of explosive used per blast, measured 
      in kilograms. This quantity is critical for ensuring the blast is 
      effective yet safe.
    
    - EquipmentType: The type of equipment being used in the mining 
      operation (e.g., Excavator, Drill, Loader, Truck). Different equipment 
      is used for different phases of mining.
    
    - EquipmentAge_years: The age of the equipment being used, in years. 
      Older equipment might be less efficient or more prone to breakdowns.
    
    - DailyProduction_tonnes: The amount of material (in tonnes) produced by 
      the mining operation each day. This is a direct measure of mining 
      productivity.
    
    Each of these features contributes to a comprehensive view of the mining 
    operation, providing insights into the geospatial aspects, the nature 
    of the resources being extracted, the methods and tools used in the 
    extraction process, and the output of the operation.

    Parameters
    ----------
    samples : int
        The number of entries (rows) in the dataset.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated mining dataset.
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
        
    Example
    -------
    >>> from gofast.datasets import make_mining_ops
    >>> samples = 1000
    >>> mining_data = make_mining_ops(samples)
    >>> print(mining_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    # Geospatial data for drilling locations
    eastings = np.random.uniform(0, 1000, samples)  # in meters
    northings = np.random.uniform(0, 1000, samples)  # in meters
    depths = np.random.uniform(0, 500, samples)  # in meters

    # Mineralogical data
    ore_types = np.random.choice(ORE_TYPE.keys(), samples)
    ore_concentrations = np.random.uniform(0.1, 20, samples)  # percentage

    # Drilling and blasting data
    drill_diameters = np.random.uniform(50, 200, samples)  # in mm
    blast_hole_depths = np.random.uniform(3, 15, samples)  # in meters
    explosive_types = np.random.choice(EXPLOSIVE_TYPE.keys(), samples)
    explosive_amounts = np.random.uniform(10, 500, samples)  # in kg

    # Equipment details
    equipment_types = np.random.choice(EQUIPMENT_TYPE, samples)
    equipment_ages = np.random.randint(0, 15, samples)  # in years

    # Production figures
    daily_productions = np.random.uniform(1000, 10000, samples)  # in tonnes

    # Construct the DataFrame
    mining_data = pd.DataFrame({
        'Easting_m': eastings,
        'Northing_m': northings,
        'Depth_m': depths,
        'OreType': ore_types,
        'OreConcentration_Percent': ore_concentrations,
        'DrillDiameter_mm': drill_diameters,
        'BlastHoleDepth_m': blast_hole_depths,
        'ExplosiveType': explosive_types,
        'ExplosiveAmount_kg': explosive_amounts,
        'EquipmentType': equipment_types,
        'EquipmentAge_years': equipment_ages,
        'DailyProduction_tonnes': daily_productions
    })
    tnames = list (is_iterable ( 
        tnames or 'DailyProduction_tonnes',exclude_string= True, transform =True )
        )
    # map to make it a little bit real.
    for typ, rtype  in zip ( ("OreType", "ExplosiveType"), 
                       (ORE_TYPE, EXPLOSIVE_TYPE )) : 
        mining_data[typ] = mining_data[typ].map (rtype) 
        
    mining_data = _manage_data(
        mining_data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size,
        noises= noises, 
        seed=seed, 
        ) 
    return mining_data

def make_sounding(
    *, samples=100, 
    num_layers=5, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a complex dataset for geophysical sounding, 
    typically used in ERT or seismic surveys.

    This function generates a DataFrame with data for multiple survey points, 
    each with a specified number of subsurface layers. For each layer, 
    the function simulates:

    - LayerDepth_m: The depth of each layer in meters.
    - Resistivity_OhmMeter: The electrical resistivity of each layer in 
      ohm-meters, a key parameter in ERT surveys.
    - SeismicVelocity_m_s: The seismic wave velocity through each layer 
      in meters per second, a parameter often measured in seismic refraction 
      surveys.
    - The dataset can be used for simulations or testing algorithms related 
      to geophysical sounding analysis. Each row in the dataset represents 
      a unique combination of survey point and layer with associated 
      geophysical properties.
      
    Parameters
    ----------
    samples : int
        The number of survey points (entries) in the dataset.

    num_layers : int
        The number of subsurface layers to simulate for each survey point.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated geophysical sounding dataset.
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
        
    Example
    -------
    >>> from gofast.datasets import make_sounding
    >>> samples = 100
    >>> num_layers = 5
    >>> sounding_data = make_sounding(samples, num_layers)
    >>> print(sounding_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    # Initializing lists to hold data
    survey_point_ids = []
    layer_depths = []
    resistivities = []
    velocities = []

    for point_id in range(samples):
        depth = 0
        for layer in range(num_layers):
            # Simulate layer depth increment and properties
            depth += np.random.uniform(1, 10)  # Depth increment in meters
            resistivity = np.random.uniform(10, 1000)  # Resistivity in ohm-meter
            velocity = np.random.uniform(500, 5000)  # Seismic wave velocity in m/s

            survey_point_ids.append(point_id)
            layer_depths.append(depth)
            resistivities.append(resistivity)
            velocities.append(velocity)

    # Constructing the DataFrame
    sounding_data = pd.DataFrame({
        'SurveyPointID': survey_point_ids,
        'LayerDepth_m': layer_depths,
        'Resistivity_OhmMeter': resistivities,
        'SeismicVelocity_m_s': velocities
    })

    tnames = list (is_iterable ( 
        tnames or 'Resistivity_OhmMeter',exclude_string= True, transform =True )
        )
    return _manage_data(
        sounding_data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size, 
        noises= noises, 
        seed=seed
        ) 

def make_medical_diagnosis(
    *,samples=1000, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a  medical dataset with  different features.

    This function creates a DataFrame with a diverse set of medical features. 
    The data is generated randomly and should be used for simulation or 
    testing purposes only. In a real-world scenario, medical datasets are 
    subject to strict privacy regulations and must be handled accordingly.

    The features listed are just examples, and you can modify or extend 
    them based on specific medical domains or research requirements. To reach 
    55 features for instance, you would need to continue adding more columns 
    with relevant medical data, ensuring a mix of demographic information, 
    vital signs, test results, and medical history.
    
    Parameters
    ----------
    samples : int
        The number of entries (patients) in the dataset.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
      
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated medical dataset.
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
        
    Example
    -------
    >>> from gofast.datasets import make_medical_diagnosis
    >>> samples = 1000
    >>> medical_data = make_medical_diagnosis(samples)
    >>> print(medical_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    # Demographic information
    ages = np.random.randint(0, 100, samples)
    genders = np.random.choice(['Male', 'Female'], samples)
    
    ethnicities = np.random.choice(WATER_QUAL_NEEDS['Ethnicity'], samples)
    weights = np.random.uniform(50, 120, samples)  # in kilograms
    heights = np.random.uniform(150, 200, samples)  # in centimeters

    # Vital signs
    blood_pressures = np.random.randint(90, 180, size=(samples, 2)) 
    # systolic and diastolic
    heart_rates = np.random.randint(60, 100, samples)
    temperatures = np.random.uniform(36.5, 38.0, samples)  # in Celsius

    # Laboratory test results
    blood_sugars = np.random.uniform(70, 150, samples)  # mg/dL
    cholesterol_levels = np.random.uniform(100, 250, samples)  # mg/dL
    hemoglobins = np.random.uniform(12, 18, samples)  # g/dL
    # ... (more lab tests)

    # Medical history flags (binary: 0 or 1)
    history_of_diabetes = np.random.randint(0, 2, samples)
    history_of_hypertension = np.random.randint(0, 2, samples)
    history_of_heart_disease = np.random.randint(0, 2, samples)
    # ... (more medical history flags)

    # Additional clinical metrics
    # ...
    # (add more features to reach at least 55 in total)

    # Combining all features into a DataFrame
    medical_dataset = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
        'Ethnicity': ethnicities,
        'Weight_kg': weights,
        'Height_cm': heights,
        'SystolicBP': blood_pressures[:, 0],
        'DiastolicBP': blood_pressures[:, 1],
        'HeartRate': heart_rates,
        'Temperature_C': temperatures,
        'BloodSugar_mg_dL': blood_sugars,
        'Cholesterol_mg_dL': cholesterol_levels,
        'Hemoglobin_g_dL': hemoglobins,
        # ...
        'HistoryOfDiabetes': history_of_diabetes,
        'HistoryOfHypertension': history_of_hypertension,
        'HistoryOfHeartDisease': history_of_heart_disease,
        # ...
        # Add additional columns for other features
    })
    tnames = list (is_iterable (tnames or [
        'HistoryOfDiabetes','HistoryOfHypertension','HistoryOfHeartDisease'],
        exclude_string= True, transform =True )
        )
    return _manage_data(
        medical_dataset,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size,
        noises= noises, 
        seed=seed
        ) 


def make_well_logging(*, 
    depth_start=0., 
    depth_end=200., 
    depth_interval=.5, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a synthetic dataset for geophysical well logging.

    This function creates a DataFrame that simulates typical well logging data, 
    often used in subsurface geological investigations. Each row represents 
    a set of measurements at a specific depth, with the depth intervals 
    defined by the user. The measurements for gamma-ray, resistivity, 
    neutron porosity, and density are generated using random values 
    within typical ranges, but they can be adjusted or extended to include 
    additional logging parameters as needed.
    
    Parameters
    ----------
    depth_start : float
        The starting depth for the well logging in meters.

    depth_end : float
        The ending depth for the well logging in meters.

    depth_interval : float
        The interval between depth measurements in meters.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated well logging dataset.
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
        
    Example
    -------
    >>> from gofast.datasets import make_well_logging
    >>> depth_start = 0.0
    >>> depth_end = 200.0
    >>> depth_interval = 0.5
    >>> well_logging_data = make_well_logging(depth_start, depth_end, depth_interval)
    >>> print(well_logging_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    depths = np.arange(depth_start, depth_end, depth_interval)

    # Simulating geophysical measurements
    gamma_ray = np.random.uniform(20, 150, len(depths))  # Gamma-ray (API units)
    resistivity = np.random.uniform(0.2, 200, len(depths))  # Resistivity (ohm-m)
    neutron_porosity = np.random.uniform(15, 45, len(depths))  # Neutron porosity (%)
    density = np.random.uniform(1.95, 2.95, len(depths))  # Bulk density (g/cm³)

    # Construct the DataFrame
    well_logging_dataset = pd.DataFrame({
        'Depth_m': depths,
        'GammaRay_API': gamma_ray,
        'Resistivity_OhmMeter': resistivity,
        'NeutronPorosity_Percent': neutron_porosity,
        'Density_g_cm3': density
    })
    tnames = list (is_iterable ( 
        tnames or 'NeutronPorosity_Percent', exclude_string= True,
        transform =True )
        )

    return _manage_data(
        well_logging_dataset,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size, 
        noises= noises, 
        seed=seed
        ) 


def make_ert(
    *, 
    samples=100, 
    equipment_type='SuperSting R8', 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a dataset for electrical resistivity tomography (ERT) 
    based on the specified equipment type.


    This function creates a DataFrame with synthetic data representing an 
    ERT survey. The data includes electrode positions, cable lengths, 
    resistivity measurements, and battery voltages 
    (when applicable, depending on the equipment type). The equipment_type 
    parameter allows you to specify the type of ERT equipment, and the generated 
    dataset reflects characteristics that might be associated with that 
    equipment. The function raises an error if an unrecognized equipment 
    type is specified.
   
    Parameters
    ----------
    samples : int
        The number of entries (rows) in the dataset.

    equipment_type : str
        The type of ERT equipment used. Should be one of 
        'SuperSting R8', 'Ministing or Sting R1', or 'OhmMapper'.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated ERT dataset.
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
        
    Raises
    ------
    ValueError
        If the equipment_type is not recognized.

    Example
    -------
    >>> from gofast.datasets import make_ert
    >>> samples = 100
    >>> equipment_type = 'SuperSting R8'
    >>> ert_data = make_ert(samples, equipment_type)
    >>> print(ert_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    if equipment_type not in ['SuperSting R8', 'Ministing or Sting R1', 
                              'OhmMapper']:
        raise ValueError("equipment_type must be one of 'SuperSting R8'," 
                         "'Ministing or Sting R1', or 'OhmMapper'")

    # Generate synthetic data
    electrode_positions = np.random.uniform(0, 100, samples)  # in meters
    cable_lengths = np.random.choice([20, 50, 100], samples)  # in meters
    resistivity_measurements = np.random.uniform(10, 1000, samples)  # in ohm-meter
    battery_voltage = np.random.choice(
        [12], samples) if equipment_type != 'OhmMapper' else np.nan  # in V

    # Construct the DataFrame
    ert_dataset = pd.DataFrame({
        'ElectrodePosition_m': electrode_positions,
        'CableLength_m': cable_lengths,
        'Resistivity_OhmMeter': resistivity_measurements,
        'BatteryVoltage_V': battery_voltage,
        'EquipmentType': equipment_type
    })
    tnames = list (is_iterable ( 
        tnames or 'Resistivity_OhmMeter', exclude_string= True, transform =True )
        )
    return _manage_data(
        ert_dataset,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size, 
        noises= noises, 
        seed=seed
        ) 


def make_tem(
    *, 
    samples=500., 
    lat_range=(34.00, 36.00), 
    lon_range=(-118.50, -117.00), 
    time_range=(0.01, 10.0), 
    measurement_range=(100, 10000), 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a dataset for a Transient Electromagnetic (TEM) 
    survey including equipment types.

    Parameters
    ----------
    samples : int
        The number of entries (rows) in the dataset.

    lat_range : tuple of float
        The range of latitude values (min_latitude, max_latitude).

    lon_range : tuple of float
        The range of longitude values (min_longitude, max_longitude).

    time_range : tuple of float
        The range of time intervals in milliseconds after the pulse (min_time, max_time).

    measurement_range : tuple of float
        The range of TEM measurements (min_measurement, max_measurement).

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated TEM survey dataset with equipment types.
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
        
    Example
    -------
    >>> from gofast.datasets import make_tem
    >>> samples = 500
    >>> lat_range = (34.00, 36.00)
    >>> lon_range = (-118.50, -117.00)
    >>> time_range = (0.01, 10.0) # milliseconds
    >>> measurement_range = (100, 10000) # arbitrary units
    >>> tem_data = make_tem(
        samples, lat_range, lon_range, time_range, measurement_range)
    >>> print(tem_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    # Equipment types
    equipment_types = [
        'Stratagem EH5-Geometric', 'IRIS Remote Field Probes',
        'Phoenix Atlas RTM System', 'Zonge GDP 24-bit Receiver']

    # Generate random geospatial data
    latitudes = np.random.uniform(lat_range[0], lat_range[1], samples)
    longitudes = np.random.uniform(lon_range[0], lon_range[1], samples)

    # Generate time intervals, measurements, and equipment types
    times = np.random.uniform(time_range[0], time_range[1], samples)
    measurements = np.random.uniform(measurement_range[0], measurement_range[1], samples)
    equipment = np.random.choice(equipment_types, samples)

    # Construct the DataFrame
    tem_survey_data = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Time_ms': times,
        'TEM_Measurement': measurements,
        'EquipmentType': equipment
    })
    tnames = list (is_iterable ( 
        tnames or 'TEM_Measurement', exclude_string= True, transform =True )
        )
    return _manage_data(
        tem_survey_data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size,
        noises= noises, 
        seed=seed
        ) 

def make_erp(*, 
    samples=1000, 
    lat_range=(34.00, 36.00), 
    lon_range =(-118.50, -117.00), 
    resistivity_range=(10, 1000),
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a dataset for geophysical analysis with easting, northing,
    longitude, latitude, positions, step, and resistivity values.

    This function creates a DataFrame with synthetic geospatial data. 
    The easting and northing values are derived from the latitude and 
    longitude values for simplicity. In real-world applications, conversions
    between these coordinate systems are more complex and often require 
    specific geospatial libraries or tools. The 'Position' and 'Step' 
    columns simulate sequential survey data, and the 'Resistivity' column 
    provides random resistivity values within the specified range. 
    The function can be customized to match specific requirements of 
    geophysical surveys or similar applications.
    
    Parameters
    ----------
    samples : int
        The number of entries (rows) in the dataset.

    lat_range : tuple of float
        The range of latitude values (min_latitude, max_latitude).

    lon_range : tuple of float
        The range of longitude values (min_longitude, max_longitude).

    resistivity_range : tuple of float
        The range of resistivity values (min_resistivity, max_resistivity).
        
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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated dataset.
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
        
    Example
    -------
    >>> from gofast.datasets import make_erp
    >>> samples = 1000
    >>> lat_range = (34.00, 36.00)
    >>> lon_range = (-118.50, -117.00)
    >>> resistivity_range = (10, 1000)
    >>> dataset = make_erp(
        samples, lat_range, lon_range, resistivity_range)
    >>> print(dataset.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    # Generate random geospatial data
    latitudes = np.random.uniform(lat_range[0], lat_range[1], samples)
    longitudes = np.random.uniform(lon_range[0], lon_range[1], samples)

    # Convert lat/lon to easting/northing (simplified, for example purposes)
    eastings = (longitudes - lon_range[0]) * 100000
    northings = (latitudes - lat_range[0]) * 100000

    # Positions and steps
    positions = np.arange(1, samples + 1)
    steps = np.random.randint(1, 10, samples)

    # Generate resistivity values
    resistivities = np.random.uniform(
        resistivity_range[0], resistivity_range[1], samples)

    # Construct the DataFrame
    data = pd.DataFrame({
        'Easting': eastings,
        'Northing': northings,
        'Longitude': longitudes,
        'Latitude': latitudes,
        'Position': positions,
        'Step': steps,
        'Resistivity': resistivities
    })
    tnames = list (is_iterable ( 
        tnames or 'Resistivity', exclude_string= True, transform =True )
        )
    return _manage_data(
        data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size, 
        noises= noises, 
        seed=seed
        ) 


def make_elogging(
    *, 
    start_date='2021-01-01', 
    end_date='2021-01-31', 
    samples=100, 
    log_levels=None, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a dataset of simulated logging data.

    Parameters
    ----------
    start_date : str
        The start date for the logging data in 'YYYY-MM-DD' format.

    end_date : str
        The end date for the logging data in 'YYYY-MM-DD' format.

    samples : int
        The number of log entries to generate.

    log_levels : list of str, optional
        A list of log levels (e.g., ['INFO', 'WARNING', 'ERROR']). 
        If None, defaults to ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'].

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated logging data with columns 
        'Timestamp', 'LogLevel', and 'Message'.
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
        
    Example
    -------
    >>> from gofast.datasets import make_elogging
    >>> start_date = '2021-01-01'
    >>> end_date = '2021-01-31'
    >>> samples = 100
    >>> log_data = make_elogging(start_date, end_date, samples)
    >>> print(log_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    if log_levels is None:
        log_levels = ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']

    # Generate random timestamps within the given range
    start_timestamp = pd.to_datetime(start_date)
    end_timestamp = pd.to_datetime(end_date)
    timestamps = [start_timestamp + timedelta(
        seconds=random.randint(0,int((end_timestamp - start_timestamp
                                      ).total_seconds()))) 
        for _ in range(samples)
        ]

    # Generate random log levels and messages
    levels = [random.choice(log_levels) for _ in range(samples)]
    messages = [f'This is a {level} message.' for level in levels]

    # Create DataFrame
    log_data = pd.DataFrame({'Timestamp': timestamps,
                             'LogLevel': levels, 
                             'Message': messages})
    log_data.sort_values(by='Timestamp', inplace=True)
    
    tnames = list (is_iterable ( 
        tnames or 'LogLevel', exclude_string= True, transform =True )
        )
    return _manage_data(
        log_data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size, 
        noises= noises, 
        seed=seed
        ) 


def make_gadget_sales(*, 
    start_date='2021-12-26', 
    end_date='2022-01-10', 
    samples=500, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a dataset of gadget sales data for girls and boys after the 
    Christmas holiday.

    This function generates a DataFrame with random sales data entries, 
    including the sale date, type of gadget, gender (either 'Girl' or 'Boy'), 
    and the number of units sold. The function allows customization of the 
    date range and the number of entries. The gadget_types and genders 
    lists can be modified to include different categories or more specific 
    items as per your requirements.
    
    Parameters
    ----------
    start_date : str
        The start date for the sales data in 'YYYY-MM-DD' format.

    end_date : str
        The end date for the sales data in 'YYYY-MM-DD' format.

    samples : int
        The number of sales entries to generate.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the gadget sales data with columns 
        'SaleDate', 'Gadget', 'Gender', and 'UnitsSold'.
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
        
    Example
    -------
    >>> from gofast.datasets import make_gadget_sales
    >>> start_date = '2021-12-26'
    >>> end_date = '2022-01-10'
    >>> samples = 100
    >>> sales_data = make_gadget_sales(start_date, end_date, samples)
    >>> print(sales_data.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    gadget_types = ['Smartphone', 'Tablet', 'Laptop', 'Smartwatch', 'Headphones']
    genders = ['Girl', 'Boy']

    # Generate random sale dates within the given range
    start_timestamp = pd.to_datetime(start_date)
    end_timestamp = pd.to_datetime(end_date)
    sale_dates = [start_timestamp + timedelta(days=random.randint(
        0, (end_timestamp - start_timestamp).days)) for _ in range(samples)]

    # Generate random gadget types, genders, and units sold
    gadgets = [random.choice(gadget_types) for _ in range(samples)]
    gender = [random.choice(genders) for _ in range(samples)]
    units_sold = [random.randint(1, 20) for _ in range(samples)]

    # Create DataFrame
    sales_data = pd.DataFrame({'SaleDate': sale_dates, 
                               'Gadget': gadgets, 'Gender': gender, 
                               'UnitsSold': units_sold})
    sales_data.sort_values(by='SaleDate', inplace=True)
    
    tnames = list (is_iterable ( 
        tnames or 'UnitsSold', exclude_string= True, transform =True )
        )
    return _manage_data(
        sales_data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size, 
        noises= noises, 
        seed=seed
        ) 


def make_retail_store(
    *, samples=1000, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    noises: float=None, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generates a retail score dataset for machine learning purposes 
    with mixed data types.
    
    The dataset will simulate a hypothetical scenario, for instance, customer 
    data for a retail store, with features like age, income, shopping frequency, 
    last purchase amount, preferred shopping category, and a binary target 
    variable indicating whether the customer is likely to respond to a new 
    marketing campaign.
    
    This function captures a mix of linear and non-linear relationships 
    between features and the target variable. Such a dataset can be useful 
    for testing various machine learning algorithms, especially those used for 
    classification tasks. Remember, the relationships and distributions 
    here are arbitrary and for demonstration purposes; they might not 
    reflect real-world scenarios accurately.

    Parameters
    ----------
    samples : int
        The number of entries (rows) in the dataset.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
       
    Returns
    -------
    pd.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame containing the generated dataset.
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
    Example
    -------
    >>> from gofast.datasets import make_retail_store
    >>> samples = 1000
    >>> dataset = make_retail_store(samples)
    >>> print(dataset.head())

    """
    # Random seed for reproducibility
    np.random.seed(seed)

    # Generating numerical data
    ages = np.random.randint(18, 70, size=samples)
    incomes = np.random.normal(50000, 15000, samples).clip(20000, 100000)
    shopping_frequency = np.random.randint(1, 10, size=samples)  # frequency per month
    last_purchase_amount = np.random.exponential(100, samples).clip(10, 500)

    # Generating categorical data
    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 
                  'Health & Beauty']
    preferred_category = np.random.choice(categories, size=samples)

    # Generating target variable (binary)
    # Here, we can simulate some complex relationships
    likelihood_to_respond = (0.3 * np.random.normal(size=samples) +
                             0.1 * (ages / 70) +
                             0.2 * (incomes / 100000) +
                             0.3 * (shopping_frequency / 10) -
                             0.1 * (last_purchase_amount / 500))
    target = (likelihood_to_respond > np.random.normal(0.5, 0.1, samples)
              ).astype(int)

    # Construct the DataFrame
    data = pd.DataFrame({
        'Age': ages,
        'Income': incomes,
        'ShoppingFrequency': shopping_frequency,
        'LastPurchaseAmount': last_purchase_amount,
        'PreferredCategory': preferred_category,
        'LikelyToRespond': target
    })

    tnames = list (is_iterable ( 
        tnames or 'Income', exclude_string= True, transform =True )
        )
    return _manage_data(
        data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        test_size=test_size,
        noises= noises, 
        seed=seed
        ) 

def make_cc_factors(
    *, samples=1000,  
    noises=.1, 
    as_frame:bool =..., 
    return_X_y:bool = ..., 
    split_X_y:bool= ..., 
    tnames:list=None,  
    test_size:float =.3, 
    seed:int | np.random.RandomState = None, 
    **kws
    ):
    """
    Generate a synthetic climate change dataset with the most influential 
    features.

    Parameters
    ----------
    samples : int, default=1000.
        Number of samples in the dataset.

    noises : float, default='10%'
        Probability of a value being missing in the dataset, by default 0.1.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pandas.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame representing the climate change dataset.
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
        
    Examples 
    --------
    >>> from gofast.datasets import make_cc_factors
    >>> # Parameters for the dataset
    >>> num_data_points = 1000
    >>> # Generating the climate change dataset with missing values
    >>> climate_change_data = make_cc_factors(num_data_points)
    >>> # Display information about the generated dataset
    >>> print(f"Dataset shape: {climate_change_data.shape}")
    >>> print(f"Sample data:\n{climate_change_data[:5]}")
    """

    # Features influencing climate change
    features= {
        "GHG": "Greenhouse Gas Emissions (CO2, Methane, Nitrous Oxide)",
        "Def&Land": "Deforestation and Land Use Changes",
        "FossilFuels": "Burning of Fossil Fuels (Coal, Oil, Natural Gas)",
        "IndustProc": "Industrial Processes and Manufacturing",
        "Agri&Livestock": "Agriculture and Livestock Farming (Methane from Cattle)",
        "Transport": "Transportation (Road, Air, Maritime)",
        "EnergyProd": "Energy Production and Consumption",
        "Urban&Infra": "Urbanization and Infrastructure Development",
        "WasteMgmt": "Waste Management and Landfills",
        "MeltPolar": "Melting of Polar Ice Caps and Glaciers",
        "ChgLandAlbedo": "Changes in Land Surface Albedo",
        "SolarIrr": "Solar Irradiance and Variability",
        "Aerosols": "Aerosols and Particulate Matter in the Atmosphere",
        "OzoneDepl": "Ozone Depletion in the Stratosphere",
        "ChgOceanCirc": "Changes in Ocean Circulation and Currents",
        "OceanAcid": "Ocean Acidification due to CO2 Absorption",
        "Permafrost": "Permafrost Thawing and Release of Methane",
        "ChgAtmWater": "Changes in Atmospheric Water Vapor",
        "LandDeg&SoilErosion": "Land Degradation and Soil Erosion",
        "HumanAct&Biodiv": "Human Activities Impacting Biodiversity",
        "NatDisasters": "Natural Disasters (Floods, Hurricanes, Wildfires)",
        "Feedbacks": "Feedback Mechanisms (Positive/Negative Climate Feedbacks)"
    }
    # Random seed for reproducibility
    np.random.seed(seed)
    
    # # Generate synthetic data for the features
    # data = np.random.rand(samples, len(features))

    # # Introduce missing values randomly (replace some values with NaN)
    # num_missing = int(np.round(samples * len(features) * missing_percentage))
    # indices_to_replace = np.random.choice(range(samples * len(features)), 
    #                                       num_missing, replace=False)
    # data_flattened = data.flatten()
    # data_flattened[indices_to_replace] = np.nan
    # data = data_flattened.reshape((samples, len(features)))

    # df = pd.DataFrame ( data , columns = features )
    noises = assert_ratio(noises)
    data = np.random.randn(samples, len(features)) * 10  # Random data generation
    missing_mask = np.random.random(size=data.shape) < noises
    data[missing_mask] = np.nan  # Introduce missing values based on the probability
    
    # Generating a Pandas DataFrame
    cc_data = pd.DataFrame(data, columns=list( features.keys()))
    
    tnames = list (is_iterable ( 
        tnames or 'Feedbacks', 
        exclude_string= True, transform =True )
        )
    cc_data = add_noises_to(cc_data, noises = noises )
    
    return _manage_data(
        cc_data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        features_descr=features, 
        test_size=test_size, 
        seed=seed
        ) 


def make_water_demand(
    *, samples=700, 
    as_frame =..., 
    return_X_y = ..., 
    noises:float=None, 
    split_X_y= ..., 
    tnames=None,  
    test_size =.3, 
    seed = None, 
    ):
    """
    Generate a synthetic water demand needs dataset.

    Parameters
    ----------
    samples : int, default=700
        Number of samples or data points in the dataset.

    noises : float, Optional
        Probability of a value being missing in the dataset.

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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns
    -------
    pandas.DataFrame if ``as_frame=True`` and ``return_X_y=False``
        A DataFrame representing the water demand needs dataset.
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
        
     Examples
     --------
     >>> from gofast.datasets import make_water_demand 
     >>> b = make_water_demand ()
     >>> b.frame
     Out[80]: 
          Agri Demand  ...         SDG6_Challenge
     0      25.098231  ...        Water Pollution
     1      34.665733  ...         Water Scarcity
     2      20.573921  ...  Ecosystem Degradation
     3      72.900736  ...         Lack of Access
     4      55.853812  ...  Ecosystem Degradation
     ..           ...  ...                    ...
     695    48.090530  ...         Water Scarcity
     696    97.867586  ...                    NaN
     697    89.923326  ...  Ecosystem Degradation
     698          NaN  ...         Water Scarcity
     699          NaN  ...         Lack of Access

     [700 rows x 39 columns]
    """
    # Random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize an empty dictionary to store data for each feature
    data_dict = {}

    # Generate synthetic data for water needs features
    for feature in WATER_QUAN_NEEDS.keys():
        data_dict[feature] = np.random.uniform(0, 100, samples)

    # Generate synthetic data for categorical features
    WATER_QUAL_NEEDS ["Economic Status"]= np.random.uniform(
        1000, 50000, samples).round(2)

    for feature, possible_values in WATER_QUAL_NEEDS.items():
        # first skip when feature is Region to compute later 
        if feature =='Region': 
            continue 
        
        data_dict[feature] = np.random.choice(possible_values, samples)

    # now get the feature Ehnicity and found 
    # random region where the language is spoken 
    data_dict["Ethnicity"]
    regions =[]
    for language  in data_dict["Ethnicity"]: 
        ethnicity_lang_value = np.random.choice (
            WATER_QUAL_NEEDS["Region"][language], 1 )[0]
        regions.append ( ethnicity_lang_value ) 
    # then create the data "REgion 
    data_dict ["Region"] = regions 

    # Generate synthetic data for SDG6 challenges
    sdg6_challenges = list(SDG6_CHALLENGES.keys())
    data_dict["SDG6_Challenge"] = np.random.choice(sdg6_challenges, samples)

    # Create a DataFrame from the data dictionary
    water_data  = pd.DataFrame(data_dict)

    tnames = list (is_iterable ( 
        tnames or 'Drinking', 
        exclude_string= True, transform =True )
        )
    
    return _manage_data(
        water_data,
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        tnames=tnames, 
        descr= { **WATER_QUAN_NEEDS, **WATER_QUAL_NEEDS,**SDG6_CHALLENGES}, 
        test_size=test_size, 
        noises = noises, 
        seed=seed
        ) 
# --func -utilities -----  

def _manage_data(
    data, /, 
    as_frame =..., 
    return_X_y = ..., 
    split_X_y= ..., 
    tnames=None,  
    test_size =.3, 
    noises=None, 
    seed = None, 
    **kws
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
        
    tnames: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns 
    ---------
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
    y =None, 
    as_frame, return_X_y, split_X_y = ellipsis2false(
        as_frame, return_X_y, split_X_y )
    
    frame = data.copy() 
    if return_X_y : 
        y = data [tnames] 
        data.drop( columns = tnames, inplace =True )
        
    feature_names = (is_in_if(list( frame.columns), tnames, return_diff =True )
                     if tnames else list(frame.columns ))
    
    # Noises only in the data not in target  
    data = add_noises_to(data, noises = noises )
    if not as_frame: 
        data = np.array (data )
        y = np.array(y ) 
            
    if split_X_y: 
        return train_test_split ( 
            data, y , test_size =assert_ratio (test_size),
            random_state=seed) 

    if return_X_y : 
        return data, y 
    
    if as_frame:
        frame [feature_names] = add_noises_to(
            frame [feature_names], noises =noises )
        return frame
    
    return Boxspace(
        data=data,
        target=frame[tnames].values ,
        frame=frame,
        tnames=tnames,
        target_names = tnames,
        feature_names=feature_names,
        )
 
def _get_item_from ( spec , /,  default_items, default_number = 7 ): 
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
    

# --------------------------------GLOBAL DATA ---------------------------------
# Features representing water needs
WATER_QUAN_NEEDS= {
    "Agri Demand": "Agricultural Water Demand",
    "Indus Demand": "Industrial Water Demand",
    "Domestic Demand": "Domestic Water Demand",
    "Municipal Demand": "Municipal Water Demand",
    "Livestock Needs": "Livestock Water Needs",
    "Irrigation Req": "Irrigation Water Requirements",
    "Hydropower Gen": "Hydropower Generation",
    "Aquaculture Usage": "Aquaculture Water Usage",
    "Mining Consumption": "Mining Water Consumption",
    "Thermal Plant Consumption": "Thermal Power Plant Water Consumption",
    "Ecosystems": "Water for Ecosystems",
    "Forestry": "Water for Forestry",
    "Recreation": "Water for Recreation",
    "Urban Dev": "Water for Urban Development",
    "Drinking": "Water for Drinking",
    "Sanitation": "Water for Sanitation",
    "Food Processing": "Water for Food Processing",
    "Textile Industry": "Water for Textile Industry",
    "Paper Industry": "Water for Paper Industry",
    "Chemical Industry": "Water for Chemical Industry",
    "Pharma Industry": "Water for Pharmaceutical Industry",
    "Construction": "Water for Construction",
    "Energy Production": "Water for Energy Production",
    "Oil Refining": "Water for Oil Refining",
    "Metals Production": "Water for Metals Production",
    "Auto Manufacturing": "Water for Automobile Manufacturing",
    "Electronics Manufacturing": "Water for Electronics Manufacturing",
    "Plastics Manufacturing": "Water for Plastics Manufacturing",
    "Leather Industry": "Water for Leather Industry",
    "Beverage Industry": "Water for Beverage Industry",
    "Pulp & Paper Industry": "Water for Pulp and Paper Industry",
    "Sugar Industry": "Water for Sugar Industry",
    "Cement Industry": "Water for Cement Industry",
    "Fertilizer Industry": "Water for Fertilizer Industry",
}

# Define categorical feature values
WATER_QUAL_NEEDS= {
    "Water Quality": ["Excellent",
                      "Good", 
                      "Fair", 
                      "Poor", 
                      "Very Poor",
                      "Toxic", 
                      "Polluted", 
                      "Eutrophic", 
                      "Saline",
                      "Acidic/Alkaline"
                      ],
    "Ethnicity": [
        "English", 
        "Mandarin Chinese", 
        "Spanish", 
        "French", 
        "Arabic", 
        "Hindi",
        "Bengali", 
        "Russian", 
        "Portuguese",
        "Japanese",
        "Swahili", 
        "Hausa",
        "Yoruba",
        "Zulu", 
        "Amharic",
        "Agni",
        "Baoule", 
        "Bron",
        "Asante"
        ],
    "Region": {
        "English": [
            "United States", 
            "United Kingdom",
            "Canada",
            "Australia", 
            "South Africa"
                    ],
        "Mandarin Chinese": [
            "China (Mainland China)",
            "Taiwan",
            "Singapore", 
            "Malaysia", 
            "Indonesia"
             ],
        "Spanish": [
            "Mexico",
            "United States (primarily in areas with a large Hispanic population)",
            "Spain", 
            "Colombia",
            "Argentina"
                    ],
        "French": [
            "France",
            "Democratic Republic of the Congo",
            "Canada (particularly in Quebec)", 
            "Belgium",
            "Cote d'Ivoire (Ivory Coast)"
                   ],
        "Arabic": [
            "Egypt", 
            "Saudi Arabia", 
            "Algeria", 
            "Morocco", 
            "Sudan"
                   ],
        "Hindi": [
            "India",
            "Nepal",
            "Fiji",
            "Trinidad and Tobago",
            "Guyana"
                  ],
        "Bengali": [
            "Bangladesh",
            "India (particularly in the state of West Bengal)",
            "West Bengal (India) is a major region."
            ],
        "Russian": [
            "Russia (primarily in the European part)",
            "Kazakhstan", 
            "Ukraine",
            "Belarus",
            "Kyrgyzstan"
                    ],
        "Portuguese": [
            "Brazil", 
            "Portugal",
            "Mozambique", 
            "Angola",
            "Guinea-Bissau"
                       ],
        "Japanese": [
            "Japan (natively spoken)",
            "Brazil (has a significant Japanese-speaking community)",
            "Hawaii, USA (also has a Japanese-speaking community)",
            "Peru (small Japanese-speaking community)",
            "Canada (particularly in Vancouver and Toronto)"
                     ],
        "Swahili": [
            "Kenya", 
            "Tanzania",
            "Uganda", 
            "Rwanda", 
            "Burundi",
            "Democratic Republic of Congo"
                    ],
        "Hausa": [
            "Nigeria","Niger"
                  ],
        "Yoruba": [
            "Nigeria","Benin", "Togo"
                   ],
        "Zulu": ["South Africa (particularly in the KwaZulu-Natal province)"
                 ],
        "Amharic": ["Ethiopia"],
        "Agni": ["Cote d'Ivoire"],
        "Baoule": ["Cote d'Ivoire"],
        "Bron": ["Cote d'Ivoire","Ghana"],
        "Asante": ["Ghana", "Cote d'Ivoire"],
        },
        # Random GDP per capita values
        # np.random.uniform(1000, 50000, num_samples).round(2),
    "Economic Status": [], # will define later    
}

# SDG6 Challenges dictionary with shorthand keys
SDG6_CHALLENGES = {
    "Lack of Access": "Access",
    "Water Scarcity": "Scarcity",
    "Water Pollution": "Pollution",
    "Ecosystem Degradation": "Ecosystems",
    "Governance Issues": "Governance",
}

ORE_TYPE = {
    'Type1': 'Gold Ore',
    'Type2': 'Iron Ore',
    'Type3': 'Copper Ore',
    'Type4': 'Silver Ore',
    'Type5': 'Lead Ore',
    'Type6': 'Zinc Ore',
    'Type7': 'Nickel Ore',
    'Type8': 'Tin Ore',
    'Type9': 'Bauxite',
    'Type10': 'Cobalt Ore',
    'Type11': 'Chromite',
    'Type12': 'Uranium Ore',
    'Type13': 'Manganese Ore',
    'Type14': 'Platinum Ore',
    'Type15': 'Tantalum Ore',
    'Type16': 'Vanadium Ore',
    'Type17': 'Molybdenum Ore',
    'Type18': 'Titanium Ore',
    'Type19': 'Lithium Ore',
    'Type20': 'Tungsten Ore',
    'Type21': 'Antimony Ore',
    'Type22': 'Mercury Ore',
    'Type23': 'Sulfur Ore',
    'Type24': 'Graphite Ore',
    'Type25': 'Diamond Ore',
    'Type26': 'Rare Earth Element Ores',
    'Type27': 'Phosphate Ore',
    'Type28': 'Gypsum Ore',
    'Type29': 'Fluorite Ore',
    'Type30': 'Barite Ore',
    'Type31': 'Asbestos Ore',
    'Type32': 'Boron Ore',
    'Type33': 'Potash Ore'
}

EXPLOSIVE_TYPE = {
    'Explosive1': 'ANFO (Ammonium Nitrate Fuel Oil)',
    'Explosive2': 'Water Gel Explosives',
    'Explosive3': 'Emulsion Explosives',
    'Explosive4': 'Dynamite',
    'Explosive5': 'Nitroglycerin',
    'Explosive6': 'Slurry Explosives',
    'Explosive7': 'Binary Explosives',
    'Explosive8': 'Boosters',
    'Explosive9': 'Detonating Cord',
    'Explosive10': 'C-4 (Plastic Explosive)',
    'Explosive11': 'Ammonium Nitrate',
    'Explosive12': 'Black Powder',
    'Explosive13': 'TNT (Trinitrotoluene)',
    'Explosive14': 'RDX (Cyclotrimethylenetrinitramine)',
    'Explosive15': 'PETN (Pentaerythritol Tetranitrate)',
    'Explosive16': 'ANFO Prills',
    'Explosive17': 'Cast Boosters',
    'Explosive18': 'Ammonium Nitrate Emulsion',
    'Explosive19': 'Nitrocellulose',
    'Explosive20': 'Aluminized Explosives',
    'Explosive21': 'Pentolite',
    'Explosive22': 'Semtex',
    'Explosive23': 'Nitroguanidine',
    'Explosive24': 'HMX (Cyclotetramethylenetetranitramine)',
    'Explosive25': 'Amatol',
    'Explosive26': 'Tetryl',
    'Explosive27': 'Composition B',
    'Explosive28': 'Water Gels with Sensitizers',
    'Explosive29': 'Nitrate Mixture Explosives',
    'Explosive30': 'Perchlorate Explosives',
    'Explosive31': 'Detonators (Non-Electric)',
    'Explosive32': 'Electric Detonators',
    'Explosive33': 'Electronic Detonators'
}

EQUIPMENT_TYPE = [
    'Excavator', 
    'Drill', 
    'Loader', 
    'Truck',
    "Articulated Haulers",
    "Asphalt Pavers",
    "Backhoe Loaders",
    "Blasthole Drills",
    "Bulldozers",
    "Cable Shovels",
    "Continuous Miners",
    "Conveyor Systems",
    "Crushing Equipment",
    "Draglines",
    "Drilling Rigs",
    "Dump Trucks",
    "Electric Rope Shovels",
    "Excavators",
    "Exploration Drills",
    "Feller Bunchers",
    "Forwarders",
    "Graders",
    "Harvesters",
    "Hydraulic Mining Shovels",
    "Jaw Crushers",
    "Loaders",
    "Material Handlers",
    "Milling Equipment",
    "Motor Graders",
    "Off-Highway Trucks",
    "Pipelayers",
    "Road Reclaimers",
    "Rock Drills",
    "Rotary Drills",
    "Scrapers",
    "Skid Steer Loaders",
    "Telehandlers",
    "Track Loaders",
    "Tracked Dozers",
    "Underground Mining Loaders",
    "Underground Mining Trucks",
    "Wheel Dozers",
    "Wheel Excavators",
    "Wheel Loaders",
    "Wheel Tractor-Scrapers",
    "Hydraulic Hammer",
    "Jumbos and Drifters",
    "Longwall Miners",
    "Roof Bolters",
    "Scooptrams",
    "Shotcrete Machines",
    "Shuttle Cars",
    "Stackers",
    "Reclaimers",
    "Screening Plants",
    "Haul Trucks",
    "Feeders",
    "Gyratory Crushers",
    "Cone Crushers",
    "Impact Crushers",
    "Hammer Mills",
    "Sizers"
]


COMMON_CROPS = [
    "Wheat", 
    "Rice",
    "Corn",
    "Barley", 
    "Soybeans",
    "Oats",
    "Rye",
    "Millet",
    "Sorghum",
    "Canola",
    "Sunflower",
    "Cotton",
    "Sugar Cane", 
    "Sugar Beet",
    "Potatoes",
    "Tomatoes",
    "Onions", 
    "Cabbage",
    "Carrots",
    "Lettuce",
    "Spinach", 
    "Broccoli", 
    "Garlic",
    "Cucumbers", 
    "Pumpkins",
    "Peppers",
    "Eggplants",
    "Zucchini", 
    "Squash",
    "Peas",
    "Green Beans",
    "Lentils", 
    "Chickpeas",
    "Almonds", 
    "Walnuts",
    "Peanuts", 
    "Cashews", 
    "Pistachios", 
    "Apples",
    "Oranges",
    "Bananas", 
    "Grapes",
    "Strawberries",
    "Blueberries",
    "Raspberries",
    "Blackberries",
    "Cherries", 
    "Peaches",
    "Pears", 
    "Plums",
    "Pineapples",
    "Mangoes",
    "Avocados"
]

COMMON_PESTICIDES = [
    'Herbicide',
    'Insecticide',
    'Fungicide'
    "Glyphosate", 
    "Atrazine",
    "2,4-Dichlorophenoxyacetic acid (2,4-D)", 
    "Dicamba",
    "Paraquat", 
    "Chlorpyrifos",
    "Metolachlor",
    "Imidacloprid",
    "Thiamethoxam", 
    "Clothianidin", 
    "Acetamiprid", 
    "Fipronil",
    "Bacillus thuringiensis (Bt)",
    "Neonicotinoids",
    "Pyrethroids",
    "Carbamates",
    "Organophosphates",
    "Sulfonylureas",
    "Roundup",
    "Liberty",
    "Malathion",
    "Diazinon", 
    "DDT", 
    "Methoxychlor",
    "Aldrin", 
    "Dieldrin", 
    "Endrin",
    "Chlordane",
    "Heptachlor", 
    "Hexachlorobenzene",
    "Mirex", 
    "Toxaphene",
    "Captan", 
    "Chlorothalonil", 
    "Mancozeb",
    "Maneb", 
    "Zineb",
    "Copper Sulphate",
    "Streptomycin",
    "Tetracycline", 
    "Difenoconazole", 
    "Propiconazole",
    "Cyproconazole", 
    "Azoxystrobin",
    "Chlorantraniliprole",
    "Abamectin", 
    "Spinosad", 
    "Bifenthrin",
    "Cyfluthrin", 
    "Deltamethrin", 
    "Permethrin", 
    "Cypermethrin",
    "Metam Sodium",
    "Methyl Bromide",
    "Chloropicrin",
    "Vapam"
]
AFRICAN_COUNTRIES= [
    "Algeria",
    "Angola",
    "Benin", 
    "Botswana",
    "Burkina Faso",
    "Burundi",
    "Cabo Verde", 
    "Cameroon",
    "Central African Republic", 
    "Chad",
    "Comoros", 
    "Congo",
    "Congo Democratic Republic", 
    "Cote d'Ivoire",
    "Djibouti",
    "Egypt", 
    "Equatorial Guinea", 
    "Eritrea", 
    "Eswatini",
    "Ethiopia",
    "Gabon",
    "Gambia", 
    "Ghana",
    "Guinea",
    "Guinea-Bissau",
    "Kenya", 
    "Lesotho",
    "Liberia",
    "Libya",
    "Madagascar",
    "Malawi", 
    "Mali",
    "Mauritania",
    "Mauritius",
    "Morocco", 
    "Mozambique",
    "Namibia",
    "Niger",
    "Nigeria",
    "Rwanda", 
    "Sao Tome and Principe",
    "Senegal", 
    "Seychelles",
    "Sierra Leone",
    "Somalia",
    "South Africa",
    "South Sudan",
    "Sudan", 
    "Tanzania",
    "Togo", 
    "Tunisia", 
    "Uganda",
    "Zambia",
    "Zimbabwe"
]

