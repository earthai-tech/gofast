# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import itertools
import numpy as np
import pandas as pd 
from .._typing import List, Optional, Union, DataFrame 

class Bunch:
    """
    A utility class for storing collections of results or data attributes.
    
    This class is designed to bundle various attributes and provide a 
    convenient way to access them using attribute-style access instead of 
    dictionary-style. It overrides the `__repr__` and `__str__`
    methods to offer informative and readable representations of its contents.
    
    Methods
    -------
    __repr__()
        Provides a basic representation of the object, indicating it contains 
        results.
    
    __str__()
        Returns a string representation of the object's contents in a nicely 
        formatted way, showing each attribute and its corresponding value.
    
    Examples
    --------
    >>> from gofast.tools.box import Bunch 
    >>> results = Bunch()
    >>> results.accuracy = 0.95
    >>> results.loss = 0.05
    >>> print(results)
    accuracy   0.95
    loss       0.05
    
    >>> results
    <Bunch object containing results. Use print() to see contents>
    
    Note
    ----
    This class is intended to be a simple container for dynamically added attributes. 
    It provides a more intuitive and accessible way to handle grouped data without 
    the syntax of dictionaries, enhancing code readability and convenience.
    """
 
    def __init__(self, **kwargs):
        """
        Initialize a Bunch object with optional keyword arguments.

        Each keyword argument provided is set as an attribute of the Bunch object.

        Parameters
        ----------
        **kwargs : dict, optional
            Arbitrary keyword arguments which are set as attributes of the Bunch object.
        """
        self.__dict__.update(kwargs)
        
    def __repr__(self):
        """
        Returns a simple representation of the _Bunch object.
        
        This representation indicates that the object is a 'bunch' containing 
        results,suggesting to print the object to see its contents.
        
        Returns
        -------
        str
            A string indicating the object contains results.
        """
        return "<Bunch object containing results. Use print() to see contents>"

    def __str__(self):
        """
        Returns a detailed string representation of the _Bunch object's contents.
        
        This method lists all attributes of the _Bunch object in alphabetical order,
        along with their values in a formatted string. This provides a clear overview
        of all stored data for easy readability.
        
        Returns
        -------
        str
            A string representation of the _Bunch object's contents, including attribute
            names and their values, formatted in a tabular style.
        """
        if not self.__dict__:
            return "<empty Bunch>"
        keys = sorted(self.__dict__.keys())
        max_key_length = max(len(key) for key in keys)
        formatted_attrs = [f"{key:{max_key_length}} : {self.__dict__[key]}" for key in keys]
        return "\n".join(formatted_attrs)
    
        # keys = [key for key, _ in self.__dict__.items()]
        # keys.sort()
        # max_key_length = max([len(key) for key in keys])
        # tabulated_contents = []
        # format_string = "{:" + str(max_key_length) + "}   {}"
        # for key in keys:
        #     tabulated_contents.append(format_string.format(key, self.__dict__[key]))
        # return "\n".join(tabulated_contents)

    def __delattr__(self, name):
        """
        Delete an attribute from the Bunch object.

        Parameters
        ----------
        name : str
            The name of the attribute to delete.

        Raises
        ------
        AttributeError
            If the specified attribute does not exist.
        """
        if name in self.__dict__:
            del self.__dict__[name]
        else:
            raise AttributeError(f"'Bunch' object has no attribute '{name}'")

    def __contains__(self, name):
        """
        Check if an attribute exists in the Bunch object.

        Parameters
        ----------
        name : str
            The name of the attribute to check for existence.

        Returns
        -------
        bool
            True if the attribute exists, False otherwise.
        """
        return name in self.__dict__

    def __iter__(self):
        """
        Return an iterator over the Bunch object's attribute names and values.

        Yields
        ------
        tuple
            Pairs of attribute names and their corresponding values.
        """
        for item in self.__dict__.items():
            yield item

    def __len__(self):
        """
        Return the number of attributes stored in the Bunch object.

        Returns
        -------
        int
            The number of attributes.
        """
        return len(self.__dict__)

    def __getattr__(self, name):
        """
        Get an attribute's value, raising an AttributeError if it doesn't exist.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Raises
        ------
        AttributeError
            If the specified attribute does not exist.
        """
        raise AttributeError(f"'Bunch' object has no attribute '{name}'")

    def __eq__(self, other):
        """
        Compare this Bunch object to another for equality.

        Parameters
        ----------
        other : Bunch
            Another Bunch object to compare against.

        Returns
        -------
        bool
            True if the other object is a Bunch with the same attributes and 
            values, False otherwise.
        """
        return isinstance(other, Bunch) and self.__dict__ == other.__dict__

 
class Boxspace(dict):
    """
    A container object that extends dictionaries by enabling attribute-like 
    access to its items.
    
    `Boxspace` allows accessing values using the standard dictionary key 
    access method or directly as attributes. This feature provides a more 
    convenient and intuitive way to handle data, especially when dealing with 
    configurations or loosely structured objects.
    
    Examples
    --------
    >>> bs = Boxspace(pkg='gofast', objective='give water', version='0.1.dev')
    >>> bs['pkg']
    'gofast'
    >>> bs.pkg
    'gofast'
    >>> bs.objective
    'give water'
    >>> bs.version
    '0.1.dev'
    
    Notes
    -----
    While `Boxspace` provides a flexible way to access dictionary items, it's 
    important to ensure that key names do not conflict with the dictionary's 
    method names, as this could lead to unexpected behavior.
    """

    def __init__(self, **kwargs):
        """
        Initializes a Boxspace object with optional keyword arguments.
        
        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments which are set as the initial 
            items of the dictionary.
        """
        super().__init__(**kwargs)

    def __getattr__(self, key):
        """
        Allows attribute-like access to dictionary items.
        
        Parameters
        ----------
        key : str
            The attribute name corresponding to the dictionary key.
        
        Returns
        -------
        The value associated with 'key' in the dictionary.
        
        Raises
        ------
        AttributeError
            If the key is not found in the dictionary.
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Boxspace' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        """
        Allows setting dictionary items as attributes.
        
        Parameters
        ----------
        key : str
            The attribute name to be added or updated in the dictionary.
        value : any
            The value to be associated with 'key'.
        """
        self[key] = value

    def __setstate__(self, state):
        """
        Overrides __setstate__ to ensure the object can be unpickled correctly.
        
        This method is a no-op, effectively ignoring the pickled __dict__, which is
        necessary because `Boxspace` objects use the dictionary itself for item storage.
        """
        pass

    def __dir__(self):
        """
        Ensures that autocompletion works in interactive environments.
        
        Returns
        -------
        list
            A list of keys in the dictionary, which are exposed as attributes.
        """
        return super().__dir__() + list(self.keys()) # self.keys()
    
class AquiferGroupAnalyzer:
    """
    Analyzes and represents aquifer groups, particularly focusing on the 
    relationship between permeability coefficient ``K`` values and aquifer
    groupings. It utilizes a Mixture Learning Strategy (MXS) to impute missing
    'k' values by creating Naive Group of Aquifer (NGA) labels based on 
    unsupervised learning predictions.
    
    This approach aims to minimize bias by considering the permeability coefficient
    'k' closely tied to aquifer groups. It determines the most representative aquifer
    group for given 'k' values, facilitating the filling of missing 'k' values in the dataset.
    
    Parameters
    ----------
    group_data : dict, optional
        A dictionary mapping labels to their occurrences, representativity, and
        similarity within aquifer groups.
    
    Example
    -------
    See class documentation for a detailed example of usage.
    
    Attributes
    ----------
    group_data : dict
        Accessor for the aquifer group data.
    similarity : generator
        Yields label similarities with NGA labels.
    preponderance : generator
        Yields label occurrences in the dataset.
    representativity : generator
        Yields the representativity of each label.
    groups : generator
        Yields groups for each label.
    """

    def __init__(self, group_data=None):
        """
        Initializes the AquiferGroupAnalyzer with optional group data.
        """
        self.group_data = group_data if group_data is not None else {}

    @property
    def similarity(self):
        """Yields label similarities with NGA labels."""
        return ((label, list(rep_val[1])[0]) for label, rep_val in self.group_data.items())

    @property
    def preponderance(self):
        """Yields label occurrences in the dataset."""
        return ((label, rep_val[0]) for label, rep_val in self.group_data.items())

    @property
    def representativity(self):
        """Yields the representativity of each label."""
        return ((label, round(rep_val[1].get(list(rep_val[1])[0]), 2))
                for label, rep_val in self.group_data.items())

    @property
    def groups(self):
        """Yields groups for each label."""
        return ((label, {k: v for k, v in repr_val[1].items()})
                for label, repr_val in self.group_data.items())

    def __repr__(self):
        """
        Returns a string representation of the AquiferGroupAnalyzer object,
        formatting the representativity of aquifer groups.
        """
        formatted_data = self._format(self.group_data)
        return f"{self.__class__.__name__}({formatted_data})"

    def _format(self, group_dict):
        """
        Formats the representativity of aquifer groups into a string.
        
        Parameters
        ----------
        group_dict : dict
            Dictionary composed of the occurrence of the group as a function
            of aquifer group representativity.
        
        Returns
        -------
        str
            A formatted string representing the aquifer group data.
        """
        formatted_groups = []
        for index, (label, (preponderance, groups)) in enumerate(group_dict.items()):
            label_str = f"{'Label' if index == 0 else ' ':>17}=['{label:^3}',\n"
            preponderance_str = f"{'>32'}(rate = '{preponderance * 100:^7}%',\n"
            groups_str = f"{'>34'}'Groups', {groups}),\n"
            representativity_key, representativity_value = next(iter(groups.items()))
            representativity_str = f"{'>34'}'Representativity', ('{representativity_key}', {representativity_value}),\n"
            similarity_str = f"{'>34'}'Similarity', '{representativity_key}')])],\n"

            formatted_groups.extend([
                label_str, preponderance_str, groups_str,
                representativity_str, similarity_str
            ])
        
        return ''.join(formatted_groups).rstrip(',\n')

def DataToBox(
    data: Union[DataFrame, dict, List], 
    entity_name: Optional[str] = None, 
    use_column_as_name: bool = False, 
    keep_name_column: bool = True, 
    columns: Optional[List[str]] = None):
    """
    Transforms each row of a DataFrame or a similar iterable structure into 
    :class:`Boxspace` objects.
    
    Parameters
    ----------
    data : DataFrame, dict, or List
        The data to transform. If not a DataFrame, an attempt is made to 
        convert it into one.
    
    entity_name : str, optional
        Base name for the entities in the BoxSpace. If `use_column_as_name` is 
        True, this refersto a column name whose values are used as entity names.
    
    use_column_as_name : bool, default False
        If True, `entity_name` must be a column name in `data`, and its values 
        are used as names
        for the entities.
    
    keep_name_column : bool, default True
        Whether to keep the column used for naming entities in the data.
    
    columns : list of str, optional
        Specifies column names when `data` is a list or an array-like structure. 
        Ignored if `data` is already a DataFrame.
    
    Returns
    -------
    BoxSpace
        A BoxSpace object containing named entities corresponding to each 
        row of the input `data`.
        
    Raises
    ------
    ValueError
        If `use_column_as_name` is True but `entity_name` is not a 
        column in `data`.
    
    TypeError
        If the input `data` cannot be converted into a DataFrame.
    
    Examples
    --------
    >>> from gofast.tools.box import DataToBox 
    >>> DataToBox([2, 3, 4], entity_name='borehole')
    >>> DataToBox({"x": [2, 3, 4], "y": [8, 7, 5]}, entity_name='borehole')
    >>> DataToBox([2, 3, 4], entity_name='borehole', columns=['id'])
    >>> DataToBox({"x": [2, 3, 4], "y": [8, 7, 5], "code": ['h2', 'h7', 'h12']},
                            entity_name='code', use_column_as_name=True)
    """
    if not isinstance(data, pd.DataFrame):
        if columns is None and isinstance(data, dict):
            columns = list(data.keys())
        data = pd.DataFrame(data, columns=columns)

    if entity_name is not None and use_column_as_name:
        if entity_name not in data.columns:
            raise ValueError(f"Column '{entity_name}' must exist in the data.")
        names = data[entity_name].astype(str)
        if not keep_name_column:
            data = data.drop(columns=[entity_name])
    else:
        if entity_name is None:
            entity_name = 'object'
        names = [f"{entity_name}{i}" for i in range(len(data))]

    box_space_objects = {
        name: Boxspace(**data.iloc[i].to_dict()) for i, name in enumerate(names)}

    return Boxspace(**box_space_objects)

def data2Box(
    data, /,  
    name: str = None, 
    use_colname: bool =False, 
    keep_col_data: bool =True, 
    columns: List [str] =None 
    ): 
    """ Transform each data rows as Boxspace object. 
    
    Parameters 
    -----------
    data: DataFrame 
      Data to transform as an object 
      
    columns: list of str, 
      List of str item used to construct the dataframe if tuple or list 
      is passed. 
      
    name: str, optional 
       The object name. When string argument is given, the index value of 
       the data is is used to prefix the name data unless the `use_column_name`
       is set to ``True``. 
       
    use_colname: bool, default=False 
       If ``True`` the name must be in columns. Otherwise an error raises. 
       However, when ``use_colname=true``, It is recommended to make sure 
       whether each item in column data is distinct i.e. is unique, otherwise, 
       some data will be erased. The number of object should be less than 
       the data size along rows axis. 
       
    keep_col_data: bool, default=True 
      Keep in the data the column that is used to construct the object name.
      Otherwise, column data whom object created from column name should 
      be dropped. 
      
    Return
    --------
    Object: :class:`.BoxSpace`, n_objects = data.size 
       Object that composed of many other objects where the number is equals 
       to data size. 
       
    Examples
    --------- 
    >>> from gofast.tools.box import data2Box 
    >>> o = data2Box ([2, 3, 4], name = 'borehole')
    >>> o.borehole0
    {'0': 2}
    >>> o = data2Box ({"x": [2, 3, 4], "y":[8, 7, 5]}, name = 'borehole')
    >>> o.borehole0.y
    8
    >>> from gofast.tools.box import data2Box 
    >>> o = data2Box ([2, 3, 4], name = 'borehole', columns ='id') 
    >>> o.borehole0.id
    2
    >>> o = data2Box ({"x": [2, 3, 4], "y":[8, 7, 5], 
                       "code": ['h2', 'h7', 'h12'] }, name = 'borehole')
    >>> o.borehole1.code
    'h7'
    >>> o = data2Box ({"x": [2, 3, 4], "y":[8, 7, 5], "code": ['h2', 'h7', 'h12'] }, 
                      name = 'code', use_colname= True )
    >>> o.h7.code
    'h7'
    >>> o = data2Box ({"x": [2, 3, 4], "y":[8, 7, 5], "code": ['h2', 'h7', 'h12'] 
                       }, name = 'code', use_colname= True, keep_col_data= False  )
    >>> o.h7.code # code attribute does no longer exist 
    AttributeError: code
    """
    from .validator import _is_numeric_dtype 
    from .coreutils import is_iterable 
    
    if columns is not None: 
        columns = is_iterable (
            columns, exclude_string= True , transform =True  )
   
    if ( 
            not hasattr ( data , 'columns') 
            or hasattr ( data, '__iter__')
            ): 
        data = pd.DataFrame ( data, columns = columns )
        
    if not hasattr(data, '__array__'): 
            raise TypeError (
                f"Object accepts only DataFrame. Got {type(data).__name__}")

    if columns is not None: 
        # rename columns if given 
        data = pd.DataFrame(np.array( data), columns = columns )
        
    if name is not None: 
        # Name must be exists in the dataframe. 
        if use_colname:  
            if name not in data.columns:  
                raise ValueError (
                    f"Name {name!r} must exist in the data columns.")
            
            name =  data [name] if keep_col_data else data.pop ( name )
            
    # make name column if not series 
    if not hasattr ( name, 'name'):
        # check whether index is numeric then prefix with index 
        index = data.index 
        if _is_numeric_dtype(index, to_array= True ): 
            index = index.astype (str)
            if name is None:
                name ='obj'
        
        name = list(map(''.join, itertools.zip_longest(
            [name  for i in range ( len(index ))], index)))
        
    # for consistency # reconvert name to str 
    name = np.array (name ).astype ( str )
    
    obj = dict() 
    for i in range ( len(data)): 
        v = Boxspace( **dict ( zip ( data.columns.astype (str),
                                    data.iloc [i].values )))
        obj [ name[i]] = v 
        
    return Boxspace( **obj )


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
