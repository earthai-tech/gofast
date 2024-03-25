# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import numpy as np
import pandas as pd 


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
        
        formatted_attrs = [
            f"{key:{max_key_length}} : {self._format_iterable(self.__dict__[key])}" 
            for key in keys]
        return "\n".join(formatted_attrs)
    
    def _format_iterable(self, attr):
        """
        Processes iterable attributes of a Bunch object, providing a formatted string
        representation based on the type and content of the iterable. This method
        supports lists, tuples, sets, NumPy arrays, pandas Series, and pandas DataFrames.
    
        For numeric iterables (lists, tuples, sets, NumPy arrays), it calculates and
        includes statistics such as minimum value, maximum value, mean, and length
        (or dimensions and dtype for NumPy arrays). For pandas Series and DataFrames,
        it also considers data type and, when applicable, provides a summary including
        minimum value, maximum value, mean, number of rows, number of columns, and dtypes.
        If the Series or DataFrame contains non-numeric data or a mix of types, it adjusts
        the output to exclude statistical summaries that are not applicable.
    
        Parameters
        ----------
        attr : iterable
            The iterable attribute of the Bunch object to be formatted. This can be a
            list, tuple, set, NumPy ndarray, pandas Series, or pandas DataFrame.
    
        Returns
        -------
        str
            A string representation of the iterable attribute, formatted according to
            its type and contents. This includes a concise summary of statistical
            information for numeric data and relevant structural information for all
            types.
            
        Notes
        -----
        - Numeric summaries (minval, maxval, mean) are rounded to 4 decimal places.
        - For NumPy arrays, the shape is presented in an 'n_rows x m_columns' format.
        - For pandas DataFrames with uniform dtypes, the dtype is directly mentioned;
          if the DataFrame contains a mix of dtypes, 'dtypes=object' is used instead.
        """
     
        if isinstance(attr, (list, tuple, set)):
            numeric = all(isinstance(item, (int, float)) for item in attr)
            if numeric:
                minval, maxval= round(min(attr), 4), round(max(attr), 4)
                mean = round(np.mean(list(attr)), 4)
                return ( 
                    f"{type(attr).__name__} (minval={minval}, maxval={maxval},"
                    f" mean={mean}, len={len(attr)})"
                    )
            return f"{type(attr).__name__} (len={len(attr)})"
        
        elif isinstance(attr, np.ndarray):
            numeric = np.issubdtype(attr.dtype, np.number)
            if numeric:
                minval, maxval= round(attr.min(), 4), round(attr.max(), 4)
                mean= round(attr.mean(), 4)
                return (
                    f"Array (minval={minval}, maxval={maxval}, mean={mean}, "
                    f"ndim={attr.ndim}, shape={'x'.join(map(str, attr.shape))},"
                    f" dtype={attr.dtype})")
            return (
                f"Array (ndim={attr.ndim}, shape={'x'.join(map(str, attr.shape))},"
                f" dtype={attr.dtype})"
                )
        
        elif isinstance(attr, pd.Series):
            if attr.dtype == 'object':
                return f"Series (len={attr.size}, dtype={attr.dtype})"
            minval, maxval= round(attr.min(), 4), round(attr.max(), 4)
            mean= round(attr.mean(), 4)
            return ( f"Series (minval={minval}, maxval={maxval}, mean={mean},"
                    f" len={attr.size}, dtype={attr.dtype})"
                    )
        
        elif isinstance(attr, pd.DataFrame):
            dtypes_set = set(attr.dtypes)
            dtypes = 'object' if len(dtypes_set) > 1 else list(dtypes_set)[0]
            if 'object' not in dtypes_set:
                numeric_cols = attr.select_dtypes(include=np.number).columns.tolist()
                minval = round(attr[numeric_cols].min().min(), 4)
                maxval= round(attr[numeric_cols].max().max(), 4)
                mean=round(attr[numeric_cols].mean().mean(), 4)
                return (f"DataFrame (minval={minval}, maxval={maxval}, mean={mean}, "
                        f"n_rows={attr.shape[0]}, n_cols={attr.shape[1]}, dtypes={dtypes})")
            return f"DataFrame (n_rows={attr.shape[0]}, n_cols={attr.shape[1]}, dtypes={dtypes})"
        
        return str(attr)

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

            
class FlexDict(dict):
    """
    A `FlexDict` is a dictionary subclass that provides flexible attribute-style
    access to its items, allowing users to interact with the dictionary as if it
    were a regular object with attributes. It offers a convenient way to work with
    dictionary keys without having to use the bracket notation typically required by
    dictionaries in Python. This makes it especially useful in environments where
    quick and easy access to data is desired.

    The `FlexDict` class extends the built-in `dict` class, so it inherits all the
    methods and behaviors of a standard dictionary. In addition to the standard
    dictionary interface, `FlexDict` allows for the setting, deletion, and access
    of keys as if they were attributes, providing an intuitive and flexible
    interface for managing dictionary data.

    Examples
    --------
    Here is how you can use a `FlexDict`:

    >>> from gofast.api.structures import FlexDict
    >>> fd = FlexDict(pkg='gofast', goal='simplify tasks', version='1.0')
    >>> fd['pkg']  # Standard dictionary access
    'gofast'
    >>> fd.pkg     # Attribute access
    'gofast'
    >>> fd.goal    # Another example of attribute access
    'simplify tasks'
    >>> fd.version # Accessing another attribute
    '1.0'
    >>> fd.new_attribute = 'New Value'  # Setting a new attribute
    >>> fd['new_attribute']             # The new attribute is accessible as a key
    'New Value'

    Notes
    -----
    - While `FlexDict` adds convenience, it is important to avoid key names that
      clash with the methods and attributes of a regular dictionary. Such conflicts
      can result in unexpected behavior, as method names would take precedence over
      key names during attribute access.

    - The behavior of `FlexDict` under serialization (e.g., when using pickle) may
      differ from that of a standard dictionary due to the attribute-style access.
      Users should ensure that serialization and deserialization processes are
      compatible with `FlexDict`'s approach to attribute access.

    - Since `FlexDict` is built on the Python dictionary, it maintains the same
      performance characteristics for key access and management. However, users
      should be mindful of the additional overhead introduced by supporting
      attribute access when considering performance-critical applications.

    By providing a dictionary that can be accessed and manipulated as if it were a
    regular object, `FlexDict` offers an enhanced level of usability, particularly
    in situations where the more verbose dictionary syntax might be less desirable.
    """


    def __init__(self, **kwargs):
        """
        Initialize a FlexDict with keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs to initialize the FlexDict.
        """
        super(FlexDict, self).__init__(**kwargs)
        self.__dict__ = self

    def __getattr__(self, item):
        """
        Allows attribute-style access to the dictionary keys.

        Parameters
        ----------
        item : str
            The attribute name corresponding to the dictionary key.

        Returns
        -------
        The value associated with 'item' in the dictionary.

        Raises
        ------
        AttributeError
            If 'item' is not found in the dictionary.
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'FlexDict' object has no attribute '{item}'")

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
        Ensures that FlexDict can be unpickled correctly.
        """
        self.update(state)
        self.__dict__ = self

    def __dir__(self):
        """
        Ensures that auto-completion works in interactive environments.
        """
        return list(self.keys())

    def __repr__(self):
        """
        Provides a string representation of the FlexDict object, including the keys.
        """
        keys = ', '.join(self.keys())
        return f"<FlexDict with keys: {keys}>"

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
