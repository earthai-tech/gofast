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
    >>> from gofast.tools.box import Boxspace
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

    def to_frame(self, error='raise'):
        """
        Converts the Boxspace object into a pandas DataFrame.
    
        This method iterates through each item in the Boxspace, expecting
        each item to be either a dictionary or a Bunch object. It ensures
        that all items have consistent keys to serve as DataFrame columns,
        filling in missing data with NaN to maintain data integrity.
    
        Parameters
        ----------
        error : {'raise', 'ignore'}, default 'raise'
            How to handle errors during the conversion process.
            - If 'raise', a ValueError will be raised if the conversion fails.
            - If 'ignore', an empty DataFrame will be returned in case of failure.
    
        Returns
        -------
        pd.DataFrame
            A DataFrame representation of the Boxspace object, with each
            item represented as a row.
    
        Raises
        ------
        ValueError
            If `error` is set to 'raise' and the Boxspace object cannot be
            converted into a DataFrame due to inconsistent structures among
            its items.
    
        Examples
        --------
        >>> from gofast.tools.box import Boxspace, Bunch
        >>> box = Boxspace(a=Bunch(name='Alice', age=30),
        ...                b=Bunch(name='Bob', age=25))
        >>> df = box.to_frame()
        >>> print(df)
           name  age
        a  Alice   30
        b    Bob   25
    
        >>> # Example with inconsistent data and error handling
        >>> box = Boxspace(a=Bunch(name='Alice', age=30),
        ...                b={'name': 'Bob'})
        >>> try:
        ...     df = box.to_frame()
        ... except ValueError as e:
        ...     print(e)
        Failed to create DataFrame from Boxspace: ...
        """
        if not self:
            return pd.DataFrame()  # Return an empty DataFrame if Boxspace is empty
        
        all_keys = set()
        for item in self.values():
            if not isinstance(item, (dict, Bunch)):
                if error == 'raise':
                    raise ValueError("All items in the Boxspace must be dictionaries"
                                     " or Bunch objects for DataFrame conversion.")
                else:
                    return pd.DataFrame()
            all_keys.update(item.keys())
        
        data = []
        for item in self.values():
            row_data = {key: item.get(key, pd.NA) for key in all_keys}
            data.append(row_data)
        
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            if error == 'raise':
                raise ValueError(f"Failed to create DataFrame from Boxspace: {e}")
            else:
                df = pd.DataFrame()
        
        return df

    def __repr__(self):
        """
        Provides an enhanced representation of the Boxspace object, displaying
        its contents in a more readable and nicely formatted manner.
        
        Returns
        -------
        str
            A string representation of the Boxspace object, showcasing its
            keys and values in a well-organized layout.
        """
        if not self:
            return f"{self.__class__.__name__}()"

        max_key_length = max(len(str(key)) for key in self.keys()) + 1
        items_repr = []

        for key, value in self.items():
            key_repr = f"{str(key):<{max_key_length}}"
            value_repr = repr(value).replace('\n', '\n' + ' ' * (max_key_length + 4))
            items_repr.append(f"{key_repr} : {value_repr}")

        items_str = "\n    ".join(items_repr)
        return f"{self.__class__.__name__}(\n    {items_str}\n)"
    

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

class BoxCategoricalEncoder:
    """
    Encodes categorical features in a dataset.

    This class provides methods to convert categorical variables into numerical
    formats, supporting both label encoding and one-hot encoding. It's designed
    to work seamlessly with pandas DataFrames and Boxspace objects.

    Attributes
    ----------
    encoding_type : str
        The type of encoding to apply. Options are 'label' for label encoding
        and 'one-hot' for one-hot encoding.

    Methods
    -------
    fit_transform(data)
        Fits the encoder to the data and transforms the categorical features.
    inverse_transform(data)
        Converts numerical features back to their original categorical values.
    """

    def __init__(self, encoding_type='label'):
        self.encoding_type = encoding_type
        self.encoders = {}


    def fit(self, data):
        """
        Fits the encoder to the data, preparing it for transforming categorical
        features according to the specified `encoding_type`. This method identifies
        the categorical columns and initializes encoders for them.
    
        Parameters
        ----------
        data : pd.DataFrame, dict, or Boxspace
            The dataset containing categorical features to encode. It can be 
            a pandas DataFrame, a dictionary (which will be converted to a DataFrame),
            or a Boxspace object.
    
        Returns
        -------
        self : BoxCategoricalEncoder
            The fitted encoder instance ready for transforming data.
    
        Raises
        ------
        ValueError
            If the input `data` type is not supported or if no categorical columns
            are identified in the `data`.
    
        Example
        -------
        >>> from gofast.tools.box import Boxspace, BoxCategoricalEncoder
        >>> data = Boxspace(a='red', b='blue', c='green')
        >>> encoder = BoxCategoricalEncoder(encoding_type='label')
        >>> encoder.fit(data)
        <BoxCategoricalEncoder fitted with label encoding for ['a', 'b', 'c']>
        """
        # Ensure data is a DataFrame for processing
        if isinstance(data, Boxspace):
            data = data.to_frame()
        elif isinstance(data, dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Data must be a pandas DataFrame, a Boxspace object, or a dictionary.")
    
        # Identify categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            raise ValueError("No categorical columns identified in the data.")
    
        # Initialize encoders for each categorical column
        self.encoders = {}
        if self.encoding_type == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(data[col])
                self.encoders[col] = le
        elif self.encoding_type == 'one-hot':
            # For one-hot encoding, we don't need to fit individual encoders
            pass
        else:
            raise ValueError(f"Unsupported encoding type '{self.encoding_type}'.")
    
        return self

    def fit_transform(self, data):
        """
        Fits the encoder to the data and transforms the categorical features. This
        method combines the fitting and transforming steps for convenience.
    
        Parameters
        ----------
        data : pd.DataFrame, dict, or Boxspace
            The dataset containing categorical features to encode. It can be a
            pandas DataFrame, a dictionary of lists (which will be converted to DataFrame), 
            or a Boxspace object.
    
        Returns
        -------
        pd.DataFrame or Boxspace
            A new dataset with categorical features encoded according to the 
            `encoding_type`. The type of the returned object matches the type 
            of the input `data`.
    
        Raises
        ------
        ValueError
            If the input `data` type is not supported or if `encoding_type` 
            is not recognized.
    
        Example
        -------
        >>> from gofast.tools.box import Boxspace, BoxCategoricalEncoder
        >>> data = Boxspace(a='red', b='blue', c='red')
        >>> encoder = BoxCategoricalEncoder(encoding_type='label')
        >>> encoded_data = encoder.fit_transform(data)
        >>> print(encoded_data)
        {'a': 0, 'b': 1, 'c': 0}
        """
        self.fit(data)
        return self.transform()
    
    def inverse_transform(self, data):
        """
        Converts numerical features back to their original categorical values. 
        
        This method is primarily applicable for label encoded data.
    
        Parameters
        ----------
        data : pd.DataFrame or Boxspace
            The dataset with numerical features previously encoded by this 
            encoder, which are to be converted back to their original 
            categorical format.
    
        Returns
        -------
        pd.DataFrame or Boxspace
            The dataset with encoded features converted back to their original
            categorical values. The type of the returned object matches the 
            type of the input `data`.
    
        Raises
        ------
        NotImplementedError
            If attempting to inverse-transform data encoded with 'one-hot', as this
            operation is not straightforward with the current implementation.
    
        Example
        -------
        >>> from gofast.tools.box import Boxspace, BoxCategoricalEncoder
        >>> data = Boxspace(a=0, b=1, c=0)
        >>> encoder = BoxCategoricalEncoder(encoding_type='label')
        >>> original_data = encoder.inverse_transform(data)
        >>> print(original_data)
        {'a': 'red', 'b': 'blue', 'c': 'red'}
        """
        if not isinstance(data, (pd.DataFrame, Boxspace)):
            raise ValueError("Data must be a pandas DataFrame or a Boxspace object.")
        original_type= type (data)
        if original_type==Boxspace: 
            data = data.to_frame(error='ignore')
            
        if self.encoding_type == 'label':
            for col, le in self.encoders.items():
                if col in data.columns:
                    data[col] = le.inverse_transform(data[col])
        else:
            raise NotImplementedError(
                "Inverse transformation for 'one-hot' encoding is not implemented.")
        if original_type==Boxspace: 
            return Boxspace ( **data.to_dict (orient='series'))
        return data

def normalize_box_data(data, method='z-score'):
    """
    Normalizes the numerical features in a dataset using specified normalization
    techniques.
    
    This utility function is versatile and can handle both pandas DataFrame 
    and Boxspace objects,
    making it suitable for a wide range of data preprocessing tasks in data
    analysis and machine learning.
    
    Supported normalization methods include 'z-score' for standardization and
    'min-max' for scaling data to a fixed range [0, 1]. Z-score normalization 
    transforms the data to have a mean of 0 and a standard deviation of 1, 
    while min-max scaling adjusts the scale without distorting the differences
    in the ranges of values.
    
    Parameters
    ----------
    data : pd.DataFrame or Boxspace
        The input dataset containing numerical features to be normalized. 
        This can be a pandas DataFrame or a Boxspace object, providing 
        flexibility in data structures for normalization.
    method : str, optional
        The normalization method to apply to the numerical features in `data`.
        Supported values are:
        - 'z-score': Applies Z-score normalization.
        - 'min-max': Applies min-max scaling.
        Default value is 'z-score'.
    
    Returns
    -------
    pd.DataFrame or Boxspace
        The resulting dataset after applying the specified normalization 
        technique. The return type matches the input type (`data`), ensuring 
        consistency and ease of integration into data pipelines.
    
    Examples
    --------
    Using a pandas DataFrame:

    >>> import pandas as pd
    >>> from gofast.tools.box import normalize_box_data
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> normalized_df = normalize_box_data(df, method='min-max')
    >>> print(normalized_df)
           A         B
    0  0.0  0.0
    1  0.5  0.5
    2  1.0  1.0

    Using a Boxspace object:

    >>> from gofast.tools.box import Boxspace, normalize_box_data
    >>> box = Boxspace(item1={'A': 1, 'B': 4}, item2={'A': 2, 'B': 5}, item3={'A': 3, 'B': 6})
    >>> normalized_box = normalize_box_data(box, method='z-score')
    >>> print(normalized_box.item1)
    {'A': -1.224744871391589, 'B': -1.224744871391589}
    
    Notes
    -----
    - When applying 'min-max' scaling, the function uses the range [0, 1]. 
      This behavior can be customized by modifying the function to accept range
      parameters if needed.
    - For Boxspace objects, ensure that all items are dictionary-like with 
      numerical values for the normalization to work properly.
    """

    if method not in ['z-score', 'min-max']:
        raise ValueError(f"Unsupported normalization method: {method}")

    original_type = type(data)

    if original_type not in [pd.DataFrame, Boxspace]:
        raise TypeError("Data must be either a pd.DataFrame or Boxspace object.")

    # Convert Boxspace to DataFrame if necessary
    if original_type == Boxspace:
        data = data.to_frame(error='ignore')

    numeric_cols = data.select_dtypes(include=np.number).columns

    if not numeric_cols.any():
        raise ValueError("No numeric columns found in data for normalization.")

    try:
        if method == 'z-score':
            from scipy.stats import zscore
            data[numeric_cols] = data[numeric_cols].apply(zscore)
        elif method == 'min-max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    except Exception as e:
        raise RuntimeError(f"Normalization failed: {e}")

    # Convert back to original data type if necessary
    if original_type == Boxspace:
        return Boxspace(**data.to_dict(orient='series'))
    
    return data

def merge_boxes(*boxes):
    """
    Merges multiple Boxspace objects into a single Boxspace object, ensuring that
    all inputs are Boxspace instances and handling duplicate keys gracefully.

    Parameters
    ----------
    *boxes : Boxspace
        An arbitrary number of Boxspace objects to merge.

    Returns
    -------
    Boxspace
        A new Boxspace object containing the merged contents of all input Boxspaces.

    Raises
    ------
    TypeError
        If any of the arguments are not instances of Boxspace.

    Warning
    -------
    UserWarning
        If duplicate keys are found, the function keeps the value from the 
        first occurrence and issues a warning about the duplicate keys ignored.

    Example
    -------
    >>> from gofast.tools.box import merge_boxes, Boxspace
    >>> box1 = Boxspace(a=1, b=2)
    >>> box2 = Boxspace(c=3, d=4, a=5)  # Note the duplicate key 'a'
    >>> merged_box = merge_boxes(box1, box2)
    UserWarning: Duplicate keys were found and ignored; kept the first occurrence: ['a']
    >>> print(merged_box.a, merged_box.c)
    1 3
    """
    if not all(isinstance(box, Boxspace) for box in boxes):
        raise TypeError("All arguments must be instances of Boxspace.")

    merged_dict = {}
    duplicates = []
    for box in boxes:
        for key, value in box.items():
            if key not in merged_dict:
                merged_dict[key] = value
            else:
                duplicates.append(key)

    if duplicates:
        import warnings
        duplicates = list(set(duplicates))  # Removing duplicates from the list of duplicates
        warnings.warn(
            f"Duplicate keys were found and ignored; kept the first occurrence: {duplicates}",
            UserWarning
        )

    return Boxspace(**merged_dict)
        
    
def deep_merge_boxspace(*boxes):
    """
    Deeply merges multiple Boxspace objects into one.

    Parameters
    ----------
    *boxes : Boxspace
        Variable number of Boxspace objects to merge.

    Returns
    -------
    Boxspace
        A new Boxspace object containing the deeply merged contents of all
        input Boxspaces.

    Example
    -------
    >>> from gofast.tools.box import deep_merge_boxspace, Boxspace
    >>> box1 = Boxspace(a=1, b=Boxspace(x=2, y=3))
    >>> box2 = Boxspace(b=Boxspace(y=4, z=5), c=6)
    >>> merged_box = deep_merge_boxspace(box1, box2)
    >>> print(merged_box.b.y, merged_box.b.z, merged_box.c)
    4 5 6
    """
    if not all(isinstance(box, Boxspace) for box in boxes):
        raise TypeError("All arguments must be instances of Boxspace.")
        
    def merge(dict1, dict2):
        for k, v in dict2.items():
            if k in dict1 and isinstance(dict1[k], Boxspace) and isinstance(v, Boxspace):
                merge(dict1[k], v)
            else:
                dict1[k] = v

    merged = Boxspace()
    for box in boxes:
        merge(merged, box)
    
    return merged

def save_to_box(data, key_column, value_columns=None):
    """
    Saves data from a DataFrame to a Boxspace, using one column's 
    values as keys.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing data to be saved in a Boxspace. 
        Must be an instance of pd.DataFrame.
    key_column : str
        The column in `data` whose values will be used as keys in the Boxspace. 
        Must exist in `data`.
    value_columns : list of str, optional
        The columns in `data` whose values will be saved under each key in 
        the Boxspace.
        If None, all other columns except `key_column` are included.

    Returns
    -------
    Boxspace
        A Boxspace object containing data indexed by `key_column` values.

    Raises
    ------
    TypeError
        If `data` is not an instance of pd.DataFrame.
    KeyError
        If `key_column` is not found in `data` or any `value_columns` do not 
        exist in `data`.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.tools.box import save_to_box
    >>> df = pd.DataFrame({'id': ['item1', 'item2'], 'value1': [10, 20], 'value2': [30, 40]})
    >>> box = save_to_box(df, 'id')
    >>> print(box.item1.value1, box.item2.value2)
    10 40
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The 'data' parameter must be an instance of pd.DataFrame.")

    if key_column not in data.columns:
        raise KeyError(f"The key column '{key_column}' is not found in the DataFrame.")
    
    if value_columns is not None:
        missing_cols = [col for col in value_columns if col not in data.columns]
        if missing_cols:
            raise KeyError("The following value columns are not found in"
                           f" the DataFrame: {missing_cols}")
    else:
        value_columns = [col for col in data.columns if col != key_column]
    
    try:
        box_data = {
            getattr(row, key_column): Bunch(**row[value_columns].to_dict())
            for _, row in data.iterrows()
        }
    except Exception as e:
        raise ValueError(f"An error occurred while processing the DataFrame: {e}")
    
    return Boxspace(**box_data)


def filter_boxspace(box, condition):
    """
    Filters attributes in a Boxspace object based on a given condition.

    Parameters
    ----------
    box : Boxspace
        The Boxspace object to be filtered. Must be an instance of Boxspace.
    condition : callable
        A function that takes a key-value pair (attribute name and value) and 
        returns True if the attribute should be included in the filtered 
        Boxspace. Must be a callable object.

    Returns
    -------
    Boxspace
        A new Boxspace object containing only the attributes that meet the 
        condition.

    Raises
    ------
    TypeError
        If `box` is not an instance of Boxspace or if `condition` is not callable.

    Example
    -------
    >>> from gofast.tools.box import filter_boxspace, Boxspace 
    >>> box = Boxspace(a=1, b=2, c=3)
    >>> filtered_box = filter_boxspace(box, lambda k, v: v > 1)
    >>> print(filtered_box)
    b : 2
    c : 3
    """
    if not isinstance(box, Boxspace):
        raise TypeError("The 'box' parameter must be an instance of Boxspace.")
    if not callable(condition):
        raise TypeError("The 'condition' parameter must be callable.")
    
    try:
        filtered_items = {k: v for k, v in box.items() if condition(k, v)}
    except Exception as e:
        raise ValueError(f"An error occurred while applying the condition: {e}")

    return Boxspace(**filtered_items)


def apply_to_boxspace(box, func):
    """
    Applies a function to each value in a Boxspace object.

    Parameters
    ----------
    box : Boxspace
        The Boxspace object whose values will be transformed. 
        Must be an instance of Boxspace.
    func : callable
        A function to apply to each value in the Boxspace. 
        Must be a callable object.

    Returns
    -------
    Boxspace
        A new Boxspace object with the function applied to each original value.

    Raises
    ------
    TypeError
        If `box` is not an instance of Boxspace or if `func` is not callable.

    Example
    -------
    >>> from gofast.tools.box import apply_to_boxspace, Boxspace
    >>> box = Boxspace(a=1, b=2, c=3)
    >>> squared_box = apply_to_boxspace(box, lambda x: x**2)
    >>> print(squared_box)
    a : 1
    b : 4
    c : 9
    """
    if not isinstance(box, Boxspace):
        raise TypeError("The 'box' parameter must be an instance of Boxspace.")
    if not callable(func):
        raise TypeError("The 'func' parameter must be callable.")
    
    try:
        transformed_items = {k: func(v) for k, v in box.items()}
    except Exception as e:
        raise ValueError(f"An error occurred while applying the function: {e}")

    return Boxspace(**transformed_items)

            
def transform_boxspace_attributes(box, func, attributes=None):
    """
    Applies a function selectively to attributes of a Boxspace object.

    Parameters
    ----------
    box : Boxspace
        The Boxspace object whose attributes are to be transformed.
    func : callable
        The function to apply to selected attributes of the Boxspace. 
        The function must take a single argument and return a value.
    attributes : list of str, optional
        A list of specific attribute names within the Boxspace to which the 
        function will be applied. If None, the function is applied to all 
        attributes.

    Returns
    -------
    None
        The function modifies the Boxspace object in place, transforming 
        the values of specified attributes according to the provided function.

    Raises
    ------
    TypeError
        If `box` is not an instance of Boxspace or if `func` is not callable.
    KeyError
        If any of the specified attributes do not exist in the Boxspace.

    Example
    -------
    >>> from gofast.tools.box import transform_boxspace_attributes, Boxspace
    >>> box = Boxspace(a=1, b=2, c=3)
    >>> def increment(x): return x + 1
    >>> transform_boxspace_attributes(box, increment, attributes=['a', 'c'])
    >>> print(box.a, box.b, box.c)
    2 2 4
    """
    if not isinstance(box, Boxspace):
        raise TypeError("The 'box' parameter must be an instance of Boxspace.")
    if not callable(func):
        raise TypeError("The 'func' parameter must be callable.")
    
    target_keys = attributes if attributes is not None else box.keys()
    
    for key in target_keys:
        if key in box:
            box[key] = func(box[key])
        else:
            raise KeyError(f"Attribute '{key}' not found in Boxspace.")
  
def boxspace_to_dataframe(boxspace):
    """
    Converts a Boxspace object containing multiple Bunch objects into a 
    pandas DataFrame.

    Parameters
    ----------
    boxspace : Boxspace
        The Boxspace object to convert.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the data contained in the Boxspace.

    Raises
    ------
    ValueError
        If the input is not an instance of Boxspace or if the Boxspace 
        does not contain Bunch objects.

    Example
    -------
    >>> import pandas as pd
    >>> from gofast.tools.box import Boxspace, Bunch
    >>> box = Boxspace(item1=Bunch(a=1, b=2), item2=Bunch(a=3, b=4))
    >>> df = boxspace_to_dataframe(box)
    >>> print(df)
       a  b
    0  1  2
    1  3  4
    """
    if not isinstance(boxspace, Boxspace):
        raise ValueError("Input must be an instance of Boxspace.")

    records = []
    for item in boxspace.values():
        if not isinstance(item, (Bunch, Boxspace)):
            raise ValueError("All items in the Boxspace must be Bunch objects.")
        records.append(item.__dict__)

    return pd.DataFrame(records)

def dataframe_to_boxspace(df, index_as_key='auto'):
    """
    Converts a pandas DataFrame to a Boxspace object, with each row represented
    as a Bunch object. Offers flexible handling of DataFrame's index for naming
    the Bunch objects within the Boxspace.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert.
    index_as_key : bool or 'auto', default 'auto'
        Determines how the DataFrame's index is used as keys in the Boxspace.
        If False, keys are generated as 'row_{index}'. If True, the index values
        are used as keys. If 'auto', index values are used if they are unique
        and of string type, otherwise 'row_{index}' pattern is used.

    Returns
    -------
    Boxspace
        A Boxspace object containing the DataFrame's data, with each row 
        as a Bunch object.

    Raises
    ------
    TypeError
        If the input is not an instance of pd.DataFrame.

    Example
    -------
    >>> import pandas as pd
    >>> from gofast.tools.box import Boxspace, Bunch, dataframe_to_boxspace
    >>> df = pd.DataFrame({'a': [1, 3], 'b': [2, 4]})
    >>> box = dataframe_to_boxspace(df)
    >>> print(box.row_0.a, box.row_1.b)
    1 4
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be an instance of pd.DataFrame.")

    box = Boxspace()
    use_index_as_key = index_as_key

    # Auto-determine if index can be used as keys
    if index_as_key == 'auto':
        unique_index = df.index.is_unique
        string_index = df.index.map(type).eq(str).all()
        use_index_as_key = unique_index and string_index

    for index, row in df.iterrows():
        key = str(index) if use_index_as_key else f"row_{index}"
        box[key] = Bunch(**row.to_dict())

    return box

def list_dicts_to_boxspace(list_dicts, key_func=lambda x, i: f"item_{i}"):
    """
    Converts a list of dictionaries into a Boxspace object, with each dictionary
    converted into a Bunch object.

    Parameters
    ----------
    list_dicts : list of dict
        A list where each element is a dictionary representing an item to be 
        included in the Boxspace. Must be a list of dictionaries.
    key_func : callable, optional
        A function that generates a key for each Bunch object in the Boxspace,
        given the dictionary and its index in the list. Must be a callable object.
        By default, keys are generated as 'item_{index}'.

    Returns
    -------
    Boxspace
        A Boxspace object containing Bunch objects for each dictionary in 
        the input list.

    Raises
    ------
    TypeError
        If `list_dicts` is not a list of dictionaries or if `key_func` is not callable.

    Example
    -------
    >>> list_dicts = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 25}]
    >>> box = list_dicts_to_boxspace(list_dicts)
    >>> print(box.item_0.name, box.item_1.age)
    John 25
    """
    if not isinstance(list_dicts, list) or not all(
            isinstance(item, dict) for item in list_dicts):
        raise TypeError("The 'list_dicts' parameter must be a list of dictionaries.")
    if not callable(key_func):
        raise TypeError("The 'key_func' parameter must be callable.")
    
    box = Boxspace()
    try:
        for i, dict_item in enumerate(list_dicts):
            key = key_func(dict_item, i)
            box[key] = Bunch(**dict_item)
    except Exception as e:
        raise ValueError("An error occurred while converting the"
                         f" list of dictionaries: {e}")

    return box

def flatten_boxspace(box, parent_key='', sep='_'):
    """
    Flattens a nested Boxspace object into a single-level Boxspace with keys
    representing the path to nested attributes.

    Parameters
    ----------
    box : Boxspace
        The Boxspace object to be flattened.
    parent_key : str, optional
        The prefix to append to child keys. Used internally for recursive calls.
    sep : str, optional
        The separator used between nested keys. Default is '_'.

    Returns
    -------
    Boxspace
        A flattened Boxspace object.

    Example
    -------
    >>> from gofast.tools.box import Boxspace, flatten_boxspace
    >>> nested_box = Boxspace(a=Boxspace(b=1, c=2), d=3)
    >>> flat_box = flatten_boxspace(nested_box)
    >>> print(flat_box.a_b, flat_box.a_c, flat_box.d)
    1 2 3
    """
    if not isinstance(box, ( Boxspace, Bunch)):
        raise TypeError("The 'box' parameter must be an instance of Boxspace.")
        
    items = []
    for k, v in box.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Boxspace):
            items.extend(flatten_boxspace(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return Boxspace(**dict(items))

         
def update_boxspace_if(box, condition, update_func):
    """
    Updates attributes in a Boxspace object if they meet a specified 
    condition.

    Parameters
    ----------
    box : Boxspace
        The Boxspace object to be updated. Must be an instance of Boxspace.
    condition : callable
        A function that takes a key-value pair (attribute name and value) and 
        returns True if the update should be applied. Must be a callable object.
    update_func : callable
        A function to update the value of attributes that meet the condition.
        Must be a callable object.

    Returns
    -------
    None
        The Boxspace object is modified in place.

    Raises
    ------
    TypeError
        If `box` is not an instance of Boxspace or if either `condition` or 
        `update_func` is not callable.

    Example
    -------
    >>> from gofast.tools.box import Boxspace, update_boxspace_if
    >>> box = Boxspace(x=10, y=20, z=5)
    >>> update_boxspace_if(box, lambda k, v: v > 10, lambda x: x + 5)
    >>> print(box.x, box.y, box.z)
    10 25 5
    """
    if not isinstance(box, ( Boxspace, Bunch)):
        raise TypeError("The 'box' parameter must be an instance of Boxspace.")
    if not callable(condition):
        raise TypeError("The 'condition' parameter must be callable.")
    if not callable(update_func):
        raise TypeError("The 'update_func' parameter must be callable.")
    
    try:
        for k, v in box.items():
            if condition(k, v):
                box[k] = update_func(v)
    except Exception as e:
        raise ValueError(f"An error occurred while applying the update function: {e}")

if __name__=='__main__':
    # from gofast.tools.box import Bunch, Boxspace, save_to_box
    # import pandas as pd
    
    # Simulating some data
    df = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3'],
        'age': [25, 32, 18],
        'score': [88, 92, 95]
    })
    
    # Saving data into a Boxspace for easy access
    user_data_box = save_to_box(df, 'user_id')
    
    # Accessing data
    print(f"User2's age: {user_data_box.user2.age}")
    print(f"User3's score: {user_data_box.user3.score}")
    
    # Creating a Bunch for summary statistics
    summary_stats = Bunch(mean_age=df['age'].mean(), mean_score=df['score'].mean())
    print(f"Mean age: {summary_stats.mean_age}")
    print(f"Mean score: {summary_stats.mean_score}")
    
    # from gofast.tools.box import Boxspace, filter_boxspace, apply_to_boxspace
    
    # Example dataset of users and their attributes
    users = Boxspace(
        john=Bunch(age=28, score=85),
        doe=Bunch(age=34, score=90),
        jane=Bunch(age=22, score=95)
    )
    
    # Filtering users based on score
    high_scorers = filter_boxspace(users, lambda k, v: v.score > 90)
    print("High scorers:", high_scorers.keys())
    
    # Applying a transformation to the age attribute of each user
    aged_users = apply_to_boxspace(users, lambda v: Bunch(age=v.age + 10, score=v.score))
    print("Aged users:")
    for user, attrs in aged_users.items():
        print(f"{user}: age={attrs.age}, score={attrs.score}")
    
    # Output:
    # High scorers: dict_keys(['jane'])
    # Aged users:
    # john: age=38, score=85
    # doe: age=44, score=90
    # jane: age=32, score=95
                   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
