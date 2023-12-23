# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations
import re
import sys
from warnings import warn
import pandas as pd

from ._docstring import DocstringComponents, _core_docs
from ._gofastlog import gofastlog
from ._typing import List, Optional, DataFrame, Tuple
from .tools.baseutils import _is_readable
from .exceptions import NotFittedError
from .tools._dependency import import_optional_dependency
from .tools.funcutils import (
    sanitize_frame_cols,
    exist_features,
   _assert_all_types, 
   repr_callable_obj, 
   smart_strobj_recognition,
   inspect_data
   )
from .tools.validator import array_to_frame, check_array, build_data_if

_logger = gofastlog().get_gofast_logger(__name__)


__all__ = ["Data", "Missing", "MergeableSeries", "MergeableFrames",
           "FrameOperations"]

# +++ add base documentations +++
_base_params = dict(
    axis="""
axis: {0 or 'index', 1 or 'columns'}, default 0
    Determine if rows or columns which contain missing values are 
    removed.
    * 0, or 'index' : Drop rows which contain missing values.
    * 1, or 'columns' : Drop columns which contain missing value.
    Changed in version 1.0.0: Pass tuple or list to drop on multiple 
    axes. Only a single axis is allowed.    
    """,
    columns="""
columns: str or list of str 
    columns to replace which contain the missing data. Can use the axis 
    equals to '1'.
    """,
    name="""
name: str, :attr:`pandas.Series.name`
    A singluar column name. If :class:`pandas.Series` is given, 'name'  
    denotes the attribute of the :class:`pandas.Series`. Preferably `name`
    must correspond to the label name of the target. 
    """,
    sample="""
sample: int, Optional, 
    Number of row to visualize or the limit of the number of sample to be 
    able to see the patterns. This is usefull when data is composed of 
    many rows. Skrunked the data to keep some sample for visualization is 
    recommended.  ``None`` plot all the samples ( or examples) in the data     
    """,
    kind="""
kind: str, Optional 
    type of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar``. 
    ``corr`` plot  for dendrogram , :mod:`msno` bar,  :mod:`plt`
    and :mod:`msno` correlation  visualization respectively: 
        * ``bar`` plot counts the  nonmissing data  using pandas
        *  ``mbar`` use the :mod:`msno` package to count the number 
            of nonmissing data. 
        * dendrogram`` show the clusterings of where the data is missing. 
            leaves that are the same level predict one onother presence 
            (empty of filled). The vertical arms are used to indicate how  
            different cluster are. short arms mean that branch are 
            similar. 
        * ``corr` creates a heat map showing if there are correlations 
            where the data is missing. In this case, it does look like 
            the locations where missing data are corollated.
        * ``None`` is the default vizualisation. It is useful for viewing 
            contiguous area of the missing data which would indicate that 
            the missing data is  not random. The :code:`matrix` function 
            includes a sparkline along the right side. Patterns here would 
            also indicate non-random missing data. It is recommended to limit 
            the number of sample to be able to see the patterns. 
    Any other value will raise an error. 
    """,
    inplace="""
inplace: bool, default False
    Whether to modify the DataFrame rather than creating a new one.    
    """
)

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    base=DocstringComponents(_base_params)
)
# +++ end base documentations +++


class Data:
    def __init__(self, verbose: int = 0):
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        self.verbose = verbose
        self.data_ = None

    @property
    def data(self):
        """ return verified data """
        return self.data_

    @data.setter
    def data(self, d):
        """ Read and parse the data"""
        self.data_ = _is_readable(d)

    @property
    def describe(self):
        """ Get summary stats  as well as see the cound of non-null data.
        Here is the default behaviour of the method i.e. it is to only report  
        on numeric columns. To have have full control, do it manually by 
        yourself. 

        """
        return self.data.describe()

    def fit(self, data: str | DataFrame = None):
        """ Read, assert and fit the data.

        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.

        """

        if data is not None:
            self.data = data
        check_array(
            self.data,
            force_all_finite='allow-nan',
            dtype=object,
            input_name='Data',
            to_frame=True
        )
        # for consistency if not a frame, set to aframe
        self.data = array_to_frame(
            self.data, to_frame=True, input_name='col_', force=True
        )
        data = sanitize_frame_cols(self.data, fill_pattern='_')
        for col in data.columns:
            setattr(self, col, data[col])

        return self

    def shrunk(self,
               columns: list[str],
               data: str | DataFrame = None,
               **kwd
               ):
        """ Reduce the data with importance features

        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        columns: str or list of str 
            Columns or features to keep in the datasets

        kwd: dict, 
        additional keywords arguments from :func:`gofast.tools.mlutils.selectfeatures`

        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.

        """
        self.inspect

        self.data = selectfeatures(
            self.data, features=columns, **kwd)

        return self

    @property
    def inspect(self):
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""

        msg = ("{dobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )

        if self.data_ is None:
            raise NotFittedError(msg.format(
                dobj=self)
            )
        return 1

    def profilingReport(self, data: str | DataFrame = None, **kwd):
        """Generate a report in a notebook. 

        It will summarize the types of the columns and allow yuou to view 
        details of quatiles statistics, a histogram, common values and extreme 
        values. 

        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.

        Examples 
        ---------
        >>> from gofast.base import Data 
        >>> Data().fit(data).profilingReport()

        """
        extra_msg = ("'Data.profilingReport' method uses 'pandas-profiling'"
                     " as a dependency.")
        import_optional_dependency("pandas_profiling", extra=extra_msg)

        self.inspect

        self.data = data or self.data

        try:
            from pandas_profiling import ProfileReport
        except ImportError:

            msg = (f"Missing of 'pandas_profiling package. {extra_msg}"
                   " Cannot plot profiling report. Install it using pip"
                   " or conda.")
            warn(msg)
            raise ImportError(msg)

        return ProfileReport(self.data, **kwd)

    def rename(self,
               data: str | DataFrame = None,
               columns: List[str] = None,
               pattern: Optional[str] = None
               ):
        """ 
        rename columns of the dataframe with columns in lowercase and spaces 
        replaced by underscores. 

        Parameters 
        -----------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        columns: str or list of str, Optional 
            the  specific columns in dataframe to renames. However all columns 
            is put in lowercase. If columns not in dataframe, error raises.  

        pattern: str, Optional, 
            Regular expression pattern to strip the data. By default, the 
            pattern is ``'[ -@*#&+/]'``.

        Return
        -------
        ``self``: :class:`~gofast.base.Data` instance 
            returns ``self`` for easy method chaining.

        """
        pattern = str(pattern)

        if pattern == 'None':
            pattern = r'[ -@*#&+/]'
        regex = re.compile(pattern, flags=re.IGNORECASE)

        if data is not None:
            self.data = data

        self.data.columns = self.data.columns.str.strip()
        if columns is not None:
            exist_features(self.data, columns, 'raise')

        if columns is not None:
            self.data[columns].columns = self.data[columns].columns.str.lower(
            ).map(lambda o: regex.sub('_', o))
        if columns is None:
            self.data.columns = self.data.columns.str.lower().map(
                lambda o: regex.sub('_', o))

        return self

    # XXX TODO # use logical and to quick merge two frames
    def merge(self):
        """ Merge two series whatever the type with operator `&&`. 

        When series as dtype object as non numeric values, dtypes should be 
        change into a object 
        """
        # try :
        #     self.data []
    # __and__= __rand__ = merge

    def drop(
        self,
        labels: list[str | int] = None,
        columns: List[str] = None,
        inplace: bool = False,
        axis: int = 0, **kws
    ):
        """ Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding 
        axis, or by specifying directly index or column names. When using a 
        multi-index, labels on different levels can be removed by specifying 
        the level.

        Parameters 
        -----------
        labels: single label or list-like
            Index or column labels to drop. A tuple will be used as a single 
            label and not treated as a list-like.

        axis: {0 or 'index', 1 or 'columns'}, default 0
            Whether to drop labels from the index (0 or 'index') 
            or columns (1 or 'columns').

        columns: single label or list-like
            Alternative to specifying axis 
            (labels, axis=1 is equivalent to columns=labels)
        kws: dict, 
            Additionnal keywords arguments passed to :meth:`pd.DataFrame.drop`.

        Returns 
        ----------
        DataFrame or None
            DataFrame without the removed index or column labels or 
            None if `inplace` equsls to ``True``.

        """
        self.inspect

        data = self.data.drop(labels=labels,  inplace=inplace,
                              columns=columns, axis=axis, **kws)
        return data

    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj(self, skip='y')

    def __getattr__(self, name):
        if name.endswith('_'):
            if name not in self.__dict__.keys():
                if name in ('data_', 'X_'):
                    raise NotFittedError(
                        f'Fit the {self.__class__.__name__!r} object first'
                    )

        rv = smart_strobj_recognition(name, self.__dict__, deep=True)
        appender = "" if rv is None else f'. Do you mean {rv!r}'

        raise AttributeError(
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
        )


Data.__doc__ = """\
Data base class

Typically, we train a model with a matrix of data. Note that pandas Dataframe 
is the most used because it is very nice to have columns lables even though 
Numpy arrays work as well. 

For supervised Learning for instance, suc as regression or clasification, our 
intent is to have a function that transforms features into a label. If we 
were to write this as an algebra formula, it would be look like:
    
.. math::
    
    y = f(X)

:code:`X` is a matrix. Each row represent a `sample` of data or information 
about individual. Every columns in :code:`X` is a `feature`.The output of 
our function, :code:`y`, is a vector that contains labels (for classification)
or values (for regression). 

In Python, by convention, we use the variable name :code:`X` to hold the 
sample data even though the capitalization of variable is a violation of  
standard naming convention (see PEP8). 

Parameters 
-----------
{params.core.data}
{params.base.columns}
{params.base.axis}
{params.base.sample}
{params.base.kind}
{params.base.inplace}
{params.core.verbose}

Returns
-------
{returns.self}
   
Examples
--------
.. include:: ../docs/data.rst

""".format(
    params=_param_docs,
    returns=_core_docs["returns"],
)


class Missing (Data):
    """ Deal with missing values in Data 

    Most algorithms will not work with missing data. Notable exceptions are the 
    recent boosting libraries such as the XGBoost 
    (:doc:`gofast.documentation.xgboost.__doc__`) CatBoost and LightGBM. 
    As with many things in machine learning , there are no hard answaers for how 
    to treat a missing data. Also, missing data could  represent different 
    situations. There are three warious way to handle missing data:: 

        * Remove any row with missing data 
        * Remove any columns with missing data 
        * Impute missing values 
        * Create an indicator columns to indicator data was missing 

    Parameters
    ----------- 
    in_percent: bool, 
        give the statistic of missing data in percentage if ser to ``True``. 

    sample: int, Optional, 
        Number of row to visualize or the limit of the number of sample to be 
        able to see the patterns. This is usefull when data is composed of 
        many rows. Skrunked the data to keep some sample for visualization is 
        recommended.  ``None`` plot all the samples ( or examples) in the data 
    kind: str, Optional 
        type of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar``. 
        ``corr`` plot  for dendrogram , :mod:`msno` bar,  :mod:`plt`
        and :mod:`msno` correlation  visualization respectively: 

            * ``bar`` plot counts the  nonmissing data  using pandas
            *  ``mbar`` use the :mod:`msno` package to count the number 
                of nonmissing data. 
            * dendrogram`` show the clusterings of where the data is missing. 
                leaves that are the same level predict one onother presence 
                (empty of filled). The vertical arms are used to indicate how  
                different cluster are. short arms mean that branch are 
                similar. 
            * ``corr` creates a heat map showing if there are correlations 
                where the data is missing. In this case, it does look like 
                the locations where missing data are corollated.
            * ``None`` is the default vizualisation. It is useful for viewing 
                contiguous area of the missing data which would indicate that 
                the missing data is  not random. The :code:`matrix` function 
                includes a sparkline along the right side. Patterns here would 
                also indicate non-random missing data. It is recommended to limit 
                the number of sample to be able to see the patterns. 

        Any other value will raise an error 

    Examples 
    --------
    >>> from gofast.base import Missing
    >>> data ='data/geodata/main.bagciv.data.csv' 
    >>> ms= Missing().fit(data) 
    >>> ms.plot_.fig_size = (12, 4 ) 
    >>> ms.plot () 

    """

    def __init__(self,
                 in_percent=False,
                 sample=None,
                 kind=None,
                 drop_columns: List[str] = None,
                 **kws):

        self.in_percent = in_percent
        self.kind = kind
        self.sample = sample
        self.drop_columns = drop_columns
        self.isnull_ = None

        super().__init__(**kws)

    @property
    def isnull(self):
        """ Check the mean values  in the data  in percentge"""
        self.isnull_ = self.data.isnull().mean(
        ) * 1e2 if self.in_percent else self.data.isnull().mean()

        return self.isnull_

    def plot(self, figsize: Tuple[int] = None,  **kwd):
        """
        Vizualize patterns in the missing data.

        Parameters 
        ------------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        kind: str, Optional 
            kind of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar`` plot 
            for dendrogram , :mod:`msno` bar and :mod:`plt` visualization 
            respectively: 

                * ``bar`` plot counts the  nonmissing data  using pandas
                *  ``mbar`` use the :mod:`msno` package to count the number 
                    of nonmissing data. 
                * dendrogram`` show the clusterings of where the data is missing. 
                    leaves that are the same level predict one onother presence 
                    (empty of filled). The vertical arms are used to indicate how  
                    different cluster are. short arms mean that branch are 
                    similar. 
                * ``corr` creates a heat map showing if there are correlations 
                    where the data is missing. In this case, it does look like 
                    the locations where missing data are corollated.
                * ``None`` is the default vizualisation. It is useful for viewing 
                    contiguous area of the missing data which would indicate that 
                    the missing data is  not random. The :code:`matrix` function 
                    includes a sparkline along the right side. Patterns here would 
                    also indicate non-random missing data. It is recommended to limit 
                    the number of sample to be able to see the patterns. 

                Any other value will raise an error 

        sample: int, Optional
            Number of row to visualize. This is usefull when data is composed of 
            many rows. Skrunked the data to keep some sample for visualization is 
            recommended.  ``None`` plot all the samples ( or examples) in the data 

        kws: dict 
            Additional keywords arguments of :mod:`msno.matrix` plot. 

        Return
        -------
        ``self``: :class:`~gofast.base.Missing` instance 
            returns ``self`` for easy method chaining.


        Examples 
        --------
        >>> from gofast.base import Missing
        >>> data ='data/geodata/main.bagciv.data.csv' 
        >>> ms= Missing().fit(data) 
        >>> ms.plot(figsize = (12, 4 ) ) 


        """
        self.inspect
        from .view.plot import ExPlot

        ExPlot(fig_size=figsize).fit(self.data).plotmissing(
            kind=self.kind, sample=self.sample, **kwd)
        return self

    @property
    def get_missing_columns(self):
        """ return columns with Nan Values """
        return list(self.data.columns[self.data.isna().any()])

    def drop(self,
             data: str | DataFrame = None,
             columns: List[str] = None,
             inplace=False,
             axis=1,
             **kwd
             ):
        """Remove missing data 

        Parameters 
        -----------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        columns: str or list of str 
            columns to drop which contain the missing data. Can use the axis 
            equals to '1'.

        axis: {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are 
            removed.
            * 0, or 'index' : Drop rows which contain missing values.

            * 1, or 'columns' : Drop columns which contain missing value.
            Changed in version 1.0.0: Pass tuple or list to drop on multiple 
            axes. Only a single axis is allowed.

        how: {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we 
            have at least one NA or all NA.

            * 'any': If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.

        thresh: int, optional
            Require that many non-NA values. Cannot be combined with how.

        subset: column label or sequence of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows 
            these would be a list of columns to include.

        inplace: bool, default False
            Whether to modify the DataFrame rather than creating a new one.

        Returns 
        -------
        ``self``: :class:`~gofast.base.Missing` instance 
            returns ``self`` for easy method chaining.

        """
        if data is not None:
            self.data = data

        self.inspect
        if columns is not None:
            self.drop_columns = columns

        exist_features(self.data, self.drop_columns, error='raise')

        if self.drop_columns is None:
            if inplace:
                self.data.dropna(axis=axis, inplace=True, **kwd)
            else:
                self.data = self.data .dropna(
                    axis=axis, inplace=False, **kwd)

        elif self.drop_columns is not None:
            if inplace:
                self.data.drop(columns=self.drop_columns,
                               axis=axis, inplace=True,
                               **kwd)
            else:
                self.data.drop(columns=self.columns, axis=axis,
                               inplace=False, **kwd)

        return self

    @property
    def sanity_check(self):
        """Ensure that we have deal with all missing values. The following 
        code returns a single boolean if there is any cell that is missing 
        in a DataFrame """

        return self.data.isna().any().any()

    def replace(self,
                data: str | DataFrame = None,
                columns: List[str] = None,
                fill_value: float = None,
                new_column_name: str = None,
                return_non_null: bool = False,
                **kwd):
        """ 
        Replace the missing values to consider. 

        Use the :code:`coalease` function of :mod:`pyjanitor`. It takes a  
        dataframe and a list of columns to consider. This is a similar to 
        functionality found in Excel and SQL databases. It returns the first 
        non null value of each row. 

        Parameters 
        -----------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        columns: str or list of str 
            columns to replace which contain the missing data. Can use the axis 
            equals to '1'.

        axis: {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are 
            removed.
            * 0, or 'index' : Drop rows which contain missing values.

            * 1, or 'columns' : Drop columns which contain missing value.
            Changed in version 1.0.0: Pass tuple or list to drop on multiple 
            axes. Only a single axis is allowed.

         Returns 
         -------
         ``self``: :class:`~gofast.base.Missing` instance 
             returns ``self`` for easy method chaining.

        """

        if data is not None:
            self.data = data

        self.inspect
        exist_features(self.data, columns)

        if return_non_null:
            new_column_name = _assert_all_types(new_column_name, str)

            if 'pyjanitor' not in sys.modules:
                raise ModuleNotFoundError(" 'pyjanitor' is missing.Install it"
                                          " mannualy using conda or pip.")
            import pyjanitor as jn
            return jn.coalease(self.data,
                               columns=columns,
                               new_column_name=new_column_name,
                               )
        if fill_value is not None:
            # fill missing values with a particular values.

            try:
                self.data = self.data .fillna(fill_value, **kwd)
            except:
                if 'pyjanitor' in sys.modules:
                    import pyjanitor as jn
                    jn.fill_empty(
                        self.data, columns=columns or list(self.data.columns),
                        value=fill_value
                    )

        return self


def selectfeatures(
    df: DataFrame,
    features: List[str] = None,
    include=None,
    exclude=None,
    coerce: bool = False,
    **kwd
):
    """ Select features  and return new dataframe.  

    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param include: the type of data to retrieved in the dataframe `df`. Can  
        be ``number``. 
    :param exclude: type of the data to exclude in the dataframe `df`. Can be 
        ``number`` i.e. only non-digits data will be keep in the data return.
    :param coerce: return the whole dataframe with transforming numeric columns.
        Be aware that no selection is done and no error is raises instead. 
        *default* is ``False``
    :param kwd: additional keywords arguments from `pd.astype` function 

    :ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    """

    if features is not None:
        exist_features(df, features, error='raise')
    # change the dataype
    df = df.astype(float, errors='ignore', **kwd)
    # assert whether the features are in the data columns
    if features is not None:
        return df[features]
    # raise ValueError: at least one of include or exclude must be nonempty
    # use coerce to no raise error and return data frame instead.
    return df if coerce else df.select_dtypes(include, exclude)


class MergeableSeries:
    """
    A class that wraps a pandas Series to enable logical AND operations
    using the & operator, even for non-numeric data types.
    """

    def __init__(self, series):
        """
        Initialize the MergeableSeries object.

        Parameters
        ----------
        series : pandas.Series
            The pandas Series to be wrapped.
        """
        self.series = series

    def __and__(self, other):
        """
        Overload the & operator to perform logical AND between two Series.

        Parameters
        ----------
        other : MergeableSeries
            Another MergeableSeries object to perform logical AND with.

        Returns
        -------
        pandas.Series
            A new Series containing the result of the logical AND operation.

        Raises
        ------
        ValueError
            If 'other' is not an instance of MergeableSeries.
        Example 
        --------

        >>> from gofast.base import MergeableSeries 
        >>> s1 = MergeableSeries(pd.Series([True, False, 'non-numeric']))
        >>> s2 = MergeableSeries(pd.Series(['non-numeric', False, True]))
        >>> result = s1 & s2
        >>> print(result)
        """
        if not isinstance(other, MergeableSeries):
            raise ValueError("Operand must be an instance of MergeableSeries")

        # Convert non-numeric types to string for logical operations
        series1 = (self.series.astype(str) if self.series.dtype == 'object'
                   else self.series)
        series2 = (other.series.astype(str) if other.series.dtype == 'object'
                   else other.series)

        # Perform logical AND operation
        return series1 & series2


class FrameOperations:
    """
    A class for performing various operations on pandas DataFrames.

    This class provides methods to merge, concatenate, compare, and
    perform arithmetic operations on two or more pandas DataFrames.

    """

    def __init__(self, ):
        ...

    def fit(self, *frames, **kws):
        """ Inspect frames 

        Parameters 
        -----------
        *frames : pandas.DataFrame
            Variable number of pandas DataFrame objects to be operated on.

        kws: dict, 
           Additional keywords arguments passed to 
           func:`gofast.tools.funcutils.inspect_data`
        Returns 
        ---------
        self: Object for chainings methods. 

        """
        frames = []
        for frame in self.frames:
            frames.append(inspect_data(frames, **kws))

        self.frames = frames

        return self

    def merge_frames(self, on, how='inner', **kws):
        """
        Merge two or more DataFrames on a key column.

        Parameters
        ----------
        on : str or list
            Column or index level names to join on. Must be found 
            in both DataFrames.
        how : str, default 'inner'
            Type of merge to be performed. Options include 'left',
            'right', 'outer', 'inner'.

        kws: dict, 
           Additional keyword arguments passed to `pd.merge`
        Returns
        -------
        pandas.DataFrame
            A DataFrame resulting from the merge operation.

        Examples
        --------
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df2 = pd.DataFrame({'A': [2, 3], 'C': [5, 6]})
        >>> df_ops = DataFrameOperations(df1, df2)
        >>> df_ops.merge_dataframes(on='A')
        """
        result = self.frames[0]
        for df in self.frames[1:]:
            result = pd.merge(result, df, on=on, how=how, **kws)
        return result

    def concat_frames(self, axis=0, **kws):
        """
        Concatenate two or more DataFrames along a particular axis.

        Parameters
        ----------
        axis : {0/'index', 1/'columns'}, default 0
            The axis to concatenate along.

        kws: dict, 
           keywords arguments passed to `pandas.concat`
        Returns
        -------
        pandas.DataFrame
            A DataFrame resulting from the concatenation.

        Examples
        --------
        >>> df1 = pd.DataFrame({'A': [1, 2]})
        >>> df2 = pd.DataFrame({'B': [3, 4]})
        >>> df_ops = DataFrameOperations(df1, df2)
        >>> df_ops.concatenate_dataframes(axis=1)
        """
        return pd.concat(self.dataframes, axis=axis, **kws)

    def compare_frames(self):
        """
        Compare the dataframes for equality.

        Returns
        -------
        bool
            True if all dataframes are equal, False otherwise.

        Examples
        --------
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df2 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df_ops = DataFrameOperations(df1, df2)
        >>> df_ops.compare_dataframes()
        """
        first_df = self.dataframes[0]
        for df in self.dataframes[1:]:
            if not first_df.equals(df):
                return False

        return True

    def add_frames(self):
        """
        Perform element-wise addition of two or more DataFrames.

        Returns
        -------
        pandas.DataFrame
            A DataFrame resulting from the element-wise addition.

        Examples
        --------
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        >>> df_ops = DataFrameOperations(df1, df2)
        >>> df_ops.add_dataframes()
        """
        result = self.dataframes[0].copy()
        for df in self.dataframes[1:]:
            result = result.add(df, fill_value=0)
        return result
    
    def conditional_filter(self, conditions):
        """
        Filter the DataFrame based on multiple conditional criteria.
    
        Parameters
        ----------
        conditions : dict
            A dictionary where keys are column names and values are 
            functions that take a single argument and return a boolean.
    
        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
    
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df_ops = ComplexDataFrameOperations(df)
        >>> conditions = {'A': lambda x: x > 1, 'B': lambda x: x < 6}
        >>> df_ops.conditional_filter(conditions)
        """
        mask = pd.Series(True, index=self.dataframe.index)
        for col, condition in conditions.items():
            mask &= self.dataframe[col].apply(condition)
            
        return self.dataframe[mask]

class MergeableFrames:
    """
    A class that wraps pandas DataFrames to enable logical operations
    (like AND, OR) using bitwise operators on DataFrames.

    This class provides a way to intuitively perform logical operations
    between multiple DataFrames, especially useful for conditional
    filtering and data analysis.
    
    Parameters
    ----------
    frame : pandas.DataFrame
        The pandas DataFrame to be wrapped.
    kws: dict, 
       Additional keyword arguments  to build a frame if the array is passed 
       rather than the dataframe. 

    Attributes
    ----------
    dataframe : pandas.DataFrame
        The pandas DataFrame to be wrapped.

    Methods
    -------
    __and__(self, other)
        Overloads the & operator to perform logical AND between DataFrames.

    __or__(self, other)
        Overloads the | operator to perform logical OR between DataFrames.

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [True, False], 'B': [False, True]})
    >>> df2 = pd.DataFrame({'A': [False, True], 'B': [True, False]})
    >>> mergeable_df1 = MergeableDataFrames(df1)
    >>> mergeable_df2 = MergeableDataFrames(df2)
    >>> and_result = mergeable_df1 & mergeable_df2
    >>> or_result = mergeable_df1 | mergeable_df2
    """

    def __init__(self, frame, **kws ):

        self.frame = build_data_if(frame, force=True , **kws )

    def __and__(self, other):
        """
        Overload the & operator to perform logical AND between 
        two DataFrames.

        Parameters
        ----------
        other : MergeableFrames
            Another MergeableFrames object to perform logical AND with.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing the result of the logical AND operation.

        Raises
        ------
        ValueError
            If 'other' is not an instance of MergeableDataFrames.
        """
        if not isinstance(other, MergeableFrames):
            raise ValueError("Operand must be an instance of MergeableFrames")

        return self.frame & other.frame

    def __or__(self, other):
        """
        Overload the | operator to perform logical OR between two DataFrames.

        Parameters
        ----------
        other : MergeableDataFrames
            Another MergeableDataFrames object to perform logical OR with.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing the result of the logical OR operation.

        Raises
        ------
        ValueError
            If 'other' is not an instance of MergeableDataFrames.
        """
        if not isinstance(other, MergeableFrames):
            raise ValueError("Operand must be an instance of MergeableFrames")

        return self.frame | other.frame


