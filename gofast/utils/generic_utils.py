# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides common helper functions and for validation, 
comparison, and other generic operations
"""
import warnings
import inspect

from typing import ( 
    Union, Optional, 
    Dict, Any, List
)
import numpy as np 
import pandas as pd 

__all__ =['verify_identical_items', 'vlog', 'detect_dt_format',
          'get_actual_column_name', 'transform_contributions', 
          'exclude_duplicate_kwargs']

def verify_identical_items(
    list1, 
    list2, 
    mode: str = "unique", 
    ops: str = "check_only", 
    error: str = "raise", 
    objname: str = None, 
) -> Union[bool, list]:
    """
    Check if two lists contain identical elements according 
    to the specified mode.

    In "unique" mode, the function compares the unique elements
    in each list.
    In "ascending" mode, it compares elements pairwise in order.

    Parameters
    ----------
    list1     : list
        The first list of items.
    list2`     : list
        The second list of items.
    mode      : {'unique', 'ascending'}, default="unique"
        The mode of comparison:
          - "unique": Compare unique elements (order-insensitive).
          - "ascending": Compare each element pairwise in order.
    ops       : {'check_only', 'validate'}, default="check_only"
        If "check_only", returns True/False indicating a match.
        If "validate", returns the validated list.
    error     : {'raise', 'warn', 'ignore'}, default="raise"
        Specifies how to handle mismatches.
    objname   : str, optional
        A name to include in error messages.

    Returns
    -------
    bool or list
        Depending on `ops`, returns True/False or the validated list.

    Examples
    --------
    >>> from gofast.utils.generic_utils import verify_identical_items 
    >>> list1 = [0.1, 0.5, 0.9]
    >>> list2 = [0.1, 0.5, 0.9]
    >>> verify_identical_items(list1, list2, mode="unique", ops="validate")
    [0.1, 0.5, 0.9]
    >>> verify_identical_items(list1, list2, mode="ascending", ops="check_only")
    True

    Notes
    -----
    In "ascending" mode, both lists must have the same length, and the
    function compares each corresponding pair of elements.
    In "unique" mode, the function uses the set of unique values for
    comparison. If the lists contain mixed types, the function attempts
    to compare their string representations.
    """
    # Validate mode.
    if mode not in ("unique", "ascending"):
        raise ValueError("mode must be either 'unique' or 'ascending'")
    if ops not in ("check_only", "validate"):
        raise ValueError("ops must be either 'check_only' or 'validate'")
    if error not in ("raise", "warn", "ignore"):
        raise ValueError(
            "error must be one of 'raise', 'warn', or 'ignore'")

    # Ascending mode: compare each element in order.
    if mode == "ascending":
        if len(list1) != len(list2):
            msg = (
                f"Length mismatch in {objname or 'object lists'}: "
                f"{len(list1)} vs {len(list2)}."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings
                warnings.warn(msg, UserWarning)
            return False
        
        differences = []
        for idx, (a, b) in enumerate(zip(list1, list2)):
            if a != b:
                differences.append((idx, a, b))
        if differences:
            msg = (
                f"Differences in {objname or 'object lists'}: {differences}."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings
                warnings.warn(msg, UserWarning)
            return False
        return True if ops == "check_only" else list1

    # Unique mode: compare the unique elements of each list.
    else:
        try:
            unique1 = sorted(set(list1))
            unique2 = sorted(set(list2))
        except Exception:
            unique1 = sorted({str(x) for x in list1})
            unique2 = sorted({str(x) for x in list2})
        if unique1 != unique2:
            msg = (
                f"Inconsistent unique elements in {objname or 'object lists'}: "
                f"{unique1} vs {unique2}."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings
                warnings.warn(msg, UserWarning)
            return False
        return True if ops == "check_only" else unique1


def vlog(
    message,
    verbose=None,
    level: int = 3,
    depth: Union[int, str] = "auto",
    mode:Optional[str]=None,
    vp: bool=True
):
    """
    Log or naive messages with optional indentation and
    bracketed tags.

    This function, `vlog`, allows conditional logging
    or printing of messages based on a global or passed
    in <parameter inline> `verbose` level. By default,
    it behaves differently depending on whether
    ``mode`` is ``'log'`` or ``'naive'``. When
    :math:`mode = 'log'`, the message is printed only if
    :math:`\\text{verbose} \\geq \\text{level}`. Otherwise,
    for :math:`mode` in [``None``, ``'naive'``], the
    verbosity threshold leads to various bracketed
    prefixes (e.g. [INFO], [DEBUG], [TRACE]) unless the
    message already contains such a prefix.

    .. math::
       \\text{indentation} = 2 \\times \\text{depth}

    where :math:`\\text{depth}` is either manually
    specified or auto-derived based on `<parameter inline>`
    `level` (1 = ERROR, 2 = WARNING, 3 = INFO, 4/5 =
    DEBUG, 6/7 = TRACE).

    Parameters
    ----------
    message : str
        The text to be printed or logged.
    verbose : int, optional
        Overall verbosity threshold. If ``None``, it
        looks for a global variable named ``verbose``.
        Default is ``None``.
    level : int, default=3
        Severity or importance level of the message.
        Commonly:
          * 1 = ERROR
          * 2 = WARNING
          * 3 = INFO
          * 4,5 = DEBUG
          * 6,7 = TRACE
    depth : int or str, default="auto"
        Indentation level used for the printed message.
        If ``"auto"``, the depth is computed from
        `<parameter inline> level`.
    mode : str, optional
        Determines logging mode. If set to ``'log'``,
        prints messages only if
        :math:`\\text{verbose} \\geq \\text{level}`.
        Otherwise (if ``None`` or ``'naive'``), it
        follows a custom logic driven by `<parameter
        inline> verbose`.
    vp : bool, default=True
        If ``True``, the function automatically prepends
        bracketed tags (e.g. [INFO]) unless the message
        already contains one of [INFO], [DEBUG], [ERROR],
        [WARNING], or [TRACE].

    Returns
    -------
    None
        This function does not return anything. It either
        prints the message to stdout or omits it,
        depending on `<parameter inline> verbose`, `<parameter
        inline> level`, and ``mode``.

    Notes
    -----
    This function is helpful for selectively displaying
    or logging messages in applications that adapt to
    the user's required verbosity. By default, each
    level has a specific bracketed tag and an auto
    indentation depth.

    Examples
    --------
    >>> from gofast.utils.generic_utils import vlog
    >>> # Example with mode='log'
    >>> # This prints only if global or passed-in
    >>> # verbose >= 4.
    >>> vlog("Check debugging details.", verbose=3, level=4,
    ...      mode='log')
    >>> # Example with mode='naive'
    >>> # If verbose=2, it displays as [INFO] prefixed.
    >>> vlog("Loading data...", verbose=2, mode='naive')

    See Also
    --------
    globals : Used to retrieve the fallback `verbose`
        value if not explicitly passed.

    """
    verbosity_labels = {
        1: "[ERROR]",
        2: "[WARNING]",
        3: "[INFO]",
        4: "[DEBUG]",
        5: "[DEBUG]",
        6: "[TRACE]",
        7: "[TRACE]"
    }

    # When depth="auto", assign indentation based on `level`.
    if depth == "auto":
        if level == 1:
            depth = 0
        elif level == 2:
            depth = 2
        elif level == 3:
            depth = 0
        elif level in (4, 5):
            depth = 2
        elif level in (6, 7):
            depth = 4
        else:
            depth = 0

    # Fallback to a global 'verbose' if none given.
    actual_verbose = verbose if verbose is not None else globals().get('verbose', 0)

    # If mode == 'log', keep original approach:
    if mode == 'log':
        if actual_verbose >= level:
            # Indent and prefix with the label from `level`.
            indent = " " * (depth * 2)
            print(f"{indent}{verbosity_labels[level]} {message}")
        # Nothing else for mode='log' if verbosity is too low.
        return

    # If mode == 'naive' or None, override with the new rules:
    if mode in [None, 'naive']:
        # If actual_verbose < 1, do not print anything.
        if actual_verbose < 1:
            return

        # Otherwise, figure out the prefix based on the threshold:
        # We check if the message already includes [INFO], [DEBUG], [WARNING],
        # [ERROR], or [TRACE]. If so, skip adding any prefix.
        prefix_tags = ("[INFO]", "[DEBUG]", "[ERROR]", "[WARNING]", "[TRACE]")
        already_tagged = any(tag in message for tag in prefix_tags)

        # indent as per the computed `depth`
        indent = " " * (depth * 2)
        
        # If >=3 => prefix with [INFO] if vp is True and not already tagged
        if actual_verbose <=3:
            if vp and not already_tagged:
                print(f"{indent}[INFO] {message}")
            else:
                print(f"{indent}{message}")
            return 

        # If 3 < verbose < 5 => prefix with [DEBUG] if vp is True and not already tagged
        if 3 < actual_verbose < 5:
            if vp and not already_tagged:
                print(f"{indent}[DEBUG] {message}")
            else:
                print(f"{indent}{message}")
            return

        # If verbose >= 5 => prefix with [TRACE] if vp is True and not already tagged
        if actual_verbose >= 5:
            if vp and not already_tagged:
                print(f"{indent}[TRACE] {message}")
            else:
                print(f"{indent}{message}")
            return

def get_actual_column_name(
    df: pd.DataFrame, 
    tname: str = None, 
    actual_name: str = None, 
    error: str = 'raise',  
    default_to=None, 
) -> str:
    """
    Determines the actual target column name in the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the target column.
    tname : str, optional
        The base target name (e.g., "subsidence"). If not found in the DataFrame,
        it will attempt to find a matching column using "<tname>_actual" format.
    actual_name : str, optional
        If provided, this name will be returned as the actual target column name.
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Specifies how to handle the case when no valid column is found:
        - 'raise': Raises a `ValueError`.
        - 'warn': Issues a warning and returns `None`.
        - 'ignore': Silently returns `None`.

    Returns
    -------
    str or None
        The determined actual column name, or None if no match is found 
        and `error='warn'` or `error='ignore'`.

    Raises
    ------
    ValueError
        If no valid target column is found and `error='raise'`.

    Examples
    --------
    >>> from gofast.utils.generic_utils import get_actual_column_name
    >>> df = pd.DataFrame({'subsidence_actual': [1, 2, 3]})
    >>> get_actual_column_name(df, tname="subsidence")
    'subsidence_actual'

    >>> df = pd.DataFrame({'subsidence': [1, 2, 3]})
    >>> get_actual_column_name(df, tname="subsidence")
    'subsidence'

    >>> df = pd.DataFrame({'actual': [1, 2, 3]})
    >>> get_actual_column_name(df)
    'actual'
    
    >>> df = pd.DataFrame({'measurement': [1, 2, 3]})
    >>> get_actual_column_name(df, tname="subsidence", error="warn")
    Warning: Could not determine the actual target column in the DataFrame.
    None
    """

    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    # If `actual_name` is provided, check if it exists in df
    if actual_name and actual_name in df.columns:
        return actual_name

    # If `tname` exists in df, return it
    if tname and tname in df.columns:
        return tname

    # If `<tname>_actual` exists, return it
    if tname and f"{tname}_actual" in df.columns:
        return f"{tname}_actual"

    # If "actual" column exists, return it
    if "actual" in df.columns:
        return "actual"

    # Handle the case when no valid column is found
    msg = "Could not determine the actual target column in the DataFrame."
    if error == 'raise':
        raise ValueError(msg)
    elif error == 'warn':
        warnings.warn(msg, UserWarning)
    
    if default_to=='tname': 
        return tname 
    
    return None  # If `error='ignore'`, return None silently


def detect_dt_format(series: pd.Series) -> str:
    r"""
    Detect the datetime format of a pandas Series containing datetime values.

    This function inspects a non-null sample from the datetime Series and
    infers the format string based on its components (year, month, day, hour,
    minute, and second). It returns a format string that can be used with 
    ``strftime``. For example, if the sample indicates only a year is relevant, 
    it returns ``"%Y"``; if full date information is present, it returns 
    ``"%Y-%m-%d"``; and if time details are also present, it extends the format 
    accordingly.

    Parameters
    ----------
    series : pandas.Series
        A Series containing datetime values (dtype datetime64).

    Returns
    -------
    str
        A datetime format string (e.g., ``"%Y"``, ``"%Y-%m-%d"``, or 
        ``"%Y-%m-%d %H:%M:%S"``) that represents the resolution of the data.

    Examples
    --------
    >>> from gofast.utils.generic_utils import detect_dt_format
    >>> import pandas as pd
    >>> dates = pd.to_datetime(['2023-01-01', '2024-01-01', '2025-01-01'])
    >>> fmt = detect_dt_format(pd.Series(dates))
    >>> print(fmt)
    %Y

    Notes
    -----
    The detection logic checks if month, day, hour, minute, and second are 
    all default values (e.g., month == 1, day == 1, hour == 0, etc.) and infers
    the most compact format that still represents the data accurately.

    """
    # Validate input DataFrame
    if not isinstance(series, pd.Series):
        raise TypeError("`series` must be a pandas Series.")

    # Drop null values and pick a sample for analysis.
    sample = series.dropna().iloc[0]
    
    # Always include year.
    fmt = "%Y"
    
    # Include month if not January.
    if sample.month != 1 or sample.day != 1 or sample.hour != 0 or \
       sample.minute != 0 or sample.second != 0:
        fmt += "-%m"
    
    # Include day if not the first day.
    if sample.day != 1 or sample.hour != 0 or sample.minute != 0 or \
       sample.second != 0:
        fmt += "-%d"
    
    # Include time details if any are non-zero.
    if sample.hour != 0 or sample.minute != 0 or sample.second != 0:
        fmt += " %H"
        if sample.minute != 0 or sample.second != 0:
            fmt += ":%M"
        if sample.second != 0:
            fmt += ":%S"
    
    return fmt

def transform_contributions(
    contributions, 
    to_percent=True, 
    normalize=False, 
    norm_range=(0, 1), 
    scale_type=None, 
    zero_division='warn', 
    epsilon=1e-6, 
    log_transform=False
):
    """
    Converts the feature contributions either to a direct percentage, 
    normalizes them to a custom range, or applies a scaling strategy 
    based on the chosen parameters.

    Parameters
    ----------
    contributions : dict
        A dictionary where keys are feature names and values are the 
        feature contributions. Each value is expected to be a numerical 
        value representing the contribution of the respective feature.

    to_percent : bool, optional, default=True
        Whether to convert the contributions to percentages. If `True`, 
        each value in `contributions` will be multiplied by 100. This is 
        useful when contributions are given in decimal form but are expected 
        as percentages.

    normalize : bool, optional, default=False
        Whether to normalize the contributions using min-max scaling. If 
        `True`, the values will be scaled to the range defined in 
        ``norm_range``.

    norm_range : tuple, optional, default=(0, 1)
        A tuple specifying the range (min, max) for normalization. This range 
        is applied when `normalize` is set to `True`. The contributions will 
        be rescaled so that the minimum value maps to `norm_range[0]` and the 
        maximum value maps to `norm_range[1]`.

    scale_type : str, optional, default=None
        The scaling strategy. Options include:
        - ``'zscore'``: Performs Z-score normalization.
        - ``'log'``: Applies a logarithmic transformation to the data.
        If `None`, no scaling is applied.

    zero_division : str, optional, default='warn'
        Defines how to handle zero or missing values in the contributions. 
        Options include:
        - ``'skip'``: Skips zero values (no modification).
        - ``'warn'``: Issues a warning if zero values are found.
        - ``'replace'``: Replaces zeros with a small value defined by 
          ``epsilon`` to avoid division by zero or undefined results.

    epsilon : float, optional, default=1e-6
        A small value used to replace zeros when `zero_division` is set to 
        ``'replace'``. This prevents division by zero errors during 
        transformations like Z-score or log transformation.

    log_transform : bool, optional, default=False
        Whether to apply a logarithmic transformation to the contributions. 
        If `True`, it applies the natural logarithm to each value in the 
        `contributions` dictionary. Only positive values are valid for log 
        transformation, and zero values are either skipped or replaced 
        based on the ``zero_division`` parameter.

    Returns
    -------
    dict
        A dictionary with feature names as keys and the transformed feature 
        contributions as values. The transformation is applied according to 
        the chosen parameters.

    Notes
    -----
    - When ``normalize=True``, if the minimum and maximum values in the 
      `contributions` are the same, normalization is skipped with a warning.
    - If ``scale_type='zscore'``, the function applies Z-score normalization:
      
      .. math::
          Z = \frac{X - \mu}{\sigma}
      
      where :math:`X` is the contribution, :math:`\mu` is the mean of the 
      contributions, and :math:`\sigma` is the standard deviation of the 
      contributions.
      
    - If ``log_transform=True``, the function applies the natural logarithm:
      
      .. math::
          \text{log}(X) \text{ for } X > 0
          
    - The ``zero_division`` parameter handles zero values by either skipping, 
      warning, or replacing them with a small value (`epsilon`).

    Examples
    --------
    >>> from gofast.utils.generic_utils import transform_contributions
    >>> contributions = {
    >>>     'GWL': 2.226836617133828,
    >>>     'rainfall_mm': 12.398293851061492,
    >>>     'normalized_seismic_risk_score': 0.9402759347406523,
    >>>     'normalized_density': 4.806074194258057,
    >>>     'density_concentration': 5.666943330566496e-06,
    >>>     'geology': 1.2798872011280326e-05,
    >>>     'density_tier': 1.044039559604414e-05,
    >>>     'rainfall_category': 0.0
    >>> }
    >>> transform_contributions(contributions, to_percent=True, normalize=True)
    >>> transform_contributions(contributions, to_percent=False, scale_type='zscore')
    
    See Also
    --------
    `numpy.mean`: Compute the arithmetic mean of an array.
    `numpy.std`: Compute the standard deviation of an array.

    References
    ----------
    [1]_ "Statistical Methods for Data Transformation" by J. Smith, 
         Springer, 2020.
    """
    
    # Handle zero values based on user preference
    if zero_division == 'replace':
        contributions = {
            feature: (contribution if contribution != 0 else epsilon)
            for feature, contribution in contributions.items()
        }
    elif zero_division == 'warn' and any(
        contribution == 0 for contribution in contributions.values()
    ):
        warnings.warn(
            "Some contribution values are zero. Consider replacing them.",
            UserWarning
        )

    # Convert contributions to percentage if specified
    if to_percent:
        contributions = {
            feature: contribution * 100 
            for feature, contribution in contributions.items()
        }
    
    # Apply normalization to the specified range
    if normalize:
        min_val = min(contributions.values())
        max_val = max(contributions.values())
        
        # Check if min and max values are the
        # same to avoid division by zero
        if min_val == max_val:
            warnings.warn(
                "All contribution values are the same,"
                " cannot normalize; Skipped.",
                UserWarning
            )
        else:
            norm_range = 100 * np.asarray(
                norm_range) if to_percent else norm_range 
            
            contributions = {
                feature: (
                    ((contribution - min_val) / (max_val - min_val)) * 
                    (norm_range[1] - norm_range[0]) + norm_range[0]
                ) 
                for feature, contribution in contributions.items()
            }

    # Apply scaling (Z-score or log)
    if scale_type == 'zscore':
        mean_val = np.mean(list(contributions.values()))
        std_val = np.std(list(contributions.values()))
        
        contributions = {
            feature: (contribution - mean_val) / std_val 
            if std_val != 0 else contribution
            for feature, contribution in contributions.items()
        }
    
    elif log_transform:
        contributions = {
            feature: np.log(contribution) if contribution > 0 else 0
            for feature, contribution in contributions.items()
        }
    
    return contributions


def exclude_duplicate_kwargs(
    func: callable,
    existing_kwargs: Union[
        Dict[str, Any],
        List[str]
    ],
    user_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prevents the user from overriding existing parameters
    in a target function. The method `exclude_duplicate_kwargs`
    checks both developer-specified and function-level
    parameter names to exclude them from `user_kwargs`.
    
    .. math::
       \text{final\_kwargs} =
       \{\,(k, v) \in \text{user\_kwargs} \,\mid\,
       k \notin \text{protected\_params}\,\}
    
    Parameters
    ----------
    func : callable
        The target function whose valid parameters are
        checked. It uses Python's introspection to gather
        the acceptable parameter names.
    
    existing_kwargs : dict or list
        Developer-defined parameters to protect. Can be:
        * A dictionary of parameter-value pairs (e.g.,
          ``{'ax': ax_obj, 'data': df}``) whose keys
          are excluded from user overrides.
        * A list of parameter names (e.g., ``['ax',
          'data']``) to protect from user overrides.
    
    user_kwargs : dict
        The user-supplied keyword arguments that are
        candidates for merging with `existing_kwargs`.
        This dictionary is filtered to remove collisions
        with protected parameters.
    
    Returns
    -------
    dict
        A filtered dictionary of user-defined arguments
        that do not overlap with protected parameters.
    
    Examples
    --------
    >>> from gofast.utils.generic_utils import exclude_duplicate_kwargs
    >>> import seaborn as sns
    >>> # Developer has some base kwargs
    ... base_kwargs = {
    ...     'x': 'species',
    ...     'y': 'sepal_length',
    ...     'palette': 'viridis'
    ... }
    >>> # User tries to override 'x' with new param
    ... user_args = {
    ...     'x': 'petal_width',
    ...     'color': 'red'
    ... }
    >>> # Filter out duplicates
    ... safe_args = exclude_duplicate_kwargs(
    ...     sns.scatterplot,
    ...     base_kwargs,
    ...     user_args
    ... )
    >>> safe_args
    {'color': 'red'}
    
    Notes
    -----
    By default, if `existing_kwargs` is a dictionary,
    its keys are treated as protected parameter names.
    If it's a list, those items are protected. The
    function signature of `func` is also used to
    verify that only recognized parameters are
    protected.
    
    See Also
    --------
    inspect.signature : Used to introspect function
        parameters.
    filter_valid_kwargs : Another inline function that
        discards user params not valid for a given
        function.
    
    References
    ----------
    .. [1] David Beazley and Brian K. Jones.
       *Python Cookbook, 3rd Edition.* O'Reilly Media, 2013.
    """

    # Validate existing_kwargs
    if not isinstance(existing_kwargs, (dict, list)):
        raise TypeError(
            "existing_kwargs must be a dict or list"
        )

    # Build set of protected parameters
    protected_params = (
        existing_kwargs.keys()
        if isinstance(existing_kwargs, dict)
        else existing_kwargs
    )

    # Inspect func signature to get valid names
    sig = inspect.signature(func)
    valid_params = {
        name: param
        for name, param in sig.parameters.items()
        if param.kind not in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD
        )
    }

    # Restrict protection to valid parameters only
    exclude = [
        param
        for param in protected_params
        if param in valid_params
    ]

    # Filter out excluded params from user_kwargs
    safe_kwargs = {
        k: v
        for k, v in user_kwargs.items()
        if k not in exclude
    }

    return safe_kwargs
