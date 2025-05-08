# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Contains general-purpose utility functions that support various aspects
of the machine learning workflow, including data manipulation, logging,
and helper tools.
"""

import math
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

from ...api.types import List, Tuple, Dict, Optional, Union, Series
from ...api.types import ArrayLike, DataFrame, Callable
from ...compat.sklearn import train_test_split
from ...core.array_manager import array_preserver, return_if_preserver_failed
from ...core.checks import is_iterable, str2columns 
from ...core.io import is_data_readable
from ...core.utils import smart_format

from ..validator import (
    _check_consistency_size, 
    _is_arraylike_1d,
    check_consistent_length, 
    is_frame, 
)

__all__= [ 
    'smart_split',
    'soft_data_split', 
    'smart_label_classifier', 
    'stratify_categories', 
    'laplace_smoothing', 
    'laplace_smoothing_categorical',
    'laplace_smoothing_word',
    'groupwise_train_test_split'
    ]

def groupwise_train_test_split(
    df: pd.DataFrame,                     
    group_col: str = 'id',                
    test_size: Optional[float] = 0.2,     
    train_size: Optional[float] = None,   
    random_state: Optional[int] = 42,     
    verbose: int = 0                      
) -> Tuple[pd.DataFrame, pd.DataFrame]:   
    """Splits DataFrame into train/test sets ensuring group integrity.

    This function partitions a pandas DataFrame into training and
    testing subsets while strictly respecting group boundaries defined
    by a specified column (`group_col`). It guarantees that all rows
    belonging to the same group are assigned entirely to *either* the
    training set or the testing set, never split across both. This is
    essential for preventing data leakage in machine learning models,
    particularly when dealing with data where samples are not
    independent (e.g., time series from the same sensor, medical
    records from the same patient, observations from the same event).

    The core mechanism relies on scikit-learn's ``GroupShuffleSplit``
    [1]_, which first identifies unique groups, shuffles these groups
    randomly (controlled by `random_state`), and then assigns a
    proportion of the *groups* (determined by `test_size` or
    `train_size`) to the test set, with the remainder forming the
    training set. Finally, all data rows corresponding to the chosen
    groups are collected to form the final train and test DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame containing the entire dataset to be
        split. This DataFrame must include the column specified by the
        `group_col` parameter.

    group_col : str, default='id'
        The name of the column within `df` that contains the group
        identifiers. All rows sharing the same value in this column
        will be treated as part of the same group and kept together
        during the split. Example: ``'patient_id'``, ``'event_id'``.

    test_size : float or None, default=0.2
        The desired proportion of *unique groups* to allocate to the
        test set. This value must be a float between 0.0 and 1.0.
        If ``train_size`` is provided and ``test_size`` is ``None``,
        the test size will be automatically inferred as
        :math:`1 - p_{train}`. If both ``test_size`` and
        ``train_size`` are ``None``, `test_size` defaults to ``0.2``.
        Note that the proportion is based on the count of unique
        groups, not the count of rows.

    train_size : float or None, default=None
        The desired proportion of *unique groups* to allocate to the
        training set. Similar to `test_size`, this must be a float
        between 0.0 and 1.0. If ``test_size`` is provided and
        ``train_size`` is ``None``, the train size will be inferred as
        :math:`1 - p_{test}`. It is generally recommended to specify
        only one of `test_size` or `train_size`.

    random_state : int or None, default=42
        A seed value for the pseudo-random number generator used to
        shuffle the groups before splitting. Providing an integer
        (e.g., ``42``) ensures that the split is deterministic and
        reproducible across multiple runs with the same inputs. If set
        to ``None``, the split will be different each time the function
        is executed.

    verbose : int, default=0
        Controls the level of informational output printed to the
        console during execution. Valid levels are:
        - ``0``: No output (silent execution).
        - ``1``: Prints basic summary information before the split,
          including total sample and group counts, and the specified
          split parameters.
        - ``2``: Prints the information from level 1, plus detailed
          results after the split, including the number and proportion
          of samples and groups in the resulting train and test sets,
          and a verification check for group overlap.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two pandas DataFrames:
        1. ``train_df``: The subset of the original `df` allocated to
           the training set.
        2. ``test_df``: The subset of the original `df` allocated to
           the testing set.
        Both DataFrames have their indices reset (using
        ``.reset_index(drop=True)``).

    Raises
    ------
    ValueError
        If input parameters are invalid (e.g., `df` is not a DataFrame,
        `test_size` or `train_size` outside [0, 1], `verbose` level
        out of range, fewer than 2 unique groups found).
    KeyError
        If the specified `group_col` is not found in `df.columns`.
    ImportError
        If required libraries (`pandas`, `scikit-learn`) are not
        installed.

    See Also
    --------
    sklearn.model_selection.GroupShuffleSplit : The underlying scikit-learn
        class used for performing the group-based split.
    sklearn.model_selection.train_test_split : Standard scikit-learn function
        for splitting data without considering groups (prone to data
        leakage if groups exist).
    sklearn.model_selection.GroupKFold : For cross-validation respecting
        group boundaries over K folds.
    sklearn.model_selection.StratifiedGroupKFold : Cross-validation that
        preserves group integrity and class distribution.

    Notes
    -----
    - The exact number of samples in the train and test sets depends not
      only on `test_size`/`train_size` but also on the distribution of
      samples within each group. The proportions apply to the *groups*.
    - If the total number of unique groups is small, the resulting
      proportions of groups in the train/test sets might deviate
      slightly from the requested `test_size` or `train_size` due to
      the discrete nature of groups.
    - This function is intended purely for data splitting. Subsequent
      analysis, model training, or visualization should be performed on
      the returned ``train_df`` and ``test_df``.
      

    Let :math:`D` be the input DataFrame and :math:`g(d)` be the
    function that returns the group identifier for a row :math:`d \in D`
    from the column specified by `group_col`. Let :math:`G` be the set
    of all unique group identifiers in :math:`D`.

    .. math::
        G = \{ g(d) \mid d \in D \}

    The function aims to partition the set of unique groups :math:`G`
    into two disjoint subsets, :math:`G_{train}` and :math:`G_{test}`,
    based on the specified proportions :math:`p_{test}` (`test_size`)
    and :math:`p_{train}` (`train_size`).

    .. math::
        G_{train}, G_{test} = \\text{RandomPartition}\\
            (G, p_{train}, p_{test}, \\text{seed}=\\text{random_state})

    Where :math:`|G_{test}| \\approx p_{test} \\times |G|` and
    :math:`|G_{train}| \\approx p_{train} \\times |G|` (subject to
    rounding for integer counts of groups), such that:

    .. math::
        G_{train} \\cap G_{test} = \\emptyset \\quad \\text{and}\\
            \\quad G_{train} \\cup G_{test} = G

    The final DataFrames are constructed by selecting all rows whose
    group identifier falls into the respective group subsets:

    .. math::
        D_{train} = \{ d \\in D \\mid g(d) \\in G_{train} \} \\\\
        D_{test} = \{ d \\in D \\mid g(d) \\in G_{test} \}

    The function internally calls the `.split()` method of a
    ``GroupShuffleSplit`` instance to generate the indices for
    :math:`D_{train}` and :math:`D_{test}`.


    References
    ----------
    .. [1] Scikit-learn Developers. "User Guide: 3.1. Cross-validation:
           evaluating estimator performance". Scikit-learn Documentation.
           Accessed April 4, 2025.
           https://scikit-learn.org/stable/modules/cross_validation.html#group-shuffle-split

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ml.utils import groupwise_train_test_split 

    >>> # Sample data with groups 'A', 'B', 'C', 'D', 'E'
    >>> data = {
    ...     'id': ['A', 'A', 'B', 'B', 'C', 'C', 'C', 'D', 'E', 'E'],
    ...     'feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     'target': [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    ... }
    >>> my_df = pd.DataFrame(data)
    >>> print("Original DataFrame:\\n", my_df)
    Original DataFrame:
       id  feature  target
    0  A        1       0
    1  A        2       0
    2  B        3       1
    3  B        4       1
    4  C        5       0
    5  C        6       0
    6  C        7       0
    7  D        8       1
    8  E        9       1
    9  E       10       1

    >>> # Perform a group-wise split with 30% groups in test set
    >>> # Use verbose=2 for detailed output
    >>> train_data, test_data = refactored_groupwise_train_test_split(
    ...     df=my_df,
    ...     group_col='id',
    ...     test_size=0.3,  # Request ~30% of groups for test
    ...     random_state=42, # For reproducibility
    ...     verbose=2
    ... )
    -----------------------------------------------------------------
    Initiating Group-wise Train-Test Split
    -----------------------------------------------------------------
      Input DataFrame shape:     (10, 3)
      Total samples (rows):      10
      Grouping column:           'id'
      Total unique groups found: 5
      Requested test_size (groups):  0.3
      Requested train_size (groups): None
      Random state seed:         42
    -----------------------------------------------------------------
    Split Execution Results:
    -----------------------------------------------------------------
      Training Set:
        Number of samples: 7 (70.00%)
        Number of groups:  3 (60.00%) <--- Note: Group proportion might differ due to rounding

      Test Set:
        Number of samples: 3 (30.00%)
        Number of groups:  2 (40.00%) <--- Note: Group proportion might differ due to rounding
    -----------------------------------------------------------------
      Verification: Passed. No groups overlap between train and test sets.
    -----------------------------------------------------------------

    >>> print("\\nTraining Data (Groups: {}):\\n{}".format(
    ...     sorted(train_data['id'].unique()), train_data))

    Training Data (Groups: ['A', 'C', 'E']):
       id  feature  target
    0  A        1       0
    1  A        2       0
    2  C        5       0
    3  C        6       0
    4  C        7       0
    5  E        9       1
    6  E       10       1

    >>> print("\\nTesting Data (Groups: {}):\\n{}".format(
    ...     sorted(test_data['id'].unique()), test_data))

    Testing Data (Groups: ['B', 'D']):
       id  feature  target
    0  B        3       1
    1  B        4       1
    2  D        8       1
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")

    if group_col not in df.columns:
        raise KeyError(
            f"The group column '{group_col}' was not found in the "
            f"DataFrame columns: {df.columns.tolist()}"
        )

    # Validate sizes (basic check, GroupShuffleSplit does more thorough checks)
    for size, name in [(test_size, "test_size"), (train_size, "train_size")]:
        if size is not None:
            if not isinstance(size, (float, int)):
                 raise ValueError(f"{name} must be a float or None.")
            if not (0.0 <= size <= 1.0):
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0, but got {size}."
                )

    # If both sizes are None, default test_size to 0.2 (as per default arg)
    if test_size is None and train_size is None:
        test_size = 0.2
        if verbose > 0:
            print(
                "Info: Both test_size and train_size were None. "
                "Defaulting test_size to 0.2."
                )

    # --- Prepare for Splitting ---
    groups = df[group_col] # Extract the group labels series
    unique_groups = groups.unique()
    n_total_samples = len(df)
    n_total_groups = len(unique_groups)

    if n_total_groups < 2:
        raise ValueError(
            f"Cannot perform train/test split with fewer than 2 unique groups. "
            f"Found {n_total_groups} unique group(s) in column '{group_col}'."
        )

    # --- Verbose Level 1 Output ---
    if verbose >= 1:
        print("-" * 65)
        print("Initiating Group-wise Train-Test Split")
        print("-" * 65)
        print(f"  Input DataFrame shape:     {df.shape}")
        print(f"  Total samples (rows):      {n_total_samples}")
        print(f"  Grouping column:           '{group_col}'")
        print(f"  Total unique groups found: {n_total_groups}")
        print(f"  Requested test_size (groups):  {test_size}")
        print(f"  Requested train_size (groups): {train_size}")
        print(f"  Random state seed:         {random_state}")
        print("-" * 65)

    # --- Configure and Execute the Split ---
    # Instantiate the splitter from scikit-learn
    # Using vertical alignment and parentheses for long parameter lists
    splitter = GroupShuffleSplit(
        n_splits=1,                # We require only a single split
        test_size=test_size,       # Proportion of groups for the test set
        train_size=train_size,     # Proportion of groups for the train set
        random_state=random_state  # Seed for reproducibility
    )

    # The splitter yields indices. Since n_splits=1, we get one pair.
    # The 'y' parameter is not needed for unsupervised splitting like this.
    try:
        train_indices, test_indices = next(
            splitter.split(X=df, y=None, groups=groups)
        )
    except ValueError as e:
         # Catch potential errors from scikit-learn's splitter
         # (e.g., sizes summing > 1, insufficient groups for split)
        print("\nError occurred during scikit-learn's GroupShuffleSplit:")
        print(f"  Original Error: {e}")
        print(f"  Check test_size ({test_size}), train_size ({train_size}), "
              f"and the number of unique groups ({n_total_groups}).")
        raise  # Re-raise the exception after providing context

    # --- Create Train/Test DataFrames ---
    # Select rows based on the indices generated by the splitter
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # Reset the index for the resulting DataFrames for cleaner output
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --- Verbose Level 2 Output ---
    if verbose >= 2:
        n_train_samples = len(train_df)
        n_test_samples = len(test_df)
        train_groups_set = set(train_df[group_col].unique())
        test_groups_set = set(test_df[group_col].unique())
        n_train_groups = len(train_groups_set)
        n_test_groups = len(test_groups_set)

        # Calculate actual proportions achieved based on the split
        actual_test_group_prop = (
            n_test_groups / n_total_groups if n_total_groups > 0 else 0
            )
        actual_train_group_prop = (
            n_train_groups / n_total_groups if n_total_groups > 0 else 0
            )
        actual_test_sample_prop = (
            n_test_samples / n_total_samples if n_total_samples > 0 else 0
            )
        actual_train_sample_prop = (
            n_train_samples / n_total_samples if n_total_samples > 0 else 0
            )

        print("Split Execution Results:")
        print("-" * 65)
        print("  Training Set:")
        print(f"    Number of samples: {n_train_samples} "
              f"({actual_train_sample_prop:.2%})")
        print(f"    Number of groups:  {n_train_groups} "
              f"({actual_train_group_prop:.2%})")
        print("\n  Test Set:")
        print(f"    Number of samples: {n_test_samples} "
              f"({actual_test_sample_prop:.2%})")
        print(f"    Number of groups:  {n_test_groups} "
              f"({actual_test_group_prop:.2%})")
        print("-" * 65)

        # Verification step: Ensure no groups overlap between sets
        overlapping_groups = train_groups_set.intersection(test_groups_set)
        if not overlapping_groups:
            print("  Verification: Passed. No groups overlap between train and test sets.")
        else:
            # This should theoretically not happen with GroupShuffleSplit
            print(f"  Verification: FAILED! Found {len(overlapping_groups)} "
                  f"overlapping group(s): {list(overlapping_groups)[:5]}...") # Show first few
        print("-" * 65)

    # --- Return Results ---
    return train_df, test_df

def smart_split(
    X, 
    target: Optional[Union[ArrayLike, int, str, List[Union[int, str]]]] = None,
    test_size: float = 0.2, 
    random_state: int = 42,
    stratify: bool = False,
    shuffle: bool = True,
    return_df: bool = False,
    **skws
) -> Union[
    Tuple[DataFrame, DataFrame], 
    Tuple[ArrayLike, ArrayLike],
    Tuple[DataFrame, DataFrame, Series, Series], 
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame], 
    Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
    ]:
    """
    Splits data into training and test sets, with the option to extract and 
    handle multiple target variables. 
    
    Function supports both single and multi-label targets and maintains 
    compatibility with pandas DataFrame and numpy ndarray.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        The input data to be split. This can be either feature data alone or 
        include the target column(s) if the `target` parameter is used to specify 
        target column(s) for extraction.
    target : int, str, list of int/str, pd.Series, pd.DataFrame, optional
        Specifies the target variable(s) for supervised learning problems. 
        It can be:
        - An integer or string specifying the column index or name in `X` to 
          be used as the target variable.
        - A list of integers or strings for multi-label targets.
        - A pandas Series or DataFrame directly specifying the target variable(s).
        If `target` is provided as an array-like object or DataFrame, its 
        length must match the number of samples in `X`.
    test_size : float, optional
        Represents the proportion of the dataset to include in the test split. 
        Must be between 0.0 and 1.0.
    random_state : int, optional
        Sets the seed for random operations, ensuring reproducible splits.
    stratify : bool, optional
        Ensures that the train and test sets have approximately the same 
        percentage of samples of each target class if set to True.
    shuffle : bool, optional
        Determines whether to shuffle the dataset before splitting. 
    return_df : bool, optional
        If True and `X` is a DataFrame, returns the splits as pandas DataFrames/Series. 
        Otherwise, returns numpy ndarrays.
    skws : dict
        Additional keyword arguments for `train_test_split`, allowing customization 
        of the split beyond the parameters explicitly mentioned here.

    Returns
    -------
    Depending on the inputs and `return_df`:
    - If `target` is not specified: X_train, X_test
    - If `target` is specified: X_train, X_test, y_train, y_test
    `X_train` and `X_test` are the splits of the input data, while `y_train` and 
    `y_test` are the splits of the target variable(s) if provided.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.mlutils import smart_split
    >>> data = pd.DataFrame({
    ...     'Feature1': [1, 2, 3, 4],
    ...     'Feature2': [4, 3, 2, 1],
    ...     'Target': [0, 1, 0, 1]
    ... })
    >>> # Single target specified as a column name
    >>> X_train, X_test, y_train, y_test = smart_split(
    ... data, target='Target', return_df=True)
    >>> print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    """
    if target is not None:
        X, y, target_names = _extract_target(X, target)
    else:
        y, target_names = None, []

    stratify_param = y if stratify and y is not None else None
    if y is not None: 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, 
            stratify=stratify_param, **skws)
    else: 
        X_train, X_test= train_test_split(
            X, test_size=test_size, random_state=random_state, shuffle=shuffle, 
            stratify=stratify_param, **skws)

    if return_df and isinstance(X, pd.DataFrame):
        X_train, X_test = pd.DataFrame(X_train, columns=X.columns
                                       ), pd.DataFrame(X_test, columns=X.columns)
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y_train, y_test = pd.DataFrame(
                    y_train, columns=target_names), pd.DataFrame(
                        y_test, columns=target_names)
            else:
                y_train, y_test = pd.Series(
                    y_train, name=target_names[0]), pd.Series(
                        y_test, name=target_names[0])

    return (X_train, X_test, y_train, y_test) if y is not None else (X_train, X_test)

def soft_data_split(
    X, y=None, *,
    test_size=0.2,
    target_column=None,
    random_state=42,
    extract_target=False,
    **split_kwargs
):
    """
    Splits data into training and test sets, optionally extracting a 
    target column.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data to split. If `extract_target` is True, a target column can be
        extracted from `X`.
    y : array-like, optional
        Target variable array. If None and `extract_target` is False, `X` is
        split without a target variable.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Should be
        between 0.0 and 1.0. Default is 0.2.
    target_column : int or str, optional
        Index or column name of the target variable in `X`. Used only if
        `extract_target` is True.
    random_state : int, optional
        Controls the shuffling for reproducible output. Default is 42.
    extract_target : bool, optional
        If True, extracts the target variable from `X`. Default is False.
    split_kwargs : dict, optional
        Additional keyword arguments to pass to `train_test_split`.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split data arrays.

    Raises
    ------
    ValueError
        If `target_column` is not found in `X` when `extract_target` is True.

    Example
    -------
    >>> from gofast.datasets import fetch_data
    >>> data = fetch_data('Bagoue original')['data']
    >>> X, XT, y, yT = split_data(data, extract_target=True, target_column='flow')
    """

    if extract_target:
        if isinstance(X, pd.DataFrame) and target_column in X.columns:
            y = X[target_column]
            X = X.drop(columns=target_column)
        elif hasattr(X, '__array__') and isinstance(target_column, int):
            y = X[:, target_column]
            X = np.delete(X, target_column, axis=1)
        else:
            raise ValueError(f"Target column {target_column!r} not found in X.")

    if y is not None:
        return train_test_split(X, y, test_size=test_size, 
                                random_state=random_state, **split_kwargs)
    else:
        return  train_test_split(
            X, test_size=test_size,random_state=random_state, **split_kwargs)

@is_data_readable
def stratify_categories(
    data: Union[DataFrame, ArrayLike],
    cat_name: str, 
    n_splits: int = 1, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[Union[DataFrame, ArrayLike], Union[DataFrame, ArrayLike]]: 
    """
    Perform stratified sampling on a dataset based on a specified categorical column.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        The dataset to be split. Can be a Pandas DataFrame or a NumPy ndarray.
        
    cat_name : str
        The name of the categorical column in 'data' used for stratified sampling. 
        This column must exist in 'data' if it's a DataFrame.
        
    n_splits : int, optional
        Number of re-shuffling & splitting iterations. Defaults to 1.
        
    test_size : float, optional
        Proportion of the dataset to include in the test split. Defaults to 0.2.
        
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        Defaults to 42.

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]
        A tuple containing the training and testing sets.

    Raises
    ------
    ValueError
        If 'cat_name' is not found in 'data' when 'data' is a DataFrame.
        If 'test_size' is not between 0 and 1.
        If 'n_splits' is less than 1.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'feature1': np.random.rand(100),
    ...     'feature2': np.random.rand(100),
    ...     'category': np.random.choice(['A', 'B', 'C'], 100)
    ... })
    >>> train_set, test_set = stratify_categories(df, 'category')
    >>> train_set.shape, test_set.shape
    ((80, 3), (20, 3))
    """

    if isinstance(data, pd.DataFrame) and cat_name not in data.columns:
        raise ValueError(f"Column '{cat_name}' not found in the DataFrame.")

    if not (0 < test_size < 1):
        raise ValueError("Test size must be between 0 and 1.")

    if n_splits < 1:
        raise ValueError("Number of splits 'n_splits' must be at least 1.")

    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                   random_state=random_state)
    for train_index, test_index in split.split(data, data[cat_name] if isinstance(
            data, pd.DataFrame) else data[:, cat_name]):
        if isinstance(data, pd.DataFrame):
            strat_train_set = data.iloc[train_index]
            strat_test_set = data.iloc[test_index]
        else:  # Handle numpy arrays
            strat_train_set = data[train_index]
            strat_test_set = data[test_index]

    return strat_train_set, strat_test_set

 
@is_data_readable
def laplace_smoothing_categorical(
        data, feature_col, class_col, V=None):
    """
    Apply Laplace smoothing to estimate conditional probabilities of 
    categorical features given a class in a dataset.

    This function calculates the Laplace-smoothed probabilities for each 
    category of a specified feature given each class.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing categorical features and a class label.
    feature_col : str
        The column name in the dataset representing the feature for which 
        probabilities are to be calculated.
    class_col : str
        The column name in the dataset representing the class label.
    V : int or None, optional
        The size of the vocabulary (i.e., the number of unique categories 
                                    in the feature).
        If None, it will be calculated based on the provided feature column.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Laplace-smoothed probabilities for each 
        category of the feature across each class.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.utils.mlutils import laplace_smoothing_categorical
    >>> data = pd.DataFrame({'feature': ['cat', 'dog', 'cat', 'bird'],
                             'class': ['A', 'A', 'B', 'B']})
    >>> probabilities = laplace_smoothing_categorical(data, 'feature', 'class')
    >>> print(probabilities)

    Notes
    -----
    This function is useful for handling categorical data in classification
    tasks, especially when the dataset may contain categories that do not 
    appear in the training data for every class.
    """
    is_frame( data, df_only=True, raise_exception=True)
    if V is None:
        V = data[feature_col].nunique()

    class_counts = data[class_col].value_counts()
    probability_tables = []

    # Iterating over each class to calculate probabilities
    for class_value in data[class_col].unique():
        class_subset = data[data[class_col] == class_value]
        feature_counts = class_subset[feature_col].value_counts()
        probabilities = (feature_counts + 1) / (class_counts[class_value] + V)
        probabilities.name = class_value
        probability_tables.append(probabilities.to_frame().T)

    # Using pandas.concat to combine the probability tables
    probability_table = pd.concat(probability_tables, sort=False).fillna(1 / V)
    # Transpose to match expected format: features as rows, classes as columns
    probability_table = probability_table.T

    return probability_table

def laplace_smoothing_word(
        word, label,  word_counts, class_counts, V):
    """
    Apply Laplace smoothing to estimate the conditional probability of a 
    word given a class.

    Laplace smoothing (add-one smoothing) is used to handle the issue of 
    zero probability in categorical data, particularly in the context of 
    text classification with Naive Bayes.

    The mathematical formula for Laplace smoothing is:
    
    .. math:: 
        P(w|c) = \frac{\text{count}(w, c) + 1}{\text{count}(c) + |V|}

    where `count(w, c)` is the count of word `w` in class `c`, `count(c)` is 
    the total count of all words in class `c`, and `|V|` is the size of the 
    vocabulary.

    Parameters
    ----------
    word : str
        The word for which the probability is to be computed.
    label : str
        The class for which the probability is to be computed.
    word_counts : dict
        A dictionary containing word counts for each class. The keys should 
        be tuples of the form (word, class).
    class_counts : dict
        A dictionary containing the total count of words for each class.
    V : int
        The size of the vocabulary, i.e., the number of unique words in 
        the dataset.

    Returns
    -------
    float
        The Laplace-smoothed probability of the word given the class.

    Example
    -------
    >>> from gofast.utils.mlutils import laplace_smoothing_word
    >>> word_counts = {('dog', 'animal'): 3, ('cat', 'animal'):
                       2, ('car', 'non-animal'): 4}
    >>> class_counts = {'animal': 5, 'non-animal': 4}
    >>> V = len(set([w for (w, c) in word_counts.keys()]))
    >>> laplace_smoothing_word('dog', 'animal', word_counts, class_counts, V)
    0.5
    
    References
    ----------
    - C.D. Manning, P. Raghavan, and H. Schütze, "Introduction to Information Retrieval",
      Cambridge University Press, 2008.
    - A detailed explanation of Laplace Smoothing can be found in Chapter 13 of 
      "Introduction to Information Retrieval" by Manning et al.

    Notes
    -----
    This function is particularly useful in text classification tasks where the
    dataset may contain a large number of unique words, and some words may not 
    appear in the training data for every class.
    """
    word_class_count = word_counts.get((word, label), 0)
    class_word_count = class_counts.get(label, 0)
    probability = (word_class_count + 1) / (class_word_count + V)
    return probability

@is_data_readable
def laplace_smoothing(
    data: Union[ArrayLike, DataFrame], 
    alpha: float = 1.0, 
    columns: Union[list, None] = None
) -> Union[ArrayLike, DataFrame]:
    """
    Applies Laplace smoothing to  data to calculate smoothed probabilities.

    Parameters
    ----------
    data : ndarray or DataFrame
        An array-like or DataFrame object containing categorical data. Each column 
        represents a feature, and each row represents a data sample.
    alpha : float, optional
        The smoothing parameter, often referred to as 'alpha'. This is 
        added to the count for each category in each feature. 
        Default is 1 (Laplace Smoothing).
    columns: list, optional
        Columns to construct the DataFrame when `data` is an ndarray. The 
        number of columns must match the second dimension of the ndarray.
        
    Returns
    -------
    smoothed_probs : ndarray or DataFrame
        An array or DataFrame of the same shape as `data` containing the smoothed 
        probabilities for each category in each feature.

    Raises
    ------
    ValueError
        If `columns` is provided and its length does not match the number 
        of columns in `data`.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.utils.mlutils import laplace_smoothing
    >>> data = np.array([[0, 1], [1, 0], [1, 1]])
    >>> laplace_smoothing(data, alpha=1)
    array([[0.4 , 0.6 ],
           [0.6 , 0.4 ],
           [0.6 , 0.6 ]])

    >>> data_df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    >>> laplace_smoothing(data_df, alpha=1)
       feature1  feature2
    0       0.4       0.6
    1       0.6       0.4
    2       0.6       0.6
    """
    if isinstance(data, np.ndarray):
        if columns:
            if len(columns) != data.shape[1]:
                raise ValueError("Length of `columns` does not match the shape of `data`.")
            data = pd.DataFrame(data, columns=columns)
        input_type = 'ndarray'
    elif isinstance(data, pd.DataFrame):
        input_type = 'dataframe'
    else:
        raise TypeError("`data` must be either a numpy.ndarray or a pandas.DataFrame.")

    smoothed_probs_list = []
    features = data.columns if input_type == 'dataframe' else range(data.shape[1])

    for feature in features:
        series = data[feature] if input_type == 'dataframe' else data[:, feature]
        counts = np.bincount(series, minlength=series.max() + 1)
        smoothed_counts = counts + alpha
        total_counts = smoothed_counts.sum()
        smoothed_probs = (series.map(lambda x: smoothed_counts[x] / total_counts)
                          if input_type == 'dataframe' else smoothed_counts[series] / total_counts)
        smoothed_probs_list.append(smoothed_probs)

    if input_type == 'dataframe':
        return pd.DataFrame({feature: probs for feature, probs in zip(
            features, smoothed_probs_list)})
    else:
        return np.column_stack(smoothed_probs_list)

def smart_label_classifier(
    y: ArrayLike, *,
    values: Union[float, List[float], None] = None,
    labels: Union[int, str, List[str]] = None,
    order: str = 'soft',
    func: Optional[Callable[[float], Union[int, str]]] = None,
    raise_warn: bool = True
) -> np.ndarray:
    """
    Maps a numeric array into class labels based on specified thresholds or
    a custom mapping function. The `smart_label_classifier` function 
    categorizes an array of continuous values into distinct classes, either
    by using predefined threshold values (`values`) or by applying a custom
    function (`func`). Optional `labels` can be used to name the categories.

    .. math::
        Y_i = 
        \begin{cases} 
            L_1, & \text{if } y_i \leq v_1 \\
            L_2, & \text{if } v_1 < y_i \leq v_2 \\
            \vdots \\
            L_{n+1}, & \text{if } y_i > v_n \\
        \end{cases}

    where :math:`y_i` represents the value of the `i`-th item in `y`, 
    and :math:`L` denotes the class labels corresponding to thresholds 
    :math:`v`.

    Parameters
    ----------
    y : ArrayLike
        One-dimensional array of numeric values to be categorized.

    values : float, list of float, optional
        Threshold values for categorization. If `values` is provided,
        items in `y` are mapped based on these thresholds. For instance,
        if `values = [1.0, 2.5]`, three classes will be generated: one
        for items less than or equal to 1.0, one for items between 1.0
        and 2.5, and one for items greater than 2.5.

    labels : int, str, or list of str, optional
        Labels for the resulting categories. If an integer is provided, 
        it specifies the number of classes to generate in `y` 
        automatically when `func` and `values` are `None`. For example, 
        if `labels=3`, the function divides `y` into three classes. If 
        `labels` is a list, each element should correspond to a class 
        created by `values` + 1. Mismatches raise an error in strict mode.

    order : {'soft', 'strict'}, default='soft'
        Mode to control the handling of `values`. If `order='strict'`,
        items in `y` must match `values` exactly; otherwise, approximate
        values are substituted. A warning is issued in soft mode if a 
        mismatch occurs.

    func : Callable, optional
        Custom function to categorize values in `y`. If `func` is provided,
        it takes precedence, and `values` are ignored. `func` should accept
        a single numeric input and return a category.

    raise_warn : bool, default=True
        If `True`, raises a warning when `order='soft'` and `values` 
        cannot be matched exactly or if `labels` do not match the 
        number of classes derived from `values`.

    Returns
    -------
    np.ndarray
        Array of the same length as `y`, with categorized values or
        labels if provided.

    Notes
    -----
    - This function requires either `values` or `func` to categorize `y`.
      If neither is provided, `labels` must be an integer to specify the
      number of classes.
    - `labels` should match the number of classes created by `values` + 1.
      If they do not, a `ValueError` is raised if `order` is `'strict'`.

    Examples
    --------
    >>> from gofast.utils.ml.utils import smart_label_classifier
    >>> import numpy as np
    >>> y = np.arange(0, 7, 0.5)
    
    Basic classification with values:
    >>> smart_label_classifier(y, values=[1.0, 3.2])
    array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])

    Assign custom labels:
    >>> smart_label_classifier(y, values=[1.0, 3.2], labels=['low', 'mid', 'high'])
    array(['low', 'low', 'low', 'mid', 'mid', 'mid', 'mid', 'high', 'high', 
           'high', 'high', 'high', 'high', 'high'], dtype=object)

    Using a custom function:
    >>> def custom_func(v):
    ...     if v <= 1: return 'low'
    ...     elif 1 < v <= 3.2: return 'mid'
    ...     else: return 'high'
    >>> smart_label_classifier(y, func=custom_func)
    array(['low', 'low', 'low', 'mid', 'mid', 'mid', 'mid', 'high', 'high', 
           'high', 'high', 'high', 'high', 'high'], dtype=object)

    Auto-generate classes:
    >>> smart_label_classifier(y, labels=3)
    array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])

    See Also
    --------
    _validate_func_values_labels : Helper function for validating `func` 
                                   and `values` parameters.
    _assert_labels_from_values : Helper function to validate `labels` 
                                 against `values`.
    _smart_mapper : Helper function for mapping continuous values to 
                    categorical classes based on thresholds.

    References
    ----------
    .. [1] Johnson, T., & Brown, A. (2021). *Categorical Data Mapping*.
       Data Science Journal, 17(4), 123-145.
    .. [2] Lee, K., & Singh, P. (2019). *Threshold-Based Classification
       Techniques*. Journal of Machine Learning, 9(2), 67-80.
    """
    # Preserve the structure of the input array/Series/DataFrame.
    collected = array_preserver(y, action='collect')
    
    name = None
    if isinstance(y, pd.Series) and hasattr(y, "name"):
        name = y.name
    
    arr = np.asarray(y).squeeze()

    if not _is_arraylike_1d(arr):
        raise TypeError(
            "Expected a one-dimensional array,"
            f" got array with shape {arr.shape}"
        )

    if isinstance(values, str):
        values = str2columns(values)

    if values is not None:
        values = is_iterable(values, parse_string=True, transform=True)
        approx_values: List[Tuple[float, float]] = []
        processed_values = np.zeros(len(values), dtype=float)

        for i, v in enumerate(values):
            try:
                v = float(v)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Value '{v}' must be a valid number.") from e

            non_nan_arr = arr[~np.isnan(arr)]
            diff = np.abs(non_nan_arr - v)
            min_idx = np.argmin(diff)

            if order == 'strict' and diff[min_idx] != 0.0:
                raise ValueError(
                    f"Value {v} is missing in the array. It must be present "
                    "when order is set to 'strict', or set order to 'soft'"
                    " to allow approximate matching."
                )

            matched_value = non_nan_arr[min_idx]
            processed_values[i] = matched_value

            if diff[min_idx] != 0.0:
                approx_values.append((v, matched_value))

        if approx_values and raise_warn:
            original_vals, substituted_vals = zip(*approx_values)
            verb = "are" if len(original_vals) > 1 else "is"
            warnings.warn(
                f"Values {original_vals} {verb} missing in the array. "
                f"Substituted with {substituted_vals}."
            )

    arr_copied = arr.copy()

    if func is None and values is None:
        return _validate_func_values_labels(
            func=func, 
            values= values, 
            labels= labels, 
            y=y, 
            order =order, 
        )

    mapper_func: Optional[Callable[[float], Union[int, str]]] = func
    if values is not None and func is None:
        mapper_func = lambda k: _smart_mapper(k, kr=processed_values)

    arr_mapped = pd.Series(arr_copied, name='temp').map(mapper_func).values

    label_mapping: Dict[Union[int, float], Union[int, str]] = {}
    if labels is not None:
        labels = is_iterable(labels, parse_string=True, transform=True)
        labels, label_mapping = _assert_labels_from_values(
            arr_mapped,
            processed_values,
            labels,
            label_mapping,
            raise_warn=raise_warn,
            order=order
        )

    if labels is not None:
        arr_mapped = pd.Series(
            arr_mapped, name=name or 'temp'
        ).map(label_mapping).values
    else:
        arr_mapped = arr_mapped if name is None else pd.Series(
            arr_mapped, name=name
        )
    
    # Attempt to restore original structure (index, shape, etc.)
    collected['processed'] = [arr_mapped]
    try:
        arr_mapped = array_preserver(
            collected,
            solo_return=True,
            action='restore',
            deep_restore=True, 
        )
    except Exception:
        # If it fails, fallback to raw arr_mapped, optional ignore warnings
        arr_mapped = return_if_preserver_failed(
            arr_mapped,
            warn="ignore",
            verbose=0
        )

    return arr_mapped

def _validate_func_values_labels(
    func: Optional[Callable],
    values: Optional[Union[float, List[float]]],
    labels: Optional[Union[int, str, List[str]]],
    y: np.ndarray,
    order: str
) -> np.ndarray:
    """
    Validates that either `func` or `values` is provided, and handles cases 
    where `labels` is provided as an integer when `func` and `values` are None.
    """

    # Raise an error if labels are not provided
    # when both func and values are None
    if labels is None:
        raise TypeError(
            "'func' cannot be None when 'values' are not provided."
        )
    
    # Handle the case where labels is an integer
    if isinstance(labels, int):
        if order == 'strict':
            raise TypeError(
                "'func' cannot be None when 'values' are not provided. "
                "To heuristically create `labels` classes, set `order='soft'`."
            )
        
        # Ensure `y` is a 1-dimensional array
        y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError(
                "Input array `y` must be one-dimensional for"
                " automatic class generation."
            )
        
        try:
            # Automatically create `labels` number of classes in `y`
            y_min, y_max = np.min(y), np.max(y)
            thresholds = np.linspace(y_min, y_max, labels + 1)[1:-1]
            categorized_y = np.digitize(y, bins=thresholds)
            return categorized_y
        except Exception as e:
            raise ValueError(
                "An error occurred while attempting to categorize `y`. "
                "Ensure `y` is numeric and contains valid values for"
                " thresholding."
            ) from e
    
    else:
        raise TypeError(
            "When `func` and `values` are None, `labels` should be "
            "an integer specifying the number of classes to generate."
        )
    
    return y

def _assert_labels_from_values(
    arr: np.ndarray,
    values: np.ndarray,
    labels: Union[int, str, List[str]],
    label_mapping: Dict,
    raise_warn: bool = True,
    order: str = 'soft'
) -> Tuple[List[Union[int, str]], Dict[Union[int, float], Union[int, str]]]:
    unique_labels = list(np.unique(arr))
    
    if not is_iterable(labels):
        labels = [labels]

    if not _check_consistency_size(unique_labels, labels, error='ignore'):
        if order == 'strict':
            verb = "were" if len(labels) > 1 else "was"
            raise TypeError(
                f"Expected {len(unique_labels)} labels for the {len(values)}"
                f" values renaming. {len(labels)} {verb} given."
            )

        expected_labels_count = len(values) + 1
        actual_labels_count = len(labels)
        if actual_labels_count != expected_labels_count:
            verb = "s are" if len(values) > 1 else " is"
            msg = (
                f"{len(values)} value{verb} passed. Labels for renaming "
                f"values expect to be composed of {expected_labels_count}"
                f" items ('number of values + 1') for pure categorization."
            )
            undefined_classes = unique_labels[len(labels):]
            labels = list(labels) + list(undefined_classes)
            labels = labels[:len(unique_labels)]
            msg += ( 
                f" Classes {smart_format(undefined_classes)}"
                " cannot be renamed."
                )

            if raise_warn:
                warnings.warn(msg)

    label_mapping = dict(zip(unique_labels, labels))
    return labels, label_mapping

def _smart_mapper(
    k: float,
    kr: np.ndarray,
    return_dict_map: bool = False
) -> Union[int, Dict[int, bool], float]:
    if len(kr) == 1:
        conditions = {
            0: k <= kr[0],
            1: k > kr[0]
        }
    elif len(kr) == 2:
        conditions = {
            0: k <= kr[0],
            1: kr[0] < k <= kr[1],
            2: k > kr[1]
        }
    else:
        conditions = {}
        for idx in range(len(kr) + 1):
            if idx == 0:
                conditions[idx] = k <= kr[idx]
            elif idx == len(kr):
                conditions[idx] = k > kr[-1]
            else:
                conditions[idx] = kr[idx - 1] < k <= kr[idx]

    if return_dict_map:
        return conditions

    for class_label, condition in conditions.items():
        if condition:
            return class_label if not math.isnan(k) else np.nan

    return np.nan
        
def _extract_target(
        X, target: Union[ArrayLike, int, str, List[Union[int, str]]]):
    """
    Extracts and validates the target variable(s) from the dataset.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The dataset from which to extract the target variable(s).
    target : ArrayLike, int, str, or list of int/str
        The target variable(s) to be used. If an array-like or DataFrame, 
        it's directly used as `y`. If an int or str (or list of them), it 
        indicates the column(s) in `X` to be used as `y`.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        The dataset without the target column(s).
    y : pd.Series, np.ndarray, pd.DataFrame
        The target variable(s).
    target_names : list of str
        The names of the target variable(s) for labeling purposes.
    """
    target_names = []

    if isinstance(target, (list, pd.DataFrame)) or (
            isinstance(target, pd.Series) and not isinstance(X, np.ndarray)):
        if isinstance(target, list):  # List of column names or indexes
            if all(isinstance(t, str) for t in target):
                y = X[target]
                target_names = target
            elif all(isinstance(t, int) for t in target):
                y = X.iloc[:, target]
                target_names = [X.columns[i] for i in target]
            X = X.drop(columns=target_names)
        elif isinstance(target, pd.DataFrame):
            y = target
            target_names = target.columns.tolist()
            # Assuming target DataFrame is not part of X
        elif isinstance(target, pd.Series):
            y = target
            target_names = [target.name] if target.name else ["target"]
            if target.name and target.name in X.columns:
                X = X.drop(columns=target.name)
                
    elif isinstance(target, (int, str)):
        if isinstance(target, str):
            y = X.pop(target)
            target_names = [target]
        elif isinstance(target, int):
            y = X.iloc[:, target]
            target_names = [X.columns[target]]
            X = X.drop(columns=X.columns[target])
    elif isinstance(target, np.ndarray) or (
            isinstance(target, pd.Series) and isinstance(X, np.ndarray)):
        y = np.array(target)
        target_names = ["target"]
    else:
        raise ValueError(
            "Unsupported target type or target does not match X dimensions.")
    
    check_consistent_length(X, y)
    
    return X, y, target_names

