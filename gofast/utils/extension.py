# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides additional utilities and tools that extend 
the core capabilities of the gofast package.
"""

import numpy as np
import pandas as pd
from .validator import validate_length_range
from ..core.array_manager import to_series 
from ..core.checks import assert_ratio, check_numeric_dtype
from ..core.io import is_data_readable 

__all__ = ["reorder_importances", "spread_coverage"]

@is_data_readable 
def reorder_importances(
    data,
    axis: int = 0,
    method: str | callable = "sum",
    ascending: bool = False,
    top_n: int | None = None,
    copy: bool = True
) -> pd.DataFrame:
    """
    Reorder the rows or columns of feature importances
    (or any numeric data) in descending (or ascending)
    order, based on an aggregation method such as
    "sum", "mean", "max", or a custom callable.

    Parameters
    ----------
    data : array-like, pd.Series, or pd.DataFrame
        The data whose rows or columns we want
        to reorder. If a NumPy array or Series is
        provided, it will be converted to a
        DataFrame.
    axis : int, default=0
        Axis along which to compute the sorting
        key. If ``0``, reorder rows based on an
        aggregation across columns. If ``1``,
        reorder columns based on an aggregation
        across rows.
    method : {'sum', 'mean', 'max'} or callable, default='sum'
        Aggregation method used to evaluate
        each row (if ``axis=0``) or each column
        (if ``axis=1``). If a callable is given,
        it should accept a 1D array and return a
        scalar, e.g. ``np.median``. 
    ascending : bool, default=False
        If ``False``, sort from largest to
        smallest. If ``True``, sort from smallest
        to largest.
    top_n : int, optional
        If provided, keep only the top ``n``
        rows or columns after sorting (depending
        on ``axis``). For example, ``top_n=3``
        would keep only the 3 best rows when
        ``axis=0`` and discarding the rest.
    copy : bool, default=True
        If ``True``, return a new DataFrame
        without modifying the original. If
        ``False``, sort the data in place.
        (Note that in-place sorting is only
        valid if ``data`` is already a
        DataFrame. Otherwise, a new DataFrame
        will still be returned.)

    Returns
    -------
    pd.DataFrame
        A DataFrame sorted along the given axis
        according to the specified aggregation
        method.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.utils.extension import reorder_importances 
    >>> arr = np.array([[0.4, 0.3], [0.2, 0.8], [0.7, 0.1]])
    >>> reorder_importances(arr, axis=0, method="sum")
           0    1
    2    0.7  0.1
    0    0.4  0.3
    1    0.2  0.8

    >>> df = pd.DataFrame({
    ...     "ModelA": [0.2, 0.4, 0.1, 0.3],
    ...     "ModelB": [0.3, 0.1, 0.4, 0.2]},
    ...     index=["f1","f2","f3","f4"])
    >>> reorder_importances(df, axis=0, method="mean", top_n=2)
         ModelA  ModelB
    f2      0.4     0.1
    f1      0.2     0.3

    Notes
    -----
    - When a callable is used for ``method``, it
      is applied to each 1D slice (row or column).
    - By default, sorting is descending to quickly
      see the "largest" importances.
    """

    # 1) Convert data to a DataFrame if needed
    if isinstance(data, pd.DataFrame):
        df = data if not copy else data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame()
        if not copy:
            # Series doesn't truly allow "in-place" for sorting by aggregates
            # Return a new DataFrame anyway
            pass
    else:
        # assume NumPy array or array-like
        df = pd.DataFrame(data)

    # 2) Choose aggregator
    if isinstance(method, str):
        if method not in ("sum", "mean", "max"):
            raise ValueError(
                "method must be one of {'sum', 'mean', 'max'} "
                "or a callable."
            )
        agg_fn = getattr(np, method)
    elif callable(method):
        agg_fn = method
    else:
        raise TypeError(
            "method should be a string ('sum', 'mean', 'max') "
            "or a callable."
        )

    # 3) Compute the aggregation on the axis
    #    e.g. if axis=0, we look across columns to get row-based metric
    #    if axis=1, we look across rows to get column-based metric
    #    We'll create a series with that aggregator:
    if axis == 0:
        metric = df.apply(agg_fn, axis=1)
        # Sort row labels by that metric
        sorted_idx = metric.sort_values(ascending=ascending).index
        df = df.loc[sorted_idx]
    elif axis == 1:
        metric = df.apply(agg_fn, axis=0)
        # Sort column labels
        sorted_idx = metric.sort_values(ascending=ascending).index
        df = df[sorted_idx]
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns).")

    # 4) Optionally keep only the top_n
    if top_n is not None:
        if axis == 0:
            df = df.head(top_n)
        else:
            df = df.iloc[:, :top_n]
    
    df = to_series(df, handle_2d="passthrough") 

    return df

def spread_coverage(
    pred_q,  
    q="50%",  
    cov_col=None,  
    spread=0.2,  
    use_relative=True,  
    abs_margin=None,  
    forecast_err="symmetric",  
    spread_range=None,  
    perturbation=None,  
    return_type="df",  
    name=None,  
    nloc="prefix",  
    trailer="_",  
    verbose=0  
):
    r"""
    Spread lower (q10) and upper (q90) predictions from a given set 
    of median predictions (q50) in order to create an approximate 
    prediction interval based on spread value. The function supports
    symmetric as well as asymmetric forecast errors, and allows for 
    optional perturbation and renaming of the output series.

    Parameters
    ----------
    pred_q : array-like
        Input predictions. These values are interpreted as the 
        median predictions (i.e. `q50`) when ``q`` is set to 
        ``"50%"`` (or 0.5). If a pandas DataFrame is provided, then 
        the column specified by `cov_col` is used.
    q : {``"50%"``, ``"10%"``, ``"90%"``, 0.5, 0.1, 0.9}, default 
        ``"50%"``
        Indicates which quantile the input `pred_q` represents. For 
        example, if set to ``"50%"`` or 0.5, `pred_q` is considered 
        the median predictions (q50) from which q10 and q90 are 
        generated. If set to ``"10%"`` (or 0.1), then `pred_q` is 
        assumed to be the 10th quantile and the function computes the 
        corresponding q50 and q90. Similarly, ``"90%"`` (or 0.9) 
        indicates that `pred_q` represents the 90th quantile.
    cov_col : ``str``, optional
        When `pred_q` is provided as a DataFrame, this string specifies 
        the column name to extract the numeric predictions.
    spread : ``float``, default 0.2
        The relative spread used to compute the lower and upper 
        predictions when `use_relative` is True. In this case, the 
        lower prediction is computed as 
        :math:`q50 \\times (1 - \\text{spread})` and the upper as 
        :math:`q50 \\times (1 + \\text{spread})`.
    use_relative : ``bool``, default True
        If True, the function uses the relative spread to compute 
        the quantiles; if False, `spread` is treated as an absolute 
        margin.
    abs_margin : ``float``, optional
        Absolute margin to be used in place of `spread`. If provided, 
        the lower and upper predictions are computed as 
        :math:`q50 - \\text{abs_margin}` and 
        :math:`q50 + \\text{abs_margin}`, respectively.
    forecast_err : ``str``, default ``"symmetric"``
        Specifies the forecast error structure. Acceptable values 
        are ``"symmetric"`` and ``"asymmetric"``. For asymmetric 
        errors, `spread_range` should be provided.
    spread_range : tuple or list of two floats, optional
        For asymmetric forecast errors, specifies the lower and 
        upper spread as 
        ``(spread_low, spread_high)``. These values are used to 
        compute q10 and q90 when `forecast_err` is set to 
        ``"asymmetric"``.
    perturbation : ``float`` or list/tuple of two floats, optional
        A factor to perturb the computed q10 and q90. If a list of two 
        values is provided, the first is applied to q10 and the second 
        to q90. If a single number is provided, it is applied to both.
    return_type : ``str``, default ``"df"``
        Specifies the return type. If set to ``"df"``, the function 
        returns a pandas DataFrame with columns `q10`, `q50`, and 
        `q90`. Otherwise, it returns a tuple of NumPy arrays 
        `(q10, q50, q90)`.
    name : ``str``, optional
        An optional string to rename the output columns. For example, 
        if `name` is ``"sale"``, the columns might be renamed to 
        ``sale_q10``, ``sale_q50``, and ``sale_q90`` (depending on 
        `nloc` and `trailer`).
    nloc : ``str``, default ``"prefix"``
        Specifies whether the `name` should be used as a prefix or 
        suffix in the output column names. Valid options are 
        ``"prefix"`` and ``"suffix"``.
    trailer : ``str``, default ``"_"``
        A string used as a separator between `name` and the quantile 
        label in the output column names.
    verbose : ``int``, default 0
        Verbosity level; higher values will print debug information 
        during processing.

    Returns
    -------
    df : pandas.DataFrame
        If `return_type` is set to ``"df"``, a DataFrame with columns 
        `q10`, `q50`, and `q90` is returned.
    or
    tuple of np.ndarray
        A tuple `(q10, q50, q90)` is returned if `return_type` is not 
        ``"df"``.

    Formulation
    ------------
    Let :math:`q_{50,i}` be the median prediction for observation 
    :math:`i`, and let :math:`f_i` be defined by:

    .. math::
       f_i = \\begin{cases}
         \\text{spread} & \\text{if } \\text{forecast_err} = 
         \\text{"symmetric"} \\\\
         \\text{spread_low} & \\text{for asymmetric errors (lower)} \\\\
         \\text{spread_high} & \\text{for asymmetric errors (upper)}
       \\end{cases}

    Then, for the symmetric case,
    
    .. math::
       q_{10,i} = q_{50,i} \\times (1 - f_i) \\quad \\text{and} \\quad 
       q_{90,i} = q_{50,i} \\times (1 + f_i).

    For absolute margins, these become:

    .. math::
       q_{10,i} = q_{50,i} - \\text{abs_margin} \\quad \\text{and} \\quad
       q_{90,i} = q_{50,i} + \\text{abs_margin}.

    If perturbation is applied, then:
    
    .. math::
       q_{10,i} \\text{ is perturbed by } (1 - p_1) \\quad \\text{and} \\quad
       q_{90,i} \\text{ is perturbed by } (1 + p_2),
    
    where :math:`p_1` and :math:`p_2` are the perturbation factors.

    Examples
    --------
    1) **Basic usage with relative spread:**

       >>> from gofast.utils.extension import spread_coverage
       >>> import numpy as np
       >>> q50 = np.array([100, 150, 200])
       >>> df = spread_coverage(q50, q="50%", spread=0.2)
       >>> df
           q10    q50    q90
       0  80.0  100.0  120.0
       1 120.0  150.0  180.0
       2 160.0  200.0  240.0

    2) **Using an absolute margin:**

       >>> df = spread_coverage(q50, q="50%", 
       ...          abs_margin=20, use_relative=False)
       >>> df
           q10    q50    q90
       0  80.0  100.0  120.0
       1 130.0  150.0  170.0
       2 180.0  200.0  220.0

    3) **Returning a tuple and applying perturbation:**

       >>> q10, q50_arr, q90 = spread_coverage(
       ...         q50, q="50%", spread=0.1, 
       ...         perturbation=[0.01, 0.03], 
       ...         return_type="tuple")
       >>> q10
       array([ 89., 134., 179.])
       >>> q50_arr
       array([100., 150., 200.])
       >>> q90
       array([111., 164., 221.])

    Notes
    -----
    This function uses the inline method `assert_ratio` to standardize 
    the input quantile ``q``. The parameters `spread` and 
    `abs_margin` control the width of the prediction interval. The 
    optional `perturbation` parameter allows further fine-tuning of the 
    lower and upper predictions. The output columns can be renamed by 
    specifying `name`, `nloc`, and `trailer`. This implementation 
    assumes a symmetric error distribution by default [1]_.

    See Also
    --------
    gofast.core.checks.check_numeric_dtype` :
        Validates numeric data types in arrays.

    References
    ----------
    .. [1] Armstrong, J. S. "Prediction intervals." 
           *International Journal of Forecasting* 16.4 (2000): 521-530.
    """

    def log(level, msg):
        # Simple logger that prints messages if verbose is high enough.
        if verbose >= level:
            print(msg)

    # If pred_q is a DataFrame, extract the specified column.
    if isinstance(pred_q, pd.DataFrame):
        if cov_col is None:
            raise ValueError(
                "cov_col must be provided when pred_q is a DataFrame."
            )
        pred_q = pred_q[cov_col]

    pred_q= check_numeric_dtype(
        pred_q, ops="validate", 
        coerce=True, 
        param_names ={"X":"pred_q"}
    )
    # Convert pred_q to a NumPy array of floats.
    pred_q = np.asarray(pred_q, dtype=float)
    
    # Standardize q: convert a string like "50%" to 0.5 using the 
    # inline method assert_ratio (assumed imported).
    q = assert_ratio(q, bounds=(0, 1), 
                     exclude_values=[0, 1], 
                     name="Quantile q")
    log(3, f"Processing q={q}, forecast_err={forecast_err}")

    # Determine spread values based on forecast_err.
    if forecast_err == "symmetric":
        spread_low  = spread_high = spread
    elif forecast_err == "asymmetric":
        if spread_range is None:
            log(2, "spread_range not provided, setting default asymmetric spread.")
            spread_low, spread_high = 0.1, 0.2  # Default values.
        elif (isinstance(spread_range, (tuple, list)) and 
              len(spread_range) == 2):
            spread_low, spread_high = spread_range
        else:
            raise ValueError(
                "spread_range must be a tuple (spread_low, spread_high) "
                "for asymmetric forecast errors."
            )
    else:
        raise ValueError(
            "Invalid value for `forecast_err`. Choose 'symmetric' or "
            "'asymmetric'."
        )
    log(3, f"Spread values - Low: {spread_low}, High: {spread_high}")

    # Define pred_q50, pred_q10, and pred_q90 based on the input quantile.
    if q == 0.5:
        pred_q50 = pred_q
        pred_q10 = (pred_q50 * (1 - spread_low)
                    if use_relative else pred_q50 - spread_low)
        pred_q90 = (pred_q50 * (1 + spread_high)
                    if use_relative else pred_q50 + spread_high)
    elif q == 0.1:
        pred_q10 = pred_q
        pred_q50 = (pred_q10 / (1 - spread_low)
                    if use_relative else pred_q10 + spread_low)
        pred_q90 = (pred_q50 * (1 + spread_high)
                    if use_relative else pred_q50 + spread_high)
    elif q == 0.9:
        pred_q90 = pred_q
        pred_q50 = (pred_q90 / (1 + spread_high)
                    if use_relative else pred_q90 - spread_high)
        pred_q10 = (pred_q50 * (1 - spread_low)
                    if use_relative else pred_q50 - spread_low)
    else:
        raise ValueError(
            "Invalid value for `q`. Must be '50%', '10%', '90%' or their "
            "numerical equivalents."
        )

    # Override with absolute margin if provided.
    if abs_margin is not None:
        pred_q10 = pred_q50 - abs_margin
        pred_q90 = pred_q50 + abs_margin

    # Apply perturbations if specified.
    if perturbation is not None:
        if isinstance(perturbation, (list, tuple)) and len(perturbation) == 2:
            perturbation = validate_length_range(
                perturbation, param_name="Perturbation"
            )
            p1, p2 = perturbation
        elif isinstance(perturbation, (int, float)):
            p1 = p2 = perturbation
        else:
            raise ValueError(
                "`perturbation` should be a float or a list of two floats."
            )
        pred_q10 *= 1 - p1
        pred_q90 *= 1 + p2

    log(3, f"Final predictions - q10: {pred_q10}, q50: {pred_q50}, q90: {pred_q90}")

    _return_type = return_type 
    if _return_type =="series": 
        return_type ="df"
        
    # Construct output: return a DataFrame or a tuple.
    if return_type == "df":
        output = pd.DataFrame({
            "q10": pred_q10,
            "q50": pred_q50,
            "q90": pred_q90
        })
        # Rename columns if a name is provided.
        if name:
            if nloc == "prefix":
                output.columns = [f"{name}{trailer}{col}" 
                                  for col in output.columns]
            elif nloc == "suffix":
                output.columns = [f"{col}{trailer}{name}" 
                                  for col in output.columns]
                
        if _return_type=="series": 
            # Extract each column as an individual Series and return them.
            return (output.iloc[:, 0], output.iloc[:, 1], output.iloc[:, 2])
    
        return output
    
    return pred_q10, pred_q50, pred_q90
