# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
These functions offer a range of probability utilities suitable for large 
datasets, leveraging the power of NumPy and SciPy for efficient computation. 
They cover various aspects of probability calculations, from simple PDF and 
CDF computations to more complex scenarios like sampling from distributions.

"""

import numpy as np
import pandas as pd 
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import poisson

from .._typing import Array1D, ArrayLike,  DataFrame, Series 
from .._typing import Union, Tuple, List 
from ..decorators import Extract1dArrayOrSeries, Dataify 
from ..decorators import DataTransformer   
from ..tools.funcutils import ensure_pkg 
# from ..tools.validator import _is_arraylike_1d, _is_numeric_dtype
try: 
    import pymc3 as pm
except: pass 


@Extract1dArrayOrSeries(column=0, method='soft', verbose=True)
@ensure_pkg("pymc3")
def stochastic_volatility_model(
    arr: Union[Array1D, Series, DataFrame], 
    column: Union[str, int] = None,  
    sigma_init: float = 1.0, 
    mu: float = 0.0, 
    sampling: bool = False,
    return_inference_data: bool = True,
    n_samples: int = 1000
 ): 
    """
    Computes a stochastic volatility model using PyMC3 for time-series data.
    
    The model captures the phenomenon of heteroskedasticity in financial 
    time-series data, representing non-constant volatility. This implementation 
    provides flexibility in adjusting the initial model parameters and supports 
    both Pandas DataFrames and NumPy ndarrays as input.
    
    The stochastic volatility model is defined by the following equations:
    
    .. math::
        y[t] \sim \mathcal{N}(0, \exp(2 \cdot v[t])),
        v[t] \sim \mathcal{N}(v[t-1], \sigma^2),
    
    where :math:`y[t]` is the observed value at time :math:`t`, and :math:`v[t]` 
    is the log volatility at time :math:`t`. :math:`\sigma` is a model parameter 
    controlling the volatility of the volatility (vol of vol).
    
    Parameters
    ----------
    arr : Union[np.ndarray, pd.Series, pd.DataFrame]
        Time-series data representing returns. If a DataFrame is passed, a 
        specific column can be selected with the `column` parameter.
    column : Union[str, int], optional
        Specifies the column to use when a DataFrame is passed. It can be either 
        the column name (str) or the column index (int). Default is None, which 
        selects the first column if a DataFrame is passed.
    sigma_init : float, optional
        Initial value for the sigma parameter in the Exponential distribution, 
        which represents the prior belief about the volatility. Default is 1.0.
    mu : float, optional
        The mean parameter for the Gaussian distribution of the observed returns.
        Default is 0.0.
    sampling : bool, optional
        If True, performs sampling on the model to generate posterior 
        distributions. Default is False.
    return_inference_data : bool, optional
        If True and `sampling` is True, returns the generated samples as 
        an ArviZ InferenceData object; otherwise, returns a MultiTrace object.
        Default is True.
    n_samples : int, optional
        The number of samples to draw in the sampling process. Default is 1000.
    
    Returns
    -------
    model : pymc3.Model
        A PyMC3 model object.
    trace : Union[pm.backends.base.MultiTrace, pm.backends.arviz.InferenceData], optional
        The trace resulting from sampling, returned only if `sampling=True`.
        The format depends on `return_inference_data`.
    
    Raises
    ------
    TypeError
        If the input data type is not numeric.
    
    Examples
    --------
    >>> from gofast.stats.probs import stochastic_volatility_model
    >>> import numpy as np
    >>> returns = np.random.normal(0, 1, 100)
    >>> model = stochastic_volatility_model(returns)
    >>> model, trace = stochastic_volatility_model(returns, sampling=True, n_samples=1000)
    """
    import pymc3 as pm

    if not isinstance(arr, (np.ndarray, pd.Series)):
        raise TypeError("Input must be a NumPy ndarray or Pandas Series.")

    with pm.Model() as model:
        sigma = pm.Exponential('sigma', 1.0 / sigma_init)
        v = pm.GaussianRandomWalk('v', sigma=sigma, shape=len(arr))
        pm.Normal('y', mu=mu, sd=pm.math.exp(v), observed=arr)

        if sampling:
            with model:
                trace = pm.sample(n_samples, return_inferencedata=return_inference_data)
    if sampling:
        return model, trace
    else:
        return model

@ensure_pkg("pymc3")
def hierarchical_linear_model(
    X: ArrayLike, 
    y: Array1D, 
    groups: ArrayLike,
    mu: float = 0.0,
    sd_mu_beta: float = 10.0,
    sd_sigma_beta: float = 10.0,
    sd_sigma: float = 10.0,
    return_inference_data: bool =True,
    sampling: bool=False, 
    n_samples: int = 1000
): 
    """
    Builds a hierarchical linear model using PyMC3, ideal for grouped or 
    hierarchical data.

    Hierarchical or multilevel models are a type of regression model where 
    parameters are designed to vary at more than one level, making them 
    suitable for datasets that involve groups or hierarchies.
    
    The hierarchical linear model can be expressed using the following equations:

    .. math::
        y[i] \sim \mathcal{N}(X[i] \cdot \beta[\text{{group}}[i]], \sigma[\text{{group}}[i]])

    .. math::
        \beta[\text{{group}}[i]] \sim \mathcal{N}(\mu_{\beta}, \sigma_{\beta})

    where :math:`y[i]` is the observed value, :math:`X[i]` is the predictor
    for observation :math:`i`,  and :math:`\beta[\text{{group}}[i]]` is the 
    group-specific coefficient.
    

    Parameters
    ----------
    X : np.ndarray
        The predictor variables matrix.
    y : np.ndarray
        The response variable vector.
    groups : np.ndarray
        An array-like object indicating the grouping variable for hierarchy. 
        Each element corresponds to the group of the respective observation in X and y.
    mu : float, optional
        The mean value for the normal distribution of `mu_beta`. 
        Default is 0.0.
    sd_mu_beta : float, optional
        The standard deviation for the normal distribution of `mu_beta`.
        Default is 10.0.
    sd_sigma_beta : float, optional
        The standard deviation for the half-normal distribution of `sigma_beta`. 
        Default is 10.0.
    sd_sigma : float, optional
        The standard deviation for the half-normal distribution of `sigma`.
        Default is 10.0.
    return_inference_data : bool, optional
        If True and sampling is performed, the function returns an ArviZ 
        InferenceData object; otherwise, it returns a MultiTrace object. 
        Default is True.
    sampling : bool, optional
        If True, performs sampling on the model to generate posterior 
        distributions. Default is False.
    n_samples : int, optional
        The number of posterior samples to draw. Default is 1000.
    Returns
    -------
    model : pm.Model
        The PyMC3 model object, ready for sampling.
    trace : Union[pm.backends.base.MultiTrace, pm.backends.arviz.InferenceData], optional
        The trace resulting from the sampling process, returned only if 
        sampling is performed. The format depends on `return_inference_data`.

    Example
    -------
    >>> import numpy as np
    >>> from gofast.stats.probs import hierarchical_linear_model
    >>> X = np.random.normal(size=(100, 3))
    >>> y = np.random.normal(size=100)
    >>> groups = np.random.randint(0, 5, size=100)
    >>> model, trace = hierarchical_linear_model(X, y, groups, n_samples=1000)
    """
    
    unique_groups = np.unique(groups)
    group_idx = np.array([np.where(unique_groups == g)[0][0] for g in groups])

    with pm.Model() as model:
        # Hyperpriors
        mu_beta = pm.Normal('mu_beta', mu=0, sd=10, shape=X.shape[1])
        sigma_beta = pm.HalfNormal('sigma_beta', 10)
        sigma = pm.HalfNormal('sigma', sd=10, shape=len(unique_groups))

        # Group-specific intercepts and slopes
        beta = pm.Normal('beta', mu=mu_beta, sd=sigma_beta, shape=(
            len(unique_groups), X.shape[1]))
        # Expected value per observation
        y_est = pm.math.dot(X, beta[group_idx].T)

        # Model error
        pm.Normal('y_obs', mu=y_est, sd=sigma[group_idx], observed=y)

        if sampling and n_samples > 0:
            trace = pm.sample(n_samples, return_inferencedata=return_inference_data)
            return model, trace

    return model

@DataTransformer(mode='lazy')
@Dataify(enforce_df=False, ignore_mismatch=True, fail_silently=True ) 
def normal_pdf(
    data: ArrayLike, 
    mean: float = 0, 
    std_dev: float = 1, 
    scale: float = 1, 
    loc: float = 0,
    columns: List[str]=None, 
    **kws
    )-> ArrayLike:
    """
    Compute the Probability Density Function (PDF) for a normal 
    distribution.

    The formula for the PDF of a normal distribution is:

    .. math::
        f(x | \mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}}
        \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)

    Where :math:`\mu` is the mean, :math:`\sigma` is the standard deviation, 
    and :math:`x` is the data point.

    Parameters
    ----------
    data : ndarray
        Data for which the PDF is to be computed.
    mean : float, optional
        Mean (:math:`\mu`) of the normal distribution. Default is 0.
    std_dev : float, optional
        Standard deviation (:math:`\sigma`) of the normal distribution. 
        Default is 1.
    scale : float, optional
        Scaling factor for the PDF, not part of the standard normal 
        distribution formula. Applied as a multiplier to the result. 
        Default is 1.
    loc : float, optional
        Location parameter, used to shift the PDF along the x-axis. 
        Applied before scaling. Default is 0.

    Returns
    -------
    pdf_values : ndarray
        PDF values for the input data, scaled and shifted as specified.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.stats.probs import normal_pdf 
    >>> data = np.array([1, 2, 3])
    >>> normal_pdf(data, mean=0, std_dev=1)
    array([...])
    
    To apply scaling and location shift:

    >>> normal_pdf(data, mean=0, std_dev=1, scale=0.5, loc=2)
    array([...])
    """
    shifted_data = (data - loc) / std_dev
    pdf_values = norm.pdf(shifted_data, mean, 1, **kws) * scale
    return pdf_values

def normal_cdf(data, mean, std_dev, *args, **kws):
    """
    Compute the Cumulative Distribution Function (CDF) for
    a normal distribution.

    Parameters
    ----------
    data : ndarray
        Data for which the CDF is to be computed.
    mean : float
        Mean of the normal distribution.
    std_dev : float
        Standard deviation of the normal distribution.

    Returns
    -------
    cdf_values : ndarray
        CDF values for the input data.

    Examples
    --------
    >>> data = np.array([1, 2, 3])
    >>> normal_cdf(data, mean=0, std_dev=1)
    array([...])
    """
    return norm.cdf(data, mean, std_dev, *args, **kws)

def binomial_pmf(trials, p_success, n_successes):
    """
    Compute the Probability Mass Function (PMF) for a 
    binomial distribution.

    Parameters
    ----------
    trials : int
        Number of trials.
    p_success : float
        Probability of success in each trial.
    n_successes : int or array_like
        Number of successes to compute the PMF for.

    Returns
    -------
    pmf_values : ndarray
        PMF values for the given number of successes.

    Examples
    --------
    >>> binomial_pmf(trials=10, p_success=0.5, n_successes=5)
    0.24609375
    """
    return binom.pmf(n_successes, trials, p_success)


def poisson_logpmf(data, lambda_param, *args, **kws):
    """
    Compute the logarithm of the Probability Mass Function 
    (PMF) for a Poisson distribution.

    Parameters
    ----------
    data : ndarray
        Data for which the log PMF is to be computed.
    lambda_param : float
        Expected number of occurrences 
        (lambda parameter of the Poisson distribution).

    Returns
    -------
    log_pmf_values : ndarray
        Log PMF values for the input data.

    Examples
    --------
    >>> data = np.array([1, 2, 3])
    >>> poisson_logpmf(data, lambda_param=2)
    array([...])
    """
    return poisson.logpmf(data, lambda_param, *args, **kws)

def uniform_sampling(low, high, size):
    """
    Generate samples from a uniform distribution.

    Parameters
    ----------
    low : float
        Lower boundary of the distribution.
    high : float
        Upper boundary of the distribution.
    size : int
        Number of samples to generate.

    Returns
    -------
    samples : ndarray
        Samples from the uniform distribution.

    Examples
    --------
    >>> uniform_sampling(low=0, high=10, size=5)
    array([...])
    """
    return np.random.uniform(low, high, size )
