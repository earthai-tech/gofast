# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Created on Wed Dec 20 11:34:33 2023

These functions offer a range of probability utilities suitable for large 
datasets, leveraging the power of NumPy and SciPy for efficient computation. 
They cover various aspects of probability calculations, from simple PDF and 
CDF computations to more complex scenarios like sampling from distributions.

"""

import numpy as np
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import poisson

from ..tools._dependency import import_optional_dependency 


def stochastic_volatility_model(arr, /, ):
    """
    Stochastic Volatility Model using PyMC3.

    A stochastic volatility model captures the phenomenon of heteroskedasticity
    (non-constant volatility) in time-series data, such as financial returns.

    Parameters
    ----------
    arr : array_like
        Time-series data representing returns.

    Returns
    -------
    model : pymc3.Model
        A PyMC3 model object ready for sampling.

    Formula
    -------
    The model typically includes:
    - y[t] ~ N(0, exp(2 * v[t]))
    - v[t] ~ N(v[t-1], sigma^2)
    Where y[t] is the observed value at time t, and v[t] is the log volatility.

    Example
    -------
    >>> import numpy as np
    >>> returns = np.random.normal(0, 1, 100)
    >>> model = stochastic_volatility_model(returns)
    """
    import_optional_dependency('pymc3')
    import pymc3 as pm
    with pm.Model() as model:
        sigma = pm.Exponential('sigma', 1.0)
        v = pm.GaussianRandomWalk('v', sigma=sigma, shape=len(arr))
        pm.Normal('y', mu=0, sd=pm.math.exp(v), observed=arr) # y
        
    return model


def hierarchical_linear_model(X, y, groups):
    """
    Hierarchical Linear Model using PyMC3.

    Multilevel models are regression models in which parameters are set up to
    vary at more than one level, often used for grouped or hierarchical data.

    Parameters
    ----------
    X : array_like
        Predictor variables.
    y : array_like
        Response variable.
    groups : array_like
        Grouping variable indicating the hierarchy.

    Returns
    -------
    model : pymc3.Model
        A PyMC3 model object ready for sampling.

    Formula
    -------
    The hierarchical linear model can be expressed as:
    - y[i] ~ N(X[i] * beta[group[i]], sigma[group[i]])
    - beta[group[i]] ~ N(mu_beta, sigma_beta)
    Where y[i] is the observed value, X[i] is the predictor for observation i,
    and beta[group[i]] is the group-specific coefficient.

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 3))
    >>> y = np.random.normal(size=100)
    >>> groups = np.random.randint(0, 5, 100)
    >>> model = hierarchical_linear_model(X, y, groups)
    """
    import_optional_dependency('pymc3')
    import pymc3 as pm
    with pm.Model() as model:
        # Group-specific parameters
        mu_beta = pm.Normal('mu_beta', mu=0, sd=10, shape=X.shape[1])
        sigma_beta = pm.HalfNormal('sigma_beta', 10)
        sigma = pm.HalfNormal('sigma', sd=10, shape=len(np.unique(groups)))

        # Model error
        pm.Normal('y_est', mu=pm.math.dot(X, mu_beta[groups]),
                          sd=sigma[groups], observed=y) # y_est
        
        # Group-specific slopes and intercepts
        pm.Normal('beta', mu=mu_beta, sd=sigma_beta,
                         shape=(len(np.unique(groups)), X.shape[1])) # beta 
    return model

def normal_pdf(data, mean, std_dev):
    """
    Compute the Probability Density Function (PDF) for 
    a normal distribution.

    Parameters
    ----------
    data : ndarray
        Data for which the PDF is to be computed.
    mean : float
        Mean of the normal distribution.
    std_dev : float
        Standard deviation of the normal distribution.

    Returns
    -------
    pdf_values : ndarray
        PDF values for the input data.

    Examples
    --------
    >>> data = np.array([1, 2, 3])
    >>> normal_pdf(data, mean=0, std_dev=1)
    array([...])
    """
    return norm.pdf(data, mean, std_dev)

def normal_cdf(data, mean, std_dev):
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
    return norm.cdf(data, mean, std_dev)

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


def poisson_logpmf(data, lambda_param):
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
    return poisson.logpmf(data, lambda_param)

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
    return np.random.uniform(low, high, size)
