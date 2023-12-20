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
