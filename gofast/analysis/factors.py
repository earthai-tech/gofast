# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Factor Analysis (FA)
====================

Here, we give an overview of implementations with vizualization of some factor 
analyses. For some remaining factor analysis methods, including Maximum 
Likelihood Factor Analysis, Direct Oblimin Rotation, Varimax Rotation, and 
others, it's recommended to refer to specialized texts in statistics or use 
established statistical software like R, Python's statsmodels, or scikit-learn 
(for methods it supports).

Implementing these methods requires deep knowledge in statistics and numerical 
methods, and often these implementations are already optimized in existing 
software libraries. For learning and research purposes, you might explore 
academic textbooks or online courses that cover advanced statistical methods 
and factor analysis in detail.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.stats as statsf

from sklearn.decomposition import PCA, FactorAnalysis 
from sklearn.covariance import LedoitWolf 
from sklearn.model_selection import cross_val_score

from ..api.docstring import _core_docs
from ..api.types import ArrayLike 
from ..tools.validator import check_array 
from ..tools.coreutils import _assert_all_types 


__all__=[ 
    "ledoit_wolf_score",  
    "compare_pca_fa_scores", 
    "make_scedastic_data", 
    "rotated_factor", 
    "principal_axis_factoring", 
    "varimax_rotation", 
    "oblimin_rotation", 
    "get_pca_fa_scores", 
    "samples_hotellings_t_square", 
    "promax_rotation", 
    "spectral_fa", 
    ]

def spectral_fa(
    ar2d, /,  
    num_factors=None
    ):
    """
    Perform Spectral Method in Factor Analysis on the given data.
    
    the spectral_factor_analysis function performs the Spectral Method in 
    Factor Analysis on the input data, allowing you to obtain factor 
    loadings, common factors, and eigenvalues. 

    Parameters
    ----------
    ar2d : ndarray
        The input data matrix with shape (n_samples, n_features), where 
        n_samples is the number of observations and n_features is the 
        number of variables.
    num_factors : int, optional
        The number of factors to extract. If None, factors are not limited.

    Returns
    -------
    loadings : ndarray
        The factor loadings matrix with shape (n_features, num_factors).
    factors : ndarray
        The common factors matrix with shape (n_samples, num_factors).
    eigenvalues : ndarray
        The eigenvalues associated with the extracted factors.

    Notes
    -----
    The Spectral Method in Factor Analysis is based on the eigenvalue decomposition
    of the covariance matrix of the observed data.

    The mathematical formula for the Spectral Method is as follows:
    Let C be the covariance matrix of the observed data:
    
    .. math::
        
       C = 1/n_samples * X^T X

    Perform eigenvalue decomposition of C:
        
    .. math:: 
        
       C = V D V^T

    Where:
    - :math:`V` is the matrix of eigenvectors (columns represent eigenvectors).
    - :math:`D` is the diagonal matrix of eigenvalues.

    The factor loadings (L) can be obtained from the eigenvectors:
    
    .. math:: 
        
       L = X^T V D^{-1/2}

    The common factors (F) can be obtained as:
        
    .. math:: 
        
       F = X L

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1.2, 2.3, 3.1],
    ...                  [1.8, 3.5, 4.2],
    ...                  [2.6, 4.0, 5.3],
    ...                  [0.9, 1.7, 2.0]])
    >>> loadings, factors, eigenvalues = spectral_fa(data, num_factors=2)
    >>> print("Factor Loadings Matrix:")
    >>> print(loadings)
    >>> print("Common Factors Matrix:")
    >>> print(factors)
    >>> print("Eigenvalues:")
    >>> print(eigenvalues)
    """
    data = check_array(ar2d )
    # Calculate the covariance matrix of the data
    cov_matrix = np.cov(data, rowvar=False)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Extract factor loadings
    if num_factors is None:
        num_factors = data.shape[1]
    loadings = np.dot(data.T, eigenvectors[:, :num_factors]\
                      / np.sqrt(eigenvalues[:num_factors]))

    # Extract common factors
    factors = np.dot(data, loadings)

    return loadings, factors, eigenvalues


def promax_rotation(
    ar2d, /, 
    power: int=4
    ):
    """
    Perform Promax Rotation on factor loadings.

    The promax_rotation function performs Promax Rotation on the input 
    factor loadings matrix, allowing you to obtain the rotated factor loadings
    Parameters
    ----------
    ar2d : ndarray
        The factor loadings matrix with shape (n_features, num_factors).
    power : int, optional
        The power parameter for Promax rotation. Default is 4.

    Returns
    -------
    rotated_loadings : ndarray
        The rotated factor loadings matrix after Promax rotation.

    Notes
    -----
    Promax Rotation simplifies the interpretation of factors in 
    exploratory factor analysis.
    It transforms the original factor loadings to new loadings to 
    enhance interpretability.

    The Promax Rotation formula is given by:
    \[
    R = (L L^T)^{1/2} \cdot (L L^T)^{1/2 - 1} \cdot (L L^T)^{1/2} \cdot P
    \]
    where:
    - \(R\) is the rotated loadings matrix.
    - \(L\) is the original factor loadings matrix.
    - \(P\) is a transformation matrix derived from the power parameter.

    Examples
    --------
    >>> import numpy as np
    >>> loadings = np.array([[0.7, 0.1, 0.3],
    ...                      [0.8, 0.2, 0.4],
    ...                      [0.4, 0.6, 0.5],
    ...                      [0.3, 0.7, 0.2]])
    >>> rotated_loadings = promax_rotation(loadings, power=4)
    >>> print("Rotated Factor Loadings Matrix:")
    >>> print(rotated_loadings)
    """
    power = int (_assert_all_types(power, int, float,
                                   obj='power for a  promax rotation'))
    loadings = check_array(ar2d )
    # Perform Promax rotation using the specified power parameter
    n_features, num_factors = loadings.shape
    L = loadings
    LLT = np.dot(L, L.T)
    LLT_sqrt = np.linalg.matrix_power(LLT, 0.5)
    LLT_sqrt_inv = np.linalg.matrix_power(LLT, 0.5 - 1)
    P = np.diag(np.arange(1, num_factors + 1) ** power)
    rotated_loadings = np.dot(np.dot(np.dot(LLT_sqrt, LLT_sqrt_inv), LLT_sqrt), P)

    return rotated_loadings

def ledoit_wolf_score(
    X, 
    store_precision=True, 
    assume_centered=False,  
    **kws
    ):
    r"""Models score from Ledoit-Wolf.
    
    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.
        
    Notes
    -----
    The regularised covariance is:
        
    .. math::
        
        (1 - text{shrinkage}) * \text{cov} + \text{shrinkage} * \mu * \text{np.identity(n_features)}
    
    where :math:`\mu = \text{trace(cov)} / n_{features}`
    and shrinkage is given by the Ledoit and Wolf formula
    
    See also
    ----------
        LedoitWolf
        
    References
    ----------
    "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices",
    Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
    February 2004, pages 365-411.
    
    """
    return np.mean(cross_val_score(LedoitWolf(**kws), X))

def compare_pca_fa_scores (
    X,
    rank =10 , 
    sigma =1. , 
    n_components =5, 
    random_state = 42 , 
    verbose =0 , 
    view=True, 
   ):
    #------------------------------------------------------
    from ..models.utils import shrink_covariance_cv_score 
    # -----------------------------------------------------
    # options for n_components
    n_samples, n_features = len(X),  X.shape[1]
    n_components = np.arange(0, n_features, n_components) 
    
    rng = np.random.RandomState(random_state)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    
    #X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    X = np.dot(rng.randn(n_samples, n_features), U[:, :rank].T)
    # Adding homoscedastic noise
    X_homo = X + sigma * rng.randn(n_samples, n_features)

    # Adding heteroscedastic noise
    sigmas = sigma * rng.rand(n_features) + sigma / 2.
    X_hetero = X + rng.randn(n_samples, n_features) * sigmas


    for X, title in [(X_homo, 'Homoscedastic Noise'),
                     (X_hetero, 'Heteroscedastic Noise')]:
        pca_scores, fa_scores = get_pca_fa_scores(X, n_features)
        n_components_pca = n_components[np.argmax(pca_scores)]
        n_components_fa = n_components[np.argmax(fa_scores)]
    
        pca = PCA(svd_solver='full', n_components='mle')
        pca.fit(X)
        n_components_pca_mle = pca.n_components_
        
        if verbose:
            print("best n_components by PCA CV = %d" % n_components_pca)
            print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
            print("best n_components by PCA MLE = %d" % n_components_pca_mle)
    
        if view: 
            plt.figure()
            plt.plot(n_components, pca_scores, 'b', label='PCA scores')
            plt.plot(n_components, fa_scores, 'r', label='FA scores')
            plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
            plt.axvline(n_components_pca, color='b',
                        label='PCA CV: %d' % n_components_pca, linestyle='--')
            plt.axvline(n_components_fa, color='r',
                        label='FactorAnalysis CV: %d' % n_components_fa,
                        linestyle='--')
            plt.axvline(n_components_pca_mle, color='k',
                        label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')
        
            # compare with other covariance estimators
            plt.axhline(shrink_covariance_cv_score(X), color='violet',
                        label='Shrunk Covariance MLE', linestyle='-.')
            plt.axhline(ledoit_wolf_score(X), color='orange',
                        label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')
        
            plt.xlabel('nb of components')
            plt.ylabel('CV scores')
            plt.legend(loc='lower right')
            plt.title(title)
    
    if view: plt.show()
    
    return pca_scores, fa_scores
    
compare_pca_fa_scores.__doc__="""\
Compute PCA score and Factor Analysis scores from training X and compare  
probabilistic PCA and Factor Analysis  models.
  
Parameters 
-----------
{params.X}

n_features: int, 
    number of features that composes X 
n_components: int, default {{5}}
    number of component to retrieve. 
rank: int, default{{10}}
    Bounding for ranking 
sigma: float, default {{1.}}
    data pertubator ratio for adding heteroscedastic noise
random_state: int , default {{42}}
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
    
{params.verbose}

Returns 
---------
Tuple (pca_scores, fa_scores): 
    Scores from PCA and FA  from transformed X 
""".format(
    params =_core_docs["params"]
)    
    
def make_scedastic_data (
    n_samples= 1000, 
    n_features=50, 
    rank =  10, 
    sigma=1., 
    random_state =42
   ): 
    """ Generate a scedatic data data for probabilistic PCA and Factor Analysis for  
    model comparison. 
    
    
    Probabilistic PCA and Factor Analysis are probabilistic models. The consequence 
    is that the likelihood of new data can be used for model selection and 
    covariance estimation. Here we compare PCA and FA with cross-validation on 
    low rank data corrupted with homoscedastic noise 
    (noise variance is the same for each feature) or heteroscedastic noise 
    (noise variance is the different for each feature). In a second step we compare 
    the model likelihood to the likelihoods obtained from shrinkage covariance 
    estimators.
    
    One can observe that with homoscedastic noise both FA and PCA succeed in 
    recovering the size of the low rank subspace. The likelihood with PCA is 
    higher than FA in this case. However PCA fails and overestimates the rank 
    when heteroscedastic noise is present. Under appropriate circumstances the 
    low rank models are more likely than shrinkage models.
    
    The automatic estimation from Automatic Choice of Dimensionality for PCA. 
    NIPS 2000: 598-604 by Thomas P. Minka is also compared.
    
    # Authors: Alexandre Gramfort & Denis A. Engemann
    # License: BSD 3 clause
    
    edited by LKouadio on Tue Oct 11 16:54:26 2022
    
    By default: 
        nsamples    = 1000 
        n_features  = 50  
        rank        =10 
        
    Returns 
    ----------
    * X: sampling data 
    * X_homo: sampling data with homoscedastic noise
    * X_hetero: sampling with heteroscedastic noise
    * n_components: number of components  50 features. 
    
    """
    # Create the data
    rng = np.random.RandomState(random_state )
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    
    # Adding homoscedastic noise
    X_homo = X + sigma * rng.randn(n_samples, n_features)
    
    # Adding heteroscedastic noise
    sigmas = sigma * rng.rand(n_features) + sigma / 2.
    X_hetero = X + rng.randn(n_samples, n_features) * sigmas

    # Fit the models
    n_components = np.arange(0, n_features, 5)  # options for n_components
    
    return X, X_homo, X_hetero , n_components
    
def rotated_factor(
    X, 
    n_components=2, 
    rotation='varimax'
    ):
    """
    Perform a simple rotated factor analysis on the dataset.

    Parameters
    ----------
    X : array_like
        Input data where rows represent samples and columns represent features.
    n_components : int, optional
        The number of factors to extract.
    rotation : str, optional
        The type of rotation to apply. Options include 'varimax', 'promax', etc.

    Returns
    -------
    rotated_components : array
        The rotated factor components.
    """
    X = check_array(X)
    n_components = int (_assert_all_types(n_components, int, float,
                                          objname="n_components"))
    # Perform initial factor analysis (e.g., using PCA)
    # This is a placeholder for actual factor analysis
    initial_components = np.linalg.svd(X)[0][:, :n_components]

    # Apply rotation - this is a simplified placeholder
    if rotation == 'varimax':
        # Implement a basic varimax rotation
        pass  # Placeholder for actual varimax rotation algorithm

    return initial_components  # This would be the rotated components


def principal_axis_factoring(X, n_factors=2):
    r"""
    Perform a basic principal axis factoring.

    Principal Axis Factoring (PAF) is a method of factor analysis that aims 
    to estimate factors 
    based on the correlations between variables. Mathematically, 
    PAF can be represented as:

    PAF Objective:
    \[
    \text{Maximize: } \sum_i \left( \frac{1}{n} \sum_j r_{ij}^2 \right) - \left( \frac{1}{n^2} \sum_i \sum_j r_{ij}^2 \right)
    \]

    where \(r_{ij}\) is the correlation between the \(i\)th and \(j\)th 
    variables, and \(n\) is the number of variables.
    
    Parameters
    ----------
    X : array_like
        Input data where rows represent samples and columns represent features.
    n_factors : int, optional
        The number of factors to extract.

    Returns
    -------
    factors : array
        The extracted factors.
    """
    X = check_array(X) 
    n_factors = int (_assert_all_types(n_factors, int, float,
                                         objname="n_factors"))
    # Compute the covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Select the top n_factors eigenvectors
    factors = eigenvectors[:, :n_factors]

    return factors


def varimax_rotation(
    ar2d, /,  
    gamma=1.0, 
    q=20, 
    tol=1e-6
    ):
    r"""
    Perform Varimax (orthogonal) rotation on the factor loading matrix.

    The Varimax rotation seeks to maximize the variance of the squared loadings of a factor 
    (column) across all the variables (rows). Mathematically, it maximizes the sum of the 
    variances of the squared loadings, which can be represented as:

    Varimax Rotation Objective:
    \[
    \text{Maximize: } \sum_i \left( \frac{1}{n} \sum_j a_{ij}^2 \right)^2 - \left( \frac{1}{n^2} \sum_i \sum_j a_{ij}^2 \right)^2
    \]

    where \(a_{ij}\) is the loading of the \(j\)th variable on the \(i\)th factor, and \(n\) is the number of variables.
    
    Parameters
    ----------
    ar2d : array_like
        The factor loading matrix obtained from factor analysis. 
        Rows represent variables and columns represent factors.
    gamma : float, optional
        The normalization parameter for the rotation. Default is 1.0 for Varimax.
    q : int, optional
        The maximum number of iterations. Default is 20.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Returns
    -------
    rotated_matrix : array_like
        The rotated factor loading matrix.
        
    Examples 
    ---------
    >>> import numpy as np 
    >>> # Simulated factor loading matrix (for illustration purposes)
    >>> # In practice, this would come from your factor analysis
    >>> factor_loading_matrix = np.array([
        [0.7, 0.1],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.3, 0.7]
    ])
    >>> rotated_matrix = varimax_rotation(factor_loading_matrix)
    >>> # Display the rotated factor loading matrix
    >>> print("Rotated Factor Loading Matrix:")
    >>> print(rotated_matrix)
    """
    loading_matrix = check_array(ar2d)
    n_rows, n_cols = loading_matrix.shape
    rotation_matrix = np.eye(n_cols)
    var = 0

    for _ in range(q):
        factor_loading_rotated = np.dot(loading_matrix, rotation_matrix)
        tmp = np.diag((factor_loading_rotated ** 2).sum(axis=0)) / n_rows
        u, s, v = np.linalg.svd(np.dot(loading_matrix.T, np.dot(
            np.diag((factor_loading_rotated ** 3).sum(axis=0)),
            factor_loading_rotated)) / n_rows - np.dot(tmp, tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + tol):
            break
        var = var_new

    return np.dot(loading_matrix, rotation_matrix)


def oblimin_rotation(
    ar2d, /, 
    gamma=0.0, 
    max_iter=100, 
    tol=1e-6
    ):
    """
    Perform Oblimin (oblique) rotation on the factor loading matrix.

    Oblimin Rotation allows factors to be correlated. It seeks to 
    find rotated factor loadings that satisfy the following objective:

    Oblimin Rotation Objective:
    \[
    \sum_i \left( \sum_j (a_{ij}^2 - \gamma a_{ij}^4) \right)^2 \to \text{Maximize}
    \]

    where:
    - \(a_{ij}\) is the loading of the j-th variable on the i-th factor.
    - \(\gamma\) is a rotation parameter that controls the degree of correlation.

    Parameters
    ----------
    factor_loading_matrix : array_like
        The factor loading matrix obtained from factor analysis. 
        Rows represent variables, and columns represent factors.
    gamma : float, optional
        The rotation parameter controlling the degree of correlation 
        between factors.
        Default is 0.0 (orthogonal rotation).
    max_iter : int, optional
        The maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.

    Returns
    -------
    rotated_matrix : array_like
        The rotated factor loading matrix.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Simulated factor loading matrix (for illustration purposes)
    >>> # In practice, this would come from your factor analysis
    >>> factor_loading_matrix = np.array([
        [0.7, 0.1, 0.3],
        [0.8, 0.2, 0.4],
        [0.4, 0.6, 0.2],
        [0.3, 0.7, 0.5]
    ])
    >>> # Apply Oblimin Rotation with a specified gamma value (degree of correlation)
    >>> gamma_value = 0.2
    >>> rotated_matrix = oblimin_rotation(factor_loading_matrix, gamma=gamma_value)
    
    >>> # Display the rotated factor loading matrix
    >>> print("Original Factor Loading Matrix:")
    >>> print(factor_loading_matrix)
    >>> print("\nRotated Factor Loading Matrix (Oblimin Rotation):")
    >>>  print(rotated_matrix)

    """
    factor_loading_matrix = check_array(ar2d)
    n_vars, n_factors = factor_loading_matrix.shape
    rotated_matrix = factor_loading_matrix.copy()
    converged = False

    for _ in range(max_iter):
        prev_matrix = rotated_matrix.copy()

        for i in range(n_factors):
            for j in range(n_factors):
                if i != j:
                    cov_ij = np.sum(rotated_matrix[:, i] * rotated_matrix[:, j])
                    cov_ii = np.sum(rotated_matrix[:, i] ** 2)
                    cov_jj = np.sum(rotated_matrix[:, j] ** 2)

                    delta = (cov_ij - gamma * cov_ii * cov_jj) / cov_ii

                    rotated_matrix[:, i] = rotated_matrix[:, i] - delta * rotated_matrix[:, j]

        # Check for convergence
        if np.allclose(prev_matrix, rotated_matrix, atol=tol):
            converged = True
            break

    if not converged:
        print("Warning: Oblimin rotation did not converge.")

    return rotated_matrix


def get_pca_fa_scores(X, n_features , n_components = 5):
    n_components = np.arange(0, n_features, n_components)
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()
    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores

get_pca_fa_scores.__doc__ ="""\
Compute PCA score and Factor Analysis scores from training X. 
  
Parameters 
-----------
{params.X}

n_features: int, 
    number of features that composes X 
n_components: int, default {{5}}
    number of component to retrieve. 
Returns 
---------
Tuple (pca_scores, fa_scores): 
    Scores from PCA and FA  from transformed X 
""".format(params =_core_docs["params"])


def samples_hotellings_t_square(
    sample1: ArrayLike, 
    sample2: ArrayLike 
    ):
    r"""
    Perform Two-Sample Hotelling's T-Square test to compare two 
    multivariate samples.

    Hotelling's T-Square test assesses whether two multivariate samples come 
    from populations with equal means. It is a multivariate extension of 
    the two-sample t-test.

    Hotelling's T-Square Statistic:
    \[
    T^2 = n_1 \cdot n_2 \cdot (\mathbf{\bar{x}}_1 - \mathbf{\bar{x}}_2)^T \cdot \mathbf{S}_w^{-1} \cdot (\mathbf{\bar{x}}_1 - \mathbf{\bar{x}}_2)
    \]
    
    Degrees of Freedom (Numerator):
    \[
    df_1 = p
    \]
    
    Degrees of Freedom (Denominator):
    \[
    df_2 = n_1 + n_2 - p - 1
    \]
    
    P-Value:
    \[
    p = 1 - F(T^2 \cdot \frac{df_2}{df_1 \cdot df_2 - p + 1}, df_1, df_2)
    \]
    
    Where:
    - \(T^2\) is the Hotelling's T-Square statistic.
    - \(n_1\) and \(n_2\) are the sample sizes of the two groups.
    - \(\mathbf{\bar{x}}_1\) and \(\mathbf{\bar{x}}_2\) are the sample means of the two groups.
    - \(p\) is the number of variables (dimensions).
    - \(\mathbf{S}_w\) is the pooled within-group covariance matrix.
    - \(df_1\) is the degrees of freedom (numerator).
    - \(df_2\) is the degrees of freedom (denominator).
    - \(F\) is the F-distribution cumulative distribution function.

    Parameters
    ----------
    sample1 : array_like
        The first multivariate sample with shape (n1, p), where n1 is the number of samples
        and p is the number of variables.
    sample2 : array_like
        The second multivariate sample with shape (n2, p), where n2 is the number of samples
        and p is the number of variables.

    Returns
    -------
    statistic : float
        The Hotelling's T-Square statistic.
    df1 : int
        Degrees of freedom numerator.
    df2 : int
        Degrees of freedom denominator.
    p_value : float
        The p-value of the test.

    Examples
    --------
    >>> import numpy as np
    >>> sample1 = np.array([[1.2, 2.3], [1.8, 3.5], [2.6, 4.0]])
    >>> sample2 = np.array([[0.9, 2.0], [1.5, 3.2], [2.3, 3.8]])
    >>> statistic, df1, df2, p_value = samples_hotellings_t_square(sample1, sample2)
    >>> print(f"Hotelling's T-Square statistic: {statistic:.4f}")
    >>> print(f"Degrees of freedom (numerator): {df1}")
    >>> print(f"Degrees of freedom (denominator): {df2}")
    >>> print(f"P-value: {p_value:.4f}")
    """
    n1, p = sample1.shape
    n2 = sample2.shape[0]

    # Calculate sample means
    mean1 = np.mean(sample1, axis=0)
    mean2 = np.mean(sample2, axis=0)

    # Calculate sample covariances
    cov1 = np.cov(sample1, rowvar=False, ddof=1)
    cov2 = np.cov(sample2, rowvar=False, ddof=1)

    # Pooled within-group covariance matrix
    pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

    # Calculate Hotelling's T-Square statistic
    t_square = n1 * n2 / (n1 + n2) * np.dot(
        np.dot(mean1 - mean2, np.linalg.inv(pooled_cov)), mean1 - mean2)

    # Degrees of freedom
    df1 = p
    df2 = n1 + n2 - p - 1

    # Calculate p-value
    p_value = 1 - statsf.cdf(t_square * df2 / (df1 * df2 - p + 1), df1, df2)

    return t_square, df1, df2, p_value

