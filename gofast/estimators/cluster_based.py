# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
from numbers import Integral, Real

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.base import is_regressor, is_classifier 
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 

from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted
from ..tools._param_validation import Hidden
from ..tools._param_validation import Interval
from ..tools._param_validation import StrOptions
from ..tools._param_validation import validate_params
from ..transformers import KMeansFeaturizer
from .util import select_default_estimator
    
__all__=["KMFClassifier", "KMFRegressor"]

class KMFClassifier(BaseEstimator, ClassifierMixin):
    r"""
    A classifier that integrates k-means clustering with a base machine 
    learning estimator.

    The KMFClassifier first employs the KMeansFeaturizer algorithm to transform 
    the input data into cluster memberships based on k-means clustering [1]_. 
    Each data point is represented by its closest cluster center. This transformed 
    data, capturing the inherent clustering structure, is then used to train a 
    specified base estimator [2]_. The approach aims to enhance the performance 
    of the base estimator by leveraging the additional structure introduced by 
    the clustering process.

    The mathematical formulation of the k-means clustering involves minimizing 
    the inertia, or within-cluster sum-of-squares criterion.

    The inertia is defined as:

    .. math::

        \text{Inertia} = \sum_{i=1}^{n} (x_i - \mu_k)^2

    where :math:`x_i` represents a data point, :math:`\mu_k` is the centroid of the 
    cluster k, and the sum is taken over all the data points in the cluster.
    
    Mathematically, the KMF algorithm with no hint (target variable not included) 
    follows the standard k-means objective of minimizing the intra-cluster variance. 
    However, when the target variable y is included, the feature space is augmented 
    by stacking y with X. The KMF algorithm then minimizes a modified objective 
    function [3]_:

    .. math::

        \min_{C_1,\cdots,C_k, \mu'_1,\cdots,\mu'_k} \sum_{i=1}^{k}\\
            \left( \sum_{x y \in C_i} \|x y - \mu'_i \|_2^2 \right)

    where:
        
    - :math:`x y` denotes the augmented data point, a combination of the 
      feature vector x and the target y. This combination guides the 
      clustering process by considering both feature characteristics 
      and target variable information.
    - :math:`\mu'_i` is the centroid of the augmented data points in the 
      cluster :math:`C_i`, including target variable information, aligning
      the clusters with the underlying class structure.
    - The first summation :math:`\sum_{i=1}^{k}` sums over all k-clusters, 
      focusing on minimizing the variance within each cluster.
    - The inner summation :math:`\sum_{x y \in C_i}` represents the sum over 
      all augmented data points :math:`x y` in the cluster :math:`C_i`.
    - :math:`\|x y - \mu'_i \|_2^2` is the squared Euclidean distance between
      each augmented data point :math:`x y` and its cluster centroid 
      :math:`\mu'_i`. Minimizing this distance ensures points are as close 
        as possible to their cluster center.

    Parameters
    ----------
    base_estimator : estimator object
        The base machine learning estimator to fit on the transformed data.
        This estimator should follow the scikit-learn estimator interface. 
        If no estimator is passed, the default is ``DecisionTreeClassifier``

    n_clusters : int, default=7
        The number of clusters to form in the k-means clustering process.

    target_scale : float, default=5.0
        Scaling factor for the target variable when included in the k-means 
        clustering process. This is used to adjust the influence of the target 
        variable in the clustering process.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the centroid initialization and data shuffling,
        ensuring reproducibility of the results.

    n_components : int, optional
        Number of principal components to retain for dimensionality reduction 
        before applying k-means clustering. If not set, no dimensionality 
        reduction is performed.

    init : str, callable or array-like, default='k-means++'
        Method for initialization of the k-means centroids.

    n_init : int or 'auto', default='auto'
        Number of times the k-means algorithm will be run with different centroid 
        seeds. The best output in terms of inertia is chosen.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4
        Relative tolerance with regards to the Frobenius norm of the difference 
        in the cluster centers to declare convergence.

    copy_x : bool, default=True
        If True, the input data is centered before computation. Centring the data 
        is known to improve the numerical stability of the k-means algorithm.

    verbose : int, default=0
        Verbosity mode for logging the process of the k-means algorithm.

    algorithm : str, default='lloyd'
        K-means algorithm variant to use. The options include 'lloyd' for the 
        standard k-means algorithm and 'elkan' for the Elkan variant.
        
    to_sparse : bool, default=False
            If True, the input data `X` will be converted to a sparse matrix
            before applying the transformation. This is useful for handling
            large datasets more efficiently. If False, the data format of `X`
            is preserved.
            
    Attributes
    ----------
    featurizer_ : KMeansFeaturizer object
        The KMeansFeaturizer instance used to transform the input data.

    base_estimator_ : estimator object
        The fitted instance of the base estimator used for final predictions.

    Notes
    -----
    - The effectiveness of the KMFClassifier depends on the choice of the base
      estimator and the suitability of k-means clustering for the given dataset.
    - The KMeansFeaturizer can optionally scale the target variable and perform 
      PCA for dimensionality reduction, which can significantly affect the 
      clustering  results and hence the final prediction accuracy.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split 
    >>> from gofast.estimators.cluster_based import KMFClassifier
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split( X, y)
    >>> kmf_classifier = KMFClassifier(base_estimator=RandomForestClassifier(), n_clusters=5)
    >>> kmf_classifier.fit(X_train, y_train)
    >>> y_pred = kmf_classifier.predict(X_test)

    References
    ----------
    .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
          K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
          Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
          
    .. [2] MacQueen, J. (1967). Some methods for classification and analysis of multivariate 
           observations. Proceedings of the 5th Berkeley Symposium on Mathematical Statistics 
           and Probability. 1:281-297.
           
    .. [3] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of 
           Machine Learning Research. 12:2825-2830.
    
    """
    def __init__(
        self,
        base_estimator=None, 
        n_clusters=3, 
        target_scale=1.0, 
        random_state=None, 
        n_components=None, 
        init='k-means++', 
        n_init="auto", 
        max_iter=300, 
        tol=1e-4, copy_x=True, 
        verbose=0, 
        algorithm='lloyd', 
        to_sparse=False, 
        ):
        self.base_estimator = base_estimator 
        self.n_clusters = n_clusters
        self.target_scale = target_scale
        self.random_state = random_state
        self.n_components = n_components
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.copy_x = copy_x
        self.verbose = verbose
        self.algorithm = algorithm
        self.to_sparse=to_sparse

    def fit(self, X, y):
        """
        Fit the KMeansFeaturizer and the base estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples 
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, estimator =self, )
        if self.base_estimator is None: 
            self.base_estimator =  DecisionTreeClassifier()
        
        self.base_estimator = select_default_estimator(
            self.base_estimator, 'classification')
        
        is_classifier(self.base_estimator )
        self.featurizer_ = KMeansFeaturizer(
            n_clusters=self.n_clusters,
            target_scale=self.target_scale,
            random_state=self.random_state,
            n_components=self.n_components,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            copy_x=self.copy_x,
            verbose=self.verbose,
            algorithm=self.algorithm, 
            to_sparse=self.to_sparse, 
        )
        X_transformed = self.featurizer_.fit_transform(X, y)
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X_transformed, y)
        
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        X_transformed = self.featurizer_.transform(X)
        return self.base_estimator_.predict(X_transformed)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_prob : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        NotImplementedError
            If the base estimator does not support probability predictions.
        """
        if hasattr(self.base_estimator_, "predict_proba"):
            X_transformed = self.featurizer_.transform(X)
            return self.base_estimator_.predict_proba(X_transformed)
        else:
            raise NotImplementedError("Base estimator does not support predict_proba")
            
    def decision_function(self, X):
        """
        Compute the decision function of the samples.
    
        The decision function represents the confidence scores for samples. It is the 
        distance of each sample to the decision boundary of each class. For binary 
        classifiers, the decision function corresponds to the raw output of the model, 
        which is then transformed into a probability score.
    
        This method requires that the base classifier used in KMFClassifier has a 
        decision_function method. If the base classifier does not support 
        decision_function, an error is raised.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to compute the decision function.
    
        Returns
        -------
        decision : array-like of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class for each sample. The shape 
            depends on the number of classes and the nature of the classifier.
    
        Raises
        ------
        AttributeError
            If the base classifier does not implement a decision function.
    
        NotFittedError
            If the method is called before fitting the model with the `fit` method.
    
        Notes
        -----
        - In a multi-class setting, the decision function returns the confidence scores 
          per class. The class with the highest score is taken as the predicted class.
        - The interpretation of the decision values depends on the classifier and should 
          be used with caution, especially when comparing classifiers.
    
        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> X, y = ...  # load dataset
        >>> kmf_classifier = KMFClassifier(base_classifier=SVC(), n_clusters=5)
        >>> kmf_classifier.fit(X_train, y_train)
        >>> decision_values = kmf_classifier.decision_function(X_test)
        """
    
        # Check if the model is fitted
        check_is_fitted(self, ['featurizer_', 'base_classifier_'])
    
        # Check if the base classifier has the decision_function method
        if not hasattr(self.base_classifier_, "decision_function"):
            raise AttributeError("The base classifier does not implement a decision function.")
    
        # Transform the data using the trained KMeansFeaturizer
        X_transformed = self.featurizer_.transform(X)
    
        # Compute and return the decision function from the base classifier
        return self.base_classifier_.decision_function(X_transformed)
   
class KMFRegressor(BaseEstimator, RegressorMixin):
    r"""
    A K-Means Featurizer Regressor that combines k-means clustering with a 
    base regression estimator.

    This regressor first transforms the data into k-means cluster memberships 
    using the KMeansFeaturizer algorithm adapted to the regression task, then 
    applies a base regressor to the transformed data [1]_.

    The mathematical formulation of the k-means clustering involves minimizing 
    the inertia, or within-cluster sum-of-squares criterion.

    The inertia is defined as:

    .. math::

        \text{Inertia} = \sum_{i=1}^{n} (x_i - \mu_k)^2

    where :math:`x_i` represents a data point, :math:`\mu_k` is the centroid of the 
    cluster k, and the sum is taken over all the data points in the cluster.
    
    Mathematically, the KMF algorithm with no hint (target variable not included) 
    follows the standard k-means objective of minimizing the intra-cluster variance. 
    However, when the target variable y is included, the feature space is augmented 
    by stacking y with X. The KMF algorithm then minimizes a modified objective 
    function:

    .. math::

        \min_{C_1,\cdots,C_k, \mu'_1,\cdots,\mu'_k} \sum_{i=1}^{k}\\
            \left( \sum_{x y \in C_i} \|x y - \mu'_i \|_2^2 \right)

    where:
        
    - :math:`x y` denotes the augmented data point, a combination of the 
      feature vector x and the target y. This combination guides the 
      clustering process by considering both feature characteristics and 
      target variable information.
    - :math:`\mu'_i` is the centroid of the augmented data points in the 
      cluster :math:`C_i`, including target variable information, aligning 
      the clusters with the underlying class structure.
    - The first summation :math:`\sum_{i=1}^{k}` sums over all k-clusters, 
      focusing on minimizing the variance within each cluster.
    - The inner summation :math:`\sum_{x y \in C_i}` represents the sum over 
      all augmented data points :math:`x y` in the cluster :math:`C_i`.
    - :math:`\|x y - \mu'_i \|_2^2` is the squared Euclidean distance between 
      each augmented data point :math:`x y` and its cluster centroid 
      :math:`\mu'_i`. Minimizing this distance ensures points are as close as 
      possible to their cluster center.

    Parameters
    ----------
    base_regressor : estimator object
        The base machine learning regressor estimator to fit on the transformed 
        data. This estimator should follow the scikit-learn estimator interface.

    n_clusters : int, default=7
        The number of clusters to form in the k-means clustering process.

    target_scale : float, default=5.0
        Scaling factor for the target variable when included in the k-means 
        clustering process. This is used to adjust the influence of the target 
        variable in the clustering process.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the centroid initialization and data shuffling,
        ensuring reproducibility of the results.

    n_components : int, optional
        Number of principal components to retain for dimensionality reduction 
        before applying k-means clustering. If not set, no dimensionality 
        reduction is performed.

    init : str, callable or array-like, default='k-means++'
        Method for initialization of the k-means centroids.

    n_init : int or 'auto', default='auto'
        Number of times the k-means algorithm will be run with different centroid 
        seeds. The best output in terms of inertia is chosen.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4
        Relative tolerance with regards to the Frobenius norm of the difference 
        in the cluster centers to declare convergence.

    copy_x : bool, default=True
        If True, the input data is centered before computation. Centring the data 
        is known to improve the numerical stability of the k-means algorithm.

    verbose : int, default=0
        Verbosity mode for logging the process of the k-means algorithm.

    algorithm : str, default='lloyd'
        K-means algorithm variant to use. The options include 'lloyd' for the 
        standard k-means algorithm and 'elkan' for the Elkan variant.
        
    to_sparse : bool, default=False
            If True, the input data `X` will be converted to a sparse matrix
            before applying the transformation. This is useful for handling
            large datasets more efficiently. If False, the data format of `X`
            is preserved.
            
    Attributes
    ----------
    featurizer_ : KMeansFeaturizer object
        The featurizer used to transform the data.

    base_regressor_ : estimator object
        The base regressor used for prediction.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from gofast.estimators.cluster_based import KMFRegressor
    >>> from sklearn.datasets import load_iris 
    >>> from sklearn.model_selection import train_test_split 
    >>> X, y = load_iris (return_X_y=True) 
    >>> X_train, X_test, y_train, y_test = train_test_split( X, y) 
    >>> kmf_regressor = KMFRegressor(base_regressor=LinearRegression(), n_clusters=5)
    >>> kmf_regressor.fit(X_train, y_train)
    >>> y_pred = kmf_regressor.predict(X_test)
    
    References 
    -----------
    .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
          K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
          Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
    """
    def __init__(
        self, 
        base_regressor=None, 
        n_clusters=7, 
        target_scale=1.0, 
        random_state=None, 
        n_components=None, 
        init='k-means++', 
        n_init="auto", 
        max_iter=300, 
        tol=1e-4, 
        copy_x=True, 
        verbose=0, 
        algorithm='lloyd', 
        to_sparse=False
        ):
        self.base_regressor = base_regressor
        self.n_clusters = n_clusters
        self.target_scale = target_scale
        self.random_state = random_state
        self.n_components = n_components
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.copy_x = copy_x
        self.verbose = verbose
        self.algorithm = algorithm
        self.to_sparse=to_sparse

    def fit(self, X, y):
        """
        Fit the KMeansFeaturizer and the base regression estimator on 
        the training data.
    
        This method first validates and checks the input data X and y. Then, 
        it initializes and fits the KMeansFeaturizer to transform the input 
        data into cluster memberships. Finally, it fits the base regression 
        estimator to the transformed data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values (real numbers).
    
        Returns
        -------
        self : object
            Returns the instance itself.
    
        Raises
        ------
        ValueError
            If the base estimator is not a regression estimator.
    
        Notes
        -----
        - The fit method checks if the base estimator is suitable for regression 
          tasks.
        - Input validation is performed on the training data.
        - The KMeansFeaturizer transforms the data by applying k-means clustering, 
          potentially using the target variable to influence cluster formation.
        - The transformed data is then used to fit the base regressor.
        """
        
        if self.base_regressor is None: 
            self.base_regressor = DecisionTreeRegressor()
        
        self.base_estimator = select_default_estimator( self.base_estimator)
        
        # Check if the base estimator is a regressor
        if not is_regressor(self.base_regressor):
            name_estimator = get_estimator_name(self.base_regressor)
            raise ValueError(f"The provided base estimator {name_estimator!r}"
                             " is not a regressor.")
        # Validate the input
        X, y = check_X_y(X, y, estimator=self)
        # Instantiate and fit the KMeansFeaturizer
        self.featurizer_ = KMeansFeaturizer(
            n_clusters=self.n_clusters,
            target_scale=self.target_scale,
            random_state=self.random_state,
            n_components=self.n_components,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            copy_x=self.copy_x,
            verbose=self.verbose,
            algorithm=self.algorithm, 
            to_sparse=self.to_sparse
        )
        X_transformed = self.featurizer_.fit_transform(X, y)
        # Fit the base regressor
        self.base_regressor_ = clone(self.base_regressor)
        self.base_regressor_.fit(X_transformed, y)
        return self

    def predict(self, X):
        """
        Predict regression target for each sample in X.
    
        This method first transforms the input data using the trained 
        KMeansFeaturizer,which involves assigning each sample to the nearest 
        cluster centroid. The transformed data, now enriched with cluster 
        membership information, is then fed into the trained base regressor 
        to make predictions.
    
        The method assumes that the `fit` method has already been called 
        and that the KMeansFeaturizer and the base regressor are both trained.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to make predictions. Here, n_samples is 
            the number of samples and n_features is the number of features.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted target values per element in X.
    
        Raises
        ------
        NotFittedError
            If the method is called before fitting the model with the `fit` method.
    
        Notes
        -----
        - The quality of predictions heavily depends on both the quality of clustering
          achieved by the KMeansFeaturizer and the performance of the base regressor.
        - It is important to ensure that the input features for prediction undergo the
          same preprocessing and feature engineering steps as the training data.
    
        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression
        >>> X, y = ...  # load dataset
        >>> kmf_regressor = KMFRegressor(base_regressor=LinearRegression(), n_clusters=5)
        >>> kmf_regressor.fit(X_train, y_train)
        >>> y_pred = kmf_regressor.predict(X_test)
        """
        X= check_array(X, estimator= self, input_name= 'X')
        # Check if the model is fitted
        check_is_fitted(self, ['featurizer_', 'base_regressor_'])
        # Transform the data using the trained KMeansFeaturizer
        X_transformed = self.featurizer_.transform(X)
        # Predict using the base regressor on the transformed data
        return self.base_regressor_.predict(X_transformed)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
    
        The R^2 score, also known as the coefficient of determination, 
        provides a measure of how well the future samples are likely to be 
        predicted by the model. The best possible score is 1.0, which indicates 
        perfect prediction. This method first makes predictions on the input 
        data X using the trained model and then compares these predictions 
        with the actual values y to compute the R^2 score.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples, where n_samples is the number of samples and 
            n_features is the number of features.
    
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
    
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
    
        Raises
        ------
        NotFittedError
            If the method is called before fitting the model with the `fit` method.
    
        Notes
        -----
        - The R^2 score used here is not a symmetric function and hence can 
          be negative if the model is worse than a simple mean predictor.
        - The score method evaluates the performance of the model on the test 
          dataset. Higher scores indicate a model that better captures the 
          variability of the dataset.
        - As with any metric, the R^2 score should be considered alongside 
          other metrics and domain knowledge to evaluate the model's performance 
          comprehensively.
    
        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression
        >>> X, y = ...  # load dataset
        >>> kmf_regressor = KMFRegressor(base_regressor=LinearRegression(), n_clusters=5)
        >>> kmf_regressor.fit(X_train, y_train)
        >>> r2 = kmf_regressor.score(X_test, y_test)
        """
        # Check if the model is fitted
        check_is_fitted(self, ['featurizer_', 'base_regressor_'])
        # Make predictions and compute R^2 score
        predictions = self.predict(X)
        return r2_score(y, predictions)

@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "sample_weight": ["array-like", None],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Hidden(StrOptions({"warn"})),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left"), bool],
        "tol": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "copy_x": [bool],
        "algorithm": [
            StrOptions({"lloyd", "elkan", "auto", "full"}, deprecated={"auto", "full"})
        ],
        "return_n_iter": [bool],
    }
)
def kmf_featurizer (): 
    pass 
    
     





