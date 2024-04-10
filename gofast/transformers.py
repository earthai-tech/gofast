# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Gives some efficient tools for data manipulation 
and transformation.
"""
from __future__ import division, annotations  

import os
import itertools 
import warnings 
import numpy as np 
import pandas as pd 
from scipy import sparse
import matplotlib.pyplot as plt  

from sklearn.base import BaseEstimator,TransformerMixin 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score,  roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

try : 
    from skimage.filters import sobel, canny
    from skimage.exposure import equalize_hist
except : pass
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except : pass 

from ._gofastlog import gofastlog 
from ._typing import _F 
from .exceptions import EstimatorError, NotFittedError 
from .tools.coreutils import  parse_attrs, assert_ratio, validate_feature
from .tools.coreutils import  ellipsis2false, to_numeric_dtypes, is_iterable
from .tools.coreutils import exist_features
from .tools._dependency import import_optional_dependency 
from .tools.validator import  get_estimator_name, check_X_y, is_frame
from .tools.validator import _is_arraylike_1d, build_data_if, check_array 
from .tools.validator import check_is_fitted

EMSG = (
        "`scikit-image` is needed"
        " for this transformer. Note"
        " `skimage`is the shorthand "
        "of `scikit-image`."
        )

__docformat__='restructuredtext'
_logger = gofastlog().get_gofast_logger(__name__)

__all__= ['SequentialBackwardSelector',
          'FloatCategoricalToIntTransformer', 
          'KMeansFeaturizer',
          'AttributesCombinator', 
          'StratifyFromBaseFeature',
          'CategoryBaseStratifier', 
          'CategorizeFeatures', 
          'FrameUnion', 
          'FrameUnionFlex', 
          'DataFrameSelector',
          'BaseColumnSelector', 
          'BaseCategoricalEncoder', 
          'BaseFeatureScaler', 
          'CombinedAttributesAdder', 
          'FeaturizeX', 
          'TextFeatureExtractor', 
          'DateFeatureExtractor', 
          'FeatureSelectorByModel', 
          'PolynomialFeatureCombiner', 
          'DimensionalityReducer', 
          'CategoricalEncoder', 
          'FeatureScaler', 
          'MissingValueImputer', 
          'ColumnSelector', 
          'LogTransformer', 
          'TimeSeriesFeatureExtractor',
          'CategoryFrequencyEncoder', 
          'DateTimeCyclicalEncoder', 
          'LagFeatureGenerator', 
          'DifferencingTransformer', 
          'MovingAverageTransformer', 
          'CumulativeSumTransformer', 
          'SeasonalDecomposeTransformer', 
          'FourierFeaturesTransformer', 
          'TrendFeatureExtractor', 
          'ImageResizer', 
          'ImageNormalizer', 
          'ImageToGrayscale', 
          'ImageAugmenter', 
          'ImageChannelSelector', 
          'ImageFeatureExtractor', 
          'ImageEdgeDetector', 
          'ImageHistogramEqualizer', 
          'ImagePCAColorAugmenter', 
          'ImageBatchLoader', 
          ]

class FloatCategoricalToIntTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that detects floating-point columns in a DataFrame 
    representing categorical variables and converts them to integers.

    This transformer is useful when dealing with datasets where categorical 
    variables are represented as floating-point numbers but essentially 
    contain integer values, for example, [0.0, 1.0, 2.0] representing different 
    categories.

    Attributes
    ----------
    columns_to_transform_ : list
        List of column names in the DataFrame that are identified as 
        floating-point columns to be transformed to integers.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from gofast.transformers import FloatCategoricalToIntTransformer
    >>> data = {'category': [0.0, 1.0, 2.0, 1.0], 'value': [23.5, 12.6, 15.0, 22.1]}
    >>> df = pd.DataFrame(data)
    >>> transformer = FloatCategoricalToIntTransformer()
    >>> transformer.fit(df)
    >>> transformed = transformer.transform(df)
    >>> print(transformed)
       category  value
    0         0   23.5
    1         1   12.6
    2         2   15.0
    3         1   22.1

    Notes
    -----
    The fit method determines which columns are to be transformed by checking 
    if the unique values in each floating-point column are integers ending with 
    .0. During the transform phase, these columns are then cast to the integer 
    type, preserving their categorical nature but in a more memory-efficient 
    format.

    The transformer does not modify the input DataFrame directly; instead, it 
    returns a transformed copy.
    """
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame): 
            try : # Default construct data
                X=build_data_if(X, input_name='feature_', 
                                raise_exception=True, force=True
                ) 
            except Exception as e :
                raise TypeError(
                    "Expect a DataFrame 'X'. Got {type(X).__name__!r}"
                    ) from e 
        # Identify columns to transform based on their unique values
        self.columns_to_transform_ = []
        for col in X.columns:
            if X[col].dtype == float:
                unique_vals = np.unique(X[col])
                # Check if unique values are integers ending with .0 (e.g., 0.0, 1.0)
                if all(np.mod(unique_vals, 1) == 0):
                    self.columns_to_transform_.append(col)
        return self
    
    def transform(self, X):
        """
        Transform the DataFrame by converting identified floating-point columns
        to integers.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        X_transformed : pandas.DataFrame
            The transformed DataFrame with floating-point columns converted 
            to integers.
        """
        # Copy DataFrame to avoid modifying the original data
        X_transformed = X.copy()
        # Convert identified columns to integer type
        for col in self.columns_to_transform_:
            X_transformed[col] = X_transformed[col].astype(int)
        return X_transformed

class SequentialBackwardSelector(BaseEstimator, TransformerMixin):
    r"""
    Sequential Backward Selection (SBS)
    
    SBS is a feature selection algorithm aimed at reducing the dimensionality
    of the initial feature subspace with minimal performance decay in the
    classifier, thereby enhancing computational efficiency. In certain cases,
    SBS can even improve the predictive power of a model, particularly in
    scenarios of overfitting.
    
    The core concept of SBS is to sequentially remove features from the full
    feature subset until the remaining feature subspace contains the desired
    number of features. To determine which feature to remove at each stage,
    a criterion function :math:`J` is minimized [1]_. Essentially, the criterion
    is the difference in classifier performance before and after removing a
    particular feature. The feature removed at each stage is the one that
    maximizes this criterion, meaning it has the least impact on performance
    when removed. The SBS algorithm can be outlined in the following steps:
    
        - Initialize with :math:`k=d`, where :math:`d` is the dimensionality
          of the full feature space, :math:`X_d`.
        - Identify the feature :math:`x^{-}` that maximizes the criterion:
          :math:`x^{-} = argmax J(X_k - x)`, where :math:`x \in X_k`.
        - Remove the feature :math:`x^{-}` from the set, updating
          :math:`X_{k+1} = X_k - x^{-}; k = k - 1`.
        - Terminate if :math:`k` equals the desired number of features;
          otherwise, repeat from step 2. [2]_
    
    Parameters
    ----------
    estimator : callable or instantiated object
        A callable or an instance of a classifier/regressor with a `fit` method.
    k_features : int, default=1
        The starting number of features for selection. Must be less than the
        total number of features in the training set.
    scoring : callable or str, default='accuracy'
        Metric for scoring. Available metrics are 'precision', 'recall',
        'roc_auc', or 'accuracy'. Other metrics will raise an error.
    test_size : float or int, default=None
        If float, represents the proportion of the dataset for the test split.
        If int, represents the absolute number of test samples. Defaults to the
        complement of the train size if None. If `train_size` is also None, it
        defaults to 0.25.
    random_state : int, RandomState instance, or None, default=None
        Controls the shuffling applied to the data before the split.
        An integer value ensures reproducible results across multiple function calls.
    
    Attributes
    ----------
    feature_names_ : ndarray of shape (n_features_in,)
        This attribute stores the names of the features that the model has been 
        trained on. It is particularly useful for understanding which features 
        were present in the input dataset `X` during the `fit` method call. This 
        attribute is only defined if the input `X` has string type feature names.

    selected_indices_ : tuple
        Contains the indices of the features in the final selected subset after 
        the model fitting process. These indices correspond to the features in 
        `feature_names_` that the algorithm has identified as the most 
        significant or relevant for the model.

    feature_subsets_ : list of tuples
        A list where each tuple represents a subset of features selected at each 
        step of the Sequential Backward Selection process. It tracks the 
        evolution of feature selection over the course of the algorithm's 
        execution, showing the progressive elimination of features.

    model_scores_ : list of floats
        This list contains the scores of the model corresponding to each of the 
        feature subsets stored in `feature_subsets_`. The scores are calculated 
        during the cross-validation process within the algorithm. They provide 
        insight into the performance of the model as different features are 
        removed.

    optimal_score_ : float
        Represents the highest score achieved by the model using the feature 
        subset specified by the `k_features` parameter. It signifies the 
        performance of the model with the optimally selected features, giving 
        an idea of how well the model can perform after the feature selection 
        process.

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.datasets import fetch_data
    >>> from gofast.base import SequentialBackwardSelector
    >>> X, y = fetch_data('bagoue analysed') # data already standardized
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> knn = KNeighborsClassifier(n_neighbors=5)
    >>> sbs = SequentialBackwardSelector(knn)
    >>> sbs.fit(X_train, y_train)
    
    References
    ----------
    .. [1] Raschka, S., Mirjalili, V., Python Machine Learning, 3rd ed., Packt, 2019.
    .. [2] Ferri F., Pudil P., Hatef M., Kittler J., Comparative study of
           techniques for large-scale feature selection, pages 403-413, 1994.
           
    """

    _scorers = {
        'accuracy': accuracy_score,
        'recall': recall_score,
        'precision': precision_score,
        'roc_auc': roc_auc_score
    }

    def __init__(
        self, 
        estimator, 
        k_features=1, 
        scoring='accuracy', 
        test_size=0.25, 
        random_state=42
        ):
        self.estimator = estimator
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the Sequential Backward Selection (SBS) model to the training data.

        This method involves splitting the dataset into training and validation 
        subsets within the fit function itself. The SBS algorithm uses this 
        internal validation set, distinct from any external test set, to 
        evaluate feature importance and make selection decisions. 
        This approach ensures that the original test set remains untouched 
        and is not used inadvertently during the training process.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where each sample is represented as a feature vector. 
            `X` can be a matrix of features, or it might require preprocessing  
            (e.g., using a feature extractor or a pairwise metric) before being 
            used. n_samples is the number of samples, and n_features is the  
            number of features.
        y : array-like of shape (n_samples,)
            Target values corresponding to the training data. These are the 
            dependent variables that the model is trained to predict.

        Returns
        -------
        self : object
            The fitted `SequentialBackwardSelection` instance, allowing for 
            method chaining.
        """
        X, y = check_X_y(X, y, estimator=self, multi_output=True, to_frame=True)
        self._validate_params(X)
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        self.indices_ = tuple(range(X_train.shape[1]))
        self.subsets_ = [self.indices_]
        self.scores_ = [self._compute_score(X_train, X_test, y_train, y_test, self.indices_)]

        while len(self.indices_) > self.k_features:
            scores, subsets = [], []
            for subset in itertools.combinations(self.indices_, r=len(self.indices_)-1):
                score = self._compute_score(X_train, X_test, y_train, y_test, subset)
                scores.append(score)
                subsets.append(subset)

            best_score_index = np.argmax(scores)
            self.indices_ = subsets[best_score_index]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best_score_index])

        self.k_score_ = self.scores_[-1]
        
        return self

    def transform(self, X):
        """
        Transform the dataset to contain only the selected features.
        
        This method reduces the feature set to the ones selected during fitting.
        It handles both pandas DataFrames and numpy arrays appropriately,
        indexing into the data structure to select the specified features.
        
        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            The input samples to transform.
        
        Returns
        -------
        X_transformed : DataFrame or array-like of shape (n_samples, k_features)
            The dataset with only the selected features.
        
        Raises
        ------
        NotFittedError
            If this method is called before the instance is fitted.
        """
        if not hasattr(self, 'indices_'):
            raise NotFittedError(
                "This SequentialBackwardSelector instance is not fitted yet. "
               "Call 'fit' with appropriate arguments before using this estimator.")
        
        # Depending on the input type, use the appropriate indexing method
        if isinstance(X, pd.DataFrame):
            X_transformed = X.iloc[:, list(self.indices_)]
        else:
            X_transformed = X[:, list(self.indices_)]
        
        return X_transformed

    def _compute_score(self, X_train, X_test, y_train, y_test, indices):
        """
        Compute the score of a subset of features.

        Internally used to evaluate the performance of the model on a given
        subset of features.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        X_test : array-like of shape (n_samples, n_features)
            The testing input samples.
        y_train : array-like of shape (n_samples,)
            The training target values.
        y_test : array-like of shape (n_samples,)
            The testing target values.
        indices : array-like of shape (n_features,)
            The indices of the features to be used.

        Returns
        -------
        score : float
            The score of the estimator on the provided feature subset.
        """
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        return self.scoring(y_test, y_pred)

    def _validate_params(self, X):
        """
        Validate the parameters of the estimator.

        This method checks the compatibility of the parameters with the input data
        and raises appropriate errors if invalid parameters are detected.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples used for validation.

        Raises
        ------
        TypeError
            If the estimator does not have a 'fit' method.
        ValueError
            If `k_features` is greater than the number of features in X.
        """
        if not hasattr(self.estimator, 'fit'):
            raise TypeError("Estimator must have a 'fit' method.")
        
        self.k_features = int(self.k_features)
        if self.k_features > X.shape[1]:
            raise ValueError(
                f"k_features must be <= number of features in X ({X.shape[1]}).")

        if callable(self.scoring) or hasattr(self.scoring, '__call__'):
            self.scoring = self.scoring.__name__.replace('_score', '')
        
        if self.scoring not in self._scorers:
            valid_scorers = ", ".join(self._scorers.keys())
            raise ValueError(
                f"Invalid scoring method. Valid options are: {valid_scorers}")

        self.scoring = self._scorers[self.scoring]

    def __repr__(self):
        """
        Provide a string representation of the SequentialBackwardSelection 
        instance.

        This method is useful for debugging and provides an easy-to-read summary
        of the configuration of the SequentialBackwardSelection instance.

        Returns
        -------
        representation : str
            The string representation of the instance.
        """
        class_name = self.__class__.__name__
        params = self.get_params()
        params_str = ", ".join(f"{key}={value!r}" for key, value in params.items())
        return f"{class_name}({params_str})"

class KMeansFeaturizer(BaseEstimator, TransformerMixin):
    """Transforms numeric data into k-means cluster memberships.
     
    This transformer runs k-means on the input data and converts each data point
    into the ID of the closest cluster. If a target variable is present, it is 
    scaled and included as input to k-means in order to derive clusters that
    obey the classification boundary as well as group similar points together.
    
    Parameters 
    -------------
    n_clusters: int, default=7
       Number of initial clusters
    target_scale: float, default=5.0 
       Apply appropriate scaling and include it in the input data to k-means.
    n_components: int, optional
       Number of components for reducted down the predictor. It uses the PCA 
       to reduce down dimension to the importance components. 
    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : 'auto' or int, default=10
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs will be 10 if using
        `init='random'`, and 1 if using `init='kmeans++'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        `"auto"` and `"full"` are deprecated and they will be removed in
        Scikit-Learn 1.3. They are both aliases for `"lloyd"`.

    to_sparse : bool, default=False
            If True, the input data `X` will be converted to a sparse matrix
            before applying the transformation. This is useful for handling
            large datasets more efficiently. If False, the data format of `X`
            is preserved.
    
    Attributes 
    -----------
    km_model: KMeans featurization model used to transform

    Examples 
    --------
    >>> # (1) Use a common dataset 
    >>> import matplotlib.pyplot as plt 
    >>> from sklearn.datasets import make_moons
    >>> from gofast.plot.utils import plot_voronoi 
    >>> from gofast.datasets import load_mxs 
    >>> X, y = make_moons(n_samples=5000, noise=0.2)
    >>> kmf_hint = KMeansFeaturizer(n_clusters=50, target_scale=10).fit(X,y)
    >>> kmf_no_hint = KMeansFeaturizer(n_clusters=50, target_scale=0).fit(X, y)
    >>> fig, ax = plt.subplots(2,1, figsize =(7, 7)) 
    >>> plot_voronoi ( X, y ,cluster_centers=kmf_hint.cluster_centers_, 
                      fig_title ='KMeans with hint', ax = ax [0] )
    >>> plot_voronoi ( X, y ,cluster_centers=kmf_no_hint.cluster_centers_, 
                      fig_title ='KMeans No hint' , ax = ax[1])
    <AxesSubplot:title={'center':'KMeans No hint'}>
    >>> # (2)  Use a concrete data set 
    >>> X, y = load_mxs ( return_X_y =True, key ='numeric' ) 
    >>> # get the most principal components 
    >>> from gofast.analysis import nPCA 
    >>> Xpca =nPCA (X, n_components = 2  ) # veronoi plot expect two dimensional data 
    >>> kmf_hint = KMeansFeaturizer(n_clusters=7, target_scale=10).fit(Xpca,y)
    >>> kmf_no_hint = KMeansFeaturizer(n_clusters=7, target_scale=0).fit(Xpca, y)
    >>> fig, ax = plt.subplots(2,1, figsize =(7, 7)) 
    >>> plot_voronoi ( Xpca, y ,cluster_centers=kmf_hint.cluster_centers_, 
                      fig_title ='KMeans with hint', ax = ax [0] )
    >>> plot_voronoi ( Xpca, y ,cluster_centers=kmf_no_hint.cluster_centers_, 
                      fig_title ='KMeans No hint' , ax = ax[1])
    
    References 
    ------------
    .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
          K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
          Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3

    """
    def __init__(
        self, 
        n_clusters=2, 
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
        to_sparse=False, 
        ):
        self.n_clusters = n_clusters
        self.target_scale = target_scale
        self.random_state = random_state
        self.n_components=n_components
        self.init=init
        self.n_init=n_init
        self.max_iter=max_iter 
        self.tol=tol 
        self.copy_x=copy_x 
        self.verbose=verbose 
        self.algorithm=algorithm
        self.to_sparse=to_sparse
        
    def fit(self, X, y=None):
        """
        Fit the model to the data.

        The method runs k-means on the input data `X`. If `n_components` is specified,
        PCA is applied for dimensionality reduction before clustering. When the target 
        variable `y` is provided, it augments the feature space, enhancing the clustering 
        process to align with the target variable.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data where n_samples is the number of samples and n_features is the 
            number of features.

        y : array-like of shape (n_samples,), default=None
            Target values relative to `X`. Used to augment the feature space if provided.

        Returns
        -------
        self : object
            Returns the fitted instance.

        Raises
        ------
        ValueError
            If `n_components` is not an integer or a float.

        Examples
        --------
        >>> from sklearn.datasets import make_blobs
        >>> X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        >>> featurizer = KMeansFeaturizer(n_clusters=3, target_scale=5.0)
        >>> featurizer.fit(X, y)
        """

        # Validate inputs
        X = check_array(X, accept_sparse=True)
        if y is not None:
            y = check_array(y, ensure_2d=False)

        # Apply PCA if n_components is specified
        if self.n_components is not None:
            if not isinstance(self.n_components, (int, float)):
                raise ValueError("n_components must be an int or a float, "
                                 f"got {type(self.n_components)} instead.")
            pca = PCA(n_components=self.n_components)
            X = pca.fit_transform(X)

        # Prepare data for k-means
        if y is not None:
            y_scaled = y[:, np.newaxis] * self.target_scale
            data_for_clustering = np.hstack((X, y_scaled))
        else:
            data_for_clustering = X

        # Pre-training k-means model on data with or without target
        self.km_model_ = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            algorithm=self.algorithm,
            copy_x=self.copy_x,
            tol=self.tol,
            random_state=self.random_state
        ).fit(data_for_clustering)

        if y is not None:
            # Adjust centroids if y was used
            # Run k-means a second time to get the clusters in the original space
            # without target info. Initialize using centroids found in pre-training.
            # Go through a single iteration of cluster assignment and centroid 
            # recomputation.
            self.km_model_= KMeans(n_clusters=self.n_clusters,
                        init=self.km_model_.cluster_centers_[:,:-1], #[:, :-1]
                        n_init=1,
                        max_iter=1)
            self.km_model_.fit(X)
            
        self.cluster_centers_ = self.km_model_.cluster_centers_

        return self

    def transform(self, X):
        """
        Transform the data by appending the closest cluster ID to each sample.

        This method applies the fitted k-means model to predict the closest cluster
        for each sample in the provided dataset `X`. It then appends the cluster ID
        as an additional feature. The method handles both dense and sparse matrices
        efficiently by converting `X` to a sparse format for concatenation if 
        necessary.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            New data to transform. It can be a dense array or a sparse matrix.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_features + 1)
            The transformed data with an additional feature indicating the cluster 
            ID for each sample. The output is in sparse matrix format to optimize
            memory usage.

        Raises
        ------
        NotFittedError
            If this method is called before the model is fitted.

        Examples
        --------
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        >>> featurizer = KMeansFeaturizer(n_clusters=3)
        >>> featurizer.fit(X)
        >>> X_transformed = featurizer.transform(X)
        """
        # Check if the model is fitted
        check_is_fitted(self, 'km_model_')
        
        # Validate input
        X = check_array(X, accept_sparse=True)
        # Convert X to a sparse matrix if it's not already
        if not sparse.issparse(X) and self.to_sparse:
            X = sparse.csr_matrix(X)
        # Predict the closest cluster for each sample
        clusters = self.km_model_.predict(X)
        if self.to_sparse: 
            clusters_sparse = sparse.csr_matrix(clusters.reshape(-1, 1))
            # Concatenate the original data with the cluster labels
            X_transformed = sparse.hstack((X, clusters_sparse))
        else:
            X_transformed= np.hstack((X, clusters [:, np.newaxis] ))

        return X_transformed
    
        def __repr__(self):
            """ Pretty format for guidance following the API... """
            _t = ("n_clusters", "target_scale", "random_state", "n_components")
            outm = ( '<{!r}:' + ', '.join(
                [f"{k}={ False if getattr(self, k)==... else  getattr(self, k)!r}" 
                 for k in _t]) + '>' 
                ) 
            return  outm.format(self.__class__.__name__)
    

class StratifyFromBaseFeature(BaseEstimator, TransformerMixin):
    """
    Stratifies a dataset by categorizing a numerical attribute and returns 
    stratified training and testing sets.
    
    Useful for datasets with limited data.

    Parameters:
    ----------
    base_feature : str, optional
        Numerical feature to be categorized for stratification.

    threshold_operator : float, default=1.0
        Coefficient to normalize the numerical feature for 
        categorization.

    max_category : int, default=3
        Maximum category value. Values greater than this are grouped 
        into max_category.

    return_train : bool, default=False
        If True, returns the whole stratified training set.

    n_splits : int, default=1
        Number of re-shuffling & splitting iterations.

    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.

    random_state : int, default=42
        Controls the randomness of the dataset splitting.

    Attributes:
    ----------
    statistics_ : DataFrame or None
        Statistics about the categorization and stratification process.
    base_class_: list 
       List composed of the base feature class labels. 
       
    Example
    --------
    >>> from gofast.transformers import StratifyFromBaseFeature
    >>> st= StratifyFromBaseFeature('flow') 
    >>> a, b = st.fit_transform(data)
    >>> st.statistics_
    Out[17]: 
                 Overall (total)    Random  ...  Rand. error (%)  Strat. error (%)
    class_label                             ...                                   
    1.0                 0.320186  0.310345  ...        -3.073463          0.516408
    0.0                 0.354988  0.356322  ...         0.375629          0.375629
    3.0                 0.141531  0.183908  ...        29.941587         -2.543810
    2.0                 0.183295  0.149425  ...       -18.478103          0.334643

    [4 rows x 5 columns]
    Notes:
    ------
    The `statistics_` attribute helps evaluate the distribution of the newly
    added category in different splits and assess the effectiveness of 
    stratification.
    """

    def __init__(self, base_feature=None, threshold_operator=1.0,
                 max_category=3, return_train=False, n_splits=1,
                 test_size=0.2, random_state=42):
        self.logger = gofastlog().get_gofast_logger(self.__class__.__name__)
        self.base_feature = base_feature
        self.threshold_operator = threshold_operator
        self.max_category = max_category
        self.return_train = return_train
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fits the transformer to X for sklearn's Transformer API compatibility.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : Ignored
            Not used, present only for compatibility with sklearn's transformer
            interface.

        Returns:
        --------
        self : object
            Returns self for method chaining.
            """
        return self

    def transform(self, X):
        """
        Transforms the dataset and stratifies based on the categorized 
        attribute.

        Parameters:
        ----------
        X : DataFrame
            The input DataFrame to be stratified.

        Returns:
        -------
        tuple of DataFrames
            Stratified training and testing sets.
        """
        train_set, test_set = train_test_split(X, test_size=self.test_size,
                                               random_state=self.random_state)
        if self.base_feature is None:
            self.logger.info('Base numerical feature not provided. '
                             'Using random sampling.')
            return train_test_split(X, test_size=self.test_size,
                                    random_state=self.random_state)

        X = self._categorize_feature(X)
        # split after categorizing the base feature 
        train_set, test_set = train_test_split(
            X, test_size=self.test_size,random_state=self.random_state)
        strat_train_set, strat_test_set = self._stratify_dataset(X)
        self._calculate_statistics(X, test_set, strat_test_set) 
        self._cleanup_temp_columns(X, strat_train_set, strat_test_set)

        return (strat_train_set, strat_test_set
                ) if not self.return_train else (X, strat_test_set)

    def _categorize_feature(self, X):
        """Categorizes the numerical feature."""
        # Implement logic to categorize 'base_feature' here
        from .tools.mlutils import discretize_categories
        X = discretize_categories(X, in_cat=self.base_feature, 
            new_cat="class_label", divby =self.threshold_operator,
            higherclass = self.max_category
             )
        self.base_class_=  list(X["class_label"].value_counts().index.values)
        return X

    def _stratify_dataset(self, X):
        """Performs stratification of the dataset."""
        split = StratifiedShuffleSplit(n_splits=self.n_splits,
                                       test_size=self.test_size,
                                       random_state=self.random_state)
        for train_index, test_index in split.split(X, X['class_label']):
            return X.loc[train_index], X.loc[test_index]

    def _calculate_statistics(self, X, *test_dataframes ):
        """Calculates statistics for the stratification process.
        
        Parameters 
        -----------
        X : DataFrame
            The input DataFrame to be stratified.
        *test_dataframes: DataFrames 
           Test data before and after the stratification.  
        """
        test_set, strat_test_set = test_dataframes 
        # Implement logic to calculate statistics here
        random_stats= test_set["class_label"].value_counts()/ len(test_set)
        stratified_stats = strat_test_set["class_label"].value_counts()/len(
            strat_test_set) 
        total = X["class_label"].value_counts() /len(X)
        stats = {
             "class_label":  np.array (self.base_class_), 
             "Overall (total)":total, 
             "Random": random_stats, 
             "Stratified": stratified_stats,  
             "Rand. error (%)":  (random_stats /total -1) *100, 
             "Strat. error (%)":  (stratified_stats /total -1) *100, 
         }
        self.statistics_ = pd.DataFrame ( stats )
        # set a pandas dataframe for inspections attributes `statistics`.
        self.statistics_.set_index("class_label", inplace=True)
        
    def _cleanup_temp_columns(self, X, *dataframes):
        """Removes temporary columns used for stratification."""
        temp_columns = ["class_label"]  # Add any temporary columns used
        for df in dataframes:
            df.drop(temp_columns, axis=1, inplace=True)
        X.drop(temp_columns, axis=1, inplace=True)
        
class CategoryBaseStratifier(BaseEstimator, TransformerMixin):
    """
    Stratifies a dataset based on a specified base category 
    
    It is more representative splits into training and testing sets,
    especially useful when data is limited.

    Parameters:
    -----------
    base_column : str or int, optional
        The name or index of the column to be used for stratification. If None, 
        returns purely random sampling.

    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.

    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split.

    Attributes:
    ----------
    statistics_ : DataFrame or None
        Statistics about the stratification process, including the distribution
        of the base category in the full dataset, the random test set, and the 
        stratified test set.

    Examples:
    --------
    >>> from gofast.datasets import load_bagoue 
    >>> from gofast.transformers import CategoryBaseStratifier
    >>> X = load_bagoue ( as_frame=True) 
    >>> cobj = CategoryBaseStratifier(base_column='shape')
    >>> Xtrain_stratified, X_test_stratified= cobj.fit_transform ( X ) 
    >>> X_test_stratified.head(2) 
    Out[67]: 
           num  name      east  ...    lwi                    geol  flow
    121  122.0  b122  756473.0  ...  32.54  VOLCANO-SEDIM. SCHISTS   1.0
    207  208.0  b208  792785.0  ...  29.80  VOLCANO-SEDIM. SCHISTS   1.0

    [2 rows x 13 columns]
    Notes:
    ------
    The `statistics_` attribute provides insights into the distribution of the 
    base category across different splits, helping evaluate the effectiveness 
    of the stratification process.
    """
    
    def __init__(self, base_column=None, test_size=0.2, random_state=42):
        self.logger = gofastlog().get_gofast_logger(self.__class__.__name__)
        self.base_column = base_column
        self.test_size = test_size
        self.random_state = random_state


    def fit(self, X, y=None):
        """Does nothing, exists for compatibility with sklearn's Transformer API."""
        return self
    
    def transform(self, X):
        """
        Splits the dataset into training and testing sets using stratification.

        Parameters:
        -----------
        X : DataFrame
            The input DataFrame to split.

        Returns:
        --------
        tuple of DataFrames
            The training and testing sets after stratification.
        """
        if self.base_column is None:
            self.logger.debug('Base column not provided. Using purely random sampling.')
            return train_test_split(X, test_size=self.test_size,
                                    random_state=self.random_state)

        if isinstance(self.base_column, int):
            self.base_column = X.columns[int(self.base_column)]
        elif isinstance(self.base_column, str) and self.base_column not in X.columns:
            self.logger.warning(
                f'Base column "{self.base_column}" not found in DataFrame columns.')
            return train_test_split(
                X, test_size=self.test_size, random_state=self.random_state)

        strat_train_set, strat_test_set = self._stratify(X)
        self._calculate_statistics(X, strat_test_set)
        return strat_train_set, strat_test_set

    def _stratify(self, X):
        """
        Internal method to perform the stratification.

        Parameters:
        -----------
        X : DataFrame
            The input DataFrame to stratify.

        Returns:
        --------
        tuple of DataFrames
            The stratified training and testing sets.
        """
        from .tools.mlutils import stratify_categories 
        # stratification logic here (e.g., using pd.cut)
        strat_train_set, strat_test_set = stratify_categories(
            X, self.base_column, test_size = self.test_size, 
            random_state = self.random_state )
        return strat_train_set, strat_test_set

    def _calculate_statistics(self, X, strat_test_set):
        """
        Calculates statistics related to the stratification process.

        Parameters:
        -----------
        X : DataFrame
            The original input DataFrame.

        strat_test_set : DataFrame
            The stratified testing set.
        """
        overall_distribution = X[self.base_column].value_counts() / len(X)
        stratified_distribution = strat_test_set[self.base_column].value_counts(
            ) / len(strat_test_set)
        error = ((stratified_distribution / overall_distribution) - 1) * 100

        self.statistics_ = pd.DataFrame({
            'Overall': overall_distribution,
            'Stratified': stratified_distribution,
            'Strat. %error': error
        })


class CategorizeFeatures(BaseEstimator, TransformerMixin ): 
    """ Transform numerical features into categorical features and return 
    a new array transformed. 
    
    Parameters  
    ------------
    columns: list,
       List of the columns to encode the labels 
       
    func: callable, 
       Function to apply the label accordingly. Label must be included in 
       the columns values.
       
    categories: dict, Optional 
       Dictionnary of column names(`key`) and labels (`values`) to 
       map the labels.  
       
    get_dummies: bool, default=False 
      returns a new encoded DataFrame  with binary columns 
      for each category within the specified categorical columns.

    parse_cols: bool, default=False
      If `columns` parameter is listed as string, `parse_cols` can defaultly 
      constructs an iterable objects. 
    
    return_cat_codes: bool, default=False 
       return the categorical codes that used for mapping variables. 
       if `func` is applied, mapper returns an empty dict. 
    
    Examples
    --------
    >>> from gofast.datasets import make_mining_ops 
    >>> from gofast.transformers import CategorizeFeatures
    >>> X = make_mining_ops (samples =20, as_frame =True ) 
    >>> cf = CategorizeFeatures (columns =['OreType', 'EquipmentType']) 
    >>> Xtransformed = cf.fit_transform (X)
    >>> Xtransformed.head(7) 
       OreType  EquipmentType
    0        1              3
    1        0              3
    2        1              3
    3        1              3
    4        0              0
    5        1              0
    6        2              2
    >>> cf.cat_codes_
    {'OreType': {1: 'Type2', 0: 'Type1', 2: 'Type3'},
     'EquipmentType': {3: 'Truck', 0: 'Drill', 2: 'Loader', 1: 'Excavator'}}
    """
    
    def __init__(
        self, 
        columns: list =None, 
        func: _F=None, 
        categories: dict=None, 
        get_dummies:bool=..., 
        parse_cols:bool =..., 
        ): 
        self._logging= gofastlog().get_gofast_logger(self.__class__.__name__)
        self.columns =columns 
        self.func=func
        self.categories=categories 
        self.get_dummies =get_dummies 
        self.parse_cols=parse_cols 
 
    def fit(self, X, y=None):
        """ 
        Parameters 
        ----------
        
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            y is passed for API purpose. It does nothing.  
            
        Returns 
        --------
        ``Self`: Instanced object for methods chaining 
        
        """
        return self
    
    def transform(self, X) :
        """ 
        Transforms `X` by applying the specified operation.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_features + 1)
            The transformed array with the additional combined attribute.
        
        """
        from .tools.mlutils import codify_variables 
        # -------------------------------------------
        if _is_arraylike_1d(X): 
            raise ValueError ("One-dimensional or Series is not allowed."
                              " Use sklearn.preprocessing.LabelEncoder or "
                              " gofast.tools.smart_label_classier to encode"
                              " variables.")
        X = build_data_if(X, to_frame =True, force =True,
                          raise_warning="silence", input_name='col')
        X, self.num_columns_, self.cat_columns_ = to_numeric_dtypes(
            X, return_feature_types=True
            )
        X, self.cat_codes_ = codify_variables( 
            X ,
            columns = self.columns, 
            func =self.func, 
            categories=self.categories, 
            get_dummies=self.get_dummies,
            return_cat_codes=True
            )
        
        return X 
    

class AttributesCombinator(BaseEstimator, TransformerMixin):
    """
    Combine attributes using operators, indexes, or names.

    Create a new attribute by performing operations on selected features using
    their indexes, literal string operators, or names. This transformer is useful
    for creating new features that are combinations of existing features, which
    might improve the performance of machine learning models.

    Parameters:
    -----------
    attribute_names : list of str or str, optional
        List of feature names for combination, or a string with operator symbols.
        Decides how to combine new feature values based on the `operator` parameter.
        Example: attribute_names=['feature1', 'feature2'].

    attribute_indexes : list of int, optional
        Indexes of each feature for combining. Raises an error if any index does
        not match the dataframe or array columns. Example: attribute_indexes=[0, 1].

    operator : str, default='/' 
        Type of operation to perform. Can be one of ['/','+', '-', '*', '%'].

    Attributes:
    ----------
    _operators : dict
        A mapping of string operator symbols to NumPy functions for efficient
        computation of the desired operation.

    Methods:
    --------
    fit(X, y=None):
        Fits the transformer to `X`. In this transformer, `fit` does not perform
        any operation and returns self.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : Ignored
            This parameter is ignored and present only for compatibility with
            the sklearn transformer interface.

    transform(X):
        Transforms `X` by applying the specified operation to the selected
        features and returns the transformed data.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns:
        X_transformed : array-like of shape (n_samples, n_features + 1)
            The transformed array with the additional combined attribute.

    Examples:
    --------
    >>> import pandas as pd
    >>> from gofast.transformers import AttributesCombinator
    >>> from gofast.datasets.dload import load_bagoue
    >>> X, y = load_bagoue(as_frame=True)
    """
    _operators = {
        '/': np.divide,
        '+': np.add,
        '-': np.subtract,
        '*': np.multiply,
        '%': np.mod,
    }

    def __init__(self, attribute_names=None, attribute_indexes=None,
                 operator='/'):
        self.attribute_names = attribute_names
        self.attribute_indexes = attribute_indexes
        self.operator = operator
        self._validate_operator()

    def fit(self, X, y=None):
        """
        Fits the transformer to `X`.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : Ignored
            Not used, present only for compatibility with sklearn's transformer
            interface.

        Returns:
        --------
        self : object
            Returns self for method chaining.
        """
        return self

    def transform(self, X):
        """
        Transforms `X` by applying the specified operation.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_features + 1)
            The transformed array with the additional combined attribute.
        """
        if isinstance(X, pd.DataFrame):
            X, columns = self._validate_dataframe(X)
        else:
            columns = None

        if self.attribute_indexes:
            X = self._apply_operation(X)
            columns = columns + [self._new_feature_name(columns)] if columns else None

        return pd.DataFrame(X, columns=columns) if columns else X

    def _validate_operator(self):
        """
        Validates the operator.

        Raises:
        -------
        ValueError: If the operator is not recognized.
        """
        if self.operator not in self._operators:
            raise ValueError(f"Invalid operator '{self.operator}'. Valid"
                             f" operators are {list(self._operators.keys())}.")

    def _validate_dataframe(self, X):
        """
        Validates the DataFrame and extracts column names and indexes.

        Parameters:
        -----------
        X : DataFrame
            The DataFrame to validate.

        Returns:
        --------
        X_values : np.ndarray
            The values of the DataFrame.

        columns : list
            The column names of the DataFrame.
        """
        columns = X.columns.tolist()
        if self.attribute_names:
            if not all(name in columns for name in self.attribute_names):
                raise ValueError("Some attribute names are not in the DataFrame columns.")
            self.attribute_indexes = [columns.index(name) for name in self.attribute_names]
        return X.values, columns

    def _apply_operation(self, X):
        """
        Applies the specified operation to the selected features.

        Parameters:
        -----------
        X : np.ndarray
            The array to which the operation will be applied.

        Returns:
        --------
        X_transformed : np.ndarray
            The transformed array.
        """
        operation = self._operators[self.operator]
        result = operation(X[:, self.attribute_indexes[0]], X[:, self.attribute_indexes[1]])

        for index in self.attribute_indexes[2:]:
            result = operation(result, X[:, index])

        return np.c_[X, result]

    def _new_feature_name(self, columns):
        """
        Generates a new feature name based on the operation and involved columns.

        Parameters:
        -----------
        columns : list
            The list of column names.

        Returns:
        --------
        new_feature_name : str
            The generated feature name.
        """
        return f'{"_".join([columns[i] for i in self.attribute_indexes])}_{self.operator}'


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Combine attributes using literal string operators, indexes, or names.

    Create a new attribute by performing operations on selected features using
    either their indexes, literal string operators, or names. This class inherits
    from scikit-learn's `BaseEstimator` and `TransformerMixin` classes.

    Parameters:
    -----------
    attribute_names : list of str, optional
        List of feature names for combination. Decides how to combine new feature
        values based on the `operator` parameter. By default, it performs a ratio
        of the given attributes. For example, if `attribute_names=['lwi', 'ohmS']`,
        it will divide the 'lwi' feature by 'ohmS'.

    attribute_indexes : list of int, optional
        Indexes of each feature for combining. Raises a warning if any index does
        not match the dataframe or array columns.

    operator : str, default='/' 
        Type of operation to perform. Can be one of ['/','+', '-', '*', '%'].

    Returns:
    --------
    X : np.ndarray
        A new array containing the result of the operations specified by
        `attribute_names`, `attribute_indexes`, and `operator`. If both `attribute_names`
        and `attribute_indexes` are `None`, it will return the original array.

    Notes:
    ------
    Literal string operators can be used for operations. For example, dividing two
    numerical features can be represented as "per" separated by underscores, like
    "_per_". For instance, to create a new feature based on the division of the
    features 'lwi' and 'ohmS', you can use the `attribute_names` parameter as:
    attribute_names='lwi_per_ohmS'.

    The same literal string format is valid for other operations such as multiplication
    (_mul_), subtraction (_sub_), modulo (_mod_), and addition (_add_). Alternatively,
    you can use the indexes of features for combining by providing the `attribute_indexes`
    parameter. Multiple operations can be set by passing a list of literal string
    operators in `attribute_indexes`.

    Examples:
    --------
    >>> import pandas as pd
    >>> from gofast.transformers import CombinedAttributesAdder
    >>> from gofast.datasets.dload import load_bagoue
    >>> X, y = load_bagoue(as_frame=True)
    >>> cobj = CombinedAttributesAdder(attribute_names='lwi_per_ohmS')
    >>> Xadded = cobj.fit_transform(X)
    >>> cobj.attribute_names_
    ... ['num',
         'name',
         'east',
         'north',
         'power',
         'magnitude',
         'shape',
         'type',
         'sfi',
         'ohmS',
         'lwi',
         'geol',
         'lwi_div_ohmS']  # new attribute 'lwi_div_ohmS'
    >>> df0 = pd.DataFrame(Xadded, columns=cobj.attribute_names_)
    >>> df0['lwi_div_ohmS']
    ... 0           0.0
        1      0.000002
        2      0.000005
        3      0.000004
        4      0.000008
          
        426    0.453359
        427    0.382985
        428    0.476676
        429    0.457371
        430    0.379429
        Name: lwi_div_ohmS, Length: 431, dtype: object
    >>> cobj = CombinedAttributesAdder(attribute_names=['lwi', 'ohmS', 'power'], operator='+')
    >>> df0 = pd.DataFrame(cobj.fit_transform(X), columns=cobj.attribute_names_)
    >>> df0.iloc[:, -1]
    ... 0      1777.165142
        1      1207.551531
        2         850.5625
        3      1051.943553
        4       844.095833
            
        426      1708.8585
        427      1705.5375
        428      1568.9825
        429     1570.15625
        430      1666.9185
        Name: lwi_add_ohmS_add_power, Length: 431, dtype: object
    >>> cobj = CombinedAttributesAdder(attribute_indexes=[1, 6], operator='+')
    >>> df0 = pd.DataFrame(cobj.fit_transform(X), columns=cobj.attribute_names_)
    >>> df0.iloc[:, -1]
    ... 0        b1W
        1        b2V
        2        b3V
        3        b4W
        4        b5W
         
        426    b427W
        427    b428V
        428    b429V
        429    b430V
        430    b431V
        Name: name_add_shape, Length: 431, dtype: object
    """
    _op = {
        'times': ('times', 'prod', 'mul', '*', 'x'),
        'add': ('add', '+', 'plus'),
        'div': ('quot', '/', 'div', 'per'),
        'sub': ('sub', '-', 'less'),
        'mod': ('mod', '%'),
    }

    def __init__(
            self,
            attribute_names=None,
            attribute_indexes=None,
            operator='/', ):
        self.attribute_names = attribute_names
        self.attribute_indexes = attribute_indexes
        self.operator = operator
        self.attribute_names_ = None

    def fit(self, X, y=None):
        """
        Fit the `CombinedAttributesAdder` transformer to the input data `X`.

        Parameters:
        -----------
        X : ndarray (M x N matrix where M=m-samples, N=n-features)
            Training set; denotes data that is observed at training and
            prediction time, used as independent variables in learning.

        y : array-like, shape (M,), default=None
            Train target; denotes data that may be observed at training time
            as the dependent variable in learning, but which is unavailable
            at prediction time and is usually the target of prediction.

        Returns:
        --------
        self : object
            Returns self for easy method chaining.

        """
        return self

    def transform(self, X):
        """
        Transform data and return an array with the combined attributes.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns:
        --------
        X : {array-like, sparse matrix} of shape (n_samples, n_features + 1)
            Transformed array, where n_samples is the number of samples and
            n_features is the number of features, with the additional combined attribute.

        """
        columns = []
        self.operator = self._get_operator(
            self.operator or self.attribute_names)

        if self.operator is None:
            warnings.warn("None or invalid operator cannot be used for attribute combinations.")

        if isinstance(self.attribute_names, str):
            self.attribute_names_ = parse_attrs(self.attribute_names)

        elif isinstance(self.attribute_names, (list, tuple, np.ndarray)):
            self.attribute_names_ = self.attribute_names

        if isinstance(X, pd.DataFrame):
            # Check if attributes exist in the DataFrame
            if self.attribute_names_:
                validate_feature(X, self.attribute_names_)
            # Get the index of attributes from the DataFrame
            if self.attribute_names_:
                self.attribute_indexes = list(map(
                    lambda o: list(X.columns).index(o), self.attribute_names_)
                )

            elif self.attribute_indexes:
                try:
                    self.attribute_names_ = list(map(
                        lambda ix: list(X.columns)[ix], self.attribute_indexes)
                    )
                except IndexError:
                    raise IndexError("List of indexes is out of range.")

            columns = X.columns
            X = to_numeric_dtypes(X)
            X = X.values

        if self.attribute_indexes:
            X = self._operate(X)

        if self.attribute_names_ is not None:
            self.attribute_names_ = list(columns) + ([
                f'_{self.operator}_'.join([v for v in self.attribute_names_])
            ] if self._isfine else [])
            
        try: 
            X= pd.DataFrame ( X, columns = self.attribute_names_)
        except : pass 
    
        return X

    def _get_operator(self, operator):
        """ Get operator for combining attributes """
        for k, v in self._op.items():
            for o in v:
                if operator.find(o) >= 0:
                    self.operator = k
                    return self.operator
        return

    def _operate(self, X):
        """ Perform operations based on indexes """
        def weird_division(ix_):
            """ Replace 0. value with 1 in denominator for division calculations """
            return ix_ if ix_ != 0. else 1

        msg = ("Unsupported operand type(s)! Index provided {} doesn't match "
               "any numerical features. Combined attribute creation is not possible.")

        self._isfine = True
        Xc = X[:, self.attribute_indexes]
        cb = Xc[:, 0]
        Xc = Xc[:, 1:]

        for k in range(Xc.shape[1]):
            try:
                if self.operator == 'mod':
                    cb %= Xc[:, k]
                if self.operator == 'add':
                    cb += Xc[:, k]
                if self.operator == 'sub':
                    cb -= Xc[:, k]
                if self.operator == 'div':
                    # If the denominator contains NaN or 0, a weird division is triggered
                    # and replaces the denominator by 1
                    try:
                        cb /= Xc[:, k]
                    except ZeroDivisionError:
                        wv = np.array(
                            list(map(weird_division, Xc[:, k])))
                        cb /= wv

                    except (TypeError, RuntimeError, RuntimeWarning):
                        warnings.warn(msg.format(
                            self.attribute_indexes[1:][k]))
                if self.operator == 'x':
                    cb *= Xc[:, k]
            except:
                warnings.warn(msg.format(self.attribute_indexes[1:][k]))
                self._isfine = False

        X = np.c_[X, cb] if self._isfine else X

        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select specific attributes from a DataFrame for column transformation.

    The `DataFrameSelector` transformer is used to select specific columns,
    either numerical or categorical, from a DataFrame for further data 
    transformation, similar to scikit-learn's `make_column_transformer`.

    Parameters:
    -----------
    columns : list or array-like, default=None
        List of column names to keep in the data. If None, no columns 
        are selected.

    select_type : {'number', 'category'}, default=None
        Automatically select numerical or categorical columns based on the
        specified type. If 'number', only numerical columns will be selected;
        if 'category', only categorical columns
        will be selected. If None, no automatic selection is performed.

    parse_cols : bool, default=False
        If True, enables column name parsing to handle special characters 
        and spaces.

    return_frame : bool, default=False
        If True, returns a DataFrame after selecting columns; if False,
        returns a numpy array.

    Attributes:
    -----------
    features_ : list
        The final list of column names to be selected after considering 
        the input 'columns' and 'select_type'.

    num_features_ : list
        List of column names that are considered numerical after selection.

    cat_features_ : list
        List of column names that are considered categorical after selection.

    Returns:
    --------
    X : ndarray or DataFrame
        A new array or DataFrame composed of data from selected `columns`.

    Examples:
    ---------
    >>> from gofast.transformers import DataFrameSelector
    >>> from gofast.datasets import make_african_demo
    >>> af_data = make_african_demo(start_year='2021', end_year='2022').frame
    >>> dfr = DataFrameSelector(columns=None, select_type='number')
    >>> af_transf = dfr.fit_transform(af_data)
    >>> af_transf[:3, :]
    Out[14]: 
    array([[2.02100000e+03, 1.12942759e+08, 4.43867835e+01, 1.98222121e+01,
            7.48855579e+01, 2.83628640e+03],
           [2.02100000e+03, 1.12834803e+08, 2.73762471e+01, 6.04337242e+00,
            5.35163931e+01, 5.93341830e+03],
           [2.02100000e+03, 1.86877893e+08, 2.53328024e+01, 5.08724291e+00,
            8.10308039e+01, 9.77191812e+03]])
    """
    def __init__(
        self, 
        columns:list=None, 
        select_type:str=None, 
        parse_cols:bool=..., 
        return_frame:bool=...
        ):
        self.columns = columns 
        self.select_type = select_type
        self.parse_cols = parse_cols
        self.return_frame = return_frame

    def fit(self, X, y=None):
        """
        Fit the `DataFrameSelector` transformer to the input data `X`.

        Parameters:
        -----------
        X : ndarray (M x N matrix where M=m-samples, N=n-features)
            Training set; denotes data that is observed at training and
            prediction time, used as independent variables in learning.

        y : array-like, shape (M,), default=None
            Train target; denotes data that may be observed at training time
            as the dependent variable in learning, but which is unavailable
            at prediction time and is usually the target of prediction.

        Returns:
        --------
        self : object
            Returns self for easy method chaining.

        """
        self.parse_cols, self.return_frame = ellipsis2false(
            self.parse_cols, self.return_frame)
        self.num_features_ = []
        self.cat_features_ = []
        return self

    def transform(self, X):
        """
        Transform data and return numerical or categorical values.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns:
        --------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Transformed array, where n_samples is the number of samples and
            n_features is the number of features.

        """
        is_frame(X, df_only=True, raise_exception=True,
                 objname="DataFrameSelector")
        
        if self.columns is not None:
            self.columns = is_iterable(
                self.columns, exclude_string=True, transform=True,
                parse_string=self.parse_cols)

        if self.columns is None and self.select_type is None:
            raise TypeError(
                "Either 'columns' or 'select_type' needs to be supplied.")

        if self.columns is not None:
            validate_feature(X, self.columns)
            X = X[self.columns]

        if self.select_type is not None:
            if str(self.select_type).lower().strip(
                    ) not in "numbernumericalcategorycategories":
                emsg = ("Support only 'number' or 'category'."
                        f" Got {self.select_type!r}")
                if self.columns is None:
                     raise ValueError(emsg) 
                else: warnings.warn(emsg)
                
            _, num_feats, cat_feats = to_numeric_dtypes(
                X, return_feature_types=True)

            if str(self.select_type).lower().strip().find('num') >= 0:
                self.columns = num_feats
            elif str(self.select_type).lower().strip().find('cat') >= 0:
                self.columns = cat_feats
                
        # For consistency, shrunk X
        self.features_ = list(X.columns)
        X = X[self.columns]

        # Update the numeric and categorical features
        _, self.num_features_, self.cat_features_ = to_numeric_dtypes(
            X, return_feature_types=True)

        return X if self.return_frame else np.array(X)

class FrameUnionFlex(BaseEstimator, TransformerMixin):
    """
    FrameUnionFlex combines numerical and categorical data preprocessing
    into a single transformer. It automates the process of imputing missing
    values, scaling numerical features, and encoding categorical features,
    simplifying the creation of machine learning pipelines. This transformer
    is highly configurable, allowing for detailed specification or automatic
    detection of feature types, as well as a choice of imputation strategies,
    scaling methods, and encoding techniques.

    Parameters
    ----------
    num_attributes : list of str, optional
        Specifies the column names to be treated as numerical attributes.
        If None, numerical attributes will be automatically identified in
        the dataset based on their data type.
    cat_attributes : list of str, optional
        Specifies the column names to be treated as categorical attributes.
        If None, categorical attributes will be automatically identified in
        the dataset based on their data type.
    scale : bool, default=True
        If True, applies scaling to numerical attributes using the scaling
        method defined by `scale_mode`. This is crucial for models that are
        sensitive to feature magnitude.
    impute_data : bool, default=True
        If True, imputes missing values in numerical attributes using the
        strategy defined by `strategy`. This helps in handling datasets with
        incomplete data.
    encode : bool, default=True
        If True, encodes categorical attributes using the method defined by
        `encode_mode`. Encoding is essential for converting categorical data
        into a format that can be provided to machine learning models.
    strategy : str, default='median'
        Defines the strategy used for imputing missing values. Options include
        'mean', 'median', and 'most_frequent'.
    scale_mode : str, default='StandardScaler'
        Defines the method used for scaling numerical attributes. Options are
        'StandardScaler' for z-score normalization and 'MinMaxScaler' for
        min-max normalization. 'RobustScaler' is also available for scaling
        features using statistics that are robust to outliers.
    encode_mode : str, default='OrdinalEncoder'
        Defines the method used for encoding categorical attributes. Options
        are 'OrdinalEncoder' for ordinal encoding and 'OneHotEncoder' for
        one-hot encoding.

    Attributes
    ----------
    num_attributes_ : list
        Auto-detected or specified names of numerical attributes in the data.
    cat_attributes_ : list
        Auto-detected or specified names of categorical attributes in the data.
    attributes_ : list
        Combined list of all numerical and categorical attributes in the data.
    X_ : ndarray of shape (n_samples, n_features)
        The transformed dataset containing processed numerical and encoded
        categorical features.

    Notes
    -----
    - FrameUnionFlex is designed to be flexible and efficient, automatically
      adapting to the provided dataset while allowing for user overrides.
    - It supports handling datasets with a mix of numerical and categorical
      data, preparing them for machine learning models that require numerical
      input.
    - Users are encouraged to explicitly define `num_attributes` and
      `cat_attributes` for better control and clarity in preprocessing.

    Examples
    --------
    >>> from sklearn.datasets import fetch_california_housing
    >>> from gofast.transformers import FrameUnionFlex
    >>> data = fetch_california_housing(as_frame=True)
    >>> X = data.frame.drop('MedHouseVal', axis=1)
    >>> transformer = FrameUnionFlex(scale=True, impute_data=True,
    ...                              encode=True, scale_mode='MinMaxScaler',
    ...                              encode_mode='OneHotEncoder')
    >>> X_transformed = transformer.fit_transform(X)
    >>> print(X_transformed.shape)
    
    This example demonstrates using FrameUnionFlex to preprocess the
    California housing dataset, applying min-max scaling to numerical
    features and one-hot encoding to categorical features. The transformed
    dataset is ready for use with machine learning models.
    """
    def __init__(self, num_attributes=None, cat_attributes=None,
                 scale=True, imput_data=True, encode=True,
                 strategy='median', scale_mode='StandardScaler',
                 encode_mode='OrdinalEncoder'):

        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        
        self.num_attributes = num_attributes 
        self.cat_attributes = cat_attributes 
        self.imput_data = imput_data 
        self.strategy =strategy 
        self.scale = scale
        self.encode = encode 
        self.scale_mode = scale_mode
        self.encode_mode = encode_mode
        
    def fit(self, X, y=None):
        """
        Fit the `FrameUnion` transformer to the input data `X`.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training vector, where `n_samples` is the number of samples,
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target vector relative to X. Not used in this transformer.

        Returns:
        --------
        self : object
            Returns self.
        """
        return self
    
    def transform(self, X):
        """
        Transform `X` by applying specified preprocessing steps.

        This involves auto-detecting or using specified numerical and
        categorical attributes, imputing missing values in numerical
        attributes, scaling numerical attributes, and encoding
        categorical attributes.

        Parameters
        ----------
        X : DataFrame
            The input DataFrame to transform.

        Returns
        -------
        X_transformed : DataFrame or ndarray
            The transformed DataFrame with numerical attributes scaled
            and categorical attributes encoded. It may return a dense
            array or a sparse matrix, depending on the encoding method.
        """
        def extract_arr_columns ( attr, select_type ): 
            """ Extract array and columns from :class:`DataFrameSelector`"""
            frameobj =DataFrameSelector(columns=attr,select_type=select_type)
            arr= frameobj.fit_transform (X )
            return arr, frameobj.columns 
        
        # Construct a Frame if not aleardy dataframe 
        X = build_data_if ( X, to_frame=True, input_name='fu', force=True, 
                           raise_exception= True )
        # Validate and adjust the scale_mode parameter
        self.scale_mode = self.scale_mode if self.scale_mode in [
            "StandardScaler", "MinMaxScaler", "RobustScaler"] else "StandardScaler"
        
        # Select numerical and categorical columns
        num_array, num_columns = extract_arr_columns(self.num_attributes, 'num')
        cat_array, cat_columns = extract_arr_columns(self.cat_attributes, 'cat')
    
        # Impute numerical columns if specified
        if self.imput_data:
            num_array = SimpleImputer(
                strategy=self.strategy).fit_transform(num_array)
    
        # Scale numerical columns if specified
        if self.scale:
            scaler = ( 
                StandardScaler() if self.scale_mode == "StandardScaler" 
                else ( MinMaxScaler() if self.scale_mode =='MinMaxScaler' 
                      else RobustScaler()
                      )
                )
            num_array = scaler.fit_transform(num_array)
        # Encode categorical columns if specified
        if self.encode:
            encoder = ( 
                OrdinalEncoder() if self.encode_mode == "OrdinalEncoder" 
                else OneHotEncoder(sparse_output=True)
                )
            cat_array = encoder.fit_transform(cat_array)
            
            # Handling potential sparse matrix output from OneHotEncoder
            if isinstance(cat_array, np.ndarray):
                warnings.warn('Sparse matrix is converted to a dense Numpy array.',
                              UserWarning)
            elif self.encode_mode == "OneHotEncoder":
                warnings.warn('Using `OneHotEncoder` generates a sparse matrix.'
                              ' Consider handling sparse output accordingly.',
                              UserWarning)
        # Combine numerical and categorical arrays
        try:
            X_transformed = np.c_[num_array, cat_array]
        except ValueError as e:
            raise ValueError(f"Error concatenating transformed features: {e}")
        
        # Try to fallback to DataFrame.
        try : 
            columns = num_columns + cat_columns 
            X_transformed= pd.DataFrame (X_transformed, columns = columns)
        except : 
            pass 
        else: 
            # Keep category values as integers. 
            X_transformed = FloatCategoricalToIntTransformer(
                ).fit_transform (X_transformed)
    
        return X_transformed

class FrameUnion(BaseEstimator, TransformerMixin):
    """
    A transformer that combines numerical and categorical feature processing pipelines
    into a unified framework. This includes options for imputing missing values, 
    scaling numerical features, and encoding categorical features. Designed to be
    used within a ColumnTransformer to efficiently preprocess a DataFrame for
    machine learning models.

    Parameters
    ----------
    num_attributes : list of str, default=None
        List of column names in the DataFrame corresponding to numerical attributes.
        These columns will be processed according to the scaling and imputing
        parameters provided.

    cat_attributes : list of str, default=None
        List of column names in the DataFrame corresponding to categorical attributes.
        These columns will be processed according to the encoding parameter provided.

    scale : bool, default=True
        Determines whether numerical features should be scaled. If True, features
        will be scaled using either StandardScaler or MinMaxScaler based on the
        `scale_mode` parameter.

    impute_data : bool, default=True
        If True, missing values in the data will be imputed using SimpleImputer with
        the strategy specified by the `strategy` parameter.

    encode : bool, default=True
        Determines whether categorical features should be encoded. If True,
        features will be encoded using either OrdinalEncoder or OneHotEncoder
        based on the `encode_mode` parameter.

    strategy : str, default='median'
        The strategy used by SimpleImputer to replace missing values. Common strategies
        include 'mean', 'median', and 'most_frequent'.

    scale_mode : str, default='standard'
        Determines the scaling method to be used for numerical features. Options
        are 'standard' for StandardScaler and 'minmax' for MinMaxScaler.

    encode_mode : str, default='ordinal'
        Determines the encoding method to be used for categorical features. Options
        are 'ordinal' for OrdinalEncoder and 'onehot' for OneHotEncoder.

    Examples
    --------
    >>> from sklearn.datasets import fetch_openml
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.transformers import FrameUnion
    >>> X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    >>> num_attrs = ['age', 'fare']
    >>> cat_attrs = ['embarked', 'sex']
    >>> frame_union = FrameUnion(num_attributes=num_attrs, cat_attributes=cat_attrs,
    ...                          scale=True, impute_data=True, encode=True,
    ...                          strategy='median', scale_mode='standard', encode_mode='onehot')
    >>> X_transformed = frame_union.fit_transform(X)
    >>> print(X_transformed.shape)

    Notes
    -----
    - The `FrameUnion` transformer is particularly useful in preprocessing pipelines
      for machine learning models where both numerical and categorical data require
      different processing steps.
    - It is designed to be flexible and easily adjustable to different preprocessing
      needs by changing its parameters.
    - When `encode_mode` is set to 'onehot', the transformed data might be returned
      as a sparse matrix. Users should handle the output accordingly based on the
      requirements of downstream models or processes.
    """
    def __init__(self, num_attributes=None, cat_attributes=None, scale=True,
                 impute_data=True, encode=True, strategy='median',
                 scale_mode='standard', encode_mode='ordinal'):
        self.num_attributes = num_attributes
        self.cat_attributes = cat_attributes
        self.impute_data = impute_data
        self.scale = scale
        self.encode = encode
        self.strategy = strategy
        self.scale_mode = scale_mode.lower()
        self.encode_mode = encode_mode.lower()
        
    def fit(self, X, y=None):
        return self

    def _validate_attributes(self, X):
        if self.num_attributes is None:
            self.num_attributes = X.select_dtypes(include=np.number).columns.tolist()
        if self.cat_attributes is None:
            self.cat_attributes = X.select_dtypes(exclude=np.number).columns.tolist()

        # Validate that the provided attributes are in the DataFrame
        missing_num_attrs = set(self.num_attributes) - set(X.columns)
        missing_cat_attrs = set(self.cat_attributes) - set(X.columns)
        if missing_num_attrs or missing_cat_attrs:
            raise ValueError(f"Missing attributes in the DataFrame: "
                             f"Numerical - {missing_num_attrs}, Categorical"
                             f" - {missing_cat_attrs}")

    def transform(self, X):
        # if not dataframe construct 
        X= build_data_if(X, to_frame=True, input_name='col',
                         force=True, raise_exception=True  )
        # Validate and auto-detect attributes
        self._validate_attributes(X)

        num_pipeline = 'passthrough'
        cat_pipeline = 'passthrough'
        if self.scale:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.strategy) 
                 if self.impute_data else 'passthrough'),
                ('scaler', StandardScaler() if 
                 self.scale_mode == 'standard' else MinMaxScaler()),
            ])

        if self.encode:
            cat_pipeline = Pipeline([
                ('encoder', OrdinalEncoder() if self.encode_mode == 'ordinal'
                 else OneHotEncoder(sparse_output=True)),
            ])

        transformers = []
        if self.num_attributes:
            transformers.append(('num', num_pipeline, self.num_attributes))
        if self.cat_attributes:
            transformers.append(('cat', cat_pipeline, self.cat_attributes))

        preprocessor = ColumnTransformer(transformers=transformers)
        X_transformed = preprocessor.fit_transform(X)

        # Handling sparse output warning
        if self.encode_mode == 'onehot' and not isinstance(X_transformed, np.ndarray):
            warnings.warn('Output is a sparse matrix due to OneHotEncoder.'
                          ' Consider using .toarray() or .todense() for '
                          'downstream processing.', UserWarning)

        return X_transformed
   
class FeaturizeX(BaseEstimator, TransformerMixin ): 
    """
    Featurize X using K-Means-based featurization.

    This transformer applies K-Means clustering to the input data `X` and 
    augments it with cluster-based features. It provides options for data 
    scaling, dimensionality reduction using PCA, and the ability to split 
    data into training and testing sets.

    Parameters:
    -----------
    n_clusters: int, default=7
        The number of initial clusters for K-Means.

    target_scale: float, default=5.0
        Scales and includes the scaled values in the input data to 
        enhance K-Means clustering.

    n_components: int, optional
        Number of components for dimensionality reduction using PCA. If not 
        specified, dimensionality reduction is not applied.

    random_state: int, Optional
        The state for shuffling the data before clustering.

    split_X_y: bool, default=False
        If `True`, splits the input data `X` and target data `y` into 
        training and testing sets according to the specified `test_ratio`.

    test_ratio: int, default=0.2
        The ratio of data to keep for testing when splitting data into 
        training and testing sets.

    shuffle: bool, default=True
        Shuffles the data before splitting into training and testing sets.

    return_model: bool, default=False
        If `True`, the K-Means featurization model is included in the
        return results.

    to_sparse: bool, default=False
        If `True`, the output data `X` is converted to a sparse matrix. 
        By default,the sparse matrix is in coordinate matrix (COO) format.

    sparsity: str, default='coo'
        The kind of sparse matrix used to convert `X`. It can be 'csr' or
        'coo'. Any other value will return a coordinate matrix unless 
        `to_sparse` is set to `False`.

    Attributes
    -----------
    kmf_model_: KMeansFeaturizer
        The fitted K-Means Featurizer model.

    Examples
    ---------
    >>> import numpy as np
    >>> from gofast.transformers import FeaturizeX
    >>> X = np.random.randn(12, 7)
    >>> y = np.arange(12)
    >>> y[y < 6] = 0
    >>> y[y > 0] = 1  # For binary data

    # Example 1: Basic Usage
    >>> Xtransf = FeaturizeX(to_sparse=False).fit_transform(X)
    >>> Xtransf.shape
    (12, 8)

    # Example 2: Splitting Data
    >>> Xtransf = FeaturizeX(to_sparse=True, split_X_y=True).fit_transform(X, y)
    >>> Xtransf[0].shape, Xtransf[1].shape
    ((9, 8), (3, 8))

    # Example 3: Returning Model
    >>> *_, kmf_model = FeaturizeX(to_sparse=True, return_model=True
                                   ).fit_transform(X, y)
    >>> kmf_model
    <'KMeansFeaturizer': n_clusters=7, target_scale=5, random_state=None, 
    n_components=None>

    >>> import numpy as np
    >>> from gofast.transformers import FeaturizeX
    >>> X = np.random.randn(12, 7); y = np.arange(12)
    >>> y[y < 6] = 0; y[y > 0] = 1  # For binary data
    >>> Xtransf = FeaturizeX(to_sparse=False).fit_transform(X)
    >>> X.shape, Xtransf.shape
    ((12, 7), (12, 8))
    >>> Xtransf = FeaturizeX(to_sparse=True).fit_transform(X, y)
    >>> Xtransf
    (<12x8 sparse matrix of type '<class 'numpy.float64'>'
        with 93 stored elements in COOrdinate format>,
        array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    """

    def __init__(self, 
        n_clusters:int=7, 
        target_scale:float= 5 ,
        random_state:_F|int=None, 
        n_components: int=None,  
        model: _F =None, 
        test_ratio:float|str= .2 , 
        shuffle:bool=True, 
        to_sparse: bool=..., 
        sparsity:str ='coo'  
        ): 
        
        self.n_clusters =n_clusters 
        self.target_scale = target_scale 
        self.random_state= random_state 
        self.n_components = n_components 
        self.model=model 
        self.test_ratio=test_ratio 
        self.shuffle=shuffle 
        self.to_sparse=to_sparse 
        self.sparsity=sparsity 
        
    def fit( self, X, y =None): 
        
        """
        Parameters 
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features. 
            Note that when `n_components` is set, sparse matrix for `X` is not 
            acceptable. 

        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Return 
        ---------
        self: For chaining methods. 
        
        """
        
        return self 
    
    def transform (self, X, y=None ): 
        """ 
        Parameters 
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features. 
            Note that when `n_components` is set, sparse matrix for `X` is not 
            acceptable. 

        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns 
        -------- 
        X : NDArray shape (m_samples, n_features +1) or \
            shape (m_samples, n_sparse_features)
            Returns transformed array X NDArray of m_features plus the clusters
            features from KMF featurization procedures. The `n_sparse_features`
            is created if `to_sparse` is set to ``True``. 

        """
        
        ( Xtransf,
         * _ 
            ) =  _featurize_X(
                X, 
                y =y, 
                n_cluster = self.n_clusters, 
                target_scale=self.target_scale, 
                random_state= self.random_state,
                n_components = self.n_components, 
                model=self.model,
                test_ratio=self.test_ratio,
                shuffle=self.shuffle,
                to_sparse=self.to_sparse,
                sparsity=self.sparsity,
            )
        
        return Xtransf 
        
def _featurize_X (
    X, 
    y =None, *, 
    n_clusters:int=7, 
    target_scale:float= 5 ,
    random_state:_F|int=None, 
    n_components: int=None,  
    model: _F =None, 
    split_X_y:bool = False,
    test_ratio:float|str= .2 , 
    shuffle:bool=True, 
    return_model:bool=...,
    to_sparse: bool=..., 
    sparsity:str ='coo' 
    ): 
    """
    Featurize X using K-Means based featurization.
    
    Parameters:
    -----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training vector, where `n_samples` is the number of samples, and
        `n_features` is the number of features. Note that when `n_components` 
        is set, a sparse matrix for `X` is not acceptable.
    
    y : array-like of shape (n_samples,)
        The target vector relative to X.
    
    n_clusters: int, default=7
        The number of initial clusters.
    
    target_scale: float, default=5.0
        Apply appropriate scaling and include it in the input data to k-means.
    
    n_components: int, optional
        The number of components for reducing the predictor X. It uses PCA to reduce
        the dimension to the most important features.
    
    model: :class:`KMeansFeaturizer`
        K-Means Featurizer model. The model can be provided to featurize test data
        separated from the train data. (added in version 0.2.4)
    
    random_state: int, Optional
        The state for shuffling the data.
    
    split_X_y: bool, default=False
        Split the X, y into training data and test data according to the test size.
    
    test_ratio: int, default=0.2
        The ratio to keep for test data.
    
    shuffle: bool, default=True
        Shuffle the dataset.
    
    return_model: bool, default=False
        If `True`, return the K-Means featurization model along with the 
        transformed X.
    
    to_sparse: bool, default=False
        Convert X data to a sparse matrix. By default, the sparse matrix is in
        coordinate matrix (COO) format.
    
    sparsity: str, default='coo'
        The kind of sparse matrix used to convert `X`. It can be 'csr' or 'coo'. Any
        other value will return a coordinate matrix unless `to_sparse` is set to `False`.
    
    Returns:
    --------
    X, y: NDArray of shape (m_samples, n_features + 1) or shape \
        (m_samples, n_sparse_features)
        Returns an NDArray of m_features plus the cluster features from K-Means
        featurization procedures. The `n_sparse_features` are created if `to_sparse`
        is set to `True`.
    
    X, y, model: NDarray and K-Means Featurizer models
        Returns the transformed array X and y and the model if `return_model`
        is set to `True`.
    
    X, Xtest, y, ytest: NDArray (K-Means Featurizer), ArrayLike
        A split tuple is returned when `split_X_y` is set to `True`.
    
    Note:
    -----
    Whenever `return_model=True`, the K-Means Featurizer model 
    (:class:`KMeansFeaturizer`) is included in the return results.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from gofast.transformers import featurize_X
    >>> X = np.random.randn(12, 7); y = np.arange(12)
    >>> y[y < 6] = 0; y[y > 0] = 1  # For binary data
    >>> Xtransf, _ = featurize_X(X, to_sparse=False)
    >>> X.shape, Xtransf.shape
    ((12, 7), (12, 8))
    >>> Xtransf, y = featurize_X(X, y, to_sparse=True)
    >>> Xtransf, y
    (<12x8 sparse matrix of type '<class 'numpy.float64'>'
        with 93 stored elements in COOrdinate format>,
        array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    >>> featurize_X(X, y, to_sparse=True, split_X_y=True)
    (<9x8 sparse matrix of type '<class 'numpy.float64'>'
        with 71 stored elements in COOrdinate format>,
        <3x8 sparse matrix of type '<class 'numpy.float64'>'
        with 24 stored elements in COOrdinate format>,
        array([0, 1, 1, 0, 0, 0, 0, 1, 1]),
        array([0, 1, 1]))
    >>> *_, kmf_model = featurize_X(X, y, to_sparse=True, return_model=True)
    >>> kmf_model
    <'KMeansFeaturizer': n_clusters=7, target_scale=5, random_state=None,
    n_components=None>
    """

    # set False to value use 
    # ellipsis...
    return_model, to_sparse  =ellipsis2false(return_model, to_sparse )

    # if sparse convert X  to sparse matrix 
    if to_sparse: 
        sparsity= str(sparsity).lower().strip() 
        d_sparsity  = dict ( csr =  sparse.csr_matrix , 
            coo=  sparse.coo_matrix ) 
        sparse_func = sparse.coo_matrix  if sparsity not in (
            'coo', 'csr')  else d_sparsity.get (sparsity ) 
    
    # reduce down feature to two. 
    kmf_data = []
    if n_components: 
        from gofast.analysis import nPCA 
        X =nPCA (X, n_components = n_components  ) 
        
    if split_X_y: 
        X, test_data , y, y_test = train_test_split ( 
            X, y ,test_size = assert_ratio(test_ratio) , 
            random_state = random_state ,
            shuffle =shuffle)
        
    # create a kmeaturization with hint model
    if model: 
        if get_estimator_name(model ) !='KMeansFeaturizer': 
            raise EstimatorError(
                "Wrong model estimator. Expect 'KMeansFeaturizer'"
                f" as the valid estimator. Got {get_estimator_name (model)!r}")
            
        if callable ( model ): 
            model = model (n_clusters=n_clusters, 
            target_scale=target_scale, 
            random_state = random_state)
    else: 
        model = KMeansFeaturizer(
            n_clusters=n_clusters, 
            target_scale=target_scale, 
            random_state = random_state, 
            ).fit(X,y)
        
        ### Use the k-means featurizer to generate cluster features
        # transf_cluster = model.transform(X)
        # Xkmf= np.concatenate (
        #    (X, transf_cluster), axis =1 )
        Xkmf = model.transform(X)
        ### Form new input features with cluster features
        # training_with_cluster

    if to_sparse: 
        Xkmf= sparse_func(Xkmf )

    kmf_data.append(Xkmf)
    kmf_data.append(y) 
    if split_X_y: 
        test_with_cluster= model.transform(test_data)
        if sparse: 
            test_with_cluster= sparse_func(test_with_cluster)
 
        kmf_data.insert(1,test_with_cluster )
        kmf_data.append( y_test)

        
    return  tuple (kmf_data ) + (model, ) \
        if return_model else tuple(kmf_data )


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transform text data into TF-IDF features.

    Parameters
    ----------
    max_features : int, default=1000
        Maximum number of features to extract with TF-IDF.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Vectorizer used for converting text data into TF-IDF features.

    Examples
    --------
    >>> text_data = ['sample text data', 'another sample text']
    >>> extractor = TextFeatureExtractor(max_features=500)
    >>> features = extractor.fit_transform(text_data)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the text data using TfidfVectorizer.

    transform(X, y=None)
        Transform the input text data into TF-IDF features.

    """
    def __init__(self, max_features=1000):
        """
        Initialize the TextFeatureExtractor.

        Parameters
        ----------
        max_features : int, default=1000
            Maximum number of features to extract with TF-IDF.

        """
        self.max_features=max_features 
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the text data using TfidfVectorizer.

        Parameters
        ----------
        X : list or array-like, shape (n_samples,)
            Text data to be transformed.

        y : array-like, shape (n_samples,), optional, default=None
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features)
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input text data into TF-IDF features.

        Parameters
        ----------
        X : list or array-like, shape (n_samples,)
            Text data to be transformed.

        y : array-like, shape (n_samples,), optional, default=None
            Target values. Not used in this transformer.

        Returns
        -------
        X_tfidf : sparse matrix, shape (n_samples, n_features)
            Transformed data with TF-IDF features.

        """
        return self.vectorizer.transform(X)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract year, month, and day features from date columns.

    Parameters
    ----------
    date_format : str, default="%Y-%m-%d"
        The date format to use for parsing date columns.

    Examples
    --------
    >>> date_data = pd.DataFrame({'date': ['2021-01-01', '2021-02-01']})
    >>> extractor = DateFeatureExtractor()
    >>> features = extractor.fit_transform(date_data)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the date data.

    transform(X, y=None)
        Transform the input date data into year, month, and day features.

    """
    def __init__(self, date_format="%Y-%m-%d"):
        """
        Initialize the DateFeatureExtractor.

        Parameters
        ----------
        date_format : str, default="%Y-%m-%d"
            The date format to use for parsing date columns.

        """
        self.date_format = date_format
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the date data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Date data to be transformed.

        y : array-like, shape (n_samples,), optional, default=None
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input date data into year, month, and day features.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Date data to be transformed.

        y : array-like, shape (n_samples,), optional, default=None
            Target values. Not used in this transformer.

        Returns
        -------
        new_X : DataFrame, shape (n_samples, n_features * 3)
            Transformed data with year, month, and day features for each date column.

        """
        new_X = X.copy()
        for col in X.columns:
            new_X[col + '_year'] = pd.to_datetime(X[col], format=self.date_format).dt.year
            new_X[col + '_month'] = pd.to_datetime(X[col], format=self.date_format).dt.month
            new_X[col + '_day'] = pd.to_datetime(X[col], format=self.date_format).dt.day
        return new_X



class FeatureSelectorByModel(BaseEstimator, TransformerMixin):
    """
    Select features based on importance weights of a model.

    Parameters
    ----------
    estimator : estimator object, default=RandomForestClassifier()
        The base estimator from which the transformer is built.

    threshold : string, float, optional, default='mean'
        The threshold value to use for feature selection.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification()
    >>> selector = FeatureSelectorByModel()
    >>> X_reduced = selector.fit_transform(X, y)

    Methods
    -------
    fit(X, y)
        Fit the transformer to the data using the provided estimator.

    transform(X, y=None)
        Transform the input data by selecting features based on 
        importance weights.

    """
    def __init__(self, estimator=None, threshold='mean'):
        """
        Initialize the FeatureSelectorByModel.

        Parameters
        ----------
        estimator : estimator object, default=RandomForestClassifier()
            The base estimator from which the transformer is built.

        threshold : string, float, optional, default='mean'
            The threshold value to use for feature selection.

        """
        self.estimator =estimator 
        self.threshold = threshold 
        
    def fit(self, X, y):
        """
        Fit the transformer to the data using the provided 
        estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        if self.estimator is None:
            self.estimator = RandomForestClassifier()
        self.selector = SelectFromModel(
            self.estimator, threshold=self.threshold)
        
        self.selector.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input data by selecting features based on 
        importance weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_reduced : array-like, shape (n_samples, n_selected_features)
            Transformed data with selected features.

        """
        return self.selector.transform(X)

class PolynomialFeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Generate polynomial and interaction features.

    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial features.

    interaction_only : bool, default=False
        If True, only interaction features are produced.

    Examples
    --------
    >>> X = np.arange(6).reshape(3, 2)
    >>> combiner = PolynomialFeatureCombiner(degree=2)
    >>> X_poly = combiner.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using PolynomialFeatures.

    transform(X, y=None)
        Transform input data by generating polynomial and 
        interaction features.

    """
    def __init__(self, degree=2, interaction_only=False):
        """
        Initialize the PolynomialFeatureCombiner.

        Parameters
        ----------
        degree : int, default=2
            The degree of the polynomial features.

        interaction_only : bool, default=False
            If True, only interaction features are produced.

        """
        self.degree=degree 
        self.interaction_only= interaction_only 
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using PolynomialFeatures.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.poly = PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only)
        
        self.poly.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform input data by generating polynomial and interaction 
        features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_poly : array-like, shape (n_samples, n_poly_features)
            Transformed data with polynomial and interaction features.

        """
        return self.poly.transform(X)

class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Reduce dimensionality of the data using PCA.

    Parameters
    ----------
    n_components : int, float, None or str
        Number of components to keep. If n_components is not set, 
        all components are kept.

    Examples
    --------
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> reducer = DimensionalityReducer(n_components=1)
    >>> X_reduced = reducer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using PCA.

    transform(X, y=None)
        Transform input data by reducing dimensionality using PCA.

    """
    def __init__(self, n_components=0.95):
        """
        Initialize the DimensionalityReducer.

        Parameters
        ----------
        n_components : int, float, None or str, default=0.95
            Number of components to keep. If n_components is not set,
            all components are kept.

        """
        self.n_components = n_components 
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using PCA.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.reducer = PCA(n_components=self.n_components)
        self.reducer.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform input data by reducing dimensionality using PCA.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_reduced : array-like, shape (n_samples, n_components)
            Transformed data with reduced dimensionality.

        """
        return self.reducer.transform(X)


class BaseCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric array.

    Parameters
    ----------
    drop : {'first', 'if_binary', None}, default='first'
        Specifies a methodology to use to drop one of the categories per 
        feature.

    Attributes
    ----------
    encoder_ : OneHotEncoder object
        The fitted OneHotEncoder instance.

    Examples
    --------
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = CategoricalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using OneHotEncoder.

    transform(X, y=None)
        Transform categorical features into one-hot numeric arrays.

    """
    def __init__(self, drop='first'):
        """
        Initialize the CategoricalEncoder.

        Parameters
        ----------
        drop : {'first', 'if_binary', None}, default='first'
            Specifies a methodology to use to drop one of the categories per 
            feature.

        """
        self.encoder = OneHotEncoder(drop=drop)
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using OneHotEncoder.

        Parameters
        ----------
        X : list or array-like, shape (n_samples, n_features)
            Training data containing categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.encoder.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform categorical features into one-hot numeric arrays.

        Parameters
        ----------
        X : list or array-like, shape (n_samples, n_features)
            Input data containing categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_encoded : sparse matrix, shape (n_samples, n_encoded_features)
            Transformed data with one-hot encoding for categorical features.

        """
        return self.encoder.transform(X)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric array.

    This transformer should be applied to categorical features in a dataset 
    before applying it to a machine learning model.

    Parameters
    ----------
    categorical_features : list of str, 
        List of column names to be considered as categorical features.
        If None, features should be detected automatically.

    drop : {'first', 'if_binary', None}, default=None
        Specifies a methodology to use to drop one of the categories per feature.

    Attributes
    ----------
    encoders_ : dict of {str: OneHotEncoder}
        Dictionary containing the OneHotEncoders for each categorical feature.

    Examples
    --------
    >>> from sklearn.compose import ColumnTransformer
    >>> transformer = ColumnTransformer(transformers=[
    ...     ('cat', CategoricalEncoder(categorical_features=['color', 'brand']),
             ['color', 'brand'])
    ... ])
    >>> X = pd.DataFrame({'color': ['red', 'blue', 'green'],
                          'brand': ['ford', 'toyota', 'bmw']})
    >>> transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using OneHotEncoder for each specified
        categorical feature.

    transform(X, y=None)
        Transform categorical features into one-hot numeric arrays.

    """
    def __init__(self, categorical_features=None, drop=None):
        """
        Initialize the CategoricalEncoder2.

        Parameters
        ----------
        categorical_features : list of str
            List of column names to be considered as categorical features.

        drop : {'first', 'if_binary', None}, default=None
            Specifies a methodology to use to drop one of the categories 
            per feature.

        """
        self.categorical_features = categorical_features
        self.drop = drop
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using OneHotEncoder for each
        specified categorical feature.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        is_frame(X, df_only= True, raise_exception=True,
                 objname="CategoricalEncoder")
        
        if self.categorical_features is None: 
            *_, self.categorical_features = to_numeric_dtypes(
                X, return_feature_types= True )
            
        self.encoders_ = {
            feature: OneHotEncoder(drop=self.drop) 
            for feature in self.categorical_features}
        
        for feature in self.categorical_features:
            self.encoders_[feature].fit(X[[feature]])
            
        return self
    
    def transform(self, X, y=None):
        """
        Transform categorical features into one-hot numeric arrays.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_encoded : array-like, shape (n_samples, n_encoded_features)
            Transformed data with one-hot encoding for categorical features.

        """
        outputs = []
        for feature in self.categorical_features:
            outputs.append(self.encoders_[feature].transform(X[[feature]]))
        return np.hstack(outputs)

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.

    This transformer should be applied to numeric features in a dataset before
    applying it to a machine learning model.

    Parameters
    ----------
    numeric_features : list of str
        List of column names to be considered as numeric features.

    Attributes
    ----------
    scaler_ : StandardScaler
        The instance of StandardScaler used for scaling.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> numeric_features = ['age', 'income']
    >>> pipeline = Pipeline(steps=[
    ...     ('scaler', FeatureScaler(numeric_features=numeric_features))
    ... ])
    >>> X = pd.DataFrame({'age': [25, 35, 50], 'income': [50000, 80000, 120000]})
    >>> pipeline.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using StandardScaler for specified 
        numeric features.

    transform(X, y=None)
        Standardize specified numeric features by removing the mean and 
        scaling to unit variance.

    """
    def __init__(self, numeric_features=None ):
        """
        Initialize the FeatureScaler2.

        Parameters
        ----------
        numeric_features : list of str
            List of column names to be considered as numeric features.

        """
        self.numeric_features = numeric_features
        self.scaler_ = StandardScaler()
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using StandardScaler for specified 
        numeric features.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the numeric features to be scaled.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self._is_frame= isinstance (X, pd.DataFrame)
        return self
    
    def transform(self, X):
        """
        Standardize specified numeric features by removing the mean and 
        scaling to unit variance.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the numeric features to be scaled.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_scaled : DataFrame, shape (n_samples, n_features)
            DataFrame with standardized numeric features.

        """
        X = build_data_if(X, input_name='scale_col', force=True, 
                          raise_exception= True )
        
        if self.numeric_features is None: 
            _, self.numeric_features, _ = to_numeric_dtypes(
                X, return_feature_types= True )
        
        exist_features(X, features= self.numeric_features)
        
        self.scaler_.fit(X[self.numeric_features])
        
        X_transformed= X.copy()
        X_transformed[self.numeric_features] = ( 
            self.scaler_.transform(X_transformed[self.numeric_features])
            )
        if not self._is_frame: 
            X_transformed = np.asarray(X_transformed)
            
        return X_transformed


class BaseFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Attributes
    ----------
    scaler_ : StandardScaler object
        The fitted StandardScaler instance.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = FeatureScaler()
    >>> X = [[0, 15], [1, -10]]
    >>> scaler.fit(X)
    >>> scaler.transform(X)
    """

    def __init__(self):
        """
        Initialize the FeatureScaler.

        """
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using StandardScaler.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data containing the features to be scaled.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.scaler.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Standardize features by removing the mean and scaling to unit variance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data containing the features to be scaled.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_scaled : array-like, shape (n_samples, n_features)
            Transformed data with standardized features.

        """
        return self.scaler.transform(X)


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputation transformer for completing missing values.

    This transformer can be applied to both numeric and categorical features 
    in a dataset to impute missing values using the mean, median, or most 
    frequent value.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy. If "mean", then replace missing values 
        using the mean along each column. Can also be "median" or "most_frequent".

    Attributes
    ----------
    imputer_ : SimpleImputer
        The instance of SimpleImputer used for imputation.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> imputer = MissingValueImputer(strategy='mean')
    >>> X = pd.DataFrame({'age': [25, np.nan, 50], 'income': [50000, 80000, np.nan]})
    >>> imputer.fit_transform(X)
    >>> imputer = MissingValueImputer(strategy='mean')
    >>> X = [[1, 2], [np.nan, 3], [7, 6]]
    >>> imputer.fit(X)
    >>> imputer.transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. No actual computation is needed for 
        this transformer.

    transform(X, y=None)
        Impute missing values using the specified strategy.

    """
    def __init__(self, strategy='mean'):
        """
        Initialize the MissingValueImputer2.

        Parameters
        ----------
        strategy : str, default='mean'
            The imputation strategy. If "mean", then replace missing values 
            using the mean along each column. Can also be "median" or 
            "most_frequent".

        """
        self.strategy = strategy
        self.imputer_ = SimpleImputer(strategy=strategy)
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. No actual computation is needed 
        for this transformer.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the features to be imputed.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.columns_=[] 
        if isinstance( X, pd.DataFrame): 
            self.columns_= list(X.columns)
            
        self.imputer_.fit(X)
        return self
    
    def transform(self, X):
        """
        Impute missing values using the specified strategy.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the features to be imputed.
        Returns
        -------
        X_transformed : DataFrame, shape (n_samples, n_features)
            DataFrame with missing values imputed based on the chosen strategy.

        """
        # Apply imputation transformation
        X_transformed = self.imputer_.transform(X)
        
        try: 
            if hasattr(self, 'columns_') and self.columns_:
                X_transformed = pd.DataFrame(
                    X_transformed, columns=self.columns_)
        except:
            # Fallback if columns_ attribute doesn't exist or is empty
            X_transformed = pd.DataFrame(
                X_transformed) if self.columns else X_transformed 
    
        return X_transformed

class BaseColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select columns from a DataFrame for processing.

    Parameters
    ----------
    column_names : list of str
        List of column names to select.

    Examples
    --------
    >>> import pandas as pd
    >>> selector = ColumnSelector(column_names=['A', 'B'])
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> selector.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. No actual computation is needed 
        for this transformer.

    transform(X, y=None)
        Select specified columns from the input DataFrame.

    """
    def __init__(self, column_names):
        """
        Initialize the ColumnSelector.

        Parameters
        ----------
        column_names : list of str
            List of column names to select.

        """
        self.column_names = column_names
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. No actual computation is needed for
        this transformer.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the columns to be selected.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Select specified columns from the input DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the columns to be selected.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_selected : DataFrame, shape (n_samples, n_selected_features)
            DataFrame with only the selected columns.

        """
        if isinstance(X, pd.DataFrame):
            return X[self.column_names]
        else:
            raise TypeError("Input must be a pandas DataFrame")

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select columns from a DataFrame for processing.

    Parameters
    ----------
    column_names : list of str
        List of column names to select.

    Examples
    --------
    >>> selector = ColumnSelector(column_names=['age', 'income'])
    >>> X = pd.DataFrame({'age': [25, 35, 50], 'income': [50000, 80000, 120000], 
                          'gender': ['male', 'female', 'male']})
    >>> selector.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. Checks whether the specified column 
        names exist in the DataFrame.

    transform(X, y=None)
        Select specified columns from the input DataFrame.

    """
    def __init__(self, column_names):
        """
        Initialize the ColumnSelector2.

        Parameters
        ----------
        column_names : list of str
            List of column names to select.

        """
        self.column_names = column_names
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. Checks whether the specified 
        column names exist in the DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the columns to be selected.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        exist_features(X, self.column_names)
        return self
    
    def transform(self, X, y=None):
        """
        Select specified columns from the input DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the columns to be selected.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_selected : DataFrame, shape (n_samples, n_selected_features)
            DataFrame with only the selected columns.

        """
        is_frame (X, df_only=True, raise_exception= True,
                  objname="LogTransformer" ) 
        
        return X[self.column_names]

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a natural logarithm transformation to numeric features.

    Use this transformer on skewed numeric features to reduce 
    their skewness.

    Parameters
    ----------
    numeric_features : list of str
        List of column names to be considered as numeric features
        for log transformation.

    epsilon : float, default=1e-6
        A small constant to add to input data to avoid taking log of zero.

    Examples
    --------
    >>> transformer = LogTransformer(numeric_features=['income'],
                                     epsilon=1e-6)
    >>> X = pd.DataFrame({'income': [50000, 80000, 120000]})
    >>> transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. No actual computation is needed for 
        this transformer.

    transform(X, y=None)
        Apply the natural logarithm transformation to specified numeric features.

    """
    def __init__(self, numeric_features, epsilon=1e-6):
        """
        Initialize the LogTransformer.

        Parameters
        ----------
        numeric_features : list of str
            List of column names to be considered as numeric features
            for log transformation.

        epsilon : float, default=1e-6
            A small constant to add to input data to avoid taking log of zero.

        """
        self.numeric_features = numeric_features
        self.epsilon = epsilon
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. No actual computation is needed for 
        this transformer.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the columns to be transformed.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Apply the natural logarithm transformation to specified numeric features.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the columns to be transformed.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_transformed : DataFrame, shape (n_samples, n_features)
            DataFrame with the natural logarithm transformation applied to 
            specified numeric features.

        """
        is_frame (X, df_only=True, raise_exception= True,
                  objname="LogTransformer" ) 
            
        X_transformed = X.copy()
        for feature in self.numeric_features:
            X_transformed[feature] = np.log(X_transformed[feature] + self.epsilon)
        return X_transformed


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract common statistical features from time series data for each column.

    Parameters
    ----------
    rolling_window : int
        The size of the moving window to compute the rolling statistics.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> extractor = TimeSeriesFeatureExtractor(rolling_window=5)
    >>> X = pd.DataFrame({'time_series': np.random.rand(100)})
    >>> features = extractor.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. No actual computation is needed for 
        this transformer.

    transform(X, y=None)
        Extract common statistical features from time series data for each column.

    Notes
    -----
    TimeSeriesFeatureExtractor is a transformer that extracts common statistical 
    features from time series data for each column. These features include the
    rolling mean, rolling standard deviation, rolling minimum, rolling maximum,
    and rolling median.

    The `rolling_window` parameter specifies the size of the moving window used
    to compute the rolling statistics. Larger window sizes result in smoother
    statistical features.

    """
    def __init__(self, rolling_window):
        """
        Initialize the TimeSeriesFeatureExtractor.

        Parameters
        ----------
        rolling_window : int
            The size of the moving window to compute the rolling statistics.

        """
        self.rolling_window = rolling_window
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. No actual computation is needed
        for this transformer.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the time series data.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Extract common statistical features from time series data for each column.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the time series data to extract features from.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        features : DataFrame, shape (n_samples, n_features * 5)
            DataFrame with extracted statistical features for each column.
            Features include rolling mean, rolling standard deviation, 
            rolling minimum, rolling maximum, and rolling median.

        """
        X = build_data_if (
            X, to_frame=True, force=True, 
           raise_warning='mute', input_name='tsfe')
        # Rolling statistical features
        return X.rolling(window=self.rolling_window).agg(
            ['mean', 'std', 'min', 'max', 'median'])


class CategoryFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables based on the frequency of each category.

    Parameters
    ----------
    categorical_features : list of str
        List of column names to be considered as categorical features.

    Examples
    --------
    >>> encoder = CategoryFrequencyEncoder(categorical_features=['brand'])
    >>> X = pd.DataFrame({'brand': ['apple', 'apple', 'samsung', 'samsung',
                                    'nokia']})
    >>> encoded_features = encoder.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data by calculating the frequency of 
        each category in the categorical features.

    transform(X, y=None)
        Encode categorical features based on the calculated frequency mappings.

    Notes
    -----
    CategoryFrequencyEncoder is a transformer that encodes categorical
    variables based on the frequency of each category. It replaces categorical 
    values with their corresponding frequency values, allowing machine learning
    models to capture relationships based on category frequencies.

    The `fit` method calculates the frequency of each category in the specified
    categorical features and stores the mappings. The `transform` method 
    encodes the categorical features using these mappings.

    """
    def __init__(self, categorical_features):
        """
        Initialize the CategoryFrequencyEncoder.

        Parameters
        ----------
        categorical_features : list of str
            List of column names to be considered as categorical features.

        """
        self.categorical_features = categorical_features
        self.frequency_maps_ = None
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data by calculating the frequency of 
        each category in the categorical features.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the categorical features.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        X = build_data_if (X, to_frame=True, force =True, 
                           raise_warning='mute', input_name='cf')
        self.frequency_maps_ = {
            feature: X[feature].value_counts(normalize=True).to_dict() 
            for feature in self.categorical_features
        }
        return self
    
    def transform(self, X, y=None):
        """
        Encode categorical features based on the calculated frequency mappings.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        encoded_data : DataFrame, shape (n_samples, n_features)
            DataFrame with categorical features encoded based on category frequencies.

        """
        X_transformed = X.copy()
        for feature in self.categorical_features:
            X_transformed[feature] = X[feature].map(self.frequency_maps_[feature])
        return X_transformed

class DateTimeCyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode datetime columns as cyclical features using sine and cosine
    transformations.

    Parameters
    ----------
    datetime_features : list of str
        List of datetime column names to be encoded as cyclical.

    Examples
    --------
    >>> encoder = DateTimeCyclicalEncoder(datetime_features=['timestamp'])
    >>> X = pd.DataFrame({'timestamp': pd.date_range(start='1/1/2018', 
                                                     periods=24, freq='H')})
    >>> encoded_features = encoder.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X, y=None)
        Encode datetime columns as cyclical features using sine and cosine 
        transformations.

    Notes
    -----
    DateTimeCyclicalEncoder is a transformer that encodes datetime columns as 
    cyclical features using sine and cosine transformations. This encoding 
    is useful for capturing cyclical patterns in time-based data.

    The `fit` method is a no-op, as this transformer does not have any 
    trainable parameters. The `transform` method encodes the specified 
    datetime columns as cyclical features, adding sine and cosine components
    for the hour of the day.

    """
    def __init__(self, datetime_features):
        """
        Initialize the DateTimeCyclicalEncoder.

        Parameters
        ----------
        datetime_features : list of str
            List of datetime column names to be encoded as cyclical.

        """
        self.datetime_features = datetime_features
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Encode datetime columns as cyclical features using sine and 
        cosine transformations.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing datetime columns to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        encoded_data : DataFrame, shape (n_samples, n_features + 2 * num_datetime_features)
            DataFrame with datetime columns encoded as cyclical features using 
            sine and cosine transformations.

        """
        X = build_data_if ( X, raise_warning ='mute', to_frame=True,
                           force=True, input_name ='dtc')
        X_transformed = X.copy()
        for feature in self.datetime_features:
            dt_col = pd.to_datetime(X_transformed[feature])
            X_transformed[feature + '_sin_hour'] = np.sin(
                2 * np.pi * dt_col.dt.hour / 24)
            X_transformed[feature + '_cos_hour'] = np.cos(
                2 * np.pi * dt_col.dt.hour / 24)
        return X_transformed

class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates lag features for time series data to help capture temporal dependencies,
    enhancing model performance by providing historical context.

    This transformer is designed to work with time series data where capturing
    patterns based on past values is crucial. It can create one or multiple lag
    features based on the specified lag periods.

    Parameters
    ----------
    lags : int or list of ints
        Specifies the lag periods for which features should be generated.
        For example, `lags=1` generates a single lag feature (t-1) for each
        data point. A list, e.g., `lags=[1, 2, 3]`, generates multiple lag
        features (t-1, t-2, t-3).
    time_col : str, optional
        The name of the column representing time in the input DataFrame.
        This parameter allows specifying which column to treat as the temporal
        dimension. If None (default), the DataFrame's index is used.
    value_col : str, optional
        Specifies the column from which to generate lag features. Useful for
        DataFrames with multiple columns when only one contains the time series
        data. If None (default), the first column of the DataFrame is used.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.pipeline import make_pipeline
    >>> from gofast.transformers import LagFeatureGenerator
    >>> np.random.seed(0)  # For reproducible output
    >>> X = pd.DataFrame({
    ...     'time': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    ...     'value': np.random.randn(100).cumsum()
    ... })
    >>> generator = LagFeatureGenerator(lags=[1, 2, 3], time_col='time', value_col='value')
    >>> lag_features = generator.fit_transform(X)
    >>> print(lag_features.head())

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and does nothing during fitting.

    transform(X, y=None)
        Generates lag features for the specified value column in the input
        DataFrame, creating new columns for each specified lag period.

    Notes
    -----
    The LagFeatureGenerator is a transformer that can be an essential part of
    the preprocessing pipeline for time series forecasting models. By introducing
    lag features, it allows models to leverage historical data, potentially
    improving forecast accuracy by capturing seasonal trends, cycles, and other
    temporal patterns.
    """
    def __init__(self, lags, time_col=None, value_col=None):
        self.lags = np.atleast_1d(lags).astype(int)
        self.time_col = time_col
        self.value_col = value_col

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self

    def transform(self, X, y=None):
        """
        Generate lag features for the input DataFrame to capture temporal 
        dependencies.

        Parameters
        ----------
        X : DataFrame
            Input DataFrame containing columns for which to generate lag
            features.
        y : Not used.

        Returns
        -------
        DataFrame
            DataFrame with lag features added to capture temporal dependencies.
        """
        X = build_data_if (
            X, 
            to_frame =True, 
            raise_warning='silence', 
            force=True, 
            input_name='lf' 
        )
        
        if self.time_col and self.time_col not in X.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in input data.")
        if self.value_col and self.value_col not in X.columns:
            self.value_col = X.columns[0]  # Default to first column if not specified
        
        # Use specified value column or default to first column
        value_data = X[self.value_col] if self.value_col else X.iloc[:, 0]

        # Generate lag features
        for lag in self.lags:
            X[f'lag_{lag}'] = value_data.shift(lag)

        return X
 
class DifferencingTransformer(BaseEstimator, TransformerMixin):
    """
    Initializes the DifferencingTransformer, a transformer designed to
    apply differencing to time series data, making it stationary by removing
    trends and seasonality. Differencing involves subtracting the current value
    from a previous value at a specified lag.

    Parameters
    ----------
    periods : int, default=1
        Specifies the lag, i.e., the number of periods to shift for calculating
        the difference. A period of 1 means subtracting the current value from
        the immediately preceding value, and so on.
    
    zero_first : bool, default=False
        If True, the first value after differencing, which is NaN due to the
        shifting, is replaced with zero. This option is useful for models that
        cannot handle NaN values and ensures the output series has the same
        length as the input.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from gofast.transformers import DifferencingTransformer
    >>> np.random.seed(42)  # For reproducible output
    >>> X = pd.DataFrame({'value': np.random.randn(100).cumsum()})
    >>> transformer = DifferencingTransformer(periods=1, zero_first=True)
    >>> stationary_data = transformer.fit_transform(X)
    >>> print(stationary_data.head())

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and thus the fitting process does nothing except for input
        validation.
    
    transform(X, y=None)
        Applies differencing to the input DataFrame to produce a stationary
        series by removing trends and seasonality.

    Notes
    -----
    Differencing is a common preprocessing step for time series forecasting,
    where data must be stationary to meet the assumptions of various
    forecasting models. The DifferencingTransformer simplifies this process,
    making it easy to integrate into a preprocessing pipeline.
    """
    def __init__(self, periods=1, zero_first=False):
        """
        Initialize the DifferencingTransformer.

        Parameters
        ----------
        periods : int, default=1
            The number of periods to shift for calculating the difference.
        replace_first_with_zero : bool, default=False
            If True, replaces the first differenced value with zero to avoid 
            NaN values.
            This can be useful when the differenced series is used as input to 
            models that do not support NaN values.

        """
        self.periods = periods
        self.zero_first = zero_first
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Apply differencing to the input DataFrame to make it stationary.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            The input DataFrame containing columns for which to apply differencing.
            
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        DataFrame, shape (n_samples, n_features)
            The DataFrame with differenced data to make it stationary. If
            `replace_first_with_zero` is True, the first row of the differenced
            data will be set to zero instead of NaN.

        """
        # Ensure X is a DataFrame
        X = build_data_if (
            X, to_frame =True, input_name ='dt', force=True, 
            raise_warning='mute'
        )
        
        # Apply differencing
        differenced_data = X.diff(periods=self.periods)
        
        # Optionally replace the first value with zero
        if self.zero_first:
            differenced_data.iloc[0] = 0.
        
        return differenced_data
    
  
class MovingAverageTransformer(BaseEstimator, TransformerMixin):
    """
    Compute moving average for time series data.

    Parameters
    ----------
    window : int
        Size of the moving window.

    Examples
    --------
    >>> transformer = MovingAverageTransformer(window=5)
    >>> X = pd.DataFrame({'value': np.random.randn(100)})
    >>> moving_avg = transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X, y=None)
        Compute the moving average for each column in the input DataFrame.

    Notes
    -----
    MovingAverageTransformer is a transformer that calculates the moving 
    average of each column in the input DataFrame. It is useful for smoothing
    time series data to identify trends and patterns.

    The `fit` method is a no-op, as this transformer does not have any 
    trainable parameters. The `transform` method calculates the moving 
    average for each column in the DataFrame using a specified moving 
    window size.

    """
    def __init__(self, window):
        """
        Initialize the MovingAverageTransformer.

        Parameters
        ----------
        window : int
            Size of the moving window.

        """
        self.window = window
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Compute the moving average for each column in the input DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing columns for which to compute
            the moving average.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        moving_avg : DataFrame, shape (n_samples, n_features)
            DataFrame with the moving average of each column.

        """
        X = build_data_if(X, to_frame =True, input_name ='mav', force=True, 
                          raise_warning='silence')
        return X.rolling(window=self.window).mean()



class CumulativeSumTransformer(BaseEstimator, TransformerMixin):
    """
    Compute the cumulative sum for each column in the data.

    Parameters
    ----------
    None

    Examples
    --------
    >>> transformer = CumulativeSumTransformer()
    >>> X = pd.DataFrame({'value': np.random.randn(100)})
    >>> cum_sum = transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X, y=None)
        Compute the cumulative sum for each column in the input DataFrame.

    Notes
    -----
    CumulativeSumTransformer is a transformer that computes the cumulative 
    sum of each column in the input DataFrame. It is useful for creating 
    cumulative sums of time series or accumulating data over time.

    The `fit` method is a no-op, as this transformer does not have any 
    trainable parameters. The `transform` method calculates the cumulative 
    sum for each column in the DataFrame.

    """
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Compute the cumulative sum for each column in the input DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing columns for which to compute the 
            cumulative sum.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        cum_sum : DataFrame, shape (n_samples, n_features)
            DataFrame with the cumulative sum of each column.

        """
        X = build_data_if(X, to_frame =True, input_name ='cs', force=True, 
                          raise_warning='silence')
        return X.cumsum()

class SeasonalDecomposeTransformer(BaseEstimator, TransformerMixin):
    """
    Decompose time series data into seasonal, trend, and residual components.

    Parameters
    ----------
    model : str, default='additive'
        Type of seasonal component. Can be 'additive' or 'multiplicative'.

    freq : int, default=1
        Frequency of the time series.

    Examples
    --------
    >>> transformer = SeasonalDecomposeTransformer(model='additive', freq=12)
    >>> X = pd.DataFrame({'value': np.random.randn(100)})
    >>> decomposed = transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X, y=None)
        Decompose time series data into seasonal, trend, and residual components.

    Notes
    -----
    SeasonalDecomposeTransformer is a transformer that decomposes time series
    data into three components: seasonal, trend, and residual. It uses the 
    seasonal decomposition of time series (STL) method to extract these components.

    The decomposition model can be either 'additive' or 'multiplicative', 
    specified using the 'model' parameter. The 'freq' parameter defines the 
    frequency of the time series, which is used to identify the seasonal 
    component.

    The transformed data will have three columns: 'seasonal', 'trend', and 
    'residual', containing the respective components.

    """
    def __init__(self, model='additive', freq=1):
        """
        Initialize the SeasonalDecomposeTransformer transformer.

        Parameters
        ----------
        model : str, default='additive'
            Type of seasonal component. Can be 'additive' or 'multiplicative'.

        freq : int, default=1
            Frequency of the time series.

        Returns
        -------
        None

        """
        self.model = model
        self.freq = freq
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        import_optional_dependency("statsmodels")
        return self
    
    def transform(self, X, y=None):
        """
        Decompose time series data into seasonal, trend, and residual
        components.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input time series data with a single column.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        decomposed : DataFrame, shape (n_samples, 3)
            Dataframe containing the seasonal, trend, and residual 
            components.

        """
        result = seasonal_decompose(X, model=self.model, period=self.freq)
        return pd.concat([result.seasonal, result.trend, result.resid], axis=1)


class FourierFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generate Fourier series terms as features for capturing cyclical 
    patterns in time series data.

    Parameters
    ----------
    periods : list of int
        List of periods to generate Fourier features for.

    Examples
    --------
    >>> transformer = FourierFeaturesTransformer(periods=[12, 24])
    >>> X = pd.DataFrame({'time': np.arange(100)})
    >>> fourier_features = transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and does nothing during fitting.

    transform(X, y=None)
        Generate Fourier series terms as features for the input time series data.

    Notes
    -----
    FourierFeaturesTransformer is a transformer that generates Fourier series 
    terms as features for capturing cyclical patterns in time series data. It 
    computes sine and cosine terms for the specified periods using the 
    following formula:

    .. math::

        \text{{sin}}(2\pi ft) \text{{ and }} \text{{cos}}(2\pi ft)

    where:
        - \(f\) is the frequency corresponding to the period \(T\), 
        calculated as \(f = \frac{1}{T}\).
        - \(t\) is the time index of the data.

    The transformer creates two features for each specified period: one for 
    the sine term and one for the cosine term.

    """
    def __init__(self, periods):
        """
        Initialize the FourierFeaturesTransformer transformer.

        Parameters
        ----------
        periods : list of int
            List of periods to generate Fourier features for.

        Returns
        -------
        None

        """
        self.periods = periods
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Generate Fourier series terms as features for the input time series data.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input time series data with a 'time' column.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        fourier_features : DataFrame, shape (n_samples, 2 * n_periods)
            Dataframe containing the generated Fourier features.

        """
        X = build_data_if (X, to_frame =True, force=True, input_name='ft', 
                           raise_warning ='mute')
        X_transformed = pd.DataFrame(index=X.index)
        for period in self.periods:
            frequency = 1 / period
            X_transformed[f'sin_{period}'] = np.sin(
                2 * np.pi * frequency * X.index)
            X_transformed[f'cos_{period}'] = np.cos(
                2 * np.pi * frequency * X.index)
        return X_transformed


class TrendFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract linear trend features from time series data by fitting a polynomial
    of a specified order. This transformer allows for flexible handling of time
    series data where the time column and value column can be specified, making
    it suitable for datasets where the date or time information is not set as the
    DataFrame index.

    Parameters
    ----------
    order : int, default=1
        The order of the polynomial trend to fit. For example, `order=1` fits
        a linear trend, `order=2` fits a quadratic trend, and so on.
    time_col : str, optional
        The name of the column in the DataFrame that represents the time variable.
        If None (default), the DataFrame's index is used as the time variable.
    value_col : str, optional
        The name of the column in the DataFrame that represents the value to which
        the trend is to be fitted. If None (default), the first column of the DataFrame
        is used.

    Attributes
    ----------
    No public attributes.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from gofast.transformers import TrendFeatureExtractor
    >>> np.random.seed(0)  # For reproducible output
    >>> X = pd.DataFrame({
    ...     'time': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    ...     'value': np.random.randn(100).cumsum()
    ... })
    >>> transformer = TrendFeatureExtractor(order=1, time_col='time', value_col='value')
    >>> trend_features = transformer.fit_transform(X)
    >>> print(trend_features.head())

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer does not have any
        trainable parameters and, therefore, the fitting process does nothing
        except verifying the correctness of the input data.

    transform(X, y=None)
        Extract linear trend features from the provided time series data. The
        method fits a polynomial of the specified order to the time series data
        and returns the trend component as a new DataFrame.

    Notes
    -----
    The TrendFeatureExtractor is designed to work with time series data where the
    temporal ordering is crucial for analysis. By fitting a trend line to the data,
    this transformer helps in analyzing the underlying trend of the time series,
    which can be particularly useful for feature engineering in predictive modeling
    tasks.
    """
    def __init__(self, order=1, time_col=None, value_col=None):
        self.order = order
        self.time_col = time_col
        self.value_col = value_col
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

        Parameters
        ----------
        X : DataFrame
            Training data. Not used in this transformer.
        y : Not used.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Extract linear trend features from time series data.

        Parameters
        ----------
        X : DataFrame
            Input time series data.
        y : Not used.

        Returns
        -------
        trend_features : DataFrame
            DataFrame containing the extracted trend features.

        """
        if self.time_col is None:
            time_data = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Time column '{self.time_col}' not found in input data.")
            time_data = X[self.time_col]
            
        if self.value_col is None:
            value_data = X.iloc[:, 0]
        else:
            if self.value_col not in X.columns:
                raise ValueError(f"Value column '{self.value_col}' not found in input data.")
            value_data = X[self.value_col]

        # Fit and evaluate the polynomial
        trends = np.polyfit(time_data, value_data, deg=self.order)
        trend_poly = np.poly1d(trends)
        trend_features = pd.DataFrame(trend_poly(time_data),
                                       index=X.index if self.time_col is None else None,
                                       columns=[f'trend_{self.order}'])
        
        return trend_features

class ImageResizer(BaseEstimator, TransformerMixin):
    """
    Resize images to a specified size.

    Parameters
    ----------
    output_size : tuple of int
        The desired output size as (width, height).

    Examples
    --------
    >>> from skimage.transform import resize
    >>> resizer = ImageResizer(output_size=(128, 128))
    >>> image = np.random.rand(256, 256, 3)
    >>> resized_image = resizer.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Resize input images to the specified output size.

    Notes
    -----
    ImageResizer is a transformer that resizes input images to a specified 
    output size. It is commonly used for standardizing image dimensions before
    further processing or analysis.

    """
    
    def __init__(self, output_size):
        """
        Initialize the ImageResizer transformer.

        Parameters
        ----------
        output_size : tuple of int
            The desired output size as (width, height).

        Returns
        -------
        None

        """
        self.output_size = output_size
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        import_optional_dependency("skimage", extra=EMSG)
        return self
    
    def transform(self, X):
        """
        Resize input images to the specified output size.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) to be resized.

        Returns
        -------
        resized_images : ndarray, shape (output_height, output_width, channels)
            Resized image(s) with the specified output size.

        """
        from skimage.transform import resize
        return resize(X, self.output_size, anti_aliasing=True)


class ImageNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize images by scaling pixel values to the range [0, 1].

    Parameters
    ----------
    None

    Examples
    --------
    >>> normalizer = ImageNormalizer()
    >>> image = np.random.randint(0, 255, (256, 256, 3), 
                                  dtype=np.uint8)
    >>> normalized_image = normalizer.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no 
        trainable parameters and does nothing during fitting.

    transform(X)
        Normalize input images by scaling pixel values to the range [0, 1].

    Notes
    -----
    ImageNormalizer is a transformer that scales pixel values of input images 
    to the range [0, 1]. This is a common preprocessing step for images 
    before feeding them into machine learning models.

    """
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Normalize input images by scaling pixel values to the range [0, 1].

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) to be normalized.

        Returns
        -------
        normalized_images : ndarray, shape (height, width, channels)
            Normalized image(s) with pixel values scaled to the range [0, 1].

        """
        return X / 255.0



class ImageToGrayscale(BaseEstimator, TransformerMixin):
    """
    Convert images to grayscale.

    Parameters
    ----------
    keep_dims : bool, default=False
        If True, keeps the third dimension as 1 (e.g., (height, width, 1)).

    Examples
    --------
    >>> from skimage.color import rgb2gray
    >>> converter = ImageToGrayscale(keep_dims=True)
    >>> image = np.random.rand(256, 256, 3)
    >>> grayscale_image = converter.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Convert input color images to grayscale.

    Notes
    -----
    ImageToGrayscale is a transformer that converts color images to grayscale.
    Grayscale images have only one channel, while color images typically 
    have three (red, green, and blue). This transformer allows you to 
    control whether the grayscale image should have a single channel or 
    retain a third dimension with a value of 1.

    """
    def __init__(self, keep_dims=False):
        """
        Initialize an ImageToGrayscale transformer.

        Parameters
        ----------
        keep_dims : bool, default=False
            If True, keeps the third dimension as 1 (e.g., (height, width, 1)).

        Returns
        -------
        ImageToGrayscale
            Returns an instance of the ImageToGrayscale transformer.

        """
        self.keep_dims = keep_dims
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        import_optional_dependency ('skimage', extra = EMSG )
        return self
    
    def transform(self, X):
        """
        Convert input color images to grayscale.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input color image(s) to be converted to grayscale.

        Returns
        -------
        grayscale_images : ndarray, shape (height, width, 1) or (height, width)
            Grayscale image(s) after conversion. The output can have a 
            single channel or retain the third dimension with a value of 1, 
            depending on the `keep_dims` parameter.

        """
        from skimage.color import rgb2gray
        grayscale = rgb2gray(X)
        if self.keep_dims:
            grayscale = grayscale[:, :, np.newaxis]
        return grayscale

class ImageAugmenter(BaseEstimator, TransformerMixin):
    """
    Apply random transformations to images for augmentation.

    Parameters
    ----------
    augmentation_funcs : list of callable
        A list of functions that apply transformations to images.

    Examples
    --------
    >>> from imgaug import augmenters as iaa
    >>> augmenter = ImageAugmenter(augmentation_funcs=[
        iaa.Fliplr(0.5), iaa.GaussianBlur(sigma=(0, 3.0))])
    >>> image = np.random.rand(256, 256, 3)
    >>> augmented_image = augmenter.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and does nothing during fitting.

    transform(X)
        Apply random transformations to input images.

    Notes
    -----
    ImageAugmenter is a transformer that applies random augmentations to 
    input images. Data augmentation is commonly used in computer vision 
    tasks to increase the diversity of training data, which can lead to 
    improved model generalization.

    The `augmentation_funcs` parameter allows you to specify a list of 
    callable functions that apply various transformations to the input 
    images. These functions can include operations like flipping, rotating,
    blurring, and more.

    """
    def __init__(self, augmentation_funcs):
        """
        Initialize an ImageAugmenter.

        Parameters
        ----------
        augmentation_funcs : list of callable
            A list of functions that apply transformations to images.

        Returns
        -------
        ImageAugmenter
            Returns an instance of the ImageAugmenter.

        """
        self.augmentation_funcs = augmentation_funcs
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Apply random transformations to input images.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) to which random augmentations will be applied.

        Returns
        -------
        augmented_images : ndarray, shape (height, width, channels)
            Augmented image(s) after applying random transformations.

        """
        for func in self.augmentation_funcs:
            X = func(images=X)
        return X


class ImageChannelSelector(BaseEstimator, TransformerMixin):
    """
    Select specific channels from images.

    Parameters
    ----------
    channels : list of int
        The indices of the channels to select.

    Examples
    --------
    # Selects the first two channels.
    >>> selector = ImageChannelSelector(channels=[0, 1])  
    >>> image = np.random.rand(256, 256, 3)
    >>> selected_channels = selector.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Select specific channels from the input images.

    Notes
    -----
    ImageChannelSelector is a transformer that allows you to select specific
    color channels from input images. In many computer vision tasks, you may 
    only be interested in certain color channels of an image 
    (e.g., grayscale images, or selecting the red and green channels for analysis).

    The `channels` parameter allows you to specify which channels to select 
    from the input images. You can provide a list of channel indices to be
    retained.

    """
    def __init__(self, channels):
        """
        Initialize an ImageChannelSelector.

        Parameters
        ----------
        channels : list of int
            The indices of the channels to select.

        Returns
        -------
        ImageChannelSelector
            Returns an instance of the ImageChannelSelector.

        """
        self.channels = channels
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Select specific channels from the input images.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) from which channels will be selected.

        Returns
        -------
        selected_channels : ndarray, shape (height, width, n_selected_channels)
            Input image(s) with specific channels selected.

        """
        return X[:, :, self.channels]


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features from images using a pre-trained model.

    Parameters
    ----------
    model : callable
        The pre-trained model to use for feature extraction. If ``None``
        ``tensorflow.keras.applications.VGG16`` is used instead.

    Examples
    --------
    >>> from gofast.transformers import ImageFeatureExtractor
    >>> from tensorflow.keras.applications import VGG16
    >>> from tensorflow.keras.models import Model
    >>> base_model = VGG16(weights='imagenet', include_top=False)
    >>> model = Model(inputs=base_model.input, 
                      outputs=base_model.get_layer('block3_pool').output)
    >>> extractor = ImageFeatureExtractor(model=model)
    >>> image = np.random.rand(224, 224, 3)
    >>> features = extractor.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable
        parameters and does nothing during fitting.

    transform(X)
        Extract features from the input images using the pre-trained model.

    Notes
    -----
    Image feature extraction is a common task in computer vision, where 
    pre-trained models are used to obtain high-level features from images. 
    These features can be used for various downstream tasks such as image 
    classification, object detection, and more.

    This transformer allows you to use a pre-trained model for feature 
    extraction. You can specify the pre-trained model when initializing the 
    transformer, and it will extract features from input images using 
    that model.

    """
    
    def __init__(self, model=None ):
        """
        Initialize an ImageFeatureExtractor.

        Parameters
        ----------
        model : callable
            The pre-trained model to use for feature extraction.

        Returns
        -------
        ImageFeatureExtractor
            Returns an instance of the ImageFeatureExtractor.

        """
        self.model = model
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Extract features from the input images using the pre-trained model.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) from which features will be extracted.

        Returns
        -------
        features : ndarray, shape (n_samples, n_features)
            Extracted features from the input image(s).

        """
        if self.model is None: 
            import_optional_dependency('tensorflow')
            from tensorflow.keras.applications import VGG16
            from tensorflow.keras.models import Model
            base_model = VGG16(weights='imagenet', include_top=False)
            self.model = Model(inputs=base_model.input, 
                              outputs=base_model.get_layer('block3_pool'
                                                           ).output)
            
        return self.model.predict(X)



class ImageEdgeDetector(BaseEstimator, TransformerMixin):
    """
    Detect edges in images using a specified method.

    Parameters
    ----------
    method : str, default='sobel'
        The method to use for edge detection. Options include 'sobel',
        'canny', and others.

    Examples
    --------
    >>> from skimage.filters import sobel
    >>> detector = ImageEdgeDetector(method='sobel')
    >>> image = np.random.rand(256, 256)
    >>> edges = detector.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Detect edges in the input images using the specified edge detection 
        method.

    Notes
    -----
    Edge detection is a fundamental image processing technique used to identify 
    boundaries within images. It enhances the regions in an image where there 
    are significant changes in intensity or color, typically indicating
    object boundaries.

    This transformer allows you to perform edge detection using different
    methods such as Sobel and Canny.

    """
    
    def __init__(self, method='sobel'):
        """
        Initialize an ImageEdgeDetector.

        Parameters
        ----------
        method : str, default='sobel'
            The method to use for edge detection. Options include 'sobel',
            'canny', and others.

        Returns
        -------
        ImageEdgeDetector
            Returns an instance of the ImageEdgeDetector.

        """
        self.method = method
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        import_optional_dependency ('skimage', extra = EMSG )
        return self
    
    def transform(self, X):
        """
        Detect edges in the input images using the specified edge 
        detection method.

        Parameters
        ----------
        X : ndarray, shape (height, width)
            Input image(s) on which edge detection will be performed.

        Returns
        -------
        edges : ndarray, shape (height, width)
            Image(s) with edges detected using the specified method.

        Raises
        ------
        ValueError
            If an unsupported edge detection method is specified.

        Notes
        -----
        This method applies edge detection to the input image(s) using the 
        specified method, such as Sobel or Canny.

        If 'sobel' is selected as the method, the Sobel filter is applied 
        to detect edges.

        If 'canny' is selected as the method, the Canny edge detection 
        algorithm is applied.

        """
        if self.method == 'sobel':
            return sobel(X)
        elif self.method == 'canny':
            return canny(X)
        else:
            raise ValueError("Unsupported edge detection method.")



class ImageHistogramEqualizer(BaseEstimator, TransformerMixin):
    """
    Apply histogram equalization to images to improve contrast.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from skimage.exposure import equalize_hist
    >>> equalizer = ImageHistogramEqualizer()
    >>> image = np.random.rand(256, 256)
    >>> equalized_image = equalizer.transform(image)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Apply histogram equalization to the input images to improve contrast.

    Notes
    -----
    Histogram equalization is a technique used to enhance the contrast of an 
    image by redistributing the intensity values of its pixels. It works by 
    transforming the intensity histogram of the image to achieve a more uniform
    distribution.

    This transformer applies histogram equalization to input images, making 
    dark areas darker and bright areas brighter, thus enhancing the 
    visibility of details.

    """
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Apply histogram equalization to the input images to improve contrast.

        Parameters
        ----------
        X : ndarray, shape (height, width)
            Input image(s) to which histogram equalization will be applied.

        Returns
        -------
        X_equalized : ndarray, shape (height, width)
            Image(s) with histogram equalization applied for improved contrast.

        Notes
        -----
        This method applies histogram equalization to the input image(s). It
        enhances the contrast of the image by redistributing pixel intensity 
        values, making dark regions darker and bright regions brighter.

        """
        import_optional_dependency ('skimage', extra = EMSG )
        return equalize_hist(X)



class ImagePCAColorAugmenter(BaseEstimator, TransformerMixin):
    """
    Apply PCA color augmentation as described in the AlexNet paper.

    Parameters
    ----------
    alpha_std : float
        Standard deviation of the normal distribution used for PCA noise.
        This parameter controls the strength of the color augmentation.
        Larger values result in more significant color changes.

    Examples
    --------
    >>> augmenter = ImagePCAColorAugmenter(alpha_std=0.1)
    >>> image = np.random.rand(256, 256, 3)
    >>> pca_augmented_image = augmenter.transform(image)

    Attributes
    ----------
    alpha_std : float
        The standard deviation used for PCA noise.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. This transformer has no trainable 
        parameters and does nothing during fitting.

    transform(X)
        Apply PCA color augmentation to the input images.

    Notes
    -----
    PCA color augmentation is a technique used to perform color variations on 
    images. It applies a PCA transformation to the color channels of the image 
    and adds random noise to create color diversity.

    The `alpha_std` parameter controls the strength of the color augmentation.
    Smaller values (e.g., 0.1) result in subtle color changes, while larger 
    values (e.g., 1.0) result in more dramatic color shifts.

    This transformer reshapes the input images to a flattened form for PCA 
    processing and then reshapes them back to their original shape after 
    augmentation.

    """
    def __init__(self, alpha_std):
        self.alpha_std = alpha_std
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X):
        """
        Apply PCA color augmentation to the input images.

        Parameters
        ----------
        X : ndarray, shape (height, width, channels)
            Input image(s) to which PCA color augmentation will be applied.

        Returns
        -------
        X_augmented : ndarray, shape (height, width, channels)
            Augmented image(s) with PCA color changes applied.

        Notes
        -----
        This method applies PCA color augmentation to the input image(s). 
        It reshapes the input image(s) to a flattened form, performs PCA 
        transformation on the color channels, adds random noise, and reshapes 
        the result back to the original shape.

        """
        orig_shape = X.shape
        X = X.reshape(-1, 3)
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_pca = U.dot(np.diag(S)).dot(Vt)  # PCA transformation
        alpha = np.random.normal(0, self.alpha_std, size=3)
        X_augmented = X_pca + Vt.T.dot(alpha)
        return X_augmented.reshape(orig_shape)


class ImageBatchLoader(BaseEstimator, TransformerMixin):
    """
    Load images in batches from a directory, useful when
    dealing with large datasets.

    Parameters
    ----------
    batch_size : int
        Number of images to load per batch.

    directory : str
        Path to the directory containing images.

    Examples
    --------
    >>> loader = ImageBatchLoader(batch_size=32, directory='path/to/images')
    >>> for batch in loader.transform():
    >>>     process(batch)

    Attributes
    ----------
    batch_size : int
        Number of images to load per batch.

    directory : str
        Path to the directory containing images.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data.

    transform(X=None, y=None)
        Load and yield batches of images from the specified directory.

    Notes
    -----
    This transformer uses the 'skimage' library to load images. Ensure 
    that 'skimage' is installed.

    """
    
    def __init__(self, batch_size, directory):
        self.batch_size = batch_size
        self.directory = directory
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data. Not used in this transformer.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        import_optional_dependency('skimage', extra=EMSG)
        return self
    
    def transform(self, X=None, y=None):
        """
        Load and yield batches of images from the specified directory.

        Parameters
        ----------
        X : None, optional
            Ignored. This parameter is not used.

        y : None, optional
            Ignored. This parameter is not used.

        Yields
        ------
        batch_images : ndarray, shape (batch_size, height, width, channels)
            A batch of images loaded from the directory. The shape of each image
            is determined by the image dimensions and number of channels.

        Notes
        -----
        The images are loaded in batches from the directory specified during
        initialization. Each batch contains 'batch_size' images. The images are
        read using the 'plt.imread' function from the 'matplotlib' library.

        """
        image_files = [os.path.join(self.directory, fname) 
                       for fname in sorted(os.listdir(self.directory))]
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i + self.batch_size]
            batch_images = [plt.imread(file) for file in batch_files]
            yield np.array(batch_images)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
