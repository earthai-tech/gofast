# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Gives some efficient tools for data manipulation 
and transformation.
"""
from __future__ import division, annotations  

import os
import inspect
import itertools 
import warnings 
import numpy as np 
import pandas as pd 
from scipy import sparse
import matplotlib.pyplot as plt  
# from pandas.api.types import is_integer_dtype
from sklearn.base import BaseEstimator,TransformerMixin 
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.preprocessing import ( 
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder, 
    PolynomialFeatures
    )
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score,  roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
try : 
    from skimage.transform import resize
    from skimage.color import rgb2gray
    from skimage.filters import sobel, canny
    from skimage.exposure import equalize_hist
except : pass
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except : pass 

from ._gofastlog import gofastlog 
from ._typing import F 
from .exceptions import EstimatorError, NotFittedError 
from .tools.funcutils import ( 
    _assert_all_types, 
    parse_attrs , 
    to_numeric_dtypes,
    assert_ratio, 
    ellipsis2false
    )
from .tools.mlutils import (discretize_categories, stratify_categories, 
    existfeatures )
from .tools._dependency import import_optional_dependency 
from .tools.validator import ( get_estimator_name, check_X_y,
                              _is_arraylike_1d, build_data_if)

EMSG = (
        "`scikit-image` is needed"
        " for this transformer. Note"
        " `skimage`is the shorthand "
        "of `scikit-image`."
        )

__docformat__='restructuredtext'
_logger = gofastlog().get_gofast_logger(__name__)

__all__= ['SequentialBackwardSelection',
          'KMeansFeaturizer',
          'StratifiedWithCategoryAdder',
          'StratifiedUsingBaseCategory', 
          'CategorizeFeatures', 
          'FrameUnion', 
          'DataFrameSelector',
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

class SequentialBackwardSelection (BaseEstimator, TransformerMixin ):
    r"""
    Sequential Backward Selection (SBS) is a feature selection algorithm which 
    aims to reduce dimensionality of the initial feature subspace with a 
    minimum decay  in the performance of the classifier to improve upon 
    computationan efficiency. In certains cases, SBS can even improve the 
    predictive power of the model if a model suffers from overfitting. 
    
    The idea behind the SBS is simple: it sequentially removes features 
    from the full feature subset until the new feature subspace contains the 
    desired number of features. In order to determine which feature is to be 
    removed at each stage, the criterion fonction :math:`J` is needed for 
    minimization [1]_. 
    Indeed, the criterion calculated from the criteria function can simply be 
    the difference in performance of the classifier before and after the 
    removal of this particular feature. Then, the feature to be remove at each 
    stage can simply be the defined as the feature that maximizes this 
    criterion; or in more simple terms, at each stage, the feature that causes 
    the least performance is eliminated loss after removal. Based on the 
    preceding definition of SBS, the algorithm can be outlibe with a few steps:
        
        - Initialize the algorithm with :math:`k=d`, where :math:`d` is the 
            dimensionality of the full feature space, :math:`X_d`. 
        - Determine the feature :math:`x^{-}`,that maximizes the criterion: 
            :math:`x^{-}= argmax J(X_k-x)`, where :math:`x\in X_k`. 
        - Remove the feature :math:`x^{-}` from the feature set 
            :math:`X_{k+1}= X_k -x^{-}; k=k-1`.
        -Terminate if :math:`k` equals to the number of desired features; 
            otherwise go to the step 2. [2]_ 
            
    Parameters 
    -----------
    estimator: callable or instanciated object,
        callable or instance object that has a fit method. 
    k_features: int, default=1 
        the number of features from where starting the selection. It must be 
        less than the number of feature in the training set, otherwise it 
        does not make sense. 
    scoring: callable or str , default='accuracy'
        metric for scoring. availabe metric are 'precision', 'recall', 
        'roc_auc' or 'accuracy'. Any other metric with raise an errors. 
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25. 
        
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    References 
    -----------
    .. [1] Raschka, S., Mirjalili, V., 2019. Python Machine Learning, 3rd ed. Packt.
    .. [2] Ferri F., Pudil F., Hatef M., and Kittler J., Comparative study of 
        the techniques for Large-scale feature selection, pages 403-413, 1994.
    
    Attributes 
    -----------
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        
    indices_: tuple of dimensionnality X
        Collect the indices of subset of the best validated models 
        
    subsets_: list, 
        list of `indices_` 
        
    scores_: list, 
        Collection of the scores of the best model got during the
        cross-validating 
        
    k_score_: float, 
        The score of the desired feature. 
        
    Examples
    --------
    >>> from gofast.exlib.sklearn import KNeighborsClassifier , train_test_split
    >>> from gofast.datasets import fetch_data
    >>> from gofast.base import SequentialBackwardSelection
    >>> X, y = fetch_data('bagoue analysed') # data already standardized
    >>> Xtrain, Xt, ytrain,  yt = train_test_split(X, y)
    >>> knn = KNeighborsClassifier(n_neighbors=5)
    >>> sbs= SequentialBackwardSelection (knn)
    >>> sbs.fit(Xtrain, ytrain )

    """
    _scorers = dict (accuracy = accuracy_score , recall = recall_score , 
                   precision = precision_score, roc_auc= roc_auc_score 
                   )
    def __init__ (self, estimator=None , k_features=1 , 
                  scoring ='accuracy', test_size = .25 , 
                  random_state = 42 ): 
        self.estimator=estimator 
        self.k_features=k_features 
        self.scoring=scoring 
        self.test_size=test_size
        self.random_state=random_state 
        
    def fit(self, X, y) :
        """  Fit the training data 
        
        Note that SBS splits the datasets into a test and training insite the 
        fit function. :math:`X` is still fed to the algorithm. Indeed, SBS 
        will then create a new training subsets for testing (validation) and 
        training , which is why this test set is also called the validation 
        dataset. This approach is necessary to prevent our original test set 
        to becoming part of the training data. 
        
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
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `SequentialBackwardSelection` instance 
            returns ``self`` for easy method chaining.
        
        """
        X, y = check_X_y(
            X, 
            y, 
            estimator = get_estimator_name(self ), 
            to_frame= True, 
            )
        
        self._check_sbs_args(X)
        
        if hasattr(X, 'columns'): 
            self.feature_names_in = list(X.columns )
            X = X.values 
            
        Xtr, Xt,  ytr, yt = train_test_split(X, y , test_size=self.test_size, 
                                            random_state=self.random_state 
                                            )
        dim = Xtr.shape [1] 
        self.indices_= tuple (range (dim))
        self.subsets_= [self.indices_]
        score = self._compute_score(Xtr, Xt,  ytr, yt, self.indices_)
        self.scores_=[score]
        # compute the score for p indices in 
        # list indices in dimensions 
        while dim > self.k_features: 
            scores , subsets = [], []
            for p in itertools.combinations(self.indices_, r=dim-1):
                score = self._compute_score(Xtr, Xt,  ytr, yt, p)
                scores.append (score) 
                subsets.append (p)
            
            best = np.argmax (scores) 
            self.indices_= subsets [best]
            self.subsets_.append(self.indices_)
            dim -=1 # go back for -1 
            
            self.scores_.append (scores[best])
            
        # set  the k_feature score 
        self.k_score_= self.scores_[-1]
        
        return self 
        
    def transform (self, X): 
        """ Transform the training set 
        
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
        Returns 
        -------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            New transformed training set with selected features columns 
        
        """
        if not hasattr (self, 'indices_'): 
            raise NotFittedError(
                "Can't call transform with estimator not fitted yet."
                " Fit estimator by calling the 'fit' method with appropriate"
                " arguments.")
        return X[:, self.indices_]
    
    def _compute_score (self, Xtr, Xt,  ytr, yt, indices):
        """ Compute score from splitting `X` and indices """
        self.estimator.fit(Xtr[:, indices], ytr)
        y_pred = self.estimator.predict (Xt [:, indices])
        score = self.scoring (yt, y_pred)
        
        return score 

    def _check_sbs_args (self, X): 
        """ Assert SBS main arguments  """
        
        if not hasattr(self.estimator, 'fit'): 
            raise TypeError ("Estimator must have a 'fit' method.")
        try : 
            self.k_features = int (self.k_features)
        except  Exception as err: 
            raise TypeError ("Expect an integer for number of feature k,"
                             f" got {type(self.k_features).__name__!r}"
                             ) from err
        if self.k_features > X.shape [1] :
            raise ValueError ("Too many number of features."
                              f" Expect max-features={X.shape[1]}")
        if  ( 
            callable(self.scoring) 
            or inspect.isfunction ( self.scoring )
            ): 
            self.scoring = self.scoring.__name__.replace ('_score', '')
        
        if self.scoring not in self._scorers.keys(): 
            raise ValueError (
                f"Accept only scorers {list (self._scorers.keys())}"
                f"for scoring, not {self.scoring!r}")
            
        self.scoring = self._scorers[self.scoring] 
        
        self.scorer_name_ = self.scoring.__name__.replace (
            '_score', '').title ()
        
    def __repr__(self): 
        """ Represent the  Sequential Backward Selection class """
        get_params = self.get_params()  
        get_params.pop('scoring')
        if hasattr (self, 'scorer_name_'): 
            get_params ['scoring'] =self.scorer_name_ 
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     get_params.items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
    
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

    random_state: int, Optional 
       State for shuffling the data 
    
    Attributes 
    -----------
    km_model: KMeans featurization model used to transform

    Examples 
    --------
    >>> # (1) Use a common dataset 
    >>> import matplotlib.pyplot as plt 
    >>> from sklearn.datasets import make_moons
    >>> from gofast.tools.plotutils import plot_voronoi 
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
    """
    def __init__(
        self, 
        n_clusters=7, 
        target_scale=5.0, 
        random_state=None, 
        n_components=None
        ):
        self.n_clusters = n_clusters
        self.target_scale = target_scale
        self.random_state = random_state
        self.n_components=n_components

    def fit(self, X, y=None):
        """Runs k-means on the input data and finds and updated centroids.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.
            
        Returns
        -------
        self
            Fitted estimator.
        """
        if self.n_components: 
            self.n_components = int (_assert_all_types(
                self.n_components, int, float,objname ="'n_components'"))
            from gofast.analysis import nPCA 
            X =nPCA (X, n_components = self.n_components )
            
        if y is None:
            # No target variable, just do plain k-means
            km_model = KMeans(n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.random_state)
            km_model.fit(X)
            
            self.km_model_ = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self
        
        # There is target information. Apply appropriate scaling and include
        # it in the input data to k-means. 
        data_with_target = np.hstack((X, y[:,np.newaxis]*self.target_scale))
        
        # Build a pre-training k-means model on data and target
        km_model_pretrain = KMeans(n_clusters=self.n_clusters,
                            n_init=20,
                            random_state=self.random_state
                            )
        km_model_pretrain.fit(data_with_target)

        # Run k-means a second time to get the clusters in the original space
        # without target info. Initialize using centroids found in pre-training.
        # Go through a single iteration of cluster assignment and centroid 
        # recomputation.
        km_model = KMeans(n_clusters=self.n_clusters,
                    init=km_model_pretrain.cluster_centers_[:,:-1] , #[:, :-1]
                    n_init=1,
                    max_iter=1)
        km_model.fit(X)
        
        self.km_model_ = km_model
        self.cluster_centers_ = self.km_model_.cluster_centers_
        
        return self

    def transform(self, X):
        """Outputs the closest cluster ID for each input data point.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        clusters = self.km_model_.predict(X)
        return clusters[:,np.newaxis]

    def fit_transform(self, X, y=None):
        """ Fit and transform the data 
        
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to. 
        
        """
        self.fit(X, y)
        return self.transform(X, y)

    def __repr__(self):
        """ Pretty format for guidance following the API... """
        _t = ("n_clusters", "target_scale", "random_state", "n_components")
        outm = ( '<{!r}:' + ', '.join(
            [f"{k}={ False if getattr(self, k)==... else  getattr(self, k)!r}" 
             for k in _t]) + '>' 
            ) 
        return  outm.format(self.__class__.__name__)
    
class StratifiedWithCategoryAdder( BaseEstimator, TransformerMixin ): 
    """
    Stratified sampling transformer based on new generated category 
    from numerical attributes and return stratified trainset and test set.
    
    Parameters  
    ---------- 
    base_num_feature: str, 
        Numerical features to categorize. 
        
    threshold_operator: float, 
        The coefficient to divised the numerical features value to 
        normalize the data 
        
    max_category: Maximum value fits a max category to gather all 
        value greather than.
        
    return_train: bool, 
        Return the whole stratified trainset if set to ``True``.
        usefull when the dataset is not enough. It is convenient to 
        train all the whole trainset rather than a small amount of 
        stratified data. Sometimes all the stratified data are 
        not the similar equal one to another especially when the dataset 
        is not enough.
        
    Another way to stratify dataset is to get insights from the dataset and 
    to add a new category as additional mileage. From this new attributes,
    data could be stratified after categorizing numerical features. 
    Once data is tratified, the new category will be drop and return the 
    train set and testset stratified. For instance::  
        
        >>> from gofast.transformers import StratifiedWithCategoryAdder
        >>> stratifiedNumObj= StratifiedWithCatogoryAdder('flow')
        >>> stratifiedNumObj.fit_transform(X=df)
        >>> stats2 = stratifiedNumObj.statistics_
        
    Usage
    ------
    In this example, we firstly categorize the `flow` attribute using 
    the ceilvalue (see :func:`~discretizeCategoriesforStratification`) 
    and groupby other values greater than the ``max_category`` value to the 
    ``max_category`` andput in the temporary features. From this features 
    the categorization is performed and stratified the trainset and 
    the test set.
        
    Notes 
    ------
    If `base_num_feature` is not given, dataset will be stratified using 
    random sampling.
        
    """
    
    def __init__(
        self,
        base_num_feature=None,
        threshold_operator = 1.,
        return_train=False,
        max_category=3,
        n_splits=1, 
        test_size=0.2, 
        random_state=42
        ):
        
        self._logging= gofastlog().get_gofast_logger(self.__class__.__name__)
        
        self.base_num_feature= base_num_feature
        self.return_train= return_train
        self.threshold_operator=  threshold_operator
        self.max_category = max_category 
        self.n_splits = n_splits 
        self.test_size = test_size 
        self.random_state = random_state 
        
        self.base_items_ =None 
        self.statistics_=None 
        
    def fit(self, X, y=None): 
        """
        Does nothin just for scikit-learn API purpose. 
        """
        return self
    
    def transform(self, X, y=None):
        """Transform data and populate inspections attributes 
            from hyperparameters.
            
         Parameters
         ----------
         X : {array-like, sparse matrix} of shape (n_samples, n_features)
             New data to predict.

         y: Ignored
            Keep just for API purpose. 

         Returns
         -------
         X : {array-like, sparse matrix} of shape (n_samples, n_features)
             New data transformed.   
        """

        if self.base_num_feature is not None:
            in_c= 'temf_'
            # discretize the new added category from the threshold value
            X = discretize_categories(
                                     X,
                                    in_cat=self.base_num_feature, 
                                     new_cat=in_c, 
                                     divby =self.threshold_operator,
                                     higherclass = self.max_category
                 )

            self.base_items_ = list(
            X[in_c].value_counts().index.values)
        
            split = StratifiedShuffleSplit(n_splits =self.n_splits,
                                           test_size =self.test_size, 
                                           random_state =self.random_state)
            
            for train_index, test_index  in split.split(X, X[in_c]): 
                strat_train_set = X.loc[train_index]
                strat_test_set = X.loc[test_index] 
                #keep a copy of all stratified trainset.
                strat_train_set_copy = X.loc[ np.delete(X.index, test_index)]
                
        train_set, test_set = train_test_split( X, test_size = self.test_size,
                                   random_state= self.random_state)
        
        if self.base_num_feature is None or self.n_splits==0:
            
            self._logging.info('Stratification not applied! Train and test sets'
                               'were purely generated using random sampling.')
            
            return train_set , test_set 
            
        if self.base_num_feature is not None:
            # get statistic from `in_c` category proportions into the 
            # the overall dataset 
            o_ =X[in_c].value_counts() /len(X)
            r_ = test_set[in_c].value_counts()\
                /len(test_set)
            s_ = strat_test_set[in_c].value_counts()\
                /len( strat_test_set)
            r_error , s_error = ((r_/ o_)-1)*100, ((s_/ o_)-1)*100
            
            self.statistics_ = np.c_[np.array(self.base_items_), 
                                     o_,
                                     r_,
                                     s_, 
                                     r_error,
                                     s_error
                                     ]
      
            self.statistics_ = pd.DataFrame(data = self.statistics_,
                                columns =[in_c, 
                                          'Overall',
                                          'Random', 
                                          'Stratified', 
                                          'Rand. %error',
                                          'strat. %error'
                                          ])
            
            # set a pandas dataframe for inspections attributes `statistics`.
            self.statistics_.set_index(in_c, inplace=True)
            
            # remove the add category into the set 
            for set in(strat_train_set_copy, strat_train_set, strat_test_set): 
                set.drop([in_c], axis=1, inplace =True)
                
            if self.return_train: 
                strat_train_set = strat_train_set_copy 
               
            # force to remove the temporary features for splitting in 
            # the original dataset

            if in_c in X.columns: 
                X.drop([in_c], axis =1, inplace=True)
               
            return strat_train_set, strat_test_set 

    
class StratifiedUsingBaseCategory( BaseEstimator, TransformerMixin ): 
    """
    Transformer to stratified dataset to have data more representativce into 
    the trainset and the test set especially when data is not large enough.
    
    Arguments 
    ----------
    base_column: str or int, 
        Hyperparameters and can be index of the base mileage(category)
        for stratifications. If `base_column` is None, will return 
        the purely random sampling.
        
    test_size: float 
        Size to put in the test set.
        
    random_state: shuffled number of instance in the overall dataset. 
        default is ``42``.
    
    Usage 
    ------
    If data is  not large enough especially relative number of attributes
    if much possible to run therisk of introducing a significant sampling 
    biais.Therefore strafied sampling is a better way to avoid 
     a significant biais of sampling survey. For instance:: 
        
        >>> from gofast.transformers import StratifiedUsingBaseCategory 
        >>> from gofast.tools.mlutils import load_data 
        >>> df = load_data('data/geo_fdata')
        >>> stratifiedObj = StratifiedUsingBaseCategory(base_column='geol')
        >>> stratifiedObj.fit_transform(X=df)
        >>> stats= stratifiedObj.statistics_

    Notes
    ------
    An :attr:`~.statictics_` inspection attribute is good way to observe 
    the test set generated using purely random and  the 
    stratified sampling. The stratified sampling has category 
    ``base_column`` proportions almost indentical to those in the full 
    dataset whereas the test set generated using purely random sampling 
    is quite skewed. 
    
    """
    
    def __init__(self, base_column =None,test_size=0.2, random_state=42):
        self._logging= gofastlog().get_gofast_logger(self.__class__.__name__)
        
        self.base_column = base_column 
        self.test_size = test_size  
        self.random_state = random_state
        
        #create inspection attributes
        self.statistics_=None 
        self.base_flag_ =False 
        self.base_items_= None 
        
    def fit(self, X, y=None): 
        """ Does nothing , just for API purpose. 
        """
        return self
    
    def transform(self, X, y=None):
        """ 
        Split dataset into `trainset` and `testset` using stratified sampling. 
        
        If `base_column` not given will return the `trainset` and `testset` 
        using purely random sampling.
        {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
            
        Parameters 
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored

        Returns 
        --------
        strat_train_set, strat_test_set : NDArray, ( n_samples , n_features) 
            train set and test set stratified 
        
        """
        if self.base_column is None: 
            self.stratified = False 
            self._logging.debug(
                f'Base column is not given``{self.base_column}``.Test set'
                ' will be generated using purely random sampling.')
            
        train_set, test_set = train_test_split(
                X, test_size = self.test_size ,random_state= self.random_state )
        
        if self.base_column  is not None: 
            
            if isinstance(self.base_column, (int, float)):  # use index to find 
            # base colum name. 
                self.base_column = X.columns[int(self.base_column)]
                self.base_flag_ =True 
            elif isinstance(self.base_column, str):
                # check wether the column exist into dataframe
                for elm in X.columns:
                    if elm.find(self.base_column.lower())>=0 : 
                        self.base_column = elm
                        self.base_flag_=True
                        break 
                    
        if not self.base_flag_: 
            self._logging.debug(
                f'Base column ``{self.base_column}`` not found '
                f'in `{X.columns}`')
            warnings.warn(
                f'Base column ``{self.base_column}`` not found in '
                f'{X.columns}.Test set is generated using purely '
                'random sampling.')
            
        self.base_items_ = list(
            X[self.base_column].value_counts().index.values)
        
        if self.base_flag_: 
            strat_train_set, strat_test_set = \
                stratify_categories(X, self.base_column)
                
            # get statistic from `basecolumn category proportions into the 
            # the whole dataset, in the testset generated using purely random 
            # sampling and the test set generated using the stratified sampling.
            
            o_ =X[self.base_column].value_counts() /len(X)
            r_ = test_set [self.base_column].value_counts()/len(test_set)
            s_ = strat_test_set[self.base_column].value_counts()/len( strat_test_set)
            r_error , s_error = ((r_/ o_)-1)*100, ((s_/ o_)-1)*100
            
            self.statistics_ = np.c_[np.array(self.base_items_),
                                     o_,
                                     r_, 
                                     s_, 
                                     r_error,
                                     s_error]
      
            self.statistics_ = pd.DataFrame(data = self.statistics_,
                                columns =[self.base_column,
                                          'Overall', 
                                          'Random', 
                                          'Stratified',
                                          'Rand. %error',
                                          'strat. %error'
                                          ])
            
            # set a pandas dataframe for inspections attributes `statistics`.
            self.statistics_.set_index(self.base_column, inplace=True)
            
            return strat_train_set, strat_test_set 
        
        if not self.base_flag_: 
            return train_set, test_set 
        
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
        func: F=None, 
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
    
    def transform(self, X, y=None) :
        """ Transform the data and return new array. Can straightforwardly
        call :meth:`~.sklearn.TransformerMixin.fit_transform` inherited 
        from scikit_learn.
        
        """
        # now work with categorize data 
        # -------------------------------------------
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
    

class CombinedAttributesAdder(BaseEstimator, TransformerMixin ):
    """ Combined attributes from litteral string operators, indexes or names. 
    
    Create a new attribute using features index or litteral string operator.
    Inherits from scikit_learn `BaseEstimator`and `TransformerMixin` classes.
 
    Arguments 
    ----------
    *attribute_names* : list of str , optional
        List of features for combinaison. Decide to combine new feature
        values by from `operator` parameters. By default, the combinaison it 
        is ratio of the given attribute/numerical features. For instance, 
        ``attribute_names=['lwi', 'ohmS']`` will divide the feature 'lwi' by 
        'ohmS'.
                    
    *attributes_indexes* : list of int,
        index of each feature/feature for experience combinaison. User 
        warning should raise if any index does match the dataframe of array 
        columns.
            
    *operator*: str, default ='/' 
        Type of operation to perform. Can be ['/', '+', '-', '*', '%']  
        
    Returns
    --------   
    X : np.ndarray, 
        A  new array contained the new data from the `attrs_indexes` operation. 
        If `attr_names` and attr_indexes is ``None``, will return the same array 
        like beginning. 
    
    Notes
    ------
    A litteral string operator can be used. For instance dividing two numerical 
    features can be illustrated using the word "per" separated by underscore like 
    "_per_" For instance, to create a new feature based on the division of 
    the features ``lwi`` and ``ohmS``, the litteral string operator that holds
    the ``attribute_names`` could be::
        
        attribute_names='lwi_per_ohmS'
        
    The same litteral string is valid for multiplication (_mul_) , 
    substraction (_sub_) , modulo (_mod_) and addition (_add_). However, 
    indexes of features can also use rather than `attribute_names` providing
    the `operator` parameters. 
    
    Or it could be the indexes of both features in the array like 
    ``attributes_ix =[(10, 9)]`` which means the `lwi` and `ohmS` are
    found at index ``10`` and ``9``respectively. Furthermore, multiples 
    operations can be set by adding mutiples litteral string operator into a 
    list like ``attributes_ix = [ 'power_per_magnitude', 'ohmS_per_lwi']``.
        
    Examples 
    --------
    >>> import pandas as pd 
    >>> from gofast.transformers import CombinedAttributesAdder
    >>> from gofast.datasets.dload import load_bagoue 
    >>> X, y = load_bagoue (as_frame =True ) 
    >>> cobj = CombinedAttributesAdder (attribute_names='lwi_per_ohmS')
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
         'lwi_div_ohmS'] # new attributes with 'lwi'/'ohmS'
    >>> df0 = pd.DataFrame (Xadded, columns = cobj.attribute_names_)
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
    >>> cobj = CombinedAttributesAdder (
        attribute_names=['lwi', 'ohmS', 'power'], operator='+')
    >>> df0 = pd.DataFrame (cobj.fit_transform(X),
                            columns = cobj.attribute_names_)
    >>> df0.iloc [:, -1]
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
    >>> cobj = CombinedAttributesAdder (
        attribute_indexes =[1,6], operator='+')
    >>> df0 = pd.DataFrame (cobj.fit_transform(X), 
                            columns = cobj.attribute_names_)
    >>> df0.iloc [:, -1]
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
    _op ={'times': ('times', 'prod', 'mul', '*', 'x'), 
            'add': ('add', '+', 'plus'), 
            'div': ('quot', '/', 'div', 'per'), 
            'sub': ('sub', '-', 'less'), 
            'mod': ('mod', '%'),
        }
    
    def __init__(
            self, 
            attribute_names =None, 
            attribute_indexes = None, 
            operator: str='/', 
        ):

        self.attribute_names=attribute_names
        self.attribute_indexes= attribute_indexes
        self.operator=operator
        self.attribute_names_=None 
        
    def fit(self, X, y=None ):
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
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `CombinedAttributesAdder` instance 
            returns ``self`` for easy method chaining.
        
        """
        return self 
    
    def transform(self, X): 
        """ Tranform X and return new array with experience attributes
        combinaison. 
        
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

        Returns 
        --------
        X: NDarray,  Ndarray ( M x N+1 matrix) 
            returns X transformed (``M=m-samples``, & ``N=n+1-features``) 
            with attribute  combined. 
            
        .. versionadded:: 0.1.3
            
        """
        columns =[]
        self.operator = self._get_operator (
            self.operator or self.attribute_names)
        
        if self.operator is None: 
            warnings.warn("None or Invalid operator cannot be use for "
                          "attribute combinaisons.")
        if isinstance (self.attribute_names, str): 
            self.attribute_names_ = parse_attrs(self.attribute_names) 
        
        elif isinstance(self.attribute_names, 
                        (list, tuple, np.ndarray) ):
            self.attribute_names_ = self.attribute_names 
            
        if isinstance(X, pd.DataFrame) : 
            # asset wether attributes exists 
            # no raise errors, return the dataframe 
            if self.attribute_names_ : 
                existfeatures(X, self.attribute_names_  )
            # get the index of attributes from dataframe 
            if self.attribute_names_: 
                self.attribute_indexes = list(map (
                    lambda o: list(X.columns).index (o), self.attribute_names_)
                    ) 
                
            elif self.attribute_indexes : 
                # try :
                self.attribute_names_ = list(map (
                    lambda ix: list(X.columns)[ix], self.attribute_indexes)
                    ) 
                # except IndexError: 
                #     raise IndexError("List of index is out the range.")
                
            columns = X.columns 
            X= to_numeric_dtypes(X)
            X= X.values 
            
        if self.attribute_indexes: 
            X = self._operate(X)
        
        if self.attribute_names_ is not None: 
            self.attribute_names_ = list(columns) + ([
                f'_{self.operator}_'.join([ v for v in self.attribute_names_ ])
                ] if self._isfine else [])
        
        return X 
            
    def _get_operator (self, operator): 
        """ Get operator for combining  attribute """
        
        for k, v in self._op.items() :
            for o in v: 
                if operator.find(o) >=0 : 
                    self.operator = k 
                    return self.operator 
        return 
    
    def _operate (self,  X): 
        """ Operate data from indexes """
        def weird_division(ix_):
            """ Replace 0. value to 1 in denominator for division 
            calculus."""
            return ix_ if ix_!=0. else 1
        
        msg=("Unsupported operand type(s)! index provided {} doesn't match"
             " any numerical features. Experience combinaison attributes"
             " is not possible.")
        
        self._isfine=True 
        Xc =X[:, self.attribute_indexes]
        cb= Xc[:, 0 ] ; Xc=Xc[:,  1: ]
        
        for k in range (Xc.shape[1]): 
            try : 
                if self.operator =='mod': 
                    cb %= Xc[:, k]
                if self.operator =='add': 
                    cb += Xc[:, k]
                if self.operator =='sub': 
                    cb -= Xc[:, k]
                if self.operator =='div': 
                    # if the denominator contain nan or 0 
                    # a weird division is triggered and replace 
                    # the denominator by 1
                    try : 
                        cb /= Xc[:, k]
                    except ZeroDivisionError: 
                        wv= np.array(
                            list(map(weird_division, Xc[:, k])))
                        cb /=wv
    
                    except ( TypeError, RuntimeError, RuntimeWarning):
                        warnings.warn(msg.format(
                            self.attribute_indexes[1:][k])) 
                        
                if self.operator =='x': 
                    cb *= Xc[:, k]        
                    
            except: 
                warnings.warn(msg.format(self.attribute_indexes[1:][k])) 
                self._isfine =False          
            
        X =  np.c_[X, cb ]  if self._isfine else X 
        
        return X 

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ Select data from specific attributes for column transformer. 
    
    Select only numerical or categorial columns for operations. Work as the
    same like sckit-learn `make_colum_tranformer` 
    
    Arguments  
    ----------
    *attribute_names*: list or array_like 
        List of  the main columns to keep the data 
        
    *select_type*: str 
        Automatic numerical and categorial selector. If `select_type` is 
        ``num``, only numerical values in dataframe are retrieved else 
        ``cat`` for categorials attributes.
            
    Returns
    -------
    X: ndarray 
        New array with composed of data of selected `attribute_names`.
            
    Examples 
    ---------
    >>> from gofast.transformers import DataFrameSelector 
    >>> from gofast.tools.mlutils import load_data   
    >>> df = mlfunc.load_data('data/geo_fdata')
    >>> XObj = DataFrameSelector(attribute_names=['power','magnitude','sfi'],
    ...                          select_type=None)
    >>> cdf = XObj.fit_transform(df)
    
    """  
    def __init__(self, attribute_names=None, select_type =None): 
        self._logging= gofastlog().get_gofast_logger(self.__class__.__name__)
        self.attribute_names = attribute_names 
        self.select_type = select_type 
        
    def fit(self, X, y=None): 
        """ 
        Select the Data frame 
        
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
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `DataFrameSelector` instance 
            returns ``self`` for easy method chaining.
        
        """
        return self
    
    def transform(self, X): 
        """ Transform data and return numerical or categorial values.
        
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        """
       
        if isinstance(self.attribute_names, str): 
            self.attribute_names =[self.attribute_names]
            
        if self.attribute_names is not None: 
            t_= []
            for in_attr in self.attribute_names: 
                for attr_ in X.columns: 
                    if in_attr.lower()== attr_.lower(): 
                        t_.append(attr_)
                        break 
                    
            if len(t_)==0: 
                self._logging.warn(f' `{self.attribute_names}` not found in the'
                                   '`{X.columns}`.')
                warnings.warn('None attribute in the dataframe match'
                              f'`{self.attribute_names}.')
                
            if len(t_) != len(self.attribute_names): 
                mm_= set(self.attribute_names).difference(set(t_))
                warnings.warn(
                    f'Value{"s" if len(mm_)>1 else""} {list(mm_)} not found.'
                    f" Only `{t_}`match{'es' if len(t_) <1 else ''}"
                    " the dataframe features.")
                self._logging.warning(
                    f'Only `{t_}` can be considered as dataframe attributes.')
                                   
            self.attribute_names =t_
            
            return X[self.attribute_names].values 
        
        try: 
            if self.select_type.lower().find('num')>=0:
                self.select_type =='num'
            elif self.select_type.lower().find('cat')>=0: 
                self.select_type =='cat'
            else: self.select_type =None 
            
        except:
            warnings.warn(f'`Select_type`` given argument ``{self.select_type}``'
                         ' seems to be wrong. Should defaultly return the '
                         'Dataframe value.', RuntimeWarning)
            self._logging.warnings('A given argument `select_type`seems to be'
                                   'wrong %s. Use ``cat`` or ``num`` for '
                                   'categorical or numerical attributes '
                                   'respectively.'% inspect.signature(self.__init__))
            self.select_type =None 
        
        if self.select_type is None:
            warnings.warn('Arguments of `%s` arguments %s are all None. Should'
                          ' returns the dataframe values.'% (repr(self),
                              inspect.signature (self.__init__)))
            
            self._logging.warning('Object arguments are None.'
                               ' Should return the dataframe values.')
            return X.values 
        
        if self.select_type =='num':
            obj_columns= X.select_dtypes(include='number').columns.tolist()

        elif self.select_type =='cat': 
            obj_columns= X.select_dtypes(include=['object']).columns.tolist() 
 
        self.attribute_names = obj_columns 
        
        return X[self.attribute_names].values 
        
    def __repr__(self):
        return self.__class__.__name__
        
class FrameUnion (BaseEstimator, TransformerMixin) : 
    """ Unified categorial and numerical features after scaling and 
    and categorial features encoded.
    
    Use :class:`~gofast.tranformers.DataframeSelector` class to define 
    the categorial features and numerical features.
    
    Arguments
    ---------
    num_attributes: list 
        List of numerical attributes 
        
    cat_attributes: list 
        list of categorial attributes 
        
    scale: bool 
        Features scaling. Default is ``True`` and use 
        `:class:~sklearn.preprocessing.StandarScaler` 
        
    imput_data: bool , 
        Replace the missing data. Default is ``True`` and use 
        :attr:`~sklearn.impute.SimpleImputer.strategy`. 
        
    param_search: bool, 
        If `num_attributes` and `cat_attributes`are None, the numerical 
        features and categorial features` should be found automatically.
        Default is ``True``
        
    scale_mode:bool, 
        Mode of data scaling. Default is ``StandardScaler``but can be 
        a ``MinMaxScaler`` 
        
    encode_mode: bool, 
        Mode of data encoding. Default is ``OrdinalEncoder`` but can be 
        ``OneHotEncoder`` but creating a sparse matrix. Once selected, 
        the new shape of ``X`` should be different from the original 
        shape. 
    
    Example
    ------- 
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.tools.transformers import FrameUnion
    >>> X_= fetch_data ('Bagoue original').get('data=dfy1')
    >>> frameObj = FrameUnion(X_, encoding =OneHotEncoder)
    >>> X= frameObj.fit_transform(X_)
        
    """  
    def __init__(
        self,
        num_attributes =None , 
        cat_attributes =None,
        scale =True,
        imput_data=True,
        encode =True, 
        param_search ='auto', 
        strategy ='median', 
        scale_mode ='StandardScaler', 
        encode_mode ='OrdinalEncoder'
        ): 
        
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        
        self.num_attributes = num_attributes 
        self.cat_attributes = cat_attributes 
        self.param_search = param_search 
        self.imput_data = imput_data 
        self.strategy =strategy 
        self.scale = scale
        self.encode = encode 
        self.scale_mode = scale_mode
        self.encode_mode = encode_mode
        
        self.X_=None 
        self.X_num_= None 
        self.X_cat_ =None
        self.num_attributes_=None
        self.cat_attributes_=None 
        self.attributes_=None 
        
    def fit(self, X, y=None): 
        """
        Does nothing. Just for scikit-learn purpose. 
        """
        return self
    
    def transform(self, X): 
        """ Transform data and return X numerical and categorial encoded 
        values.
        
        Parameters
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            
        Returns 
        --------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            transformed arraylike, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            
        """
        
        if self.scale_mode.lower().find('stand')>=0: 
            self.scale_mode = 'StandardScaler'
        elif self.scale_mode.lower().find('min')>=0: 
            self.scale_mode = 'MinMaxScaler'
        if self.encode_mode.lower().find('ordinal')>=0: 
            self.encode_mode = 'OrdinalEncoder'
            
        elif self.encode_mode.lower().find('hot') >=0: 
            self.encode_mode = 'OneHotEncoder'
            
        numObj = DataFrameSelector(attribute_names= self.num_attributes, 
                                         select_type='num')
        catObj =DataFrameSelector(attribute_names= self.cat_attributes, 
                                         select_type='cat')
        num_arrayObj = numObj.fit_transform(X)
        cat_arrayObj = catObj.fit_transform(X)
        self.num_attributes_ = numObj.attribute_names 
        self.cat_attributes_ = catObj.attribute_names 
        
        self.attributes_ = self.num_attributes_ + self.cat_attributes_ 
        
        self.X_num_= num_arrayObj.copy()
        self.X_cat_ =cat_arrayObj.copy()
        self.X_ = np.c_[self.X_num_, self.X_cat_]
        
        if self.imput_data : 
            from sklearn.impute import SimpleImputer
            imputer_obj = SimpleImputer(missing_values=np.nan, 
                                        strategy=self.strategy)
            num_arrayObj =imputer_obj.fit_transform(num_arrayObj)
            
        if self.scale :
            if self.scale_mode == 'StandardScaler': 
                scaler = StandardScaler()
            if self.scale_mode =='MinMaxScaler':
                scaler = MinMaxScaler()
        
            num_arrayObj = scaler.fit_transform(num_arrayObj)
            
        if self.encode : 
            if self.encode_mode =='OrdinalEncoder': 
                encoder = OrdinalEncoder()
            elif self.encode_mode =='OneHotEncoder':
                encoder = OneHotEncoder(sparse_output=True)
            cat_arrayObj= encoder.fit_transform(cat_arrayObj )
            # sparse matrix of type class <'numpy.float64'>' stored 
            # element in compressed sparses raw format . To convert the sense 
            # matrix to numpy array , we need to just call 'to_array()'.
            warnings.warn(f'Sparse matrix `{cat_arrayObj.shape!r}` is converted'
                          ' in dense Numpy array.', UserWarning)
            # cat_arrayObj= cat_arrayObj.toarray()

        try: 
            X= np.c_[num_arrayObj,cat_arrayObj]
            
        except ValueError: 
            # For consistency use the np.concatenate rather than np.c_
            X= np.concatenate((num_arrayObj,cat_arrayObj), axis =1)
        
        if self.encode_mode =='OneHotEncoder':
            warnings.warn('Use `OneHotEncoder` to encode categorial features'
                          ' generates a Sparse matrix. X is henceforth '
                          ' composed of sparse matrix. The new dimension is'
                          ' {0} rather than {1}.'.format(X.shape,
                             self.X_.shape), UserWarning)
            self._logging.info('X become a spared matrix. The new shape is'
                               '{X.shape!r} against the orignal '
                               '{self.X_shape!r}')
            
        return X
        
class FeaturizeX(BaseEstimator, TransformerMixin ): 
    """
    Featurize X with the cluster based on the KMeans featurization
    
    Parameters 
    -----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features. 
        Note that when `n_components` is set, sparse matrix for `X` is not 
        acceptable. 

    y : array-like of shape (n_samples,)
        Target vector relative to X.
        
    n_clusters: int, default=7
       Number of initial clusters
       
    target_scale: float, default=5.0 
       Apply appropriate scaling and include it in the input data to k-means.
       
    n_components: int, optional
       Number of components for reduced down the predictor X. It uses the PCA 
       to reduce down dimension to the importance features. 
       
    model: :class:`KMeansFeaturizer`. 
       KMeasFeaturizer model. Model can be provided to featurize the 
       test data separated from the train data. 
       
    random_state: int, Optional 
       State for shuffling the data 
       
    split_X_y: bool, default=False, 
       Split the X, y into train data and test data  according to the test 
       size 
       
    test_ratio: int, default=0.2 
       ratio to keep for a test data. 
       
    shuffle: bool, default=True
       Suffling the data set. 
       
    return_model: bool, default =False 
       If ``True`` return the KMeans featurization mode and the transformed X.
       
    to_sparse: bool, default=False 
       Convert X data to sparse matrix, by default the sparse matrix is 
       coordinates matrix (COO) 
       
    sparsity:str, default='coo'
       Kind of sparse matrix use to convert `X`. It can be ['csr'|'coo']. Any 
       other values with return a coordinates matrix unless `to_sparse` is 
       turned to ``False``. 
 
    Returns 
    -------- 
    X : NDArray shape (m_samples, n_features +1) or \
        shape (m_samples, n_sparse_features)
        Returns transformed array X NDArray of m_features plus the clusters
        features from KMF featurization procedures. The `n_sparse_features`
        is created if `to_sparse` is set to ``True``. 

       
    Note
    -----
    Everytimes ``return_model=True``, KMF model (:class:`KMeansFeaturizer`) 
    is appended to the return results. 
    
    Examples 
    --------
    >>> import numpy as np 
    >>> from gofast.transformers import FeaturizeX 
    >>> X = np.random.randn (12 , 7 ) ; y = np.arange(12 )
    >>> y[ y < 6 ]= 0 ; y [y >0 ]= 1  # for binary data 
    >>> Xtransf = FeaturizeX (to_sparse =False).fit_transform(X)
    >>> X.shape, Xtransf.shape 
    ((12, 7), (12, 8))
    >>> Xtransf = FeaturizeX (to_sparse =True ).fit_transform(X,y )
    >>> Xtransf
    (<12x8 sparse matrix of type '<class 'numpy.float64'>'
     	with 93 stored elements in COOrdinate format>,
     array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))

    """
    def __init__(self, 
        n_clusters:int=7, 
        target_scale:float= 5 ,
        random_state:F|int=None, 
        n_components: int=None,  
        model: F =None, 
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
    random_state:F|int=None, 
    n_components: int=None,  
    model: F =None, 
    split_X_y:bool = False,
    test_ratio:float|str= .2 , 
    shuffle:bool=True, 
    return_model:bool=...,
    to_sparse: bool=..., 
    sparsity:str ='coo' 
    ): 
    """ Featurize X with the cluster based on the KMeans featurization
    
    Parameters 
    -----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features. 
        Note that when `n_components` is set, sparse matrix for `X` is not 
        acceptable. 

    y : array-like of shape (n_samples,)
        Target vector relative to X.
        
    n_clusters: int, default=7
       Number of initial clusters
       
    target_scale: float, default=5.0 
       Apply appropriate scaling and include it in the input data to k-means.
       
    n_components: int, optional
       Number of components for reduced down the predictor X. It uses the PCA 
       to reduce down dimension to the importance features. 
       
    model: :class:`KMeansFeaturizer`. 
       KMeasFeaturizer model. Model can be provided to featurize the 
       test data separated from the train data. 
       
       .. versionadded:: 0.2.4 
       
    random_state: int, Optional 
       State for shuffling the data 
       
    split_X_y: bool, default=False, 
       Split the X, y into train data and test data  according to the test 
       size 
       
    test_ratio: int, default=0.2 
       ratio to keep for a test data. 
       
    shuffle: bool, default=True
       Suffling the data set. 
       
    return_model: bool, default =False 
       If ``True`` return the KMeans featurization mode and the transformed X.
       
    to_sparse: bool, default=False 
       Convert X data to sparse matrix, by default the sparse matrix is 
       coordinates matrix (COO) 
       
    sparsity:str, default='coo'
       Kind of sparse matrix use to convert `X`. It can be ['csr'|'coo']. Any 
       other values with return a coordinates matrix unless `to_sparse` is 
       turned to ``False``. 
 
    Returns 
    -------- 
    X, y : NDArray shape (m_samples, n_features +1) or \
        shape (m_samples, n_sparse_features)
        Returns NDArray of m_features plus the clusters features from KMF 
        feturization procedures. The `n_sparse_features` is created if 
        `to_sparse` is set to ``True``. 
   X, y, model: NDarray and KMF models 
       Returns transformed array X and y and model if   ``return_model`` is 
       set to ``True``. 
   
      Array like train data X transformed  and test if `split_X_y` is set to 
      ``True``. 
    X, Xtest, y, ytest: NDArray (KMF), ArrayLike 
       Split tuple is returned when `split_X_y=True``.
       
    Note
    -----
    Everytimes ``return_model=True``, KMF model (:class:`KMeansFeaturizer`) 
    is appended to the return results. 
    
    Examples 
    --------
    >>> import numpy as np 
    >>> from gofast.transformers import featurize_X 
    >>> X = np.random.randn (12 , 7 ) ; y = np.arange(12 )
    >>> y[ y < 6 ]= 0 ; y [y >0 ]= 1  # for binary data 
    >>> Xtransf , _ = featurize_X (X, to_sparse =False)
    >>> X.shape, Xtransf.shape 
    ((12, 7), (12, 8))
    >>> Xtransf, y  = featurize_X (X,y,  to_sparse =True )
    >>> Xtransf , y
    (<12x8 sparse matrix of type '<class 'numpy.float64'>'
     	with 93 stored elements in COOrdinate format>,
     array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    >>> featurize_X (X,y,  to_sparse =True, split_X_y=True  )
    (<9x8 sparse matrix of type '<class 'numpy.float64'>'
     	with 71 stored elements in COOrdinate format>,
     <3x8 sparse matrix of type '<class 'numpy.float64'>'
     	with 24 stored elements in COOrdinate format>,
     array([0, 1, 1, 0, 0, 0, 0, 1, 1]),
     array([0, 1, 1]))
    >>> *_, kmf_model = featurize_X (X,y,  to_sparse =True, return_model =True)
    >>> kmf_model 
    <'KMeansFeaturizer':n_clusters=7, target_scale=5, random_state=None, 
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
    transf_cluster = model.transform(X)
    ### Form new input features with cluster features
    # training_with_cluster
    
    Xkmf = np.concatenate (
        (X, transf_cluster), axis =1 )
    if to_sparse: 
        Xkmf= sparse_func(Xkmf )

    kmf_data.append(Xkmf)
    kmf_data.append(y) 
    if split_X_y: 

        transf_test_cluster= model.transform(test_data)
        test_with_cluster = np.concatenate (
            (test_data, transf_test_cluster),axis =1 )
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
    """

    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X, y=None):
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
    """

    def __init__(self, date_format="%Y-%m-%d"):
        self.date_format = date_format
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
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
    """

    def __init__(self, estimator=None, threshold='mean'):
        if estimator is None:
            estimator = RandomForestClassifier()
        self.selector = SelectFromModel(estimator, threshold=threshold)
        
    def fit(self, X, y):
        self.selector.fit(X, y)
        return self
    
    def transform(self, X, y=None):
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
    """

    def __init__(self, degree=2, interaction_only=False):
        self.poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
        
    def fit(self, X, y=None):
        self.poly.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.poly.transform(X)

class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Reduce dimensionality of the data using PCA.

    Parameters
    ----------
    n_components : int, float, None or str
        Number of components to keep. If n_components is not set, all components are kept.

    Examples
    --------
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> reducer = DimensionalityReducer(n_components=1)
    >>> X_reduced = reducer.fit_transform(X)
    """

    def __init__(self, n_components=0.95):
        self.reducer = PCA(n_components=n_components)
        
    def fit(self, X, y=None):
        self.reducer.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.reducer.transform(X)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric array.

    Parameters
    ----------
    drop : {'first', 'if_binary', None}, default='first'
        Specifies a methodology to use to drop one of the categories per feature.

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
    """

    def __init__(self, drop='first'):
        self.encoder = OneHotEncoder(drop=drop)
        
    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.encoder.transform(X)
        
class CategoricalEncoder2(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric array.

    This transformer should be applied to categorical features in a dataset before
    applying it to a machine learning model.

    Parameters
    ----------
    categorical_features : list of str
        List of column names to be considered as categorical features.

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
    ...     ('cat', CategoricalEncoder(categorical_features=['color', 'brand']), ['color', 'brand'])
    ... ])
    >>> X = pd.DataFrame({'color': ['red', 'blue', 'green'], 'brand': ['ford', 'toyota', 'bmw']})
    >>> transformer.fit_transform(X)
    """
    def __init__(self, categorical_features, drop=None):
        self.categorical_features = categorical_features
        self.drop = drop
        self.encoders_ = {feature: OneHotEncoder(drop=drop) for feature in categorical_features}
        
    def fit(self, X, y=None):
        for feature in self.categorical_features:
            self.encoders_[feature].fit(X[[feature]])
        return self
    
    def transform(self, X, y=None):
        outputs = []
        for feature in self.categorical_features:
            outputs.append(self.encoders_[feature].transform(X[[feature]]))
        return np.hstack(outputs)

class FeatureScaler2(BaseEstimator, TransformerMixin):
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
    """
    def __init__(self, numeric_features):
        self.numeric_features = numeric_features
        self.scaler_ = StandardScaler()
        
    def fit(self, X, y=None):
        self.scaler_.fit(X[self.numeric_features])
        return self
    
    def transform(self, X, y=None):
        X_scaled = X.copy()
        X_scaled[self.numeric_features] = self.scaler_.transform(X[self.numeric_features])
        return X_scaled


class FeatureScaler(BaseEstimator, TransformerMixin):
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
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.scaler.transform(X)

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputation transformer for completing missing values.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy. If "mean", then replace missing values using the mean 
        along each column. Can also be "median" or "most_frequent".

    Attributes
    ----------
    imputer_ : SimpleImputer object
        The fitted SimpleImputer instance.

    Examples
    --------
    >>> from sklearn.impute import SimpleImputer
    >>> imputer = MissingValueImputer(strategy='mean')
    >>> X = [[1, 2], [np.nan, 3], [7, 6]]
    >>> imputer.fit(X)
    >>> imputer.transform(X)
    """

    def __init__(self, strategy='mean'):
        self.imputer = SimpleImputer(strategy=strategy)
        
    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.imputer.transform(X)

class MissingValueImputer2(BaseEstimator, TransformerMixin):
    """
    Imputation transformer for completing missing values.

    This transformer can be applied to both numeric and categorical features in a
    dataset to impute missing values using the mean or the most frequent value.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy. If "mean", then replace missing values using the mean 
        along each column. Can also be "median" or "most_frequent".

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
    """
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer_ = SimpleImputer(strategy=strategy)
        
    def fit(self, X, y=None):
        self.imputer_.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.imputer_.transform(X)

class ColumnSelector(BaseEstimator, TransformerMixin):
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
    """

    def __init__(self, column_names):
        self.column_names = column_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X[self.column_names]
        else:
            raise TypeError("Input must be a pandas DataFrame")

class ColumnSelector2(BaseEstimator, TransformerMixin):
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
    """
    def __init__(self, column_names):
        self.column_names = column_names
        
    def fit(self, X, y=None):
        #check whether colun names exists 
        from .tools.mlutils import existfeatures 
        existfeatures(X, self.column_names)
        return self
    
    def transform(self, X, y=None):
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
    """
    def __init__(self, numeric_features, epsilon=1e-6):
        self.numeric_features = numeric_features
        self.epsilon = epsilon
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
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
    """
    def __init__(self, rolling_window):
        self.rolling_window = rolling_window
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
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
    """
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.frequency_maps_ = None
        
    def fit(self, X, y=None):
        self.frequency_maps_ = {
            feature: X[feature].value_counts(normalize=True).to_dict() 
            for feature in self.categorical_features
        }
        return self
    
    def transform(self, X, y=None):
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
    """
    def __init__(self, datetime_features):
        self.datetime_features = datetime_features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
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
    Generate lag features for time series data to help capture 
    temporal dependencies.

    Parameters
    ----------
    lags : int or list of ints
        The number of lag periods to create features for.

    Examples
    --------
    >>> generator = LagFeatureGenerator(lags=3)
    >>> X = pd.DataFrame({'value': np.arange(100)})
    >>> lag_features = generator.fit_transform(X)
    """
    def __init__(self, lags):
        self.lags = lags if isinstance(lags, list) else [lags]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        for lag in self.lags:
            X_transformed[f'lag_{lag}'] = X_transformed.shift(lag)
        return X_transformed

class DifferencingTransformer(BaseEstimator, TransformerMixin):
    """
    Apply differencing to time series data to make it stationary.

    Parameters
    ----------
    periods : int, default=1
        Number of periods to shift for calculating the difference.

    Examples
    --------
    >>> transformer = DifferencingTransformer(periods=1)
    >>> X = pd.DataFrame({'value': np.cumsum(np.random.randn(100))})
    >>> stationary_data = transformer.fit_transform(X)
    """
    def __init__(self, periods=1):
        self.periods = periods
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.diff(periods=self.periods)

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
    """
    def __init__(self, window):
        self.window = window
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
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
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
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
    """

    def __init__(self, model='additive', freq=1):
        self.model = model
        self.freq = freq
        
    def fit(self, X, y=None):
        import_optional_dependency("statsmodels")
        return self
    
    def transform(self, X, y=None):
        
        result = seasonal_decompose(X, model=self.model, freq=self.freq)
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
    """
    def __init__(self, periods):
        self.periods = periods
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = pd.DataFrame(index=X.index)
        for period in self.periods:
            X_transformed[f'sin_{period}'] = np.sin(
                2 * np.pi * X.index / period)
            X_transformed[f'cos_{period}'] = np.cos(
                2 * np.pi * X.index / period)
        return X_transformed

class TrendFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract linear trend features from time series data.

    Parameters
    ----------
    order : int, default=1
        The order of the trend polynomial to fit.

    Examples
    --------
    >>> transformer = TrendFeatureExtractor(order=1)
    >>> X = pd.DataFrame({'time': np.arange(100), 
                          'value': np.random.randn(100)})
    >>> trend_features = transformer.fit_transform(X[['time']])
    """
    def __init__(self, order=1):
        self.order = order
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        trends = np.polyfit(X.index, X.values, deg=self.order)
        trend_poly = np.poly1d(trends)
        return pd.DataFrame(trend_poly(X.index), index=X.index,
                            columns=[f'trend_{self.order}'])

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
    """
    
    def __init__(self, output_size):
        self.output_size = output_size
        
    def fit(self, X, y=None):
        import_optional_dependency ('skimage', extra = EMSG )
        return self
    
    def transform(self, X):
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
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X / 255.0

class ImageToGrayscale(BaseEstimator, TransformerMixin):
    """
    Convert images to grayscale.

    Parameters
    ----------
    keep_dims : bool, default=False
        If True, keeps the third dimension as 1 (e.g., 
                                                 (height, width, 1)).

    Examples
    --------
    >>> from skimage.color import rgb2gray
    >>> converter = ImageToGrayscale(keep_dims=True)
    >>> image = np.random.rand(256, 256, 3)
    >>> grayscale_image = converter.transform(image)
    """
    
    def __init__(self, keep_dims=False):
        self.keep_dims = keep_dims
        
    def fit(self, X, y=None):
        import_optional_dependency ('skimage', extra = EMSG )
        return self
    
    def transform(self, X):
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
    """
    def __init__(self, augmentation_funcs):
        self.augmentation_funcs = augmentation_funcs
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
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
    """
    def __init__(self, channels):
        self.channels = channels
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, :, self.channels]

class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features from images using a pre-trained model.

    Parameters
    ----------
    model : callable
        The pre-trained model to use for feature extraction.

    Examples
    --------
    >>> from tensorflow.keras.applications import VGG16
    >>> from tensorflow.keras.models import Model
    >>> base_model = VGG16(weights='imagenet', include_top=False)
    >>> model = Model(inputs=base_model.input, 
                      outputs=base_model.get_layer('block3_pool').output)
    >>> extractor = ImageFeatureExtractor(model=model)
    >>> image = np.random.rand(224, 224, 3)
    >>> features = extractor.transform(image)
    """
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
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
    """
    
    def __init__(self, method='sobel'):
        self.method = method
        
    def fit(self, X, y=None):
        import_optional_dependency ('skimage', extra = EMSG )
        return self
    
    def transform(self, X):
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
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        import_optional_dependency ('skimage', extra = EMSG )
        return equalize_hist(X)

class ImagePCAColorAugmenter(BaseEstimator, TransformerMixin):
    """
    Apply PCA color augmentation as described in the AlexNet paper.

    Parameters
    ----------
    alpha_std : float
        Standard deviation of the normal distribution used for PCA noise.

    Examples
    --------
    >>> augmenter = ImagePCAColorAugmenter(alpha_std=0.1)
    >>> image = np.random.rand(256, 256, 3)
    >>> pca_augmented_image = augmenter.transform(image)
    """
    def __init__(self, alpha_std):
        self.alpha_std = alpha_std
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
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
    """
    
    def __init__(self, batch_size, directory):
        self.batch_size = batch_size
        self.directory = directory
        
    def fit(self, X, y=None):
        import_optional_dependency ('skimage', extra = EMSG )
        return self
    
    def transform(self, X=None, y=None):
        image_files = [os.path.join(self.directory, fname) 
                       for fname in sorted(os.listdir(self.directory))]
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i + self.batch_size]
            batch_images = [plt.imread(file) for file in batch_files]
            yield np.array(batch_images)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
