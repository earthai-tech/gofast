# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import re  
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder

from .._gofastlog import  gofastlog
from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted

_logger = gofastlog().get_gofast_logger(__name__)

__all__=[
  "MajorityVoteClassifier", "EnsembleNeuroFuzzy", "SimpleAverageRegressor",
  "SimpleAverageClassifier", "WeightedAverageRegressor", 
  "WeightedAverageClassifier",
  ]

class MajorityVoteClassifier (BaseEstimator, ClassifierMixin ): 
    r"""
    A majority vote ensemble classifier.

    This classifier combines different classification algorithms, each associated
    with individual weights for confidence. The aim is to create a stronger
    meta-classifier that balances out individual classifiers' weaknesses on
    specific datasets. The majority vote considers the weighted contribution of
    each classifier in making the final decision.

    Mathematically, the weighted majority vote is expressed as:

    .. math::

        \hat{y} = \arg\max_{i} \sum_{j=1}^{m} weights_j \chi_{A}(C_j(x)=i)

    Here, :math:`weights_j` is the weight associated with the base classifier :math:`C_j`;
    :math:`\hat{y}` is the predicted class label by the ensemble; :math:`A` is
    the set of unique class labels; :math:`\chi_A` is the characteristic
    function or indicator function, returning 1 if the predicted class of
    the j-th classifier matches :math:`i`. For equal weights, the equation 
    simplifies to:

    .. math::

        \hat{y} = \text{mode} \{ C_1(x), C_2(x), \ldots, C_m(x) \}

    Parameters
    ----------
    clfs : array-like, shape (n_classifiers)
        Different classifiers for the ensemble.

    vote : str, {'classlabel', 'probability'}, default='classlabel'
        If 'classlabel', prediction is based on the argmax of class labels.
        If 'probability', prediction is based on the argmax of the sum of
        probabilities. Recommended for calibrated classifiers.

    weights : array-like, shape (n_classifiers,), optional, default=None
        Weights for each classifier. Uniform weights are used if 'weights' 
        is None.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes)
        Unique class labels.

    classifiers_ : list
        List of fitted classifiers.

    Notes
    -----
    - This classifier assumes that all base classifiers are capable of producing
      probability estimates if 'vote' is set to 'probability'.
    - In case of a tie in 'classlabel' voting, the class label is determined
      randomly.

    See Also
    --------
    VotingClassifier : Ensemble voting classifier in scikit-learn.
    BaggingClassifier : A Bagging ensemble classifier.
    RandomForestClassifier : A random forest classifier.
    
    Examples 
    ---------
    >>> from sklearn.linear_model import LogisticRegression 
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.pipeline import Pipeline 
    >>> from sklearn.model_selection import cross_val_score, train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.impute import SimpleImputer
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.estimators.ensemble import MajorityVoteClassifier 
    >>> from gofast.tools.coreutils import select_features 
    >>> data = fetch_data('bagoue original').frame
    >>> X0 = data.iloc [:, :-1]; y0 = data ['flow'].values  
    >>> # exclude the categorical value for demonstration 
    >>> # binarize the target y 
    >>> y = np.asarray (list(map (lambda x: 0 if x<=1 else 1, y0))) 
    >>> X = selectfeatures (X0, include ='number')
    >>> X = SimpleImputer().fit_transform (X) 
    >>> X, Xt , y, yt = train_test_split(X, y)
    >>> clf1 = LogisticRegression(penalty ='l2', solver ='lbfgs') 
    >>> clf2= DecisionTreeClassifier(max_depth =1 ) 
    >>> clf3 = KNeighborsClassifier( p =2 , n_neighbors=1) 
    >>> pipe1 = Pipeline ([('sc', StandardScaler()), 
                           ('clf', clf1)])
    >>> pipe3 = Pipeline ([('sc', StandardScaler()), 
                           ('clf', clf3)])
    
    (1) -> Test the each classifier results taking individually 
    
    >>> clf_labels =['Logit', 'DTC', 'KNN']
    >>> # test the results without using the MajorityVoteClassifier
    >>> for clf , label in zip ([pipe1, clf2, pipe3], clf_labels): 
            scores = cross_val_score(clf, X, y , cv=10 , scoring ='roc_auc')
            print("ROC AUC: %.2f (+/- %.2f) [%s]" %(scores.mean(), 
                                                     scores.std(), 
                                                     label))
    ... ROC AUC: 0.91 (+/- 0.05) [Logit]
        ROC AUC: 0.73 (+/- 0.07) [DTC]
        ROC AUC: 0.77 (+/- 0.09) [KNN]
    
    (2) _> Implement the MajorityVoteClassifier
    
    >>> # test the resuls with Majority vote  
    >>> mv_clf = MajorityVoteClassifier(clfs = [pipe1, clf2, pipe3])
    >>> clf_labels += ['Majority voting']
    >>> all_clfs = [pipe1, clf2, pipe3, mv_clf]
    >>> for clf , label in zip (all_clfs, clf_labels): 
            scores = cross_val_score(clf, X, y , cv=10 , scoring ='roc_auc')
            print("ROC AUC: %.2f (+/- %.2f) [%s]" %(scores.mean(), 
                                                     scores.std(), label))
    ... ROC AUC: 0.91 (+/- 0.05) [Logit]
        ROC AUC: 0.73 (+/- 0.07) [DTC]
        ROC AUC: 0.77 (+/- 0.09) [KNN]
        ROC AUC: 0.92 (+/- 0.06) [Majority voting] # give good score & less errors 
    """     
    
    def __init__(self, clfs, weights = None , vote ='classlabel'):
        
        self.clfs=clfs 
        self.weights=weights
        self.vote=vote 
        
        self.classifier_names_={}
  
    def fit(self, X, y):
        """
        Fit classifiers 
        
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
        y: array-like, shape (M, ) ``M=m-samples``
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `MajorityVoteClassifier` instance 
            returns ``self`` for easy method chaining.
        """
        X, y = check_X_y(
            X, 
            y, 
            estimator = get_estimator_name(self ), 
            to_frame= True, 
            )
        
        self._check_clfs_vote_and_weights ()
        
        # use label encoder to ensure that class start by 0 
        # which is important for np.argmax call in predict 
        self._labenc = LabelEncoder () 
        self._labenc.fit(y)
        self.classes_ = self._labenc.classes_ 
        
        self.classifiers_ = list()
        for clf in self.clfs: 
            fitted_clf= clone (clf).fit(X, self._labenc.transform(y))
            self.classifiers_.append (fitted_clf ) 
            
        return self 
    
    def predict(self, X):
        """
        Predict the class label of X 
        
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
        maj_vote:{array_like}, shape (n_examples, )
            Predicted class label array 
        """
        check_is_fitted (self, 'classifiers_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        if self.vote =='proba': 
            maj_vote = np.argmax (self.predict_proba(X), axis =1 )
        if self.vote =='label': 
            # collect results from clf.predict 
            preds = np.asarray(
                [clf.predict(X) for clf in self.classifiers_ ]).T 
            maj_vote = np.apply_along_axis(
                lambda x : np.argmax( 
                    np.bincount(x , weights = self.weights )), 
                    axis = 1 , 
                    arr= preds 
                    
                    )
            maj_vote = self._labenc.inverse_transform(maj_vote )
        
        return maj_vote 
    
    def predict_proba (self, X): 
        """
        Predict the class probabilities an return average probabilities which 
        is usefull when computing the the receiver operating characteristic 
        area under the curve (ROC AUC ). 
        
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
        avg_proba: {array_like }, shape (n_examples, n_classes) 
            weights average probabilities for each class per example. 

        """
        check_is_fitted (self, 'classifiers_')
        probas = np.asarray (
            [ clf.predict_proba(X) for clf in self.classifiers_ ])
        avg_proba = np.average (probas , axis = 0 , weights = self.weights ) 
        
        return avg_proba 
    
    def get_params( self , deep = True ): 
        """ Overwrite the get params from `Base` class  and get 
        classifiers parameters from GridSearch . """
        
        if not deep : 
            return super().get_params(deep =False )
        if deep : 
            out = self.classifier_names_.copy() 
            for name, step in self.classifier_names_.items() : 
                for key, value in step.get_params (deep =True).items (): 
                    out['%s__%s'% (name, key)]= value 
        
        return out 
        
    def _check_clfs_vote_and_weights (self): 
        """ assert the existence of classifiers, vote type and the 
         classfifers weigths """
        l = "https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html"
        if self.clfs is None: 
            raise TypeError( "Expect at least one classifiers. ")

        if hasattr(self.clfs , '__class__') and hasattr(
                self.clfs , '__dict__'): 
            self.clfs =[self.clfs ]
      
        s = set ([ (hasattr(o, '__class__') and hasattr(o, '__dict__')) for o 
                  in self.clfs])
        
        if  not list(s)[0] or len(s)!=1:
            raise TypeError(
                "Classifier should be a class object, not {0!r}. Please refer"
                " to Scikit-Convention to write your own estimator <{1!r}>."
                .format('type(self.clfs).__name__', l)
                )
        self.classifier_names_ = {
            k : v for k, v  in _name_estimators(self.clfs)
            }
        
        regex= re.compile(r'(class|label|target)|(proba)')
        v= regex.search(self.vote)
        if v  is None : 
            raise ValueError ("Vote argument must be 'probability' or "
                              "'classlabel', got %r"%self.vote )
        if v is not None: 
            if v.group (1) is not None:  
                self.vote  ='label'
            elif v.group(2) is not None: 
                self.vote  ='proba'
           
        if self.weights and len(self.weights)!= len(self.clfs): 
           raise ValueError(" Number of classifier must be consistent with "
                            " the weights. got {0} and {1} respectively."
                            .format(len(self.clfs), len(self.weights))
                            )


class EnsembleNeuroFuzzy(BaseEstimator, RegressorMixin):
    r"""
    Neuro-Fuzzy Ensemble for Regression Tasks.

    This ensemble model leverages the strengths of multiple neuro-fuzzy models, 
    which integrate neural network learning capabilities with fuzzy logic's 
    qualitative reasoning. The ensemble averages the predictions from these 
    models to enhance prediction accuracy and robustness, especially in 
    capturing complex and non-linear relationships within data.

    Each NeuroFuzzyModel in the ensemble is assumed to be a pre-defined class 
    with its own fit and predict methods. This allows each model to specialize 
    in different aspects of the data. The ensemble model fits each neuro-fuzzy 
    estimator on the training data and averages their predictions to produce 
    the final output.

    Neuro-fuzzy models are particularly effective in scenarios where data 
    exhibits ambiguous or imprecise characteristics, commonly found in real-world 
    applications like control systems, weather forecasting, and financial modeling.

    The predictions of the Neuro-Fuzzy Ensemble are computed by averaging the 
    outputs of individual neuro-fuzzy models. The mathematical formulation is 
    described as follows:

    1. Neuro-Fuzzy Model Prediction:
       .. math::
           \hat{y}_i = f_{\text{NF}_i}(X)

       where :math:`f_{\text{NF}_i}` represents the prediction function of the 
       \(i\)-th neuro-fuzzy model, and \(X\) denotes the input features.

    2. Ensemble Prediction (Averaging):
       .. math::
           \hat{y}_{\text{ensemble}} = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_i

       where:
       - :math:`\hat{y}_i` is the prediction of the \(i\)-th neuro-fuzzy model.
       - :math:`N` is the number of neuro-fuzzy models in the ensemble.

    Parameters
    ----------
    nf_estimators : list of NeuroFuzzyModel objects
        List of neuro-fuzzy model estimators. Each estimator in the list
        should be pre-configured with its own settings and hyperparameters.

    Attributes
    ----------
    fitted_ : bool
        True if the ensemble model has been fitted.

    Examples
    --------
    >>> # Import necessary libraries
    >>> import numpy as np
    >>> from gofast.estimators.ensemble import EnsembleNeuroFuzzy
    
    >>> # Define two Neuro-Fuzzy models with different configurations
    >>> nf1 = NeuroFuzzyModel(...)
    >>> nf2 = NeuroFuzzyModel(...)
    
    >>> # Create a Neuro-Fuzzy Ensemble Regressor with the models
    >>> ensemble = EnsembleNeuroFuzzy([nf1, nf2])

    >>> # Generate random data for demonstration
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    
    >>> # Fit the ensemble on the data and make predictions
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Neuro-Fuzzy Ensemble model combines the predictions of multiple
      neuro-fuzzy estimators to provide a more accurate and robust regression
      prediction.
    - Each NeuroFuzzyModel in the ensemble should be pre-configured with its
      own settings, architecture, and hyperparameters.
      
    This method of averaging helps to mitigate potential overfitting or biases 
    in individual model predictions, leveraging the diverse capabilities of 
    each model to achieve a more accurate and reliable ensemble prediction.
    
    """

    def __init__(self, nf_estimators):
        self.nf_estimators = nf_estimators

    def fit(self, X, y):
        """
        Fit the Neuro-Fuzzy Ensemble model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        for estimator in self.nf_estimators:
            estimator.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the Neuro-Fuzzy Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values, averaged across all Neuro-Fuzzy estimators.
        """
        check_is_fitted(self, 'fitted_')
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )

        predictions = np.array([estimator.predict(X)
                                for estimator in self.nf_estimators])
        return np.mean(predictions, axis=0)


class SimpleAverageRegressor(BaseEstimator, RegressorMixin):
    r"""
    Simple Average Ensemble Regressor.

    This ensemble model performs regression tasks by averaging the predictions
    of multiple base regression models. Utilizing a simple average is an
    effective technique to reduce the variance of individual model predictions,
    thereby improving the overall robustness and accuracy of the ensemble
    predictions.

    Mathematical Formulation:
    The predictions of the Simple Average Ensemble are computed as the
    arithmetic mean of the predictions made by each base estimator:

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{pred}_i}

    where:
    - :math:`N` is the number of base estimators.
    - :math:`y_{\text{pred}_i}` is the prediction of the \(i\)-th base estimator.

    Parameters
    ----------
    base_estimators : list of objects
        A list of base regression estimators. Each estimator in the list should
        support fit and predict methods. These base estimators will be
        combined through averaging to form the ensemble model.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from gofast.estimators.ensemble import SimpleAverageRegressor
    >>> # Create an ensemble with Linear Regression and Decision Tree Regressor
    >>> ensemble = SimpleAverageRegressor([
    ...     LinearRegression(),
    ...     DecisionTreeRegressor()
    ... ])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Simple Average Ensemble is a basic ensemble technique that can be
      used to combine the predictions of multiple regression models.
    - It is particularly effective when the base estimators have diverse
      strengths and weaknesses, as it can balance out their individual
      performances.
    - This ensemble technique is computationally efficient and easy to
      implement, making it a practical choice for many regression problems.
    - While it provides robustness and improved generalization, it may not
      capture complex relationships between features and target variables as
      advanced ensemble methods like Random Forest or Gradient Boosting.
     
    This approach is particularly beneficial in scenarios where base estimators
    might overfit the training data or when individual model predictions are
    highly variable. By averaging these predictions, the ensemble can often
    achieve better generalization on unseen data.

    See Also
    --------
    VotingRegressor : A similar ensemble method provided by Scikit-Learn.
    Random Forest : An ensemble of decision trees with improved performance.
    Gradient Boosting : An ensemble technique for boosting model performance.
    """

    def __init__(self, base_estimators):
        self.base_estimators = base_estimators

    def fit(self, X, y):
        """
        Fit the Simple Average Ensemble model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        for estimator in self.base_estimators:
            estimator.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the Simple Average Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values, averaged across all base estimators.
        """
        check_is_fitted (self, 'fitted_') 

        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        predictions = [estimator.predict(X) for estimator in self.base_estimators]
        return np.mean(predictions, axis=0)
 
class SimpleAverageClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Simple Average Ensemble Classifier.

    This classifier employs a simple average of the predicted class probabilities 
    from multiple base classifiers to determine the final class labels. It is 
    effective in reducing variance and improving generalization over using a 
    single classifier.

    The `EnsembleSimpleAverageClassifier` aggregates predictions by averaging the 
    probabilities predicted by each classifier in the ensemble. This approach 
    can mitigate individual classifier biases and is particularly beneficial 
    when the classifiers vary widely in their predictions.

    Parameters
    ----------
    base_classifiers : list of objects
        List of base classifiers. Each classifier should support `fit`,
        `predict`, and `predict_proba` methods.

    Attributes
    ----------
    classifiers_ : list of fitted classifiers
        The collection of fitted base classifiers that have been trained on the 
        data.

    Notes
    -----
    - The `SimpleAverageClassifier` assumes all base classifiers implement the
    `fit`, `predict`, and `predict_proba` methods necessary for ensemble predictions.

    The Simple Average Ensemble Classifier determines the final predicted class
    label for an input `x` by computing the simple average of the predicted
    class probabilities from all classifiers:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\frac{1}{n} \sum_{i=1}^{n} \mathbf{P}_i(x)\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for input `x`.
    - :math:`n` is the number of base classifiers in the ensemble.
    - :math:`\mathbf{P}_i(x)` represents the predicted class probabilities by the 
      \(i\)-th base classifier for input `x`.
    
    This method of averaging class probabilities helps to stabilize the predictions 
    by leveraging the collective intelligence of multiple models, often resulting in 
    more reliable and robust classification outcomes.

    See Also
    --------
    - sklearn.ensemble.VotingClassifier: Scikit-learn's ensemble classifier 
      that supports various voting strategies.
    - sklearn.ensemble.StackingClassifier: Scikit-learn's stacking ensemble 
      classifier that combines multiple base estimators.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.estimators.ensemble import SimpleAverageClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> ensemble = SimpleAverageClassifier(
    ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()]
    ... )
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)
    """

    def __init__(self, base_classifiers):
        self.base_classifiers = base_classifiers
        
    def fit(self, X, y):
        """
        Fit the Simple Average Ensemble Classifier model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.classifiers_ = []
        for clf in self.base_classifiers:
            fitted_clf = clf.fit(X, y)
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """
        Predict using the Simple Average Ensemble Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'classifiers_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        # Calculate the average predicted class probabilities
        avg_proba = self._average_proba(X)

        # Determine the class with the highest average probability 
        #as the predicted class
        y_pred = np.argmax(avg_proba, axis=1)
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Simple Average 
        Ensemble Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """

        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        # Calculate the average predicted class probabilities
        avg_proba = self._average_proba(X)
        return avg_proba
    
    def _average_proba(self, X):
       """
       Calculate the average predicted class probabilities from base classifiers.

       Parameters
       ----------
       X : array-like of shape (n_samples, n_features)
           The input samples.

       Returns
       -------
       avg_proba : array-like of shape (n_samples, n_classes)
           The average predicted class probabilities.
       """
       # Get predicted probabilities from each base classifier
       probas = [clf.predict_proba(X) for clf in self.classifiers_]

       # Calculate the average predicted class probabilities
       avg_proba = np.mean(probas, axis=0)
       return avg_proba    

class WeightedAverageRegressor(BaseEstimator, RegressorMixin):
    r"""
    Weighted Average Ensemble Regressor.

    This ensemble model calculates the weighted average of the predictions from
    multiple base regression models. By multiplying each base model's prediction 
    with a corresponding weight, it allows for differential contributions of each 
    model to the final prediction. This technique is particularly effective when 
    base models vary in accuracy or are tuned to different aspects of the data.

    Weights for each model can be based on their performance, domain knowledge, 
    or other criteria, providing flexibility in how contributions are balanced.
    Compared to a simple average, this approach can lead to better performance 
    by leveraging the strengths of each base model more effectively.

    Mathematical Formulation:
    The predictions of the Weighted Average Ensemble are computed as:

    .. math::
        y_{\text{pred}} = \sum_{i=1}^{N} (weights_i \times y_{\text{pred}_i})

    where:
    - :math:`weights_i` is the weight assigned to the \(i\)-th base estimator.
    - :math:`y_{\text{pred}_i}` is the prediction made by the \(i\)-th base estimator.
    - \(N\) is the number of base estimators in the ensemble.

    Parameters
    ----------
    base_estimators : list of objects
        A list of base regression estimators. Each estimator in the list should
        support fit and predict methods. These base estimators will be
        combined through weighted averaging to form the ensemble model.
    weights : array-like of shape (n_estimators,)
        An array-like object containing the weights for each estimator in the
        `base_estimators` list. The weights determine the importance of each
        base estimator's prediction in the final ensemble prediction.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from gofast.estimators.ensemble import WeightedAverageRegressor
    >>> # Create an ensemble with Linear Regression and Decision Tree Regressor
    >>> # Weighted with 0.7 for Linear Regression and 0.3 for Decision Tree
    >>> ensemble = WeightedAverageRegressor([
    ...     LinearRegression(),
    ...     DecisionTreeRegressor()
    ... ], [0.7, 0.3])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Weighted Average Ensemble is a versatile ensemble technique that
      allows for fine-grained control over the contribution of each base
      estimator to the final prediction.
    - The weights can be determined based on the performance of the base
      models, domain knowledge, or other criteria.
    - It is particularly effective when the base models have significantly
      different levels of accuracy or are suited to different parts of the
      input space.
    - This ensemble technique provides flexibility and can improve overall
      performance compared to simple averaging or individual estimators.
    
    This method is particularly advantageous in scenarios where base estimators 
    perform differently across various segments of the data, allowing the ensemble 
    to optimize overall predictive accuracy by adjusting the influence of each model.
    
    See Also
    --------
    SimpleAverageRegressor : An ensemble that equally weights base model \
        predictions.
    VotingRegressor : A similar ensemble method provided by Scikit-Learn.
    Random Forest : An ensemble of decision trees with improved performance.
    Gradient Boosting : An ensemble technique for boosting model performance.
    """


    def __init__(self, base_estimators, weights):
        self.base_estimators = base_estimators
        self.weights = weights


    def fit(self, X, y):
        """
        Fit the Weighted Average Ensemble model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        for estimator in self.base_estimators:
            estimator.fit(X, y)
        self.fitted_ = True
        
        return self

    def predict(self, X):
        """
        Predict using the Weighted Average Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Weighted predicted values, averaged across all base estimators.
        """
        check_is_fitted (self, 'fitted_') 

        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
                
        predictions = np.array([estimator.predict(X) for estimator 
                                in self.base_estimators])
        weighted_predictions = np.average(predictions, axis=0, 
                                          weights=self.weights)
        return weighted_predictions

class WeightedAverageClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Weighted Average Ensemble Classifier.

    This classifier averages the predictions of multiple base classifiers,
    each weighted by its assigned importance. The ensemble prediction for 
    each class is the weighted average of the predicted probabilities from 
    all classifiers.

    `EnsembleWeightedAverageClassifier` class, `base_classifiers` are the 
    individual classifiers in the ensemble, and `weights` are their 
    corresponding importance weights. The `fit` method trains each 
    classifier in the ensemble, and the `predict` method calculates the 
    weighted average of the predicted probabilities from all classifiers
    to determine the final class labels.
    
    This implementation assumes that each base classifier can output 
    class probabilities (i.e., has a `predict_proba` method), which is 
    common in many scikit-learn classifiers. The effectiveness of this 
    ensemble classifier depends on the diversity and performance of the 
    base classifiers, as well as the appropriateness of their assigned 
    weights. Fine-tuning these elements is crucial for optimal performance.
    
    Parameters
    ----------
    base_classifiers : list of objects
        List of base classifiers. Each classifier should support `fit`, 
        `predict`, and `predict_proba` methods.
    weights : array-like of shape (n_classifiers,)
        Weights for each classifier in the `base_classifiers` list.

    Attributes
    ----------
    classifiers_ : list of fitted classifiers
        The collection of fitted base classifiers.

    Notes
    -----
    - It's important to carefully select and fine-tune the base classifiers 
      and their weights for optimal performance.
    - The `WeightedAverageClassifier` assumes that each base classifier 
    supports `fit`, `predict`, and `predict_proba` methods.
    
    Each base classifier predicts a class label and the associated probabilities
    for a given input. These predictions are then aggregated using a weighted 
    average approach, where weights may be determined based on the performance,
    reliability, or other relevant characteristics of each classifier.

    The final predicted class label \(C_{\text{final}}(x)\) for input \(x\) 
    is determined by a weighted average of the predicted probabilities from all 
    classifiers, formulated as follows:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} weights_i \cdot \mathbf{P}_i(x)\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for input \(x\).
    - \(n\) is the number of base classifiers in the ensemble.
    - :math:`weights_i` is the weight assigned to the \(i\)-th base classifier.
    - :math:`\mathbf{P}_i(x)` represents the predicted class probabilities by the 
      \(i\)-th base classifier for input \(x\).

    This weighted approach ensures that each classifier’s prediction influences 
    the final outcome proportionate to its assigned weight, ideally optimizing 
    the ensemble's performance across diverse or unbalanced datasets.
    
    See Also
    --------
    - sklearn.ensemble.VotingClassifier: Scikit-learn's ensemble classifier 
      that supports various voting strategies.
    - sklearn.ensemble.StackingClassifier: Scikit-learn's stacking ensemble 
      classifier that combines multiple base estimators.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.estimators.ensemble import EnsembleWeightedAverageClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> ensemble = EnsembleWeightedAverageClassifier(
    ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()],
    ...     weights=[0.7, 0.3]
    ... )
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)
    """

    def __init__(self, base_classifiers, weights):
        self.base_classifiers = base_classifiers
        self.weights = weights
        
    def fit(self, X, y):
        """
        Fit the Weighted Average Ensemble Classifier model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.classifiers_ = []
        for clf in self.base_classifiers:
            fitted_clf = clf.fit(X, y)
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """
        Predict using the Weighted Average Ensemble Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'classifiers_')
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        weighted_sum = np.zeros(
            (X.shape[0], len(np.unique(self.classifiers_[0].classes_))))

        for clf, weight in zip(self.classifiers_, self.weights):
            probabilities = clf.predict_proba(X)
            weighted_sum += weight * probabilities

        return np.argmax(weighted_sum, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Weighted Average 
        Ensemble Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        # Check if the classifier is fitted
        check_is_fitted(self, 'classifiers_')
    
        # Check input and ensure it's compatible with base classifiers
        X = check_array(X)
    
        # Get predicted probabilities from each base classifier
        probas = [clf.predict_proba(X) for clf in self.classifiers_]
    
        # Calculate the weighted average of predicted probabilities
        weighted_avg_proba = np.average(probas, axis=0, weights=self.weights)
    
        return weighted_avg_proba


