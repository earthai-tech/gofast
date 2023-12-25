gofast.estimators
===================

Overview
--------
The ``gofast.estimators`` module is a comprehensive collection of advanced machine 
learning estimators designed for both regression and classification tasks. It extends 
beyond traditional models to include innovative and hybrid approaches, blending various 
machine learning techniques to address complex data patterns and diverse problem sets.

Key Features
------------
- **BoostedDecisionTreeClassifier**: Harnesses the power of boosting with decision trees for 
  classification, focusing on sequentially correcting errors to enhance model accuracy.
- **RegressionTreeEnsembleClassifier**: An ensemble of decision tree classifiers, it aggregates 
  predictions to achieve higher stability and performance than individual trees.
- **HybridBoostedRegressionTreeEnsembleClassifier**: Combines boosted regression trees into a 
  robust ensemble, offering a layered approach to model complex relationships in classification tasks.
- **NeuroFuzzyClassifier**: Integrates neural networks with fuzzy logic for classification, 
  providing a unique blend of quantitative analysis and qualitative reasoning.
- **SimpleAverageEnsembleClassifier**: Averages predictions from a set of classifiers, offering 
  a straightforward yet effective method for classification by balancing diverse insights.
- **WeightedAverageEnsembleClassifier**: Similar to the simple average ensemble, but assigns 
  weights to each classifier, allowing for nuanced aggregation of predictions based on model 
  reliability or performance.
- **HammersteinWienerClassifier**: Adapts the Hammerstein-Wiener model for classification, 
  leveraging a combination of linear and nonlinear processing elements to capture complex 
  input-output relationships.

Usage
-----
To utilize these estimators, import the desired class from the ``gofast.estimators`` module:

.. code-block:: python

   from gofast.estimators import BoostedDecisionTreeClassifier
   from gofast.estimators import RegressionTreeEnsembleClassifier
   # ... and others

Each estimator is designed to fit seamlessly into your data processing and model 
training workflows, compatible with scikit-learn's well-known API structure. This 
ensures easy integration and a familiar interface for those accustomed to conventional 
machine learning practices.

Applications
------------
The ``gofast.estimators`` module is ideal for data scientists and machine learning 
practitioners looking for innovative solutions to challenging problems in various 
domains, such as finance, healthcare, marketing analysis, and more. 
Whether dealing with intricate patterns in data or needing robust and accurate 
classification models, this module provides the tools to push the boundaries of 
traditional machine learning.


Examples
--------

BoostedDecisionTreeClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from gofast.estimators import BoostedDecisionTreeClassifier
   classifier = BoostedDecisionTreeClassifier(n_estimators=100, max_depth=3)
   classifier.fit(X_train, y_train)
   y_pred = classifier.predict(X_test)

RegressionTreeEnsembleClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from gofast.estimators import RegressionTreeEnsembleClassifier
   ensemble = RegressionTreeEnsembleClassifier(n_estimators=50, max_depth=5)
   ensemble.fit(X_train, y_train)
   y_pred = ensemble.predict(X_test)

HybridBoostedRegressionTreeEnsembleClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from gofast.estimators import HybridBoostedRegressionTreeEnsembleClassifier
   hybrid_ensemble = HybridBoostedRegressionTreeEnsembleClassifier(n_estimators=10)
   hybrid_ensemble.fit(X_train, y_train)
   y_pred = hybrid_ensemble.predict(X_test)

NeuroFuzzyClassifier
^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from gofast.estimators import NeuroFuzzyClassifier
   nf_classifier = NeuroFuzzyClassifier()
   nf_classifier.fit(X_train, y_train)
   y_pred = nf_classifier.predict(X_test)

SimpleAverageEnsembleClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from gofast.estimators import SimpleAverageEnsembleClassifier
   avg_ensemble = SimpleAverageEnsembleClassifier(base_classifiers=[clf1, clf2])
   avg_ensemble.fit(X_train, y_train)
   y_pred = avg_ensemble.predict(X_test)

WeightedAverageEnsembleClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from gofast.estimators import WeightedAverageEnsembleClassifier
   weighted_ensemble = WeightedAverageEnsembleClassifier(base_classifiers=[clf1, clf2], weights=[0.6, 0.4])
   weighted_ensemble.fit(X_train, y_train)
   y_pred = weighted_ensemble.predict(X_test)

HammersteinWienerClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from gofast.estimators import HammersteinWienerClassifier
   hw_classifier = HammersteinWienerClassifier()
   hw_classifier.fit(X_train, y_train
