.. _mlutils:

Machine Learning Utilities (mlutils)
=====================================

.. currentmodule:: gofast.tools.mlutils

The :mod:`gofast.tools.mlutils` subpackage augments machine learning projects with a 
comprehensive suite of utilities for preprocessing, model evaluation, data handling, 
and more. It streamlines the data science workflow with easy-to-use functions that cover 
a wide range of common tasks in machine learning pipelines.

Key Features
------------

- **Data Preprocessing**: Functions for data cleaning, feature encoding, and imputation.
- **Feature Selection and Engineering**: Tools for selecting relevant features and creating new features.
- **Model Evaluation**: Utilities for assessing model performance with various metrics and validation techniques.
- **Data Splitting**: Methods for dividing datasets into training, validation, and testing sets.
- **Serialization and Deserialization**: Capabilities for saving and loading models and data.
- **Handling Class Imbalance**: Strategies for addressing imbalanced datasets.

Function Descriptions
---------------------

fetch_tgz_from_url
~~~~~~~~~~~~~~~~~~

Automatically downloads and extracts files from a `.tgz` archive located at a specified URL, streamlining the process of acquiring and preparing data for analysis.

.. code-block:: python

    from gofast.tools.mlutils import fetch_tgz_from_url
    fetch_tgz_from_url(
        data_url="https://example.com/data.tgz",
        data_path="path/to/save/data",
        tgz_filename="dataset.tgz"
    )

evaluate_model
~~~~~~~~~~~~~~

Assesses the performance of machine learning models using cross-validation, offering insights into the effectiveness and stability of models across different folds of the data.

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from gofast.tools.mlutils import evaluate_model
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    model = RandomForestClassifier()
    scores = evaluate_model(model, X, y, scoring='accuracy')

select_features
~~~~~~~~~~~~~~~

Facilitates feature selection based on importance, correlation, or custom criteria to enhance model performance by reducing dimensionality and focusing on relevant predictors.

.. code-block:: python

    from gofast.tools.mlutils import select_features
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    selected_features = select_features(X, y, method='mutual_info', threshold=0.05)

get_global_score
~~~~~~~~~~~~~~~~

Calculates a global score for a model across multiple evaluation metrics, providing a comprehensive overview of model performance.

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from gofast.tools.mlutils import get_global_score
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    model = RandomForestClassifier()
    global_score = get_global_score(model, X, y)

get_correlated_features
~~~~~~~~~~~~~~~~~~~~~~~

Identifies features that are highly correlated with each other, aiding in the detection and removal of redundant features to prevent multicollinearity issues.

.. code-block:: python

    import pandas as pd
    from gofast.tools.mlutils import get_correlated_features
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100) * 0.5,
        'C': np.random.rand(100) * 2,
    })
    correlated_features = get_correlated_features(df, threshold=0.8)

codify_variables
~~~~~~~~~~~~~~~~

Transforms categorical variables into numerical codes, facilitating their use in machine learning models that require numerical input.

.. code-block:: python

    import pandas as pd
    from gofast.tools.mlutils import codify_variables
    df = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A']})
    df_encoded = codify_variables(df, columns=['Category'])

categorize_target
~~~~~~~~~~~~~~~~~

Converts a continuous target variable into categorical bins or classes based on specific thresholds, enabling regression tasks to be treated as classification problems.

.. code-block:: python

    import numpy as np
    from gofast.tools.mlutils import categorize_target
    y = np.array([0.1, 0.4, 0.6, 0.8, 0.3])
    y_categorized = categorize_target(y, bins=[0, 0.3, 0.6, 1], labels=["low", "medium", "high"])

resampling
~~~~~~~~~~

Adjusts the distribution of classes in a dataset to address imbalances, using techniques such as oversampling the minority class or undersampling the majority class.

.. code-block:: python

    import numpy as np
    from gofast.tools.mlutils import resampling
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    X_resampled, y_resampled = resampling(X, y, method='oversample')

bin_counting
~~~~~~~~~~~~

Facilitates the bin counting approach for categorical features, which can be particularly useful in high-cardinality categorical data by converting categories into counts or frequencies.

.. code-block:: python

    import pandas as pd
    from gofast.tools.mlutils import bin_counting
    df = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'A']})
    df_binned = bin_counting(df, 'Category')

labels_validator
~~~~~~~~~~~~~~~~

Ensures the specified labels exist within the target array, aiding in validation and error checking for classification tasks.

.. code-block:: python

    import numpy as np
    from gofast.tools.mlutils import labels_validator
    y = np.array([0, 1, 2, 1, 0])
    assert labels_validator(y, labels=[0, 1, 2])

rename_labels_in
~~~~~~~~~~~~~~~~

Renames the specified labels within a target array to new names, enhancing readability and interpretability of classification outcomes.

.. code-block:: python

    import numpy as np
    from gofast.tools.mlutils import rename_labels_in
    y = np.array([0, 1, 2, 1, 0])
    y_renamed = rename_labels_in(y, new_names={0: "Low", 1: "Medium", 2: "High"})


soft_scaler
~~~~~~~~~~~

Applies scaling to numerical features in a dataset while preserving categorical features unchanged, supporting multiple scaling methods.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.tools.mlutils import soft_scaler
    X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
    X_scaled = soft_scaler(X, kind='StandardScaler')
    df = pd.DataFrame({'Income': [50000, 60000, 70000], 'Age': [25, 35, 45], 'Gender': ['Male', 'Female', 'Male']})
    df_scaled = soft_scaler(df, kind='MinMaxScaler', feature_range=(0, 1))

select_feature_importances
~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects features based on their importance scores from a fitted model, aiding in dimensionality reduction and model simplification.

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from gofast.tools.mlutils import select_feature_importances
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    model = RandomForestClassifier().fit(X, y)
    X_selected = select_feature_importances(model, X, threshold=0.05)

make_pipe
~~~~~~~~~

Creates a comprehensive data preprocessing pipeline, integrating various preprocessing steps like scaling, encoding, and imputation into a unified process.

.. code-block:: python

    from gofast.tools.mlutils import make_pipe
    from sklearn.datasets import make_classification
    X, _ = make_classification(n_samples=100, n_features=4, random_state=42)
    pipeline = make_pipe(X, num_features=[0, 1], cat_features=[2, 3])

build_data_preprocessor
~~~~~~~~~~~~~~~~~~~~~~~

Constructs an advanced data preprocessing pipeline capable of handling complex transformations, feature engineering, and selection tasks.

.. code-block:: python

    from gofast.tools.mlutils import build_data_preprocessor
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    preprocessor = build_data_preprocessor(X, num_features=[0, 1], cat_features=[2, 3], feature_interaction=True)

load_saved_model
~~~~~~~~~~~~~~~~

Loads a previously saved machine learning model from a file, supporting both `pickle` and `joblib` formats, enabling model reusability and deployment.

.. code-block:: python

    from gofast.tools.mlutils import load_saved_model
    model = load_saved_model(file_path="path/to/saved_model.pkl")

bi_selector
~~~~~~~~~~~

Automatically differentiates between numerical and categorical features in a dataset, facilitating targeted preprocessing steps for each feature type.

.. code-block:: python

    import pandas as pd
    from gofast.tools.mlutils import bi_selector
    df = pd.DataFrame({'num_feature': [1, 2, 3], 'cat_feature': ['A', 'B', 'C']})
    num_features, cat_features = bi_selector(df)

get_target
~~~~~~~~~~

Extracts the target variable from a dataset based on specified criteria, supporting both DataFrame and array-like inputs.

.. code-block:: python

    import pandas as pd
    from gofast.tools.mlutils import get_target
    df = pd.DataFrame({'feature1': [1, 2, 3], 'target': ['A', 'B', 'A']})
    y, df_without_target = get_target(df, 'target')

extract_target
~~~~~~~~~~~~~

Extract the target variable from a dataset, facilitating separate handling of features and target in machine learning workflows.

.. code-block:: python

    import pandas as pd
    from gofast.tools.mlutils import extract_target
    df = pd.DataFrame({'feature1': [1, 2, 3], 'target': ['A', 'B', 'A']})
    y, df_without_target = extract_target(df, 'target')
    df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'target': [0, 1, 0]})
    y, df_without_target = extract_target(df, 'target', drop=True)

stats_from_prediction
~~~~~~~~~~~~~~~~~~~~~

Generates statistical summaries and accuracy metrics from actual and predicted values, aiding in the evaluation of model performance.

.. code-block:: python

    from gofast.tools.mlutils import stats_from_prediction
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    stats = stats_from_prediction(y_true, y_pred)

fetch_tgz
~~~~~~~~~

Automatically fetches, downloads, and extracts datasets from `.tgz` files, simplifying the data acquisition process.

.. code-block:: python

    from gofast.tools.mlutils import fetch_tgz
    fetch_tgz(data_url="http://example.com/data.tgz", data_path="/local/path", tgz_filename="data.tgz")

fetch_model
~~~~~~~~~~~

Retrieves a machine learning model saved with Python's `pickle` or `joblib` module, supporting efficient model sharing and deployment.

.. code-block:: python

    from gofast.tools.mlutils import fetch_model
    model, best_params, best_scores = fetch_model(file_path="model.pkl", model_name="BestModel")

load_csv
~~~~~~~~

Loads a CSV file into a pandas DataFrame, streamlining the initial data loading step in data analysis and machine learning projects.

.. code-block:: python

    import pandas as pd
    from gofast.tools.mlutils import load_csv
    df = load_csv("path/to/dataset.csv")

discretize_categories
~~~~~~~~~~~~~~~~~~~~~

Transforms a continuous variable into discrete categories, facilitating the use of categorical data analysis techniques on numerical data.

.. code-block:: python

    import numpy as np
    from gofast.tools.mlutils import discretize_categories
    y = np.array([10, 20, 30, 40, 50])
    y_discretized = discretize_categories(y, bins=3, labels=["low", "medium", "high"])
    y_bin_discretized = discretize_categories(y, bins=[0, 10, 30, 50], labels=["low", "medium", "high"])


stratify_categories
~~~~~~~~~~~~~~~~~~~

Performs stratified sampling based on a specified categorical column, ensuring that each split of the data has approximately the same percentage of samples of each target class.

.. code-block:: python
    
    import numpy as np 
    import pandas as pd
    from gofast.tools.mlutils import stratify_categories
    df = pd.DataFrame({'feature': [1, 2, 3, 4, 5], 'category': ['A', 'B', 'A', 'B', 'A']})
    train_set, test_set = stratify_categories(df, 'category')
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    X_train, X_test, y_train, y_test = stratify_categories(X, y, test_size=0.25)
     

serialize_data
~~~~~~~~~~~~~~

Serializes and saves Python objects, such as datasets or models, to a file, providing a convenient way to store and share data.

.. code-block:: python

    from gofast.tools.mlutils import serialize_data
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    serialize_data(data, filename="data.pkl")

deserialize_data
~~~~~~~~~~~~~~~~

Loads Python objects from a serialized file, simplifying the process of retrieving saved datasets or models for analysis or prediction.

.. code-block:: python

    from gofast.tools.mlutils import deserialize_data
    data = deserialize_data(filename="data.pkl")

soft_data_split
~~~~~~~~~~~~~~~

Splits data into training and test sets, accommodating a variety of configurations and supporting complex data structures.

.. code-block:: python

    from gofast.tools.mlutils import soft_data_split
    X, y = soft_data_split(data, target_column='target', test_size=0.2)

laplace_smoothing
~~~~~~~~~~~~~~~~~

Applies Laplace smoothing to numerical data, often used in Bayesian inference to handle zero probabilities in categorical data.

.. code-block:: python

    from gofast.tools.mlutils import laplace_smoothing
    frequencies = [5, 3, 0, 2]
    smoothed_frequencies = laplace_smoothing(frequencies)

laplace_smoothing_categorical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifically applies Laplace smoothing to categorical data, aiding in handling categories with zero occurrences.

.. code-block:: python

    import pandas as pd 
    from gofast.tools.mlutils import laplace_smoothing_categorical
    category_counts = {'A': 5, 'B': 3, 'C': 0, 'D': 2}
    smoothed_counts = laplace_smoothing_categorical(category_counts)
    df = pd.DataFrame({'category': ['A', 'B', 'C', 'A', 'B', 'B', 'C', 'A']})
    df_smoothed = laplace_smoothing_categorical(df, 'category')

laplace_smoothing_word
~~~~~~~~~~~~~~~~~~~~~~

Applies Laplace smoothing to word frequencies in text data, useful in natural language processing applications such as text classification.

.. code-block:: python

    from gofast.tools.mlutils import laplace_smoothing_word
    word_counts = {'the': 5, 'a': 3, 'is': 0, 'on': 2}
    smoothed_word_counts = laplace_smoothing_word(word_counts)
    word_counts = {'word1': 5, 'word2': 3, 'word3': 0}
    smoothed_word_counts = laplace_smoothing_word(word_counts)

handle_imbalance
~~~~~~~~~~~~~~~~

Provides strategies to address imbalanced datasets, such as oversampling the minority class or undersampling the majority class.

.. code-block:: python

    import numpy as np
    from gofast.tools.mlutils import handle_imbalance
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    X_resampled, y_resampled = handle_imbalance(X, y, strategy='oversample')

smart_split
~~~~~~~~~~~

Intelligently splits datasets into training and testing sets while handling multiple target variables and supporting advanced splitting strategies.

.. code-block:: python
    
    import numpy as np
    from gofast.tools.mlutils import smart_split
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    X_train, X_test, y_train, y_test = smart_split(X, y, test_size=0.2, stratify=y)


This documentation is a comprehensive guide to utilizing the `gofast.tools.mlutils` subpackage, covering its key functionalities and providing examples for its primary functions.