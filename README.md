<img src="docs/_static/gofast_logo.svg"><br>

-----------------------------------------------------

# gofast: AIO Machine Learning Package 🚀

**gofast** is a comprehensive, all-in-one (AIO) machine learning package meticulously 
crafted to streamline and accelerate every facet of your data science workflow. By 
offering a suite of high-speed tools and utilities, **gofast** empowers users to efficiently
navigate the critical stages of data analysis, processing, and modeling with 
exceptional precision and ease. Whether you're handling large datasets, performing
complex computations, or developing sophisticated models, **gofast** ensures that 
each step is executed with optimal performance and reliability, enhancing both 
productivity and accuracy in your machine learning endeavors.

---

## 📈 Table of Contents

- [Features](#features)
- [Project Goals](#project-goals)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🌟 Features

### 🚀 Fast Preprocessing
Simplify and expedite your data preparation with **gofast**’s efficient preprocessing tools. 
Ensure your data is clean, formatted, and ready for analysis in record time.

- **Automatic Data Type Conversion:** Seamlessly convert data types to their optimal formats.
- **Missing Value Handling:** Quickly identify and handle missing values with customizable strategies.
- **Outlier Detection:** Utilize advanced techniques like IQR-based outlier detection to maintain 
data integrity.

### ⚡ Efficient Processing
Leverage robust processing utilities optimized for speed without compromising accuracy. Tackle 
large datasets and complex computations with ease.

- **Vectorized Operations:** Perform operations on entire columns or DataFrames without explicit loops.
- **Parallel Processing:** Utilize multi-core architectures to accelerate data processing tasks.
- **Memory Optimization:** Efficiently manage memory usage, enabling processing of large-scale data.

### ✅ Streamlined Validation
Implement quick and reliable validation methods to assess your models effectively. **gofast** aids in 
fine-tuning model performance, ensuring you achieve the best possible results.

- **Data Integrity Checks:** Comprehensive checks for missing values, duplicates, and data consistency.
- **Model Performance Metrics:** Easily compute and visualize key performance indicators.
- **Automated Reporting:** Generate detailed reports summarizing data and model validation results.

---

## 🎯 Project Goals

1. **Enhance Productivity**  
   Reduce the time spent on routine data tasks. **gofast** is designed to make your workflow more 
   efficient, allowing you to focus on innovation and problem-solving.

2. **User-Friendly**  
   Whether you're a beginner or an expert, **gofast** offers an intuitive and accessible interface 
   for all users in the machine learning community.

3. **Community-Driven**  
   We believe in the power of collaboration. **gofast** is open-source, welcoming contributions and 
   suggestions from the community to continuously improve and evolve.

---

## 🛠 Installation

**gofast** is currently under active development and is **not yet** available on PyPI. Once released, 
you can install **gofast** via:

```bash
pip install gofast
```

### 🔧 Installing from Source

If you wish to try out the latest features before the official release, you can install **gofast** 
directly from the GitHub repository:

```bash
git clone https://github.com/earthai-tech/gofast.git
cd gofast
pip install -e .
```

---

## ⚡ Quick Start

Get up and running with **gofast** in minutes! Follow this step-by-step example 
to understand how **gofast** can enhance your machine learning workflow by simplifying
data integrity verification and model optimization.

First, ensure that **gofast** is installed. Since it's not yet available on PyPI, 
install it directly from the [GitHub repository](#installing-from-Source)


### 🧰 Import Required Libraries

Begin by importing the necessary libraries and modules. **gofast** integrates 
seamlessly with popular Python libraries like NumPy, Pandas, and Scikit-learn 
to provide a robust machine learning environment.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from gofast.utils import build_df 
from gofast.dataops import verify_data_integrity 
from gofast.models.optimize import Optimizer
```

### 📊 Load and Prepare the Dataset

Use Scikit-learn's `load_iris` function to load the Iris dataset. Then, 
construct a Pandas DataFrame with appropriate column names using **gofast**'s 
`build_df` utility. Enable the integrity check to automatically verify the d
ataset's quality.

```python
# Load the Iris dataset
X, y = load_iris(return_X_y=True)
columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Build DataFrame with integrity check
X_df = build_df(X, columns=columns, check_integrity=True)
```

**Output:**

```
================================================================================
                                 Data Integrity                                 
--------------------------------------------------------------------------------
missing_values    : Series ~ values: <mean: 0.0000 - len: 4 - dtype: int64>
duplicates        : 1
n_numeric_columns : 4
outliers          : Dict ~ len:11 - values: <mean: 12.1429 - numval: 7 -
                    nonnumval: 4 - dtype: mixed - exist_nan: False>
outlier_values    : Dict ~ len:8 - values: <mean: 27.0000 - numval: 3 -
                    nonnumval: 5 - dtype: mixed - exist_nan: False>
integrity_checks  : Failed
================================================================================
```

### 🔍 Verify Data Integrity

Utilize **gofast**'s `verify_data_integrity` function to perform a comprehensive 
check on your DataFrame. This function evaluates missing values, duplicates, 
and outliers, providing detailed reports to help you assess and improve data quality.

```python
# Perform data integrity verification
is_valid, report = verify_data_integrity(X_df)

# Display the outlier values report
print(report.outlier_values)
```

**Output:**

```
OutlierValues(
  {

       sepal width (cm) : ['15__4.4', '32__4.1', '33__4.2', '60__2.0'] # mean at index 15, 4.4 is an outlier.

  }
)

[ 1 entries ]
```

### 🔄 Split the Dataset

Divide the dataset into training and testing subsets using Scikit-learn's 
`train_test_split`. This prepares the data for model training and evaluation.

```python
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)
```

### 🛠️ Define and Optimize Models

Define multiple machine learning estimators and their corresponding hyperparameter
 grids. Use **gofast**'s `Optimizer` to perform Grid Search Cross-Validation (GSCV) 
 and identify the best-performing models and hyperparameters.

```python
# Define estimators and their hyperparameter grids
estimators = {
    'SVC': SVC(),
    'SGDClassifier': SGDClassifier()
}

param_grids = {
    'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']},
    'SGDClassifier': {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}
}

# Initialize the Optimizer with GSCV strategy
optimizer = Optimizer(estimators, param_grids, strategy='GSCV', n_jobs=1)

# Fit the models and perform hyperparameter tuning
results = optimizer.fit(X_train, y_train)

# Display the optimization results
print(results)
```

**Output:**

```
                  Optimized Results                       

==============================================================
|                            SVC                             |
--------------------------------------------------------------
                        Model Results                         
==============================================================
Best estimator       : SVC
Best parameters      : {'C': 1, 'kernel': 'linear'}
Best score           : 0.9583
nCV                  : 5
Params combinations  : 4
==============================================================

                   Tuning Results (*=score)                   
==============================================================
        Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
--------------------------------------------------------------
0   (1, linear) 0.9583 0.0456        0.9583        0.0456    1
1      (1, rbf)   0.95 0.0612          0.95        0.0612    2
2  (10, linear)   0.95 0.0612          0.95        0.0612    2
3     (10, rbf)   0.95 0.0612          0.95        0.0612    2
==============================================================


==============================================================
|                       SGDClassifier                        |
--------------------------------------------------------------
                        Model Results                         
==============================================================
Best estimator       : SGDClassifier
Best parameters      : {'alpha': 0.001, 'max_iter': 100}
Best score           : 0.8833
nCV                  : 5
Params combinations  : 4
==============================================================

                   Tuning Results (*=score)                   
==============================================================
        Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
--------------------------------------------------------------
0  (0.0001, 50)   0.65 0.1799          0.65        0.1799    4
1 (0.0001, 100) 0.7333  0.176        0.7333         0.176    3
2   (0.001, 50)    0.8 0.1219           0.8        0.1219    2
3  (0.001, 100) 0.8833 0.0717        0.8833        0.0717    1
==============================================================
```

### 🏆 Evaluate and Select the Best Model

After optimization, review the results to select the best-performing models based on 
your evaluation metrics.

```python
# Example: Accessing the best estimator for SVC
best_svc = results.SVC['best_estimator_'] # or results['[SVC]['best_estimator_']
print("Best SVC Parameters:", best_svc.get_params())

# Example: Accessing the best estimator for SGDClassifier
best_sgd = results.SGDClassifier['best_estimator_']
print("Best SGDClassifier Parameters:", best_sgd.get_params())
```

**Output:**

```
Best SVC Parameters: {'C': 1, 'break_ties': False, 'cache_size': 200, 
                     'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr',
                      'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'max_iter': -1, 
                      'probability': False, 'random_state': None, 'shrinking': True, 
                      'tol': 0.001, 'verbose': False
                      }
Best SGDClassifier Parameters: {'alpha': 0.001, 'average': False, 'class_weight': None, 
                               'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.0,
                                'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal',
                                'loss': 'hinge', 'max_iter': 100, 'n_iter_no_change': 5,
                                'n_jobs': None, 'penalty': 'l2', 'power_t': 0.5, 'random_state': None,
                                 'shuffle': True, 'tol': 0.001, 'validation_fraction': 0.1, 
                                 'verbose': 0, 'warm_start': False
                                 }
```

### 📈 Visualize Model Performance

Leverage **gofast**'s reporting tools to visualize and interpret model performance 
metrics effectively.

```python
from gofast.reporting import ReportGenerator

# Generate and save a comprehensive report for model optimization
ReportGenerator.generate(
    report_data=results,
    output_file='output/model_optimization_report.html',
    include_plots=True
)
```

**Output:**

A detailed HTML report named `model_optimization_report.html` is generated in the `output` directory, 
showcasing the optimization results with visual plots for better insights.


---

## 📝 Usage Examples

Explore how **gofast** can enhance various aspects of your machine learning workflow 
through practical examples. Below are detailed demonstrations covering data assistance,
data engineering, and specialized metrics computations.



### 🔍 Data Assistant

Leverage **gofast**'s `data_assistant` to gain insightful details and actionable 
recommendations for processing your dataset. This utility simplifies the initial 
exploration and preparation steps, ensuring your data is primed for analysis.

```python
from gofast.datasets import simulate_world_mineral_reserves
from gofast.dataops import data_assistant

# Simulate a dataset representing world mineral reserves
reserves_data = simulate_world_mineral_reserves().frame

# Analyze the dataset to receive detailed insights and processing recommendations
data_assistant(reserves_data)
```

**Explanation:**

1. **Dataset Simulation:**
   - **`simulate_world_mineral_reserves`:** Generates a synthetic dataset mimicking real-world mineral reserves across different regions of the world.
   - **`.frame`:** Accesses the Pandas DataFrame representation of the simulated data.

2. **Data Assistance:**
   - **`data_assistant`:** Analyzes the `reserves_data` DataFrame to provide comprehensive details such as data types, missing values, duplicates, and other relevant metrics. It may also offer recommendations for data cleaning and preprocessing based on the initial analysis.

**Output:**

```
[SKIP] We skip the assistant display to let the user explore it him/herself.
```


### ⚙️ Data Engineering

Utilize **gofast**'s advanced data engineering tools to clean your dataset, engineer 
meaningful features, and assess feature contributions. This streamlined process ensures 
your data is both high-quality and optimized for model training.

```python
from gofast.utils import nan_ops 
from gofast.dataops import corr_engineering
from gofast.feature_selection import display_feature_contributions

# Step 1: Clean the data by handling missing values intelligently
reserves_cleaned = nan_ops(
    reserves_data, 
    ops='sanitize',    # Operation mode: 'sanitize' for cleaning data
    action='fill'      # Action to perform: 'fill' missing values
)

# Step 2: Engineer features by analyzing and transforming correlations
reserves_transformed = corr_engineering( 
    reserves_cleaned,  
    target='technology_used', 
    method='pearson', 
    threshold_features=0.8,      # Threshold for feature correlation
    threshold_target=0.1,        # Threshold for target correlation
    action='drop',               # Action to perform: 'drop' highly correlated features
    strategy='average',          # Strategy for handling correlated features
    precomputed=False,           # Whether correlations are precomputed
    analysis='dual',             # Analyze correlations with both numeric and categorical data
    show_corr_results=True,      # Display correlation results
)

# Step 3: Select the target variable for modeling
y = reserves_transformed['technology_used']

# Step 4: Display feature contributions to understand their impact on the target variable
summary = display_feature_contributions(
    reserves_transformed, 
    y, 
    prefit=False        # Whether the model is pre-fitted
)

# Step 5: Print the feature contributions summary
print(summary)
```

**Explanation:**

1. **Handling Missing Values:**
   - **`nan_ops`:** Cleans the `reserves_data` by filling missing values based on specified strategies.
     - **`ops='sanitize'`:** Specifies the operation mode as sanitization.
     - **`action='fill'`:** Chooses to fill missing values, potentially using mean, median, or other imputation methods.

2. **Feature Engineering through Correlation Analysis:**
   - **`corr_engineering`:** Analyzes and transforms the dataset by examining feature correlations.
     - **`target='technology_used'`:** Defines the target variable for correlation with other features.
     - **`method='pearson'`:** Utilizes Pearson correlation for assessing linear relationships.
     - **`threshold_features=0.8`:** Sets a high correlation threshold to identify and handle highly correlated features.
     - **`threshold_target=0.1`:** Sets a lower threshold for features correlated with the target variable.
     - **`action='drop'`:** Specifies that highly correlated features should be dropped to reduce multicollinearity.
     - **`strategy='average'`:** Determines how to handle multiple correlated features, such as averaging their values or selecting one representative feature.
     - **`analysis='dual'`:** Indicates that both numeric and categorical data will be analyzed for correlations.
     - **`show_corr_results=True`:** Enables the display of correlation results for user review.

3. **Target Variable Selection:**
   - **`y = reserves_transformed['technology_used']`:** Extracts the target variable from the transformed dataset for subsequent modeling.

4. **Feature Contribution Analysis:**
   - **`display_feature_contributions`:** Evaluates and displays the contribution of each feature towards predicting the target variable.
     - **`prefit=False`:** Indicates that the model used for assessing feature contributions is not pre-fitted and will be trained as part of this function.

5. **Output:**
   
    ```
    ============================
    Feature Contributions Table 
    ----------------------------
    quantity           : 0.0345
    estimated_reserves : 0.1051
    reserve_life       : 0.0665
    region             : 0.0393
    location           : 0.105
    mineral_type       : 0.0912
    extraction_cost    : 0.0715
    grade              : 0.0579
    ownership          : 0.0748
    regulatory_status  : 0.0889
    market_demand      : 0.0431
    ============================
    ```

   *Explanation:* This table summarizes how each feature contributes to predicting the
    `technology_used` target variable. Higher values indicate a more significant impact on 
    the prediction.



### 📊 Metric Specials

Enhance your model evaluation with specialized metrics computations provided by **gofast**. The 
`miv_score` function calculates Mean Impact Values (MIV) through feature perturbations, offering deeper 
insights into feature importance relative to model predictions.

```python
from gofast.metrics_special import miv_score
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y_iris = iris['target']

# Compute Mean Impact Values (MIV) for feature importance
miv_results = miv_score(
    perturbation=0.05, # 5% pertubation
    X=X_iris, 
    y=y_iris, 
    plot_type='bar', 
    percent=True,     # Display MIV in percentage
    relative=True,    # Compute MIV relative to original predictions
)

# Display the MIV results
print(miv_results)
print(miv_results.feature_contributions_)
```

**Explanation:**

1. **Metric Computation:**
   - **`miv_score`:** Calculates the Mean Impact Values (MIV) for each feature by perturbing feature values and observing the effect on model predictions.
     - **`pertubation`:** The magnitude of perturbation applied to each feature during MIV computation. A value of ``0.05`` corresponds to a 5% perturbation of the feature'soriginal value.
     - **`X=X_iris`:** The feature set from the Iris dataset.
     - **`y=y_iris`:** The target variable from the Iris dataset.
     - **`plot_type='bar'`:** Specifies that the MIV results will be visualized as a bar chart.
     - **`percent=True`:** Displays MIV values as percentages for easier interpretation.
     - **`relative=True`:** Computes MIV relative to the original predictions, providing a normalized view of feature importance.

2. **Output Display:**
   - **`print(miv_results)`:** Prints a formatted report of the MIV results.
   - **`print(miv_results.feature_contributions_)`:** Outputs the raw dictionary containing feature contributions.

**Sample Output:**

```
=============================================================================
                                M.I.V Results                                
=============================================================================
feature_contributions_ : Dict (minval=0.3333, maxval=2.0, mean=1.25, items=4)
original_predictions_  : None
petal length (cm)      : 2.0
petal width (cm)       : 1.6667
sepal length (cm)      : 1.0
sepal width (cm)       : 0.3333
=============================================================================

miv_results.feature_contributions_
Out[8]: 
{'sepal length (cm)': 0.9999999999999999,
 'sepal width (cm)': 0.3333333333333333,
 'petal length (cm)': 1.9999999999999998,
 'petal width (cm)': 1.6666666666666667}
```

*Explanation:* The MIV results indicate the relative importance of each feature in predicting the target 
variable. For instance, `petal length (cm)` has the highest impact (2.0), suggesting it is the most 
influential feature in the model's predictions.

---

By following the **Quick Start** guide and these **Usage Examples**, you can harness the full potential 
of **gofast** to streamline your machine learning workflow, data analysis, engineering, and model 
evaluation processes. Whether you're verifying data integrity, engineering robust features, or evaluating
model metrics, **gofast** provides the tools you need to enhance your machine learning workflow 
effectively.

## 📚 Documentation

Comprehensive documentation for **gofast** is currently under development and
will be available soon. Our upcoming documentation will include detailed guides,
API references, tutorials, and practical examples to help you maximize the 
potential of **gofast** in your machine learning projects. Visit our
[GitHub repository](https://github.com/earthai-tech/gofast)  to stay up-to-date, 
to access the latest updates, contribute to the project, and participate in discussions.

### Contribute to the Documentation:

We believe in the power of community collaboration. Once the initial documentation is released, 
we welcome contributions from users and developers to enhance and expand the available resources. 
Whether it's improving existing guides, adding new tutorials, or refining the API reference, 
your input is invaluable in making **gofast** a more robust and user-friendly tool.

---


## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving 
documentation, or adding new features, your help is invaluable.

### 🛠 How to Contribute

1. **Fork the Repository**  
   Click the "Fork" button at the top right of the repository page.

2. **Clone Your Fork**  
   ```bash
   git clone https://github.com/earthai-tech/gofast.git
   cd gofast
   ```

3. **Create a New Branch**  
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**  
   Implement your feature or fix the bug.

5. **Commit Your Changes**  
   ```bash
   git commit -m "Add your descriptive commit message"
   ```

6. **Push to Your Fork**  
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**  
   Navigate to the original repository and click "Compare & pull request."


---

## 📄 License

Distributed under the [BSD-3 Clause License](https://github.com/earthai-tech/gofast/blob/main/LICENSE).


## 🙏 Acknowledgments

- **Pandas**: For providing the powerful data manipulation capabilities.
- **NumPy**: For efficient numerical operations.
- **Scikit-learn**: For foundational machine learning algorithms.
- **Community Contributors**: Special thanks to all the contributors who have helped shape **gofast**.

---
## 📫 Contact

For any inquiries, suggestions, or support, please reach out to us at [contact@gofast.org](mailto:etanoyau@gmail.com).
See more about the developer [here](https://earthai-tech.github.io/).

Stay tuned for the official launch of the **gofast** documentation and embark on a streamlined,
efficient machine learning journey with us!

_Join us in making machine learning workflows faster and more efficient. With gofast, you're not just processing data; you're accelerating toward breakthroughs and innovations._

