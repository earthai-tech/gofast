"""
Regression Prediction Application
=================================

Perform quick regression predictions on a dataset using various scikit-learn
regressors. The script includes feature selection, preprocessing, and optional
hyperparameter tuning (random search by default). It handles saving outputs
such as trained models, prediction files, and metrics logs.

.. math::
   \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }

Parameters
----------
-d, --data : str
    Path to the file to read and predict (preferably in CSV format).
-t, --target : str
    Name of the target column in the dataset.
-c, --columns : list of str, optional
    List of columns to select for prediction. If omitted, all columns are used.
-th, --threshold : float, default=0.1
    Threshold for selecting relevant features based on correlation.
    Default is 0.1 (10%).
-ts, --test-size : float, default=0.3
    Ratio for splitting the dataset into training and testing sets.
    Default is 0.3.
-ft, --tune-params, --fine-tune : bool, default=False
    If set, fine-tunes the estimators with their default parameters.
-v, --verbose : int, default=1
    Verbosity level (0=quiet to 7=debug). Default is 1.
-cm, --corr-method : str, default='pearson'
    Correlation method for feature relevance. Default is 'pearson'.
-o, --output : str, optional
    Destination path (prefix) to save prediction file and metrics text.
-s, --show-fig : bool, default=True
    Display prediction figures like regression score plots. Default is True.
--accept-dt : bool, default=False
    If set, accepts datetime columns; otherwise, datetime columns are dropped.
--dt-as : str, optional
    How to convert datetime columns if accepted (e.g., 'numeric', 'integer',
    'float', 'object'). Default is None.
-sm, --save-model : bool, default=True
    Whether to save the trained models as a joblib file. Default is True.
--search : str, default='RSCV'
    Search strategy for hyperparameter tuning (e.g., 'RSCV', 'GSCV').
-cv, --cross-validation : int, default=3
    Number of cross-validation folds. Default is 3.
--helpdoc : bool, default=False
    Display the script documentation and exit.

Raises
------
ValueError
    If the target is not continuous or other data inconsistencies occur.

Examples
--------
>>> python reg_pred.py -d data.csv -t target_column
>>> python reg_pred.py --data data.csv --target target --columns feature1 feature2 feature3 --tune-params --verbose 2

Notes
-----
This application is designed for regression tasks and assumes the target variable
is continuous. It leverages the gofast library for data preprocessing, feature
selection, and model optimization.

.. math::
   \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }

See Also
--------
``gofast.preprocessing.build_data_preprocessor``,
``gofast.feature_selection.select_relevant_features``,
``gofast.models.optimize.Optimizer``,
``gofast.plot.plot_r2``

References
----------
.. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
    Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning
    in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
"""

# This module provides a quick regression prediction application
# for a CSV (or supported) dataset using scikit-learn regressors.
# The script includes feature selection, preprocessing, and optional
# hyperparameter tuning (random search by default). It handles saving
# outputs such as trained models, prediction files, and metrics logs.

import argparse
import os
import sys
import logging 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from sklearn.utils.multiclass import type_of_target
except ImportError:
    # Fallback if sklearn version does not have type_of_target
    from gofast.core.utils import type_of_target

# from gofast._gofastlog import gofastlog
from gofast.api.util import to_snake_case
from gofast.core.array_manager import to_numeric_dtypes, to_array, to_series
from gofast.core.io import read_data, export_data, print_script_info, show_usage
from gofast.core.checks import check_datetime
from gofast.utils.base_utils import nan_ops, extract_target, select_features
from gofast.utils.ml.preprocessing import build_data_preprocessor
from gofast.utils.ml.feature_selection import select_relevant_features
from gofast.utils.io_utils import save_job
from gofast.models.optimize import Optimizer
from gofast.plot import plot_r2

# Initialize a logger using gofast's logging framework.
# logger = gofastlog().get_gofast_logger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# A module-level flag to ensure docstring is printed only once
_DOCSTRING_PRINTED = False

def reg_pred_app(
    data_path,
    target_col,
    columns=None,
    threshold=0.1,
    test_size=0.3,
    tune_params=False,
    verbose=1,
    corr_method='pearson',
    output=None,
    show_fig=True,
    dt_as=None,
    accept_dt=False,
    save_model=True,
    search='RSCV',
    cv=3
):
    # 1) Attempt to read the dataset from `data_path` with gofast's read_data,
    #    which supports various file formats (e.g., CSV, XLSX).
    try:
        data = read_data(data_path)
    except Exception as e:
        # Log an error message and instruct user about CSV data as a fallback.
        logger.error(
            f"Failed to read data from {data_path}. "
            "Please ensure the file exists and is properly formatted. "
            "CSV is recommended. Details: %s", e
        )
        raise

    # 2) Convert columns with numeric-like strings to numeric dtype,
    #    while preserving other data types as is (categorical remains categorical).
    data = to_numeric_dtypes(data)

    # 3) Check for datetime columns. Depending on `accept_dt` and `dt_as`,
    #    datetime columns might be dropped or converted to numeric representation.
    data = check_datetime(
        data,
        ops='validate',
        consider_dt_as=dt_as,
        accept_dt=accept_dt
    )

    # 4) If `columns` is provided, subset the DataFrame to only those columns.
    if columns is not None:
        data = select_features(data, features=columns)

    # 5) Extract target column from the DataFrame, returning (y, X).
    target, data = extract_target(
        data,
        target_names=target_col,
        return_y_X=True
    )

    # 6) Convert the extracted target to a 1D array, then to a Series.
    #    This ensures the target is properly shaped for regression.
    target = to_array(target, accept='1d', force_conversion=True)
    try:
        target = to_series(target)
    except Exception as e:
        # If the target has multiple columns or an incompatible shape, raise an error.
        logger.error(
            "The target must be a single column. Multi-column target "
            "detection is not supported for regression. Details: %s", e
        )
        raise

    # 7) Remove rows with NaN in target or features, ensuring data integrity.
    target, data = nan_ops(
        target,
        witness_data=data,
        ops='sanitize',
        data_kind='target',
        process='do',
        action='drop'
    )

    # 8) Determine target type via sklearn's type_of_target or fallback.
    #    Raise an error if it's not recognized as continuous, because
    #    this script is for regression tasks only.
    try:
        target_type = type_of_target(target)
    except Exception as e:
        logger.error("Error in determining target type: %s", e)
        raise

    if target_type not in ['continuous', 'continuous-multioutput']:
        raise ValueError(
            f"Detected target_type='{target_type}', but only 'continuous' "
            "or 'continuous-multioutput' are supported for regression. "
            "Consider using a classification script instead."
        )

    # 9) Convert column names to snake_case to standardize usage.
    data.columns = data.columns.str.lower()
    try:
        snake_cols = [to_snake_case(col) for col in data.columns]
        data.columns = snake_cols
    except Exception as e:
        logger.warning(
            "Unable to convert columns to snake_case. Using raw columns. "
            "Details: %s", e
        )

    # 10) Optionally show a correlation heatmap if `show_fig` is True.
    if show_fig:
        corr_mat = data.corr(method=corr_method)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_mat,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=2
        )
        plt.title("Correlation Heatmap")
        plt.show()

    # 11) Select relevant features based on correlation with the target.
    #     This uses a threshold for absolute correlation. The user can
    #     specify `corr_method` (e.g., 'pearson', 'spearman').
    relevant_features = select_relevant_features(
        data,
        target=target,
        threshold=threshold,
        method=corr_method
    )

    # 12) Filter the DataFrame to only the selected features.
    processed_data = data[relevant_features]
    y = target

    # 13) Split data into train/test sets with a default test_size=0.3.
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data,
        y,
        test_size=test_size,
        random_state=42
    )

    # 14) Build a pipeline-based preprocessor that applies scaling (StandardScaler)
    #     and one-hot encoding to handle categorical features.
    processor = build_data_preprocessor(X_train, y_train, label_encoding='onehot')
    X_train_scaled = processor.fit_transform(X_train)
    X_test_scaled = processor.transform(X_test)

    # 15) Define a helper function to tune parameters if `tune_params` is True.
    def _tuning_ways(search_method='RSCV', cv_folds=cv):
        # Provide param grids for random search or other strategies.
        estimators = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "SVR": SVR()
        }
        param_grids = {
            "Linear Regression": {
                'fit_intercept': [True, False],
                # 'normalize': [True, False] # if needed in older sklearn
            },
            "Random Forest": {
                'n_estimators': np.arange(50, 200),
                'max_features': [1.0, 'sqrt', 'log2'],
                'max_depth': np.arange(3, 20),
                'min_samples_split': np.arange(2, 10),
                'min_samples_leaf': np.arange(1, 4)
            },
            "SVR": {
                'C': np.random.uniform(1, 100, size =100),
                'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 5)),
                'kernel': ['rbf', 'linear']
            }
        }
        
        # Initialize gofast's Optimizer for model hyperparameter tuning.
        optimizer = Optimizer(
            estimators,
            param_grids,
            strategy=search_method,
            cv=cv_folds,
            n_jobs=1  # Parallel jobs if needed.
        )
        optimizer.fit(X_train_scaled, y_train)
        if verbose >= 1:
            print(optimizer)

        # Retrieve best estimators from the tuning results.
        best_estimators = {
            "Linear Regression": optimizer.summary.LinearRegression['best_estimator_'],
            "Random Forest": optimizer.summary.RandomForestRegressor['best_estimator_'],
            "SVR": optimizer.summary.SVR['best_estimator_']
        }
        return best_estimators

    # 16) Either tune parameters or define default models for demonstration.
    if tune_params:
        models = _tuning_ways(search_method=search, cv_folds=cv)
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "SVR": SVR(kernel='rbf', C=10, gamma=0.1)
        }

    # 17) Prepare to store metrics and predictions for each model.
    metrics = {m_name: {} for m_name in models.keys()}
    predictions = {}

    # 18) If no output path is specified, define default filenames for text metrics and predictions.
    #     Otherwise, use the directory or basename from `output` to place results.
    if output is None:
        output_text = 'metrics.txt'
        output_pred = 'predictions.csv'
    else:
        base_name = os.path.basename(output)
        dir_name = os.path.dirname(output)
        output_text = os.path.join(dir_name, f"{base_name}_metrics.txt")
        output_pred = os.path.join(dir_name, f"{base_name}_pred.csv")

    # 19) Train each model and gather metrics. For SVR, use scaled data.
    for name, model in models.items():
        if name == "SVR":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        predictions[name] = y_pred
        metrics[name]["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics[name]["MAE"] = mean_absolute_error(y_test, y_pred)
        metrics[name]["R^2"] = r2_score(y_test, y_pred)

    # 20) Optionally save the trained models to disk if `save_model=True`.
    if save_model:
        save_job(
            models,
            savefile=os.path.join(
                os.path.dirname(output_text),
                f"{os.path.basename(output_text).replace('metrics.txt','')}models.joblib"
            )
        )

    # 21) If requested, visualize predictions and metrics with `plot_r2`
    #     plus a bar chart comparing the different models.
    if show_fig:
        # last_model_name = list(models.keys())[-1]
        # last_model_pred = predictions[last_model_name]
        svr_model_pred = predictions['SVR']
        lr_model_pred =  predictions["Linear Regression"]
        rf_model_pred =  predictions["Random Forest"]
        plot_r2(
            # y_test, last_model_pred,
            y_test, lr_model_pred, svr_model_pred, rf_model_pred, 
            other_metrics=['MAE', 'RMSE'],
            # fig_size=(6, 4),
            titles = ["Linear Regression", "SVR", "Random Forest" ]
            # titles=f"{last_model_name} - Actual vs. Predicted"
        )

        # Optionally, compare models in a single figure:
        metric_names = ["RMSE", "MAE", "R^2"]
        colors = ['blue', 'orange', 'green']
        x = np.arange(len(metric_names))
        plt.figure(figsize=(10, 6))
        width = 0.25  # width of each bar
        for i, (m_name, m_vals) in enumerate(metrics.items()):
            plt.bar(x + i * width, m_vals.values(), width=width,
                    label=m_name, color=colors[i % len(colors)])
        plt.xticks(x + width, metric_names)
        plt.ylabel("Score")
        plt.title("Model Performance Metrics")
        plt.legend()
        plt.show()

    # 22) Export the predictions for each model. The `export_data` can handle
    #     dictionaries of arrays and produce a combined DataFrame.
    try:
        export_data(predictions, output_pred, index =False )
    except Exception as e:
        logger.warning(
            "Failed to export predictions to %s. "
            "Details: %s", output_pred, e
        )

    # 23) Write out the metrics to a text file for review.
    try:
        with open(output_text, 'w') as f:
            for m_name, m_vals in metrics.items():
                f.write(f"Model: {m_name}\n")
                for metric_name, val in m_vals.items():
                    f.write(f"    {metric_name}: {val:.4f}\n")
                f.write("\n")
    except Exception as e:
        logger.warning(
            "Failed to write metrics to %s. Details: %s", output_text, e
        )

    # 24) If `verbose` is high, print the final metrics in the console as well.
    if verbose >= 2:
        logger.info("Final Model Metrics:")
        for m_name, m_vals in metrics.items():
            logger.info("%s => %s", m_name, m_vals)

    # The function completes here, having saved predictions, metrics,
    # and optionally the trained model objects.
    return models, metrics, predictions

def main():

    global _DOCSTRING_PRINTED
    if not _DOCSTRING_PRINTED:
        print_script_info(__doc__)
        _DOCSTRING_PRINTED = True
        
    # Creates an argument parser for the reg_pred script.
    parser = argparse.ArgumentParser(
        prog="reg_pred",
        description="Perform quick regression predictions using scikit-learn and gofast."
    )
    
    # Required arguments: data path and target column name.
    parser.add_argument(
        "-d", "--data",
        required=True,
        help="Path to the file to read (CSV recommended)."
    )
    parser.add_argument(
        "-t", "--target",
        required=True,
        help="Name of the target column in the dataset."
    )

    # Optional arguments for feature columns, threshold, test size, and so on.
    parser.add_argument(
        "-c", "--columns",
        nargs="*",
        default=None,
        help="List of columns to select for prediction (space-separated). "
             "If omitted, all columns are used."
    )
    parser.add_argument(
        "-th", "--threshold",
        type=float,
        default=0.1,
        help="Threshold for selecting relevant features based on correlation. Default=0.1"
    )
    parser.add_argument(
        "-ts", "--test-size",
        type=float,
        default=0.3,
        help="Ratio for test split. Default=0.3"
    )
    parser.add_argument(
        "-ft", "--tune-params","--fine-tune",
        action="store_true",
        help="If set, fine-tune the models' hyperparameters with a search strategy."
    )
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=quiet to 7=debug). Default=1"
    )
    parser.add_argument(
        "-cm", "--corr-method",
        default="pearson",
        help="Correlation method for feature relevance. Default='pearson'."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Destination path (prefix) to save prediction file and metrics text."
    )
    parser.add_argument(
        "-s", "--show-fig",
        action="store_true",
        help="If set, display plots for correlations and model performance. Default off."
    )
    parser.add_argument(
        "--accept-dt",
        action="store_true",
        help="Accept datetime columns. If not set, datetime columns are dropped. Default=False."
    )
    parser.add_argument(
        "--dt-as",
        default=None,
        help="How to convert datetime columns if accepted. e.g. 'numeric'. Default=None"
    )
    parser.add_argument(
        "-sm", "--save-model",
        action="store_true",
        default=True,
        help="Whether to save the trained models as a joblib file. Default=True."
    )
    parser.add_argument(
        "--search",
        default="RSCV",
        help="Search strategy for hyperparameter tuning. e.g. 'RSCV' or 'GSCV'. Default='RSCV'."
    )
    parser.add_argument(
        "-cv", "--cross-validation", 
        type=int,
        default=3,
        help="Number of cross-validation folds. Default=3."
    )
    parser.add_argument(
        "--helpdoc", 
        action="store_true", 
        help='Script documentation'
        )
    
    # 1) If no arguments or only the script name, show usage and exit.
    if len(sys.argv) == 1:
        show_usage(parser, script_name="reg_pred")
        sys.exit(0)
        
    # Parses the arguments from the command line.
    args = parser.parse_args()

    # 3) If user specifically asked for helpdoc, show usage and exit.
    if args.helpdoc:
        print_script_info(__doc__)
        sys.exit(0)
        
    # Calls the reg_pred_app function with parsed arguments. 
    # Note that some flags might have different names from function parameters.
    reg_pred_app(
        data_path=args.data,
        target_col=args.target,
        columns=args.columns,
        threshold=args.threshold,
        test_size=args.test_size,
        tune_params=args.tune_params,
        verbose=args.verbose,
        corr_method=args.corr_method,
        output=args.output,
        show_fig=args.show_fig,
        dt_as=args.dt_as,
        accept_dt=args.accept_dt,
        save_model=args.save_model,
        search=args.search,
        cv=args.cross_validation
    )

# If script is run directly, call main().
if __name__ == "__main__":
    main()
