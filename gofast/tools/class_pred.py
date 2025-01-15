# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

r"""
Classification Prediction Application
=====================================

Perform quick classification predictions on a dataset using various
scikit-learn classifiers. This script includes feature selection,
preprocessing, and optional hyperparameter tuning (random search by
default). It also handles saving outputs such as trained models,
prediction files, and metrics logs.

.. math::
   \\text{Accuracy} = \\frac{\\text{Number of correct predictions}}
                            {\\text{Total number of predictions}}

Parameters
----------
``-d, --data`` : str
    Path to the file to read and predict (preferably in CSV format).

``-t, --target`` : str
    Name of the target column in the dataset.

``-c, --columns`` : list of str, optional
    List of columns to select for classification. If omitted, 
    all columns are used.

``-th, --threshold`` : float, default=0.1
    Threshold for selecting relevant features based on correlation.
    Default is 0.1 (10%).

``-ts, --test-size`` : float, default=0.3
    Ratio for splitting the dataset into training and testing sets.
    Default is 0.3.

``-ft, --tune-params, --fine-tune`` : bool, default=False
    If set, fine-tunes the classifiers using a hyperparameter search
    strategy.

``-v, --verbose`` : int, default=1
    Verbosity level (0=quiet to 7=debug). Default is 1.

``-cm, --corr-method`` : str, default='pearson'
    Correlation method for feature relevance. Default is 'pearson'.

``-o, --output`` : str, optional
    Destination path (prefix) to save the prediction file and the metrics
    text file.

``-s, --show-fig`` : bool, default=True
    If set, displays plots of correlation heatmaps and model performance
    (confusion matrices and ROC curves). Default is True.

``--accept-dt`` : bool, default=False
    If set, accepts datetime columns. Otherwise, datetime columns
    are dropped by default.

``--dt-as`` : str, optional
    How to convert datetime columns if accepted (e.g., ``'numeric'``,
    ``'float'``, etc.). Default is None.

``-sm, --save-model`` : bool, default=True
    Whether to save trained model objects as joblib files. Default is True.

``--search`` : str, default='RSCV'
    Search strategy for hyperparameter tuning. Possible values include
    ``'RSCV'`` (RandomizedSearchCV) or ``'GSCV'`` (GridSearchCV).

``-cv, --cross-validation`` : int, default=3
    Number of cross-validation folds during hyperparameter tuning.
    Default is 3.

``--helpdoc`` : bool, default=False
    If set, displays the script's documentation and exits immediately.

Raises
------
ValueError
    If the target variable is not suitable for classification or if
    data inconsistencies are detected (e.g., multi-column target).

Examples
--------
>>> python class_pred.py -d data.csv -t target_column
>>> python class_pred.py --data data.csv --target target --columns feat1 feat2 \
...   --tune-params --verbose 2

Notes
-----
- This application is designed for classification tasks, where the
  target variable is binary or multiclass. It leverages the ``gofast``
  library for data preprocessing (including optional encoding),
  feature selection, and model optimization.
  
- The default classifiers include Logistic Regression, Random Forest,
  and SVC. By specifying the ``-ft`` flag, you can trigger
  hyperparameter tuning via the ``gofast.models.optimize.Optimizer``
  framework.

.. math::
   \\text{Precision} = \\frac{\\text{True Positive}}
                            {\\text{True Positive} + \\text{False Positive}}

.. math::
   \\text{Recall} = \\frac{\\text{True Positive}}
                         {\\text{True Positive} + \\text{False Negative}}

See Also
--------
``gofast.utils.ml.preprocessing.build_data_preprocessor`` :
    Builds a pipeline for feature scaling/encoding.
``gofast.utils.ml.feature_selection.select_relevant_features`` :
    Selects features via correlation thresholding.
``gofast.models.optimize.Optimizer`` :
    Performs random or grid search for hyperparameter tuning.
``gofast.plot.plot_cm`` :
    Plots confusion matrices and optional ROC curves for multiple models.

References
----------
.. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
       Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
       Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M.,
       & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python.
       *Journal of Machine Learning Research*, 12, 2825–2830.
"""

import argparse
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # noqa 

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)

from gofast.api.util import to_snake_case
from gofast.compat.sklearn import type_of_target 
from gofast.core.array_manager import (
    to_numeric_dtypes,
    to_array,
    to_series
)
from gofast.core.io import (
    read_data,
    export_data,
    print_script_info,
    show_usage
)
from gofast.core.checks import check_datetime
from gofast.dataops import ( 
    handle_unique_identifiers, 
    data_assistant,  
    corr_engineering_in 
)
from gofast.utils.base_utils import (
    extract_target,
    select_features, 
    map_values, 
)
from gofast.utils.data_utils import nan_ops 
from gofast.utils.ml.preprocessing import (
    build_data_preprocessor,
    soft_encoder, 
    handle_minority_classes 
)
from gofast.utils.ml.feature_selection import select_relevant_features
from gofast.utils.io_utils import save_job
from gofast.models.optimize import Optimizer
from gofast.transformers.feature_engineering import FloatCategoricalToInt
from gofast.plot import plot_cm

# Configure Python's logging system to handle verbosity from 0..7
# (0=silent, 1=minimal, >=2=info, >=5=debug, etc.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

_DOCSTRING_PRINTED = False  # Ensures script docs only print once.

def class_pred_app(
    data_path : str,
    target_col : str,
    columns : list = None,
    threshold : float= 0.1,
    test_size : float = 0.2,
    tune_params : bool = False,
    verbose : int = 1,
    corr_method : str = 'pearson',
    output : str  = None,
    show_fig : bool = True,
    dt_as : str  = None,
    accept_dt : bool = False,
    save_model : bool = True,
    search  : str = 'RSCV',
    cv : int = 3
):
    """
    Classification pipeline for scikit-learn estimators with optional
    hyperparameter tuning. 
    (Docstring left minimal. See instructions for usage.)
    """

    # (1) Read dataset from path, handle exception if fails.
    if verbose >= 2:
        logger.info("Reading data from %s ...", data_path)
    try:
        data = read_data(data_path)
    except Exception as e:
        logger.error(
            "Failed to read data from %s. Check file existence/format. "
            "CSV is recommended. Error: %s", data_path, e
        )
        raise

    # (2) Convert numeric-like strings to numeric dtype, keep other types as is.
    if verbose >= 3:
        logger.info("Converting numeric-like columns to numeric types.")
    data = to_numeric_dtypes(data, verbose=verbose )

    # (3) Handle datetime columns based on dt_as, accept_dt.
    if verbose >= 3:
        logger.info("Processing datetime columns with dt_as=%s, accept_dt=%s",
                    dt_as, accept_dt)
    data = check_datetime(
        df= data,
        ops = 'validate',
        consider_dt_as = dt_as,
        accept_dt  = accept_dt
    )

    # (4) Subset columns if user specified a list of features.
    if columns is not None:
        if verbose >= 2:
            logger.info("Subsetting columns: %s", columns)
        data = select_features(data, features=columns)

    # (5) Extract target column from the DataFrame.
    if verbose >= 2:
        logger.info("Extracting target '%s' from dataset.", target_col)
    target, data = extract_target(
        data,
        target_names=target_col,
        return_y_X=True
    )

    # (6) Convert target to 1D array, then Series.
    target = to_array(
        target,
        accept='1d',
        force_conversion=True, 
        verbose=verbose, 
    )
    try:
        target = to_series(target)
    except Exception as e:
        logger.error(
            "The target must be a single column. Encountered error: %s", e
        )
        raise

    # (7) Remove any rows with NaN in target or features.
    if verbose >= 3:
        logger.info("Dropping rows with NaN values in target/features.")
        
    target, data = nan_ops(
        target,
        auxi_data= data,
        ops = 'sanitize',
        data_kind = 'target',
        process = 'do',
        action = 'drop', 
        verbose=verbose, 
    )
    # Then suppress the unique idenfifiers in the data snce it is not usefull 
    # for machine learning training.
    data_n = handle_unique_identifiers(
        data, 
        threshold =0.95,  
        action ='drop', 
        view =True
    )
    if len(data_n) ==0: 
        # then display in-depth analysis of the dataset the gofast assistant
        # tool to let the user focus on 
        # the recommendation proposed by gofast assistant tool details 
        data_assistant(data)
        # after assitant has 
        raise ValueError (
            "Too much identifiers found in the dataframe." 
            " Consider revisiting the whole dataframe and follow "
            " recommendations details proposed by assistant tool above."
            )
        
    data = data_n.copy() # remake the copy to redefine the data for next steps 
    
    # (7.1) Soft-encode target for classification: ensures int codes.
    target, target_map = soft_encoder(target, return_cat_codes=True)
    # soft_encoded the data for for correlation analysis
    data_enc= soft_encoder(data)
    
    target.columns = (
        'target'
        if 'col_' in target.columns[0]
        else target.columns
    )
    # Convert any float categories to int
    target = FloatCategoricalToInt().fit_transform(target)
    target = to_series(target)

    # (8) Ensure target is recognized as binary/multiclass. If not, raise error.
    try:
        target_type = type_of_target(target)
    except Exception as e:
        logger.error("Error determining target type: %s", e)
        raise
    if target_type not in (
        'binary',
        'multiclass',
        'multiclass-multioutput'
    ):
        raise ValueError(
            f"Detected target_type='{target_type}'. Only 'binary' or "
            "'multiclass' are supported for classification."
        )

    # (9) Convert column names to snake_case for consistency.
    data.columns = data.columns.str.lower()
    try:
        snake_cols = [to_snake_case(c) for c in data.columns]
        data.columns = snake_cols
        data_enc.columns = snake_cols 
    except Exception as e:
        if verbose >= 4:
            logger.warning(
                "Column name conversion to snake_case failed; using raw. %s", e
            )

    # (10) Optionally show correlation heatmap.
    # drop the correlated features and eretrun
    data= corr_engineering_in(
            data,
            analysis='dual_merge', 
            action='drop',  
            threshold=0.8, 
            cmap ='coolwarm',
            show_corr_results=False,
            view=show_fig, 
            linewidths=2
        )
    
    if show_fig:
        if verbose >= 2:
            logger.info("Generating correlation heatmap using %s method.",
                        corr_method
        )
        # corr_mat = data_enc.corr(method=corr_method)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(
        #     corr_mat,
        #     annot=True,
        #     cmap='coolwarm',
        #     fmt=".2f",
        #     linewidths=2
        # )
        # plt.title("Correlation Heatmap")
        # plt.show()

    # (11) Select relevant features based on correlation threshold w.r.t. target.
    if verbose >= 2:
        logger.info("Selecting relevant features with threshold=%.2f using %s.",
                    threshold, corr_method)
    relevant_features = select_relevant_features(
        data = data,
        target = target,
        threshold=threshold,
        method = corr_method
    )

    # (12) Keep only the relevant columns in data.
    processed_data = data[relevant_features] if relevant_features else data 
    y = target
    
    # XXX let try to handle minory classes if few members are detected.
    y, processed_data = handle_minority_classes(
        y, data=processed_data, 
        verbose =verbose,
        techn='drop', 
        min_count=5 # 5 to be sure to have stratification at least 2 samples in both sides 
        # training and testing, 
        )
    if len(y) ==0: # mean is empty
        raise TypeError(
            "Inconsistent target with all minority classes. Consider"
            " collecting more data or ..."
            )
    
    # (13) Train/test split
    if verbose >= 2:
        logger.info("Splitting data with test_size=%.2f", test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data,
        y,
        test_size= test_size,
        random_state=42, 
        stratify=y
    )

    # (14) Build a pipeline with StandardScaler + onehot for categorical
    #      or other transformations if needed.
    if verbose >= 2:
        logger.info("Building data preprocessor pipeline (scaling + onehot).")
    processor= build_data_preprocessor(
        X_train,
        y_train,
        label_encoding='onehot'
    )
    X_train_scaled = processor.fit_transform(X_train)
    X_test_scaled = processor.transform(X_test)

    # (15) Function to tune parameters if requested.
    def _tuning_ways(
        search_method='RSCV',
        cv_folds     = cv
    ):
        # Prepare a dictionary of possible classifiers
        estimators = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest" : RandomForestClassifier(random_state=42),
            "SVC" : SVC()
        }
        # Param grids for random or grid search
        param_grids = {
            "Logistic Regression": {
                'fit_intercept': [True, False]
            },
            "Random Forest": {
                'n_estimators' : np.arange(50, 200),
                'max_features' : [1.0, 'sqrt', 'log2'],
                'max_depth' : np.arange(3, 20),
                'min_samples_split': np.arange(2, 10),
                'min_samples_leaf': np.arange(1, 4)
            },
            "SVC": {
                'C' : np.random.uniform(1, 100, size=100),
                'gamma' : ['scale', 'auto'] + list(np.logspace(-3, 1, 5)),
                'kernel': ['rbf', 'linear']
            }
        }
        opt = Optimizer(
            estimators  = estimators,
            param_grids = param_grids,
            strategy = search_method,
            cv  = cv_folds,
            n_jobs = 1
        )
        if verbose >= 2:
            logger.info("Starting hyperparameter tuning with strategy=%s", search_method)
        opt.fit(X_train_scaled, y_train)
        if verbose >= 1:
            print(opt)

        # Return best estimators from the tuning results
        best_models = {
            "Logistic Regression" : opt.summary.LogisticRegression['best_estimator_'],
            "Random Forest" : opt.summary.RandomForestClassifier['best_estimator_'],
            "SVC" : opt.summary.SVC['best_estimator_']
        }
        if verbose >= 4:
            logger.debug("Best estimators from search:\n%s", best_models)
        return best_models

    # (16) Either run parameter tuning or define defaults
    if tune_params:
        if verbose >= 2:
            logger.info("Tuning parameters for classifiers.")
        models = _tuning_ways(search_method=search, cv_folds=cv)
    else:
        if verbose >= 2:
            logger.info("Using default classifier parameters.")
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest"  : RandomForestClassifier(
                n_estimators=100, random_state=42
            ),
            "SVC" : SVC(kernel='rbf', C=10, gamma=0.1)
        }

    # (17) Prepare dictionary for metrics and predicted values
    metrics     = {m: {} for m in models}
    predictions = {}

    # (18) If user hasn't specified an output path, define default files
    if output is None:
        output_text = 'metrics.txt'
        output_pred = 'predictions.csv'
    else:
        base_name = os.path.basename(output)
        dir_name = os.path.dirname(output)
        output_text = os.path.join(dir_name, f"{base_name}_metrics.txt")
        output_pred = os.path.join(dir_name, f"{base_name}_pred.csv")

    # (19) Train each model, gather predictions, compute metrics
    for m_name, model in models.items():
        if verbose >= 2:
            logger.info("Fitting model: %s", m_name)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        predictions[m_name]         = y_pred
        metrics[m_name]["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics[m_name]["Precision"] = precision_score(
            y_test,
            y_pred,
            average='macro'  # or 'weighted' if needed
        )
        metrics[m_name]["Recall"] = recall_score(
            y_test,
            y_pred,
            average='macro'
        )

    # (20) If saving trained models, store them in a joblib file
    if save_model:
        if verbose >= 2:
            logger.info("Saving trained models to disk.")
        try:
            out_modelpath = os.path.join(
                os.path.dirname(output_text),
                f"{os.path.basename(output_text).replace('metrics.txt','')}models.joblib"
            )
            save_job(models, savefile=out_modelpath)
        except Exception as e:
            if verbose >= 1:
                logger.warning("Could not save models. Error: %s", e)

    # (21) If user wants to show figures, produce confusion matrices for each model 
    #      plus optional ROC (plot_cm).
    if show_fig:
        if verbose >= 2:
            logger.info("Generating confusion matrices & ROC if applicable.")
        # Prepare predicted arrays in the order of models dictionary
        y_pred_list = []
        model_names = []
        for nm in models:
            y_pred_list.append(predictions[nm])
            model_names.append(nm)

        # Plot them side by side with ROC 
        plot_cm(
            y_test,
            *y_pred_list,
            add_roc_curves = True,
            pos_label  = 1,
            titles = model_names
        )
        plt.show()

        # Optionally produce a bar chart comparing different models 
        # for Accuracy, Precision, Recall
        metric_names = ["Accuracy", "Precision", "Recall"]
        colors = ["blue", "orange", "green"]
        x = np.arange(len(metric_names))
        plt.figure(figsize=(10, 6))
        width = 0.25
        for i, (nm, vals) in enumerate(metrics.items()):
            plt.bar(
                x + i*width,
                vals.values(),
                width=width,
                label=nm,
                color=colors[i % len(colors)]
            )
        plt.xticks(x + width, metric_names)
        plt.ylabel("Score")
        plt.title("Model Performance Metrics")
        plt.legend()
        plt.show()

    # (22) Export predictions using gofast's export_data (merges them in a DataFrame).
    if verbose >= 2:
        logger.info("Exporting model predictions to '%s'.", output_pred)
        
    if target_map.results: 
        # Not empty dict, then map the values and add to 
        # the existing predictions  
       predictions = map_values (
           predictions, map_dict = target_map.results,
           action ='append', 
           )
    try:
        export_data(predictions, output_pred, index=False)
    except Exception as e:
        logger.warning("Failed to export predictions to %s. Error: %s", output_pred, e)

    # (23) Write out the metrics to a text file.
    if verbose >= 2:
        logger.info("Writing metrics to '%s'.", output_text)
    try:
        with open(output_text, 'w') as f:
            for nm, vals in metrics.items():
                f.write(f"Model: {nm}\n")
                for metric_name, val in vals.items():
                    f.write(f"    {metric_name}: {val:.4f}\n")
                f.write("\n")
    except Exception as e:
        logger.warning("Failed to write metrics to %s. Error: %s", output_text, e)

    # (24) If verbose >=2, print final metrics to console.
    if verbose >= 2:
        logger.info("Final Model Metrics:")
        for nm, vals in metrics.items():
            logger.info("%s => %s", nm, vals)

    return models, metrics, predictions


def main():
    global _DOCSTRING_PRINTED
    if not _DOCSTRING_PRINTED:
        print_script_info(__doc__)
        _DOCSTRING_PRINTED = True

    parser = argparse.ArgumentParser(
        prog = "class_pred",
        description = "Perform quick classification predictions using scikit-learn and gofast."
    )

    # Required arguments
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

    # Optional arguments
    parser.add_argument(
        "-c", "--columns",
        nargs="*",
        default=None,
        help="List of columns for the model (space-separated). If omitted, uses all."
    )
    parser.add_argument(
        "-th", "--threshold",
        type=float,
        default=0.1,
        help="Threshold for relevant feature selection. Default=0.1"
    )
    parser.add_argument(
        "-ts", "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio. Default=0.2"
    )
    parser.add_argument(
        "-ft", "--tune-params",
        "--fine-tune",
        action="store_true",
        help="If set, fine-tunes models with hyperparameter search."
    )
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=quiet .. 7=debug). Default=1"
    )
    parser.add_argument(
        "-cm", "--corr-method",
        default="pearson",
        help="Correlation method for feature relevance. Default='pearson'."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Prefix for saving predictions and metrics text."
    )
    parser.add_argument(
        "-s", "--show-fig",
        action="store_true",
        help="If set, display correlation and performance plots. Default off."
    )
    parser.add_argument(
        "--accept-dt",
        action="store_true",
        help="Accept datetime columns. If not set, they're dropped. Default=False."
    )
    parser.add_argument(
        "--dt-as",
        default=None,
        help="Convert datetime columns as 'numeric', etc. Default=None."
    )
    parser.add_argument(
        "-sm", "--save-model",
        action="store_true",
        default=True,
        help="Whether to save trained models. Default=True."
    )
    parser.add_argument(
        "--search",
        default="RSCV",
        help="Strategy for hyperparameter tuning, e.g. 'RSCV' or 'GSCV'. Default='RSCV'."
    )
    parser.add_argument(
        "-cv", "--cross-validation",
        type=int,
        default=3,
        help="Number of CV folds. Default=3."
    )
    parser.add_argument(
        "--helpdoc",
        action="store_true",
        help="Display script documentation and exit."
    )

    # Show usage if no arguments
    if len(sys.argv) == 1:
        show_usage(parser, script_name="class_pred")
        sys.exit(0)

    args = parser.parse_args()

    if args.helpdoc:
        print_script_info(__doc__)
        sys.exit(0)

    class_pred_app(
        data_path = args.data,
        target_col = args.target,
        columns = args.columns,
        threshold = args.threshold,
        test_size = args.test_size,
        tune_params = args.tune_params,
        verbose = args.verbose,
        corr_method = args.corr_method,
        output = args.output,
        show_fig  = args.show_fig,
        dt_as = args.dt_as,
        accept_dt = args.accept_dt,
        save_model= args.save_model,
        search = args.search,
        cv  = args.cross_validation
    )


if __name__ == "__main__":
    main()
