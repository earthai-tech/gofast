"""
Plot Subsidence Prediction Application
======================================

**Script Name:** plot_subsi_p_app.py

**Description:**
This application facilitates data processing and visualization for land subsidence
prediction using real datasets. It encompasses steps such as data loading, scaling,
outlier handling, transformation, merging, and various plotting functionalities to
analyze and visualize subsidence patterns spatially and categorically.

Designed for flexibility, users can specify key parameters such as file paths,
feature names, scaling ranges, and visualization options, allowing the application
to work seamlessly with diverse datasets.

**Usage:**
This application is intended to be used as part of the `gofast.tools` subpackage.
Users can manipulate external parameters via command-line arguments to tailor the
tool to their specific datasets and visualization needs.

**Example:**
```bash
python plot_subsi_p_app.py \
    --data_path /path/to/main/data \
    --prediction_path /path/to/hwm/predictions \
    --tft_prediction_path /path/to/tft/predictions \
    --batch_file zhongshan_filtered_final_data.csv \
    --prediction_file hwm.ls.predicted_1.csv \
    --tft_prediction_file tft.ls.predicted_2024_2030_1.csv \
    --visualize_years 2025 2027 2030 \
    --output_visualization plot_output.png
```

**Dependencies:**
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- geopandas
- shapely
- gofast

Ensure all dependencies are installed before running the application.

**Author:** Daniel
**Date:** 2024-12-17
"""

import os
import argparse
import warnings
import logging
from typing import Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

import gofast as gf
from gofast.plot.utils import (
    plot_spatial_features,
    plot_distributions,
    plot_categorical_feature,
    plot_spatial_distribution,
)
from gofast.utils.base_utils import remove_outliers, handle_outliers
from gofast.utils.spatial_utils import spatial_sampling, batch_spatial_sampling
from gofast.utils.data_utils import (
    long_to_wide,
    wide_to_long,
    merge_datasets,
    dual_merge,
    to_categories,
)
from gofast.utils.mathext import rescale_data
from gofast.transformers import OutlierHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_data(
    data_path: str,
    prediction_path: str,
    tft_prediction_path: str,
    batch_file: str,
    prediction_file: str,
    tft_prediction_file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the main dataset and prediction data into pandas DataFrames.

    :param data_path: Path to the main dataset.
    :type data_path: str
    :param prediction_path: Path to the HWM prediction data.
    :type prediction_path: str
    :param tft_prediction_path: Path to the TFT prediction data.
    :type tft_prediction_path: str
    :param batch_file: Filename of the batch data.
    :type batch_file: str
    :param prediction_file: Filename of the HWM prediction data.
    :type prediction_file: str
    :param tft_prediction_file: Filename of the TFT prediction data.
    :type tft_prediction_file: str
    :return: Tuple containing final_data, prediction_data, and tft_prediction_data DataFrames.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    """
    try:
        # Load the main dataset
        final_data_path = os.path.join(data_path, batch_file)
        final_data = pd.read_csv(final_data_path)
        logger.info("Main data loaded successfully from %s", final_data_path)

        # Load HWM prediction data
        prediction_data_path = os.path.join(prediction_path, prediction_file)
        prediction_data = pd.read_csv(prediction_data_path)
        logger.info("HWM prediction data loaded successfully from %s", prediction_data_path)

        # Load TFT prediction data
        tft_prediction_data_path = os.path.join(tft_prediction_path, tft_prediction_file)
        tft_prediction_data = pd.read_csv(tft_prediction_data_path)
        logger.info("TFT prediction data loaded successfully from %s", tft_prediction_data_path)

        return final_data, prediction_data, tft_prediction_data

    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise


def scale_prediction_data(
    tft_prediction_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
    tft_scale_range: Tuple[float, float] = (0, 300),
    hwm_scale_range: Tuple[float, float] = (0, 412),
    tft_column: str = 'cumulative_subsidence',
    hwm_column: str = 'predicted_subsidence'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale the prediction data for TFT and HWM predictions.

    :param tft_prediction_data: DataFrame containing TFT prediction data.
    :type tft_prediction_data: pandas.DataFrame
    :param prediction_data: DataFrame containing HWM prediction data.
    :type prediction_data: pandas.DataFrame
    :param tft_scale_range: Tuple defining the scaling range for TFT predictions.
    :type tft_scale_range: tuple
    :param hwm_scale_range: Tuple defining the scaling range for HWM predictions.
    :type hwm_scale_range: tuple
    :param tft_column: Column name for TFT predictions.
    :type tft_column: str
    :param hwm_column: Column name for HWM predictions.
    :type hwm_column: str
    :return: Tuple containing scaled tft_prediction_data and hwm_prediction_data.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    try:
        # Scale TFT prediction data
        tft_prediction_data[tft_column] = rescale_data(
            tft_prediction_data[tft_column],
            range=tft_scale_range
        )
        logger.info("TFT prediction data scaled to range %s.", tft_scale_range)
        logger.debug("TFT %s min: %.2f, max: %.2f", tft_column,
                     tft_prediction_data[tft_column].min(),
                     tft_prediction_data[tft_column].max())

        # Scale HWM prediction data
        prediction_data[hwm_column] = rescale_data(
            prediction_data[hwm_column],
            range=hwm_scale_range
        )
        logger.info("HWM prediction data scaled to range %s.", hwm_scale_range)
        logger.debug("HWM %s min: %.2f, max: %.2f", hwm_column,
                     prediction_data[hwm_column].min(),
                     prediction_data[hwm_column].max())

        return tft_prediction_data, prediction_data

    except Exception as e:
        logger.error("Failed to scale prediction data: %s", e)
        raise


def transform_data(
    hwm_prediction_data: pd.DataFrame,
    prediction_path: str,
    tft_prediction_data: pd.DataFrame,
    tft_prediction_path: str,
    hwm_output_file: str = 'hwm.ls.predicted_wide_2024_2025.csv',
    tft_output_file: str = 'tft.ls.predicted_wide_2024_2025.csv'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform prediction data from long to wide format.

    :param hwm_prediction_data: DataFrame with HWM predictions.
    :type hwm_prediction_data: pandas.DataFrame
    :param prediction_path: Path to save HWM wide data.
    :type prediction_path: str
    :param tft_prediction_data: DataFrame with TFT predictions.
    :type tft_prediction_data: pandas.DataFrame
    :param tft_prediction_path: Path to save TFT wide data.
    :type tft_prediction_path: str
    :param hwm_output_file: Filename for the wide-format HWM predictions.
    :type hwm_output_file: str
    :param tft_output_file: Filename for the wide-format TFT predictions.
    :type tft_output_file: str
    :return: Tuple containing hwm_wide_data and tft_wide_data DataFrames.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    try:
        # Transform HWM prediction data to wide format
        hwm_wide_data = long_to_wide(
            hwm_prediction_data,
            index_columns=['longitude', 'latitude'],
            pivot_column='year',
            value_column='predicted_subsidence',
            exclude_value_from_name=True,
            savefile=os.path.join(prediction_path, hwm_output_file)
        )
        logger.info("HWM prediction data transformed to wide format and saved to %s", 
                    os.path.join(prediction_path, hwm_output_file))

        # Transform TFT prediction data to wide format
        tft_wide_data = long_to_wide(
            tft_prediction_data,
            index_columns=['longitude', 'latitude'],
            pivot_column='year',
            value_column='cumulative_subsidence',
            exclude_value_from_name=True,
            savefile=os.path.join(tft_prediction_path, tft_output_file)
        )
        logger.info("TFT prediction data transformed to wide format and saved to %s", 
                    os.path.join(tft_prediction_path, tft_output_file))

        return hwm_wide_data, tft_wide_data

    except Exception as e:
        logger.error("Failed to transform data: %s", e)
        raise


def plot_initial_data(
    data0: pd.DataFrame,
    prediction_data: pd.DataFrame,
    tft_prediction_data: pd.DataFrame
) -> None:
    """
    Display the first few rows of the datasets for inspection.

    :param data0: Copy of the main dataset.
    :type data0: pandas.DataFrame
    :param prediction_data: HWM prediction DataFrame.
    :type prediction_data: pandas.DataFrame
    :param tft_prediction_data: TFT prediction DataFrame.
    """
    logger.info("Displaying initial data samples...")
    print("Main Data Sample:")
    print(data0.head())
    print("\nHWM Prediction Data Sample:")
    print(prediction_data.head())
    print("\nTFT Prediction Data Sample:")
    print(tft_prediction_data.head())


def plot_features(
    hwm_prediction_data: pd.DataFrame,
    data0: pd.DataFrame,
    data_transformed: pd.DataFrame,
    visualize_years: List[int],
    output_path: str
) -> None:
    """
    Plot various spatial and categorical features.

    :param hwm_prediction_data: HWM prediction DataFrame.
    :type hwm_prediction_data: pandas.DataFrame
    :param data0: Main dataset DataFrame.
    :type data0: pandas.DataFrame
    :param data_transformed: Transformed dataset after outlier handling.
    :type data_transformed: pandas.DataFrame
    :param visualize_years: List of years to visualize.
    :type visualize_years: list
    :param output_path: Path to save the plots.
    :type output_path: str
    """
    logger.info("Plotting features...")
    try:
        # Plot spatial features for HWM predictions
        plot_spatial_features(
            hwm_prediction_data,
            features=['predicted_subsidence'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps='coolwarm',
            axis_off=True,
            save_path=os.path.join(output_path, 'hwm_spatial_features.png')
        )
        logger.info("HWM spatial features plotted.")

        # Plot geological_category
        plot_categorical_feature(
            data0,
            feature='geological_category',
            date_col='year',
            x_col='longitude',
            y_col='latitude',
            figsize=(10, 8),
            save_path=os.path.join(output_path, 'geological_category.png')
        )
        logger.info("Geological category plotted.")

        # Convert rainfall_category to categorical if not already
        if not pd.api.types.is_categorical_dtype(data0['rainfall_category']):
            data0['rainfall_category'] = data0['rainfall_category'].astype('category')

        # Plot rainfall_category
        plot_categorical_feature(
            data0,
            feature='rainfall_category',
            date_col='year',
            x_col='longitude',
            y_col='latitude',
            figsize=(10, 8),
            cmap='hsv',
            save_path=os.path.join(output_path, 'rainfall_category.png')
        )
        logger.info("Rainfall category plotted.")

        # Plot rainfall_mm after transformation
        plot_spatial_features(
            data_transformed,
            features=['rainfall_mm'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps=['plasma'],
            axis_off=True,
            save_path=os.path.join(output_path, 'rainfall_mm_plasma.png')
        )
        logger.info("Rainfall_mm plasma plot saved.")

        plot_spatial_features(
            data_transformed,
            features=['rainfall_mm'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps=['PuBu'],
            axis_off=True,
            save_path=os.path.join(output_path, 'rainfall_mm_pubu.png')
        )
        logger.info("Rainfall_mm PuBu plot saved.")

        # Plot normalized_density
        plot_spatial_features(
            data_transformed,
            features=['normalized_density'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps=['plasma'],
            axis_off=True,
            save_path=os.path.join(output_path, 'normalized_density.png')
        )
        logger.info("Normalized density plotted.")

        # Plot GWL
        plot_spatial_features(
            data0,
            features=['GWL'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps=['Blues'],
            axis_off=True,
            save_path=os.path.join(output_path, 'GWL.png')
        )
        logger.info("Groundwater Levels (GWL) plotted.")

    except Exception as e:
        logger.error("Failed to plot features: %s", e)
        raise


def handle_outliers_in_data(data0: pd.DataFrame, method: str = 'iqr',
                            threshold: float = 3, fill_strategy: str = 'interpolate',
                            interpolate_method: str = 'cubic',
                            batch_size: int = 512) -> pd.DataFrame:
    """
    Handle outliers in the dataset using the OutlierHandler transformer.

    :param data0: Main dataset DataFrame.
    :type data0: pandas.DataFrame
    :param method: Outlier detection method.
    :type method: str
    :param threshold: Threshold for outlier detection.
    :type threshold: float
    :param fill_strategy: Strategy to fill outliers.
    :type fill_strategy: str
    :param interpolate_method: Interpolation method if fill_strategy is 'interpolate'.
    :type interpolate_method: str
    :param batch_size: Batch size for processing.
    :type batch_size: int
    :return: DataFrame after outlier handling.
    :rtype: pandas.DataFrame
    """
    logger.info("Handling outliers in the data...")
    try:
        transformer = OutlierHandler(
            method=method,
            threshold=threshold,
            fill_strategy=fill_strategy,
            interpolate_method=interpolate_method,
            batch_size=batch_size,
            verbose=True
        )
        data_transformed = transformer.fit_transform(data0)
        logger.info("Outlier handling completed.")
        logger.debug("Transformed data shape: %s, Original data shape: %s",
                     data_transformed.shape, data0.shape)
        return data_transformed

    except Exception as e:
        logger.error("Failed to handle outliers: %s", e)
        raise


def plot_outlier_handled_features(
    data_transformed: pd.DataFrame,
    visualize_years: List[int],
    output_path: str
) -> None:
    """
    Plot features after outlier handling.

    :param data_transformed: DataFrame after outlier handling.
    :type data_transformed: pandas.DataFrame
    :param visualize_years: List of years to visualize.
    :type visualize_years: list
    :param output_path: Path to save the plots.
    :type output_path: str
    """
    logger.info("Plotting features after outlier handling...")
    try:
        # Plot rainfall_mm after outlier handling
        plot_spatial_features(
            data_transformed,
            features=['rainfall_mm'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps=['plasma'],
            axis_off=True,
            save_path=os.path.join(output_path, 'rainfall_mm_post_outlier.png')
        )
        logger.info("Rainfall_mm after outlier handling plotted.")

        plot_spatial_features(
            data_transformed,
            features=['rainfall_mm'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps=['PuBu'],
            axis_off=True,
            save_path=os.path.join(output_path, 'rainfall_mm_pubu_post_outlier.png')
        )
        logger.info("Rainfall_mm PuBu after outlier handling plotted.")

        # Plot normalized_density after outlier handling
        plot_spatial_features(
            data_transformed,
            features=['normalized_density'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps=['plasma'],
            axis_off=True,
            save_path=os.path.join(output_path, 'normalized_density_post_outlier.png')
        )
        logger.info("Normalized density after outlier handling plotted.")

        # Plot normalized_seismic_risk_score after outlier handling
        plot_spatial_features(
            data_transformed,
            features=['normalized_seismic_risk_score'],
            dates=visualize_years,
            x_col='longitude',
            y_col='latitude',
            colormaps=['seismic'],
            axis_off=True,
            save_path=os.path.join(output_path, 'seismic_risk_post_outlier.png')
        )
        logger.info("Seismic risk score after outlier handling plotted.")

    except Exception as e:
        logger.error("Failed to plot outlier handled features: %s", e)
        raise


def merge_and_plot(
    tft_prediction_data: pd.DataFrame,
    batch_data: pd.DataFrame,
    prediction_path: str,
    visualize_years: List[int],
    output_path: str
) -> pd.DataFrame:
    """
    Merge TFT predictions with batch data and plot distributions.

    :param tft_prediction_data: TFT prediction DataFrame.
    :type tft_prediction_data: pandas.DataFrame
    :param batch_data: Batch data DataFrame.
    :type batch_data: pandas.DataFrame
    :param prediction_path: Path for saving merged data.
    :type prediction_path: str
    :param visualize_years: List of years to visualize.
    :type visualize_years: list
    :param output_path: Path to save the plots.
    :type output_path: str
    :return: Merged DataFrame.
    :rtype: pandas.DataFrame
    """
    logger.info("Merging TFT predictions with batch data...")
    try:
        merged_data = pd.merge(
            tft_prediction_data,
            batch_data[['longitude', 'latitude', 'subsidence']],
            how='inner',
            on=['longitude', 'latitude']
        )
        logger.info("Merged TFT prediction with batch data.")

        # Plot distributions
        plot_distributions(
            merged_data,
            features=['subsidence', 'predicted_subsidence', 'cumulative_subsidence'],
            save_path=os.path.join(output_path, 'merged_distributions.png')
        )
        logger.info("Distributions plotted for merged data.")

        # Dual merge with threshold
        merged_test = dual_merge(
            batch_data[['longitude', 'latitude', 'subsidence']],
            tft_prediction_data,
            feature_cols=['longitude', 'latitude'],
            threshold=0.05,
            how='inner',
            find_closest=True
        )
        logger.info("Performed dual merge with threshold.")

        # Plot distributions after dual merge
        plot_distributions(
            merged_test,
            features=['subsidence', 'predicted_subsidence', 'cumulative_subsidence'],
            save_path=os.path.join(output_path, 'dual_merged_distributions.png')
        )
        logger.info("Distributions plotted for dual merged data.")

        # Plot spatial distribution for cumulative_subsidence
        plot_spatial_distribution(
            merged_test,
            category_column='cumulative_subsidence',
            categorical=True,
            continuous_bins=[0., 25, 100, 200],
            categories=['minimal', 'moderate', 'severe'],
            filter_categories=None,
            save_path=os.path.join(output_path, 'cumulative_subsidence_spatial.png')
        )
        logger.info("Spatial distribution for cumulative subsidence plotted.")

        return merged_test

    except Exception as e:
        logger.error("Failed to merge and plot data: %s", e)
        raise


def categorize_and_plot(
    tft_pred_data: pd.DataFrame,
    visualize_years: List[int],
    output_path: str,
    tft_batch_path: str,
    bins: List[float] = [-np.inf, 10., 20., np.inf],
    categories: List[str] = ['minimal', 'moderate', 'severe']
) -> pd.DataFrame:
    """
    Categorize subsidence data and plot spatial distributions.

    :param tft_pred_data: TFT prediction DataFrame.
    :type tft_pred_data: pandas.DataFrame
    :param visualize_years: List of years to visualize.
    :type visualize_years: list
    :param output_path: Path to save the plots.
    :type output_path: str
    :param tft_batch_path: Path for saving categorized data.
    :type tft_batch_path: str
    :param bins: List of bin edges for categorization.
    :type bins: list
    :param categories: List of category labels.
    :type categories: list
    :return: Categorized TFT prediction DataFrame.
    :rtype: pandas.DataFrame
    """
    logger.info("Categorizing subsidence data...")
    try:
        # Create categorical column based on bins
        tft_pred_data['subsidence_category'] = pd.cut(
            tft_pred_data['predicted_subsidence'],
            bins=bins,
            labels=categories,
            include_lowest=True,
            right=False
        )
        logger.info("Subsidence data categorized into %s.", categories)

        # Display sample of categorization
        print(tft_pred_data[['predicted_subsidence', 'subsidence_category']].head())

        # Alternative categorization using equal range
        tft_pred_data_eqr = to_categories(
            tft_pred_data.copy(),
            column='predicted_subsidence',
            bins=[-np.inf, 10., 25., np.inf],
            categories=['minimal', 'moderate', 'severe'],
            method='equal_range',
            include_lowest=True,
            category_name='subsidence_category_eqr',
            right=False
        )
        logger.info("Subsidence data categorized using equal range.")

        # Plot spatial distribution for specified years
        plot_spatial_distribution(
            tft_pred_data_eqr[tft_pred_data_eqr['year'].isin(visualize_years)],
            category_column='subsidence_category_eqr',
            categories=categories[::-1],
            cmap='coolwarm',
            show_grid=False,
            save_path=os.path.join(output_path, 'subsidence_category_spatial.png')
        )
        logger.info("Spatial distribution for subsidence categories plotted.")

        # Save categorized data
        categorized_csv_path = os.path.join(
            tft_batch_path, 'combined_tft.ls.predicted_2024_2030_eqr.csv'
        )
        tft_pred_data_eqr.to_csv(categorized_csv_path, index=False)
        logger.info("Categorized TFT prediction data saved to %s", categorized_csv_path)

        return tft_pred_data_eqr

    except Exception as e:
        logger.error("Failed to categorize and plot subsidence data: %s", e)
        raise


def combine_tft_predictions(tft_batch_path: str, output_file: str = 'combined_tft.ls.predicted_2024_2030.csv') -> pd.DataFrame:
    """
    Combine TFT prediction data from multiple batches into a single DataFrame.

    :param tft_batch_path: Path where TFT prediction batches are stored.
    :type tft_batch_path: str
    :param output_file: Filename for the combined TFT prediction data.
    :type output_file: str
    :return: Combined TFT prediction DataFrame.
    :rtype: pandas.DataFrame
    """
    logger.info("Combining TFT prediction data from multiple batches...")
    try:
        # Define the list of TFT prediction files
        tft_files = [
            'tft.ls.predicted_2024_2030_1.csv',
            'tft.ls.predicted_2024_2030_2.csv',
            'tft.ls.predicted_2024_2030_3.csv',
            'tft.ls.predicted_2024_2030_4.csv',
            'tft.ls.predicted_2024_2030_5.csv',
        ]

        # Read and concatenate all TFT prediction files
        tft_pred_data_list = []
        for file in tft_files:
            file_path = os.path.join(tft_batch_path, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                tft_pred_data_list.append(df)
                logger.info("Loaded TFT prediction file: %s", file_path)
            else:
                logger.warning("TFT prediction file %s does not exist. Skipping.", file_path)

        if not tft_pred_data_list:
            logger.error("No TFT prediction files found in %s.", tft_batch_path)
            raise FileNotFoundError("No TFT prediction files found.")

        tft_pred_data = pd.concat(tft_pred_data_list, ignore_index=True)
        logger.info("All TFT prediction data combined.")

        # Save combined TFT prediction data
        combined_tft_path = os.path.join(tft_batch_path, output_file)
        tft_pred_data.to_csv(combined_tft_path, index=False)
        logger.info("Combined TFT prediction data saved to %s", combined_tft_path)

        return tft_pred_data

    except Exception as e:
        logger.error("Failed to combine TFT prediction data: %s", e)
        raise


def main():
    """
    Main function to execute the subsidence prediction visualization workflow.
    """
    parser = argparse.ArgumentParser(
        description="Plot Subsidence Prediction Application using XTFT and HWM Predictions."
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the main dataset directory.'
    )
    parser.add_argument(
        '--prediction_path',
        type=str,
        required=True,
        help='Path to the HWM prediction data directory.'
    )
    parser.add_argument(
        '--tft_prediction_path',
        type=str,
        required=True,
        help='Path to the TFT prediction data directory.'
    )
    parser.add_argument(
        '--batch_file',
        type=str,
        default='zhongshan_filtered_final_data.csv',
        help='Filename of the batch data CSV.'
    )
    parser.add_argument(
        '--prediction_file',
        type=str,
        default='hwm.ls.predicted_1.csv',
        help='Filename of the HWM prediction CSV.'
    )
    parser.add_argument(
        '--tft_prediction_file',
        type=str,
        default='tft.ls.predicted_2024_2030_1.csv',
        help='Filename of the TFT prediction CSV.'
    )
    parser.add_argument(
        '--visualize_years',
        type=int,
        nargs='+',
        default=[2025, 2027, 2030],
        help='List of years to visualize predictions.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save the output plots.'
    )
    parser.add_argument(
        '--tft_batch_path',
        type=str,
        required=True,
        help='Path to save combined TFT prediction data.'
    )

    args = parser.parse_args()

    # Suppress warnings for clarity
    warnings.filterwarnings('ignore')
    # tf.get_logger().setLevel('ERROR')

    try:
        # Ensure output directories exist
        os.makedirs(args.output_path, exist_ok=True)
        os.makedirs(args.tft_batch_path, exist_ok=True)

        # Load data
        final_data, prediction_data, tft_prediction_data = load_data(
            data_path=args.data_path,
            prediction_path=args.prediction_path,
            tft_prediction_path=args.tft_prediction_path,
            batch_file=args.batch_file,
            prediction_file=args.prediction_file,
            tft_prediction_file=args.tft_prediction_file
        )

        # Create a copy for manipulation
        data0 = final_data.copy()

        # Plot initial data samples
        plot_initial_data(data0, prediction_data, tft_prediction_data)

        # Scale prediction data
        tft_prediction_data_scaled, hwm_prediction_data_scaled = scale_prediction_data(
            tft_prediction_data,
            prediction_data
        )

        # Transform data to wide format
        hwm_wide_data, tft_wide_data = transform_data(
            hwm_prediction_data=hwm_prediction_data_scaled,
            prediction_path=args.prediction_path,
            tft_prediction_data=tft_prediction_data_scaled,
            tft_prediction_path=args.tft_prediction_path
        )

        # Plot features before outlier handling
        plot_features(
            hwm_prediction_data=hwm_prediction_data_scaled,
            data0=data0,
            data_transformed=data0,  # Assuming no transformation yet
            visualize_years=args.visualize_years,
            output_path=args.output_path
        )

        # Handle outliers in data
        data_transformed = handle_outliers_in_data(data0)

        # Plot features after outlier handling
        plot_outlier_handled_features(
            data_transformed=data_transformed,
            visualize_years=args.visualize_years,
            output_path=args.output_path
        )

        # Load batch data for merging
        batch_data_path = os.path.join(args.data_path, args.batch_file)
        batch_data = pd.read_csv(batch_data_path)
        logger.info("Batch data loaded from %s", batch_data_path)

        # Merge and plot
        merged_test = merge_and_plot(
            tft_prediction_data=tft_prediction_data_scaled,
            batch_data=batch_data,
            prediction_path=args.prediction_path,
            visualize_years=args.visualize_years,
            output_path=args.output_path
        )

        # Categorize and plot
        tft_pred_data_eqr = categorize_and_plot(
            tft_pred_data=tft_prediction_data_scaled,
            visualize_years=args.visualize_years,
            output_path=args.output_path,
            tft_batch_path=args.tft_batch_path,
            bins=[-np.inf, 10., 25., np.inf],
            categories=['minimal', 'moderate', 'severe']
        )

        # Combine TFT prediction data from multiple batches
        tft_pred_data_combined = combine_tft_predictions(
            tft_batch_path=args.tft_batch_path,
            output_file='combined_tft.ls.predicted_2024_2030.csv'
        )

        # Plot spatial features for combined TFT data
        plot_spatial_features(
            tft_pred_data_combined,
            features=['predicted_subsidence'],
            dates=[2024, 2027, 2030],
            x_col='longitude',
            y_col='latitude',
            colormaps=['viridis'],
            axis_off=True,
            save_path=os.path.join(args.output_path, 'combined_tft_spatial_features.png')
        )
        logger.info("Combined TFT spatial features plotted.")

        # Save combined TFT prediction data
        combined_tft_path = os.path.join(
            args.tft_batch_path, 'combined_tft.ls.predicted_2024_2030.csv'
        )
        tft_pred_data_combined.to_csv(combined_tft_path, index=False)
        logger.info("Combined TFT prediction data saved to %s", combined_tft_path)

        # Categorize and plot for combined data
        tft_pred_data_eqr_combined = categorize_and_plot(
            tft_pred_data=tft_pred_data_combined,
            visualize_years=args.visualize_years,
            output_path=args.output_path,
            tft_batch_path=args.tft_batch_path,
            bins=[-np.inf, 10., 20., np.inf],
            categories=['minimal', 'moderate', 'severe']
        )

        logger.info("Subsidence prediction visualization workflow completed successfully.")

    except Exception as e:
        logger.error("Application failed: %s", e)
        raise


if __name__ == "__main__":
    main()
