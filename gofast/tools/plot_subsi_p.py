"""
plot_subsi_p.py
========================

**Script Name:** subsidence_prediction.py

**Description:**
This script performs data processing and visualization for land
subsidence prediction using real datasets. It includes steps such as
data loading, scaling, outlier handling, transformation, merging, and
various plotting functionalities to analyze and visualize subsidence
patterns spatially and categorically.

**Usage:**
To use this script as part of the `gofast.tools` subpackage, import the
necessary functions or execute the script directly.

**Example:**

.. code-block:: python

    from gofast.tools import plot_subsi_p

    # Execute the main function
    plot_subsi_p.main()

**Dependencies:**
- pandas
- numpy
- matplotlib
- seaborn
- geopandas
- shapely
- gofast

**Author:** Daniel
**Date:** 2024-12-17
"""

# =============================================================================
# Importing necessary libraries
# =============================================================================
import os
import warnings
from typing import Union, Optional, List

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
from gofast.utils.baseutils import remove_outliers, handle_outliers
from gofast.utils.spatialutils import spatial_sampling, batch_spatial_sampling
from gofast.utils.datautils import (
    long_to_wide,
    wide_to_long,
    merge_datasets,
    dual_merge,
    to_categories,
)
from gofast.utils.mathext import rescale_data
from gofast.transformers import OutlierHandler

# =============================================================================
# Suppress warnings to keep the output clean
# =============================================================================
warnings.filterwarnings('ignore')


def load_data(
    data_path: str,
    prediction_path: str,
    tft_prediction_path: str,
    batch_file: str,
    prediction_file: str,
    tft_prediction_file: str,
) -> tuple:
    """
    Load the main dataset and prediction data into pandas DataFrames.

    :param data_path: Path to the main dataset.
    :param prediction_path: Path to the HWM prediction data.
    :param tft_prediction_path: Path to the TFT prediction data.
    :param batch_file: Filename of the batch data.
    :param prediction_file: Filename of the HWM prediction data.
    :param tft_prediction_file: Filename of the TFT prediction data.
    :return: Tuple containing final_data, prediction_data, and tft_prediction_data DataFrames.
    """
    # Load the main dataset
    final_data_path = os.path.join(data_path, 'zhongshan_filtered_final_data.csv')
    final_data = pd.read_csv(final_data_path)

    # Load HWM prediction data
    prediction_data_path = os.path.join(prediction_path, prediction_file)
    prediction_data = pd.read_csv(prediction_data_path)

    # Load TFT prediction data
    tft_prediction_data_path = os.path.join(tft_prediction_path, tft_prediction_file)
    tft_prediction_data = pd.read_csv(tft_prediction_data_path)

    return final_data, prediction_data, tft_prediction_data


def scale_prediction_data(
    tft_prediction_data: pd.DataFrame,
    prediction_data: pd.DataFrame
) -> tuple:
    """
    Scale the prediction data for TFT and HWM predictions.

    :param tft_prediction_data: DataFrame containing TFT prediction data.
    :param prediction_data: DataFrame containing HWM prediction data.
    :return: Tuple containing scaled tft_prediction_data and hwm_prediction_data.
    """
    # Scale TFT prediction data
    tft_prediction_data = rescale_data(
        tft_prediction_data,
        range=(0, 300),
        columns='cumulative_subsidence'
    )
    print(tft_prediction_data['cumulative_subsidence'].min())
    print(tft_prediction_data['cumulative_subsidence'].max())

    # Scale HWM prediction data
    hwm_prediction_data = rescale_data(
        prediction_data,
        range=(0, 412),
        columns='predicted_subsidence'
    )
    print(hwm_prediction_data['predicted_subsidence'].min())
    print(hwm_prediction_data['predicted_subsidence'].max())

    return tft_prediction_data, hwm_prediction_data


def transform_data(
    hwm_prediction_data: pd.DataFrame,
    prediction_path: str,
    tft_prediction_data: pd.DataFrame,
    tft_prediction_path: str
) -> tuple:
    """
    Transform prediction data from long to wide format.

    :param hwm_prediction_data: DataFrame with HWM predictions.
    :param prediction_path: Path to save HWM wide data.
    :param tft_prediction_data: DataFrame with TFT predictions.
    :param tft_prediction_path: Path to save TFT wide data.
    :return: Tuple containing hwm_wide_data and tft_wide_data DataFrames.
    """
    # Transform HWM prediction data
    hwm_wide_data = long_to_wide(
        hwm_prediction_data,
        index_columns=['longitude', 'latitude'],
        pivot_column='year',
        value_column='predicted_subsidence',
        exclude_value_from_name=True,
        savefile=os.path.join(
            prediction_path, 'hwm.ls.predicted_wide_2024_2025.csv'
        )
    )

    # Transform TFT prediction data
    tft_wide_data = long_to_wide(
        tft_prediction_data,
        index_columns=['longitude', 'latitude'],
        pivot_column='year',
        value_column='cumulative_subsidence',
        exclude_value_from_name=True,
        savefile=os.path.join(
            tft_prediction_path, 'tft.ls.predicted_wide_2024_2025.csv'
        )
    )

    return hwm_wide_data, tft_wide_data


def plot_initial_data(
    data0: pd.DataFrame,
    prediction_data: pd.DataFrame,
    tft_prediction_data: pd.DataFrame
) -> None:
    """
    Display the first few rows of the datasets for inspection.

    :param data0: Copy of the main dataset.
    :param prediction_data: HWM prediction DataFrame.
    :param tft_prediction_data: TFT prediction DataFrame.
    """
    print(data0.head())
    print(prediction_data.head())
    print(tft_prediction_data.head())


def plot_features(
    hwm_prediction_data: pd.DataFrame,
    data0: pd.DataFrame,
    data_transformed: pd.DataFrame
) -> None:
    """
    Plot various spatial and categorical features.

    :param hwm_prediction_data: HWM prediction DataFrame.
    :param data0: Main dataset DataFrame.
    :param data_transformed: Transformed dataset after outlier handling.
    """
    # Plot spatial features for HWM predictions
    plot_spatial_features(
        hwm_prediction_data,
        features=['predicted_subsidence'],
        dates=[2025, 2027, 2030],
        x_col='longitude',
        y_col='latitude',
        colormaps='coolwarm',
        axis_off=True,
    )

    # Plot geological_category
    plot_categorical_feature(
        data0,
        feature='geological_category',
        date_col='year',
        x_col='longitude',
        y_col='latitude',
        figsize=(10, 8)
    )

    # Convert rainfall_category to categorical
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
    )

    # Plot rainfall_mm after transformation
    plot_spatial_features(
        data_transformed,
        features=['rainfall_mm'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['plasma'],
        axis_off=True,
    )

    # Additional plots
    plot_spatial_features(
        data_transformed,
        features=['rainfall_mm'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['PuBu'],
        axis_off=True,
    )

    plot_spatial_features(
        data_transformed,
        features=['normalized_density'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['plasma'],
        axis_off=True,
    )

    plot_spatial_features(
        data0,
        features=['GWL'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['Blues'],
        axis_off=True,
    )


def handle_outliers_in_data(data0: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers in the dataset using the OutlierHandler transformer.

    :param data0: Main dataset DataFrame.
    :return: DataFrame after outlier handling.
    """
    transformer = OutlierHandler(
        method='iqr',
        threshold=3,
        fill_strategy='interpolate',
        interpolate_method='cubic',
        batch_size=512,
        verbose=True
    )
    data_transformed = transformer.fit_transform(data0)
    print(data_transformed.shape, data0.shape)
    return data_transformed


def plot_outlier_handled_features(
    data_transformed: pd.DataFrame,
    data_outliers_removed: pd.DataFrame
) -> None:
    """
    Plot features after outlier handling.

    :param data_transformed: DataFrame after initial outlier handling.
    :param data_outliers_removed: DataFrame after additional outlier handling.
    """
    plot_spatial_features(
        data_transformed,
        features=['rainfall_mm'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['plasma'],
        axis_off=True,
    )

    plot_spatial_features(
        data_transformed,
        features=['rainfall_mm'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['PuBu'],
        axis_off=True,
    )

    plot_spatial_features(
        data_transformed,
        features=['normalized_density'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['plasma'],
        axis_off=True,
    )

    plot_spatial_features(
        data_outliers_removed,
        features=['normalized_seismic_risk_score'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['seismic'],
        axis_off=True,
    )

    plot_spatial_features(
        data_outliers_removed,
        features=['normalized_seismic_risk_score'],
        dates=[2015, 2018, 2023],
        x_col='longitude',
        y_col='latitude',
        colormaps=['seismic'],
        axis_off=True,
    )


def merge_and_plot(
    tft_prediction_data: pd.DataFrame,
    batch_data: pd.DataFrame,
    prediction_path: str
) -> pd.DataFrame:
    """
    Merge TFT predictions with batch data and plot distributions.

    :param tft_prediction_data: TFT prediction DataFrame.
    :param batch_data: Batch data DataFrame.
    :param prediction_path: Path for saving merged data.
    :return: Merged DataFrame.
    """
    merged_data = pd.merge(
        tft_prediction_data,
        batch_data[['longitude', 'latitude', 'subsidence']],
        how='inner',
        on=['longitude', 'latitude']
    )

    # Plot distributions
    plot_distributions(
        merged_data,
        features=['subsidence', 'predicted_subsidence', 'cumulative_subsidence']
    )

    # Dual merge with threshold
    merged_test = dual_merge(
        batch_data[['longitude', 'latitude', 'subsidence']],
        tft_prediction_data,
        feature_cols=['longitude', 'latitude'],
        threshold=0.05,
        how='inner',
        find_closest=True
    )

    # Plot distributions after dual merge
    plot_distributions(
        merged_test,
        features=['subsidence', 'predicted_subsidence', 'cumulative_subsidence']
    )

    # Plot spatial distribution for cumulative_subsidence
    plot_spatial_distribution(
        merged_test,
        category_column='cumulative_subsidence',
        categorical=True,
        continuous_bins=[0., 25, 100, 200],
        categories=['minimal', 'moderate', 'severe'],
        filter_categories=None
    )

    return merged_test


def categorize_and_plot(
    tft_pred_data: pd.DataFrame,
    data_path: str,
    tft_batch_path: str
) -> pd.DataFrame:
    """
    Categorize subsidence data and plot spatial distributions.

    :param tft_pred_data: TFT prediction DataFrame.
    :param data_path: Path to save categorized data.
    :param tft_batch_path: Path for batch data.
    :return: Categorized TFT prediction DataFrame.
    """
    # Define bin edges based on min and max
    min_subsidence = tft_pred_data['predicted_subsidence'].min()
    max_subsidence = tft_pred_data['predicted_subsidence'].max()
    range_subsidence = max_subsidence - min_subsidence
    bin_width = range_subsidence / 3

    # Define the bins and labels
    bins = [
        min_subsidence,
        min_subsidence + bin_width,
        min_subsidence + 2 * bin_width,
        max_subsidence
    ]
    labels = ['minimal', 'moderate', 'severe']

    # Create categorical column
    tft_pred_data['subsidence_category'] = pd.cut(
        tft_pred_data['predicted_subsidence'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    print(tft_pred_data[['predicted_subsidence', 'subsidence_category']].head())

    # Alternative categorization using equal range
    tft_pred_data2 = to_categories(
        tft_pred_data.copy(),
        column='predicted_subsidence',
        bins=[-np.inf, 10., 25., np.inf],
        categories=['minimal', 'moderate', 'severe'],
        method='equal_range',
        include_lowest=True,
        category_name='subsidence_category_eqr',
        right=False
    )

    # Plot spatial distribution for year 2030
    plot_spatial_distribution(
        tft_pred_data2[tft_pred_data2['year'] == 2030],
        categories=['minimal', 'moderate', 'severe'][::-1],
        category_column='subsidence_category_eqr',
        cmap='coolwarm',
        show_grid=False,
    )

    # Save categorized data
    tft_pred_data2.to_csv(
        os.path.join(
            tft_batch_path, 'combined_tft.ls.predicted_2024_2030_eqr.csv'
        ),
        index=False
    )

    return tft_pred_data2


def main():
    """
    Main function to execute the subsidence prediction workflow.
    """
    # Define paths
    data_path = r'C:\Users\Daniel\Documents\zongshan_codes\tft_prediction_ok'
    data_in_batches_path = r'J:\zhongshan_project\data_in_batches'
    batch_file = 'zh_tft_data_batch1.csv'

    prediction_path = r'J:\zhongshan_project\zhongshan_prediction_ok\batch_1\hwm'
    prediction_file = 'hwm.ls.predicted_1.csv'

    tft_prediction_path = r'J:\zhongshan_project\zhongshan_prediction_ok\batch_1\tft'
    tft_prediction_file = 'tft.ls.predicted_2024_2030_1.csv'

    tft_batch_path = r'J:\zhongshan_project\zhongshan_prediction_ok\_tft'

    # Load data
    final_data, prediction_data, tft_prediction_data = load_data(
        data_path,
        prediction_path,
        tft_prediction_path,
        batch_file,
        prediction_file,
        tft_prediction_file
    )

    # Create a copy for manipulation
    data0 = final_data.copy()

    # Plot initial data
    plot_initial_data(data0, prediction_data, tft_prediction_data)

    # Scale prediction data
    tft_prediction_data, hwm_prediction_data = scale_prediction_data(
        tft_prediction_data, prediction_data
    )

    # Transform data to wide format
    hwm_wide_data, tft_wide_data = transform_data(
        hwm_prediction_data, prediction_path,
        tft_prediction_data, tft_prediction_path
    )

    # Plot features
    plot_features(hwm_prediction_data, data0, data0)

    # Handle outliers
    data_transformed = handle_outliers_in_data(data0)

    # Plot features after outlier handling
    plot_outlier_handled_features(data_transformed, data_transformed)

    # Load batch data
    batch_data_path = os.path.join(data_in_batches_path, batch_file)
    batch_data = pd.read_csv(batch_data_path)

    # Merge and plot
    merged_test = merge_and_plot(
        tft_prediction_data, batch_data, prediction_path
    )

    # Categorize and plot
    tft_pred_data2 = categorize_and_plot(
        tft_prediction_data, data_path, tft_batch_path
    )

    # Combine TFT prediction data from multiple batches
    tft_files = [
        'tft.ls.predicted_2024_2030_1.csv',
        'tft.ls.predicted_2024_2030_2.csv',
        'tft.ls.predicted_2024_2030_3.csv',
        'tft.ls.predicted_2024_2030_4.csv',
        'tft.ls.predicted_2024_2030_5.csv',
    ]
    tft_pred_data_list = [
        pd.read_csv(os.path.join(tft_batch_path, file)) for file in tft_files
    ]
    tft_pred_data = pd.concat(tft_pred_data_list)

    # Plot spatial features for combined TFT data
    plot_spatial_features(
        tft_pred_data,
        features=['predicted_subsidence'],
        dates=[2024, 2027, 2030],
        x_col='longitude',
        y_col='latitude',
        colormaps=['viridis'],
        axis_off=True,
    )

    # Save combined TFT prediction data
    combined_tft_path = os.path.join(
        tft_batch_path, 'combined_tft.ls.predicted_2024_2030.csv'
    )
    tft_pred_data.to_csv(combined_tft_path, index=False)

    # Define bins and categorize
    tft_pred_data2 = to_categories(
        tft_pred_data.copy(),
        column='predicted_subsidence',
        bins=[-np.inf, 10., 20., np.inf],
        categories=['minimal', 'moderate', 'severe'],
        method='equal_range',
        include_lowest=True,
        category_name='subsidence_category',
        right=False,
        savefile=os.path.join(
            data_path, 'tft.ls.categories_long_2024_2025.csv'
        ),
    )

    # Transform to wide format with categories
    tft_wide_cat_new_data = long_to_wide(
        tft_pred_data2,
        index_columns=['longitude', 'latitude'],
        pivot_column='year',
        value_column='subsidence_category',
        exclude_value_from_name=True,
        savefile=os.path.join(
            data_path, 'tft.ls.categories_wide_2024_2025.csv'
        )
    )


if __name__ == "__main__":
    main()
