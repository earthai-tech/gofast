
"""
Hammerstein-Wiener Model for Land Subsidence Prediction
=======================================================

This module provides a robust and professional tool for predicting land subsidence
using the Hammerstein-Wiener Model (HWM). The tool is designed to handle data
preprocessing, model training, prediction, and visualization in a streamlined
manner, allowing users to manipulate and extend functionalities externally.

Features:
---------
- **Data Loading and Preprocessing**: Loads datasets, handles missing values, encodes
  categorical variables, scales numerical features, and creates lag features to capture
  temporal dependencies.
- **Model Training**: Initializes and trains a Hammerstein-Wiener Regressor with support
  for early stopping and model checkpointing.
- **Prediction**: Generates predictions for future years based on the trained model.
- **Visualization**: Visualizes the prediction results geographically.
- **Reproducibility**: Records package versions and saves scalers and models for
  reproducibility.

Usage:
------
This script is intended to be used as part of the `gofast.tools` subpackage. Users can
manipulate external parameters such as data paths, model hyperparameters, and other
configurations to tailor the tool to their specific tasks.

Example:
--------
```bash
python hwm_subsi_p.py --data_path /path/to/data --epochs 100 --batch_size 32
```

Dependencies:
-------------
- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- hwm
- matplotlib

Ensure all dependencies are installed before running the script.

"""

import os
import argparse
import warnings
import logging

import pandas as pd
import numpy as np
import joblib
import sklearn 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
try: 
    import hwm 
except ImportError : 
    raise ValueError(
        "Running this script expect the `hwm` to be instaled."
        " Installed it using ``pip install hwm``. "
    )
from hwm.estimators import HammersteinWienerRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

target='subsidence'

def load_data(data_path):
    """
    Load the dataset from the specified path.

    :param data_path: Path to the data directory.
    :type data_path: str
    :return: Loaded DataFrame.
    :rtype: pandas.DataFrame
    """
    try:
        file_path = os.path.join(data_path, 'zhongshan_filtered_final_data.csv')
        filtered_data = pd.read_csv(file_path)
        logger.info("Data loaded successfully from %s", file_path)
        return filtered_data.copy()
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise


def preprocess_data(filtered_data, data_path, lags=3):
    """
    Preprocess the data by encoding categorical variables, scaling numerical features,
    and creating lag features.

    :param filtered_data: Original DataFrame.
    :type filtered_data: pandas.DataFrame
    :param data_path: Path to save encoders and scalers.
    :type data_path: str
    :param lags: Number of lag features to create.
    :type lags: int
    :return: Processed DataFrame and fitted scalers/encoders.
    :rtype: tuple[pandas.DataFrame, dict]
    """
    logger.info("Starting data preprocessing...")
    try:
        data0 = filtered_data.copy()

        # Define categorical and numerical features
        categorical_features = ['geological_category', 'rainfall_category']
        numerical_features = [
            'longitude', 'latitude', 'year', 'GWL',
            'normalized_density', 'normalized_seismic_risk_score'
        ]
        target = 'subsidence'

        # One-Hot Encoding for categorical features
        encoder = OneHotEncoder(sparse_output=False)
        encoded_cats = encoder.fit_transform(data0[categorical_features])
        encoded_cat_columns = encoder.get_feature_names_out(categorical_features)
        encoded_cat_df = pd.DataFrame(
            encoded_cats, columns=encoded_cat_columns, index=data0.index
        )
        logger.info("Categorical variables encoded.")

        # Combine numerical and encoded categorical features
        processed_data = pd.concat(
            [data0[numerical_features], encoded_cat_df],
            axis=1
        )

        # Scale numerical features
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(processed_data[numerical_features])
        processed_data[numerical_features] = scaled_numerical
        logger.info("Numerical features scaled.")

        # Scale the target variable
        target_scaler = MinMaxScaler()
        processed_data[target] = target_scaler.fit_transform(data0[[target]])
        logger.info("Target variable scaled.")

        # Create lag features for the target variable
        processed_data = create_lag_features(processed_data, target, lags)
        logger.info("Lag features created.")

        # Drop rows with NaN values resulting from lag features
        missing_values = processed_data.isnull().sum()
        logger.info("Missing values after lag feature creation:\n%s", missing_values)
        processed_data.dropna(inplace=True)
        logger.info("Dropped rows with missing values.")

        # Collect package versions for reproducibility
        pkg_versions = {
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "matplotlib": mpl.__version__,
            "hwm": hwm.__version__
        }

        # Save scalers and encoder
        scalers_path = os.path.join(data_path, 'hwm.scalers.joblib')
        joblib.dump(
            {
                'scaler': scaler,
                'target_scaler': target_scaler,
                'encoder': encoder,
                "__version__": pkg_versions
            },
            scalers_path
        )
        logger.info("Scalers and encoder saved to %s", scalers_path)

        return processed_data, scalers_path

    except Exception as e:
        logger.error("Data preprocessing failed: %s", e)
        raise


def create_lag_features(df, target_col, lags):
    """
    Create lag features for the target variable.

    :param df: DataFrame containing the data.
    :type df: pandas.DataFrame
    :param target_col: Name of the target column.
    :type target_col: str
    :param lags: Number of lag features to create.
    :type lags: int
    :return: DataFrame with lag features.
    :rtype: pandas.DataFrame
    """
    for lag in range(1, lags + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def train_model(processed_data, data_path, lags=3):
    """
    Train the Hammerstein-Wiener Regressor using the processed data.

    :param processed_data: Processed DataFrame.
    :type processed_data: pandas.DataFrame
    :param data_path: Path to save the trained model.
    :type data_path: str
    :param lags: Number of lag features used.
    :type lags: int
    :return: Trained Hammerstein-Wiener model.
    :rtype: HammersteinWienerRegressor
    """
    logger.info("Starting model training...")
    try:
        # Define features and target
        target = 'subsidence'
        feature_columns = list(processed_data.columns.difference([target]))
        X = processed_data[feature_columns]
        y = processed_data[target]

        # Split the data into training and validation sets based on year
        train_mask = processed_data['year'] <= 2023
        X_train, X_val = X[train_mask], X[~train_mask]
        y_train, y_val = y[train_mask], y[~train_mask]

        logger.info("Data split into training and validation sets.")

        # Initialize the Hammerstein-Wiener Regressor
        hwr = HammersteinWienerRegressor(
            nonlinear_input_estimator=None,
            nonlinear_output_estimator=None,
            p=2,
            loss='mse',
            output_scale=None,
            time_weighting='linear',
            batch_size='auto',
            max_iter=100,
            optimizer='adam',
            learning_rate=0.001,
            early_stopping=True,
            n_iter_no_change=5,
            verbose=1
        )
        logger.info("Hammerstein-Wiener Regressor initialized.")

        # Fit the model
        hwr.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Save the trained model
        model_path = os.path.join(data_path, 'hwm.best_model.joblib')
        joblib.dump({'hwm_best_model': hwr}, model_path)
        logger.info("Trained model saved to %s", model_path)

        return hwr

    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise


def make_predictions(processed_data, filtered_data, model, scalers_path, data_path, forecast_years=range(2024, 2031), lags=3):
    """
    Generate predictions for future years using the trained model.

    :param processed_data: Processed DataFrame.
    :type processed_data: pandas.DataFrame
    :param filtered_data: Original filtered DataFrame.
    :type filtered_data: pandas.DataFrame
    :param model: Trained Hammerstein-Wiener model.
    :type model: HammersteinWienerRegressor
    :param scalers_path: Path to the saved scalers and encoder.
    :type scalers_path: str
    :param data_path: Path to save the predictions.
    :type data_path: str
    :param forecast_years: Range of years to forecast.
    :type forecast_years: range
    :param lags: Number of lag features used.
    :type lags: int
    :return: DataFrame containing predictions.
    :rtype: pandas.DataFrame
    """
    logger.info("Starting prediction for future years...")
    try:
        # Load scalers and encoder
        scalers = joblib.load(scalers_path)
        scaler = scalers['scaler']
        target_scaler = scalers['target_scaler']
        encoder = scalers['encoder']

        # Reset indices to align processed_data and filtered_data
        final_data = filtered_data.reset_index(drop=True)
        processed_data = processed_data.reset_index(drop=True)

        # Prepare for prediction
        predictions = []
        unique_locations = final_data[['longitude', 'latitude']].drop_duplicates()

        for _, location in unique_locations.iterrows():
            # Filter processed_data to get the last known data for this location (2023)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                location_data = processed_data[
                    (final_data['longitude'] == location['longitude']) &
                    (final_data['latitude'] == location['latitude'])
                ]

            if location_data.empty:
                logger.debug(
                    "No data found for location (%s, %s). Skipping...",
                    location['longitude'], location['latitude']
                )
                continue

            # Get the last known data for this location (2023)
            last_data = location_data.iloc[-1]

            # Prepare the feature vector for this location
            last_features = last_data.drop('subsidence').values.reshape(1, -1)

            # Initialize cumulative prediction with the last known subsidence
            cumulative_subsidence = final_data.loc[
                (final_data['longitude'] == location['longitude']) &
                (final_data['latitude'] == location['latitude'])
            ].iloc[-1][target]

            # Predict for each future year
            for year in forecast_years:
                # Create a full numerical input for scaling
                placeholder_row = pd.DataFrame(
                    [np.zeros(len(scaler.feature_names_in_))],
                    columns=scaler.feature_names_in_
                )
                placeholder_row['year'] = year
                year_scaled_input = scaler.transform(placeholder_row)[0]

                # Update the 'year' feature in the last_features array
                year_idx = list(processed_data.columns).index('year')
                last_features[0, year_idx] = year_scaled_input[
                    list(processed_data.columns).index('year')
                ]

                # Predict the change in subsidence (scaled)
                delta_scaled = model.predict(last_features)
                if delta_scaled.ndim > 1:
                    delta_scaled = delta_scaled.ravel()

                delta_original = target_scaler.inverse_transform([[delta_scaled[0]]])[0, 0]

                # Update cumulative subsidence
                cumulative_subsidence += delta_original

                # Update lag features
                for lag in range(lags, 0, -1):
                    lag_col = f'subsidence_lag_{lag}'
                    if lag == 1:
                        last_features[0, list(processed_data.columns).index(lag_col)] = delta_scaled[0]
                    else:
                        prev_lag_col = f'subsidence_lag_{lag - 1}'
                        last_features[0, list(processed_data.columns).index(lag_col)] = last_features[
                            0, list(processed_data.columns).index(prev_lag_col)
                        ]

                # Store the result
                predictions.append({
                    'longitude': location['longitude'],
                    'latitude': location['latitude'],
                    'year': year,
                    'predicted_subsidence': delta_original,
                    'cumulative_subsidence': cumulative_subsidence
                })

        # Convert predictions to a DataFrame
        predictions_df = pd.DataFrame(predictions)
        logger.info("Predictions generated successfully.")

        # Save predictions to CSV
        predictions_csv_path = os.path.join(data_path, 'hwm.ls.predicted.csv')
        predictions_df.to_csv(predictions_csv_path, index=False)
        logger.info("Predictions saved to %s", predictions_csv_path)

        return predictions_df

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise


def visualize_predictions(predictions_df, data_path, visualize_years=[2025, 2027, 2030]):
    """
    Visualize the prediction results for selected years.

    :param predictions_df: DataFrame containing predictions.
    :type predictions_df: pandas.DataFrame
    :param data_path: Path to save the visualization.
    :type data_path: str
    :param visualize_years: List of years to visualize.
    :type visualize_years: list
    """
    logger.info("Starting visualization of predictions...")
    try:
        visualize_data = predictions_df[predictions_df['year'].isin(visualize_years)].copy()

        # Initialize the plot
        fig, axes = plt.subplots(
            1, len(visualize_years), figsize=(18, 6), constrained_layout=True
        )

        for i, year in enumerate(visualize_years):
            ax = axes[i]
            year_data = visualize_data[visualize_data['year'] == year]
            sc = ax.scatter(
                year_data['longitude'],
                year_data['latitude'],
                c=year_data['cumulative_subsidence'],
                cmap='viridis',
                s=10,
                alpha=0.8
            )
            ax.set_title(f'Cumulative Subsidence Prediction - {year}')
            ax.axis('off')  # Remove axes for cleaner plots

        # Add a single colorbar to the figure
        cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.1)
        cbar.set_label('Cumulative Subsidence (mm)')

        # Save the figure
        output_filename = os.path.join(data_path, 'hwm_quantile_predictions_visualization.png')
        fig.savefig(output_filename, dpi=300)
        plt.close(fig)
        logger.info("Visualization saved to %s", output_filename)

    except Exception as e:
        logger.error("Visualization failed: %s", e)
        raise


def main(args):
    """
    Main function to execute the Hammerstein-Wiener Model workflow.

    :param args: Command-line arguments.
    :type args: argparse.Namespace
    """
    data_path = args.data_path
    lags = args.lags
    forecast_years = args.forecast_years

    # Suppress warnings for clarity
    warnings.filterwarnings('ignore')

    # Load data
    filtered_data = load_data(data_path)

    # Preprocess data
    processed_data, scalers_path = preprocess_data(filtered_data, data_path, lags=lags)

    # Train model
    hwr_model = train_model(processed_data, data_path, lags=lags)

    # Make predictions
    predictions_df = make_predictions(
        processed_data,
        filtered_data,
        hwr_model,
        scalers_path,
        data_path,
        forecast_years=forecast_years,
        lags=lags
    )

    # Visualize predictions
    visualize_predictions(predictions_df, data_path, visualize_years=[2025, 2027, 2030])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hammerstein-Wiener Model for Land Subsidence Prediction."
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the data directory.'
    )
    parser.add_argument(
        '--lags',
        type=int,
        default=3,
        help='Number of lag features to create for the target variable.'
    )
    parser.add_argument(
        '--forecast_years',
        type=int,
        nargs='+',
        default=list(range(2024, 2031)),
        help='List of years to forecast subsidence.'
    )

    args = parser.parse_args()
    main(args)
