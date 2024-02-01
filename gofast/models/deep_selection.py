# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Time Series Prediction with LSTM')
    parser.add_argument('--folder_path', type=str, default='', 
                        help='Path to the CSV folder containing the data')
    parser.add_argument('--folder_list', nargs='+', default='', 
                        help='Paths to folders containing CSV files to be merged')
    parser.add_argument('--side_folder_list', nargs='+', default='', 
                        help='Paths to folders containing CSV files to be merged for side data')
    parser.add_argument('--num_rows', type=int, default=None, 
                        help='Number of rows to load from the CSV file')
    parser.add_argument('--n_past', type=int, default=12, 
                        help='Number of past time steps to consider')
    parser.add_argument('--n_future', type=int, default=1, 
                        help='Number of future time steps to predict')
    parser.add_argument('--lemda', type=float, default=0.5, 
                        help='Lambda value for custom loss function')
    parser.add_argument('--n_units', nargs='+', default='128', 
                        help='Number of units in each LSTM layer')
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--initial_lr', type=float, default=0.001, 
                        help='Initial learning rate')
    parser.add_argument('--inp', nargs='+', type=str, default=['train_custom'], 
                        help='List of actions to perform: train_custom, train_mse, test_custom, test_mse, infer')
    parser.add_argument('--patience', type=int, default=5, 
                        help='Patience for early stopping')
    parser.add_argument('--save_metrics', action='store_true', default=False, 
                        help='Save training metrics to a JSON file')
    return parser.parse_args()

def load_data(folder_path, rows):
    all_dataframes = []
    for root, _, files in os.walk(folder_path):
        existing_csv_file = os.path.join(root, "merged_data.csv")
        if os.path.exists(existing_csv_file):
            dataframe = pd.read_csv(existing_csv_file)
            all_dataframes.append(dataframe)
        else:
            csv_files = [file for file in files if file.endswith(".csv")]
            if csv_files:
                dataframes = [pd.read_csv(os.path.join(root, csv_file), skiprows=2, nrows=rows) for csv_file in csv_files]
                concatenated_dataframe = pd.concat(dataframes, ignore_index=True)
                concatenated_dataframe.sort_values(by=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
                concatenated_dataframe.to_csv(os.path.join(root, "merged_data.csv"), index=False)
                all_dataframes.append(concatenated_dataframe)
    return pd.concat(all_dataframes, ignore_index=True) if all_dataframes else None

def load_data_from_folders(folder_paths, num_rows=None):
    all_dataframes = []
    for folder_path in folder_paths:
        existing_csv_file = os.path.join(folder_path, "merged_data_folder.csv")
        if os.path.exists(existing_csv_file):
            all_dataframes.append(pd.read_csv(existing_csv_file))
        else:
            csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
            if csv_files:
                dataframes = [pd.read_csv(os.path.join(folder_path, csv_file), skiprows=2, nrows=num_rows) for csv_file in csv_files]
                merged_dataframe = pd.concat(dataframes, ignore_index=True)
                merged_dataframe.sort_values(by=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
                merged_dataframe.to_csv(os.path.join(folder_path, "merged_data_folder.csv"), index=False)
                all_dataframes.append(merged_dataframe)
    return pd.concat(all_dataframes, ignore_index=True) if all_dataframes else None

def create_side_by_side_dataframe(folder_paths, num_rows=None):
    all_dataframes = [pd.read_csv(os.path.join(folder_path, "merged_data_folder.csv")) 
                      for folder_path in folder_paths if os.path.exists(os.path.join(folder_path, "merged_data_folder.csv"))]
    return pd.concat(all_dataframes, axis=1) if all_dataframes else pd.DataFrame()

def preprocess_data(data, A, n_past, n_future, split_ratio=0.8, n_infer=200):
    target_column = 'Wind Speed'
    wind_speed_data = data[target_column].values[:-1]
    data_modified = data.iloc[:-1].copy()
    A_modified = A.iloc[:-1].copy()
    A_numpy = A_modified.values
    X = np.dot(np.linalg.pinv(A_numpy), wind_speed_data)
    data_modified['Estimated Wind Speed'] = np.dot(A_numpy, X)
    min_vals, max_vals = data_modified.min(), data_modified.max()
    data_modified = data_modified.apply(lambda x: (x - min_vals[x.name]) / (max_vals[x.name] - min_vals[x.name]) if min_vals[x.name] != max_vals[x.name] else 1)
    
    def create_datasets(data_modified, n_past, n_future):
        x, x_with_estimate, y_actual, y_estimated = [], [], [], []
        for i in range(len(data_modified) - n_past - n_future + 1):
            x.append(data_modified.iloc[i: i + n_past].drop('Estimated Wind Speed', axis=1).values)
            x_with_estimate.append(data_modified.iloc[i: i + n_past].values)
            y_actual.append(data_modified[target_column].iloc[i + n_past: i + n_past + n_future].values)
            y_estimated.append(data_modified['Estimated Wind Speed'].iloc[i + n_past: i + n_past + n_future].values)
        return np.array(x), np.array(x_with_estimate), np.array(y_actual), np.array(y_estimated)

    x_train, x_train_with_estimate, y_train_actual, y_train_estimated = create_datasets(data_modified, n_past, n_future)
    
    x_train_infer = x_train[-n_infer:]
    y_train_actual_infer = y_train_actual[-n_infer:]
    y_train_estimated_infer = y_train_estimated[-n_infer:]
    x_train_with_estimate_infer = x_train_with_estimate[-n_infer:]

    x_train = x_train[:-n_infer]
    y_train_actual = y_train_actual[:-n_infer]
    y_train_estimated = y_train_estimated[:-n_infer]
    x_train_with_estimate = x_train_with_estimate[:-n_infer]

    indices = np.random.permutation(x_train.shape[0])
    split_idx = int(len(indices) * split_ratio)
    return (x_train[indices[:split_idx]], x_train[indices[split_idx:]],
            x_train_with_estimate[indices[:split_idx]], x_train_with_estimate[indices[split_idx:]],
            y_train_actual[indices[:split_idx]], y_train_actual[indices[split_idx:]],
            y_train_estimated[indices[:split_idx]], y_train_estimated[indices[split_idx:]],
            x_train_infer, y_train_actual_infer, y_train_estimated_infer, min_vals, max_vals)

# def create_lstm_model(input_shape, n_units_list, n_future):
#     model = Sequential()
#     for i, n_units in enumerate(n_units_list):
#         is_return_sequences = i < len(n_units_list) - 1
#         model.add(LSTM(units=int(n_units), activation="relu", return_sequences=is_return_sequences, input_shape=input_shape if i == 0 else None))
#         model.add(Dropout(0.2))
#     model.add(Dense(units=n_future))
#     return model

def custom_loss(y_true, y_pred, y_estimated, lambda_value):
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    additional_term = tf.reduce_mean(tf.square(y_true - y_estimated), axis=-1)
    return mse + lambda_value * additional_term

def train_model(model, x_train, y_train_actual, y_train_estimated, lambda_value, num_epochs, batch_size, initial_lr, checkpoint_dir, patience, use_custom_loss=True):
    optimizer = Adam(learning_rate=initial_lr)
    best_loss = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss, val_loss = train_epoch(model, optimizer, x_train, y_train_actual, y_train_estimated, lambda_value, batch_size, use_custom_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            model.save(os.path.join(checkpoint_dir, 'best_model'), save_format='tf')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    plot_training_validation_loss(train_losses, val_losses, checkpoint_dir)

def train_epoch(model, optimizer, x_train, y_train_actual, y_train_estimated, lambda_value, batch_size, use_custom_loss):
    epoch_train_loss, epoch_val_loss = [], []
    for x_batch, y_actual_batch, y_estimated_batch in data_generator(x_train, y_train_actual, y_train_estimated, batch_size):
        with tf.GradientTape() as tape:
            y_pred_batch = model(x_batch, training=True)
            loss = custom_loss(y_actual_batch, y_pred_batch, y_estimated_batch, lambda_value) if use_custom_loss else tf.keras.losses.mean_squared_error(y_actual_batch, y_pred_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_train_loss.append(tf.reduce_mean(loss).numpy())

    return np.mean(epoch_train_loss), calculate_validation_loss(model, x_train, y_train_actual, y_train_estimated, lambda_value, use_custom_loss)

def calculate_validation_loss(model, x_val, y_val_actual, y_val_estimated, lambda_value, use_custom_loss):
    val_preds = model.predict(x_val)
    val_loss = custom_loss(y_val_actual, val_preds, y_val_estimated, lambda_value) if use_custom_loss else tf.keras.losses.mean_squared_error(y_val_actual, val_preds)
    return np.mean(val_loss.numpy())

def plot_training_validation_loss(train_losses, val_losses, checkpoint_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'training_validation_loss.png'))

def evaluate_model(model_path, x_test, y_test, batch_size, max_value, min_value):
    model = load_model(model_path)
    predictions = model.predict(x_test, batch_size=batch_size)
    predictions_denormalized = denormalize(predictions, min_value, max_value)
    y_test_denormalized = denormalize(y_test, min_value, max_value)
    mse = mean_squared_error(y_test_denormalized, predictions_denormalized)
    mae = mean_absolute_error(y_test_denormalized, predictions_denormalized)
    r2 = r2_score(y_test_denormalized, predictions_denormalized)
    model_size = os.path.getsize(model_path)
    test_metrics = {'mse': mse, 'mae': mae, 'r2_score': r2, 'model_size_bytes': model_size}
    with open(model_path + '_test_metrics.json', 'w') as json_file:
        json.dump(test_metrics, json_file)

def denormalize(data, min_value, max_value):
    return data * (max_value - min_value) + min_value

def main():
    args = parse_arguments()

    if args.folder_path:
        data = load_data(args.folder_path, args.num_rows)
    elif args.folder_list:
        data = load_data_from_folders(args.folder_list, args.num_rows)
    else:
        raise ValueError("No valid data source specified.")

    side_merge_data = create_side_by_side_dataframe(args.side_folder_list, args.num_rows)
    (X_train, X_test, X_train_with_estimate, X_test_with_estimate,
     y_train_actual, y_test_actual, y_train_estimated, y_test_estimated,
     x_train_infer, y_train_actual_infer, y_train_estimated_infer,
     min_vals, max_vals) = preprocess_data(data, side_merge_data, args.n_past, args.n_future)

    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), n_units_list=args.n_units, n_future=args.n_future)
    model_path = 'best_model_saved_model'

    for action in args.inp:
        if action == 'train_custom':
            train_model(model, X_train, y_train_actual, y_train_estimated, args.lemda, args.num_epochs, args.batch_size, args.initial_lr, model_path, args.patience)
        elif action == 'train_mse':
            train_model(model, X_train, y_train_actual, y_train_estimated, args.lemda, args.num_epochs, args.batch_size, args.initial_lr, model_path, args.patience, use_custom_loss=False)
        elif action == 'test_custom':
            evaluate_model(os.path.join(model_path, 'best_model_custom'), X_test, y_test_actual, args.batch_size, max_vals['Wind Speed'], min_vals['Wind Speed'])
        elif action == 'test_mse':
            evaluate_model(os.path.join(model_path, 'best_model_mse'), X_test, y_test_actual, args.batch_size, max_vals['Wind Speed'], min_vals['Wind Speed'])
        elif action == 'infer':
            perform_inference(model_path, x_train_infer, y_train_actual_infer, min_vals['Wind Speed'], max_vals['Wind Speed'])

def perform_inference(model_path, x_train_infer, y_train_actual_infer, min_wind_speed, max_wind_speed):
    model_custom = load_model(os.path.join(model_path, 'best_model_custom'))
    model_mse = load_model(os.path.join(model_path, 'best_model_mse'))
    predicted_custom = model_custom.predict(x_train_infer)
    predicted_mse = model_mse.predict(x_train_infer)
    compare_predictions(predicted_custom, predicted_mse, y_train_actual_infer, min_wind_speed, max_wind_speed)

def compare_predictions(predicted_custom, predicted_mse, y_train_actual, min_val, max_val):
    predicted_custom_denorm = denormalize(predicted_custom, min_val, max_val)
    predicted_mse_denorm = denormalize(predicted_mse, min_val, max_val)
    y_actual_denorm = denormalize(y_train_actual, min_val, max_val)
    plot_predictions(predicted_custom_denorm, predicted_mse_denorm, y_actual_denorm, 'predicted_vs_actual_wind_speed.png')
    plot_errors(predicted_custom_denorm, predicted_mse_denorm, y_actual_denorm, 'prediction_errors.png')

def plot_predictions(predicted_custom, predicted_mse, actual, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_custom, label='Predicted Wind Speed (Custom Loss)', color='blue')
    plt.plot(predicted_mse, label='Predicted Wind Speed (MSE)', color='green')
    plt.plot(actual, label='Actual Wind Speed', color='orange')
    plt.title('Actual vs Predicted Wind Speed')
    plt.xlabel('Time Steps')
    plt.ylabel('Wind Speed')
    plt.legend()
    plt.savefig(filename)

def plot_errors(predicted_custom, predicted_mse, actual, filename):
    error_custom = np.abs(actual - predicted_custom)
    error_mse = np.abs(actual - predicted_mse)
    plt.figure(figsize=(12, 6))
    plt.plot(error_custom, label='Error (Custom Loss)', color='blue')
    plt.plot(error_mse, label='Error (MSE)', color='green')
    plt.title('Error in Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(filename)
    
def create_lstm_model(input_shape, n_units_list, n_future):
    model = Sequential()
    for i, n_units in enumerate(n_units_list):
        is_return_sequences = i < len(n_units_list) - 1
        if i == 0:
            model.add(LSTM(units=n_units, activation="tanh", return_sequences=is_return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=n_units, activation="tanh", return_sequences=is_return_sequences))
        model.add(Dropout(0.2))
    model.add(Dense(units=n_future))
    return model

if __name__ == "__main__":
    main()
