# -*- coding: utf-8 -*-

import time 
import json 
import numpy as np 

from ._base import Callback 

class BaseData(Callback):
    def __init__(self, model=None):
        super().__init__(model)
        self.data_statistics = {}


    def on_epoch_start(self, epoch, logs=None):
        """
        Called at the start of each epoch to handle pre-epoch operations, 
        including logging, data transformations, and tracking statistics.
    
        This method is intended to be called at the beginning of every epoch
        during training. It initializes and tracks epoch-level information, such 
        as the start time, and can be extended to apply data transformations, 
        handle logging, and manage other custom behavior before the training on 
        each batch starts.
    
        Parameters
        ----------
        epoch : int
            The current epoch index, starting from 0 for the first epoch.
        
        logs : dict, optional
            A dictionary of logs from the previous epoch (if applicable). This 
            can include information like the loss or accuracy values from the 
            previous epoch, though this argument may be empty at the start of 
            the first epoch.
    
        Notes
        -----
        - This method is a placeholder for implementing any custom logic that 
          needs to be executed before the epoch begins. For instance, it can 
          handle data augmentation, reset certain statistics, or trigger other 
          callbacks for pre-epoch processing.
        - It is crucial to manage time efficiently here as it can impact the overall 
          performance and tracking of the training process.
        """
    
        # Initialize the statistics for this epoch if not already initialized
        if epoch not in self.data_statistics:
            self.data_statistics[epoch] = {}
    
        # Track the start time of the epoch
        epoch_start_time = time.time()
        self.data_statistics[epoch]['epoch_start'] = epoch_start_time
    
        # Log a message about epoch start, useful for debugging or tracking training
        if self.verbose:
            print(f"Epoch {epoch + 1} started at {epoch_start_time:.4f} seconds.")
    
        # Log any additional pre-epoch information here (e.g., data distribution)
        # This can be extended to log specific data statistics or operations
        self._log_epoch_start_statistics(epoch)
    
        # Apply data transformations or pre-processing operations for this epoch
        # This allows for the augmentation of data or other transformations before training starts
        self._apply_data_transformation(epoch)
    
        # Handle data distribution if applicable (logging or transformations)
        # This method is often used to track class distribution or data integrity checks
        self._check_data_integrity(epoch)
    
        # Additional functionality can be implemented here (like resetting variables, counters, etc.)
        # For instance, any epoch-specific learning rate scheduling or optimizer adjustments
        self._handle_epoch_specific_operations(epoch)
    
    def _log_epoch_start_statistics(self, epoch):
        """ 
        This helper method logs specific statistics that are relevant at the start of an epoch. 
        It could include the size of the training set, data statistics (e.g., class distribution), 
        or any other useful information.
        """
        # Example: Log the size of the training data at the start of the epoch
        if hasattr(self.model, 'train_data'):
            train_data_size = len(self.model.train_data)
            self.data_statistics[epoch]['train_data_size'] = train_data_size
            if self.verbose:
                print(f"Epoch {epoch + 1}: Training data size: {train_data_size} samples.")
    
        # Additional statistics could include class distribution, feature statistics, etc.
        self._log_data_distribution(epoch)
    
    def _log_data_distribution(self, epoch):
        """ 
        Logs the data distribution (e.g., class counts) at the beginning of each epoch. 
        This could be helpful to monitor data balancing or to track shifts in data 
        distributions over epochs.
        """
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            labels = data.get('labels')  # Assuming the data is a dictionary with 'labels'
            
            if labels is not None:
                class_counts = {label: sum(labels == label) for label in set(labels)}
                self.data_statistics[epoch]['data_distribution'] = class_counts
                if self.verbose:
                    print(f"Epoch {epoch + 1}: Data distribution: {class_counts}")
            else:
                print(f"Warning: No 'labels' field found in the training data for epoch {epoch + 1}")
    
    def _apply_data_transformation(self, epoch):
        """ 
        Applies any required data transformations (e.g., normalization, augmentation) 
        at the start of the epoch.
        """
        if hasattr(self.model, 'train_data'):
            # Example transformation: Apply any data augmentation or preprocessing steps
            data = self.model.train_data
            transformed_data = self._transform_data(data)  # Assuming a method to transform the data
            self.model.train_data = transformed_data
            if self.verbose:
                print(f"Epoch {epoch + 1}: Applied data transformations.")
    
    def _transform_data(self, data):
        """ 
        Apply custom data transformations (e.g., augmentation, feature scaling).
        This can be extended to add any preprocessing or data augmentation routines.
        """
        # Placeholder: In practice, this could be any transformation such as normalizing features,
        # performing data augmentation, or filtering data.
        return data  # No transformation by default, can be extended
    
    def _check_data_integrity(self, epoch):
        """ 
        Check the integrity of the data at the start of the epoch (e.g., ensure no missing values, 
        verify data types). This is a placeholder method that can be extended based on requirements.
        """
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            # Example check: ensure that the data is not empty or corrupted
            if len(data) == 0:
                raise ValueError(f"Epoch {epoch + 1}: Training data is empty. Please check the data pipeline.")
            if not all(isinstance(x, np.ndarray) for x in data.values()):
                raise TypeError(f"Epoch {epoch + 1}: Data contains invalid types. Expected numpy arrays.")

    def on_epoch_end(self, epoch, logs=None):
        """ Called at the end of each epoch. """
        if epoch in self.data_statistics:
            self.data_statistics[epoch]['epoch_end'] = time.time()
            # Log data distribution after the epoch ends
            self._log_data_distribution(epoch)
        # Placeholder for post-epoch operations, can be extended
        pass

    def on_batch_start(self, batch, logs=None):
        """ Called before processing each batch. """
        # Placeholder for pre-batch operations, can be extended
        pass

    def on_batch_end(self, batch, logs=None):
        """ Called after processing each batch. """
        # Placeholder for post-batch operations, can be extended
        pass
  
class DataOpsCallback(BaseData):
    def __init__(self, model=None, data_operations=None, verbose=1):
        super().__init__(model)
        self.data_operations = data_operations if data_operations is not None else []
        self.verbose = verbose

    def on_epoch_start(self, epoch, logs=None):
        """ Apply data transformations at the start of each epoch. """
        if self.verbose:
            print(f"Epoch {epoch + 1}: Starting DataOps...")
        self.data_statistics[epoch] = {'epoch_start': time.time()}
        self._apply_data_operations(epoch)

    def on_epoch_end(self, epoch, logs=None):
        """ Log data operations statistics at the end of each epoch. """
        if epoch in self.data_statistics:
            self.data_statistics[epoch]['epoch_end'] = time.time()
        self._log_data_operations(epoch)

    def _apply_data_operations(self, epoch):
        """ Apply the list of data operations to the training data. """
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            for operation in self.data_operations:
                data = operation(data)
                if self.verbose:
                    print(f"Applied operation: {operation.__name__}")
            self.model.train_data = data  # Update model's training data after transformations

    def _log_data_operations(self, epoch):
        """ Log data operations applied during this epoch. """
        if epoch in self.data_statistics:
            self.data_statistics[epoch]['data_operations'] = [
                operation.__name__ for operation in self.data_operations
            ]
            if self.verbose:
                print(f"Epoch {epoch + 1}: Operations applied: {', '.join(self.data_statistics[epoch]['data_operations'])}")


class BaseData(Callback):
    """
    Base class for handling data operations during training. 

    This class provides hooks for interacting with training data at various stages
    such as the beginning of an epoch, the end of an epoch, and at the batch level.
    It can be extended for specific use cases like data augmentation, logging, or
    performing other on-the-fly data modifications.

    Attributes:
        model: The model associated with the callback.
        data_statistics: A dictionary that logs statistics during training, e.g., data distribution.
    """
    
    def __init__(self, model=None):
        super().__init__(model)
        self.data_statistics = {}

    def on_epoch_start(self, epoch, logs=None):
        """ Called at the start of each epoch. """
        self.data_statistics[epoch] = {'epoch_start': time.time()}
        # Can be extended to handle pre-epoch operations
        pass

    def on_epoch_end(self, epoch, logs=None):
        """ Called at the end of each epoch. """
        if epoch in self.data_statistics:
            self.data_statistics[epoch]['epoch_end'] = time.time()
            # Example: Log data distribution at the end of the epoch
            self._log_data_distribution(epoch)
        # Can be extended to handle post-epoch operations
        pass

    def on_batch_start(self, batch, logs=None):
        """ Called at the start of each batch. """
        # Can be extended for batch-specific operations
        pass

    def on_batch_end(self, batch, logs=None):
        """ Called at the end of each batch. """
        # Can be extended to log batch-level statistics
        pass
    
    def _log_data_distribution(self, epoch):
        """ Logs data distribution at the end of each epoch (or batch). """
        # Example: Assume data is passed through model's `train_data`
        # In practice, this can track various statistics like class distribution, feature stats, etc.
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            # Example logging: Track class distribution of labels
            labels = data['labels']  # Assume data is a dictionary with 'labels'
            class_counts = {label: sum(labels == label) for label in set(labels)}
            self.data_statistics[epoch]['data_distribution'] = class_counts
            print(f"Epoch {epoch + 1} data distribution: {class_counts}")

    def _apply_data_transformation(self, data):
        """ Applies any data transformations (like augmentation or preprocessing). """
        # This can be overridden to add specific transformations for each use case.
        # For example, augmenting data, normalizing features, etc.
        return data  # Return unmodified data by default
    
    def _get_data_statistics(self, data):
        """ Collects basic statistics from the data. """
        # Example: Assuming data has a 'labels' field for classification tasks
        if 'labels' in data:
            unique_labels, counts = np.unique(data['labels'], return_counts=True)
            return dict(zip(unique_labels, counts))
        return {}

    def reset(self):
        """ Reset the callback, clearing any saved statistics. """
        self.data_statistics = {}


class DataAugmentationCallback(BaseData):
    """
    Callback for applying data augmentation to training data.

    This callback applies data augmentation techniques during training. It overrides 
    methods for data modification and augmentation. The user can customize this
    class by modifying the `apply_augmentation` method to implement specific augmentation logic.

    Attributes:
        augmentation_function: A function that applies data augmentation to input data.
    """
    
    def __init__(self, model, augmentation_function, verbose=1):
        super().__init__(model)
        self.augmentation_function = augmentation_function
        self.verbose = verbose
    
    def on_batch_start(self, batch, logs=None):
        """ Modify the data on the fly before each batch. """
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            augmented_data = self.apply_augmentation(data)
            self.model.train_data = augmented_data
            if self.verbose:
                print(f"Augmented data for batch {batch + 1}")
    
    def apply_augmentation(self, data):
        """ Apply the augmentation function to the data. """
        return self.augmentation_function(data)


class DataLoggingCallback(BaseData):
    """
    Callback for logging statistics and other information related to training data.

    This callback allows logging of important metrics related to the data at different stages
    of the training process. It can be extended for tasks like tracking input features,
    class distributions, etc.

    Attributes:
        log_file: File path for storing the logged data.
    """
    
    def __init__(self, model, log_file="data_log.json", verbose=1):
        super().__init__(model)
        self.log_file = log_file
        self.verbose = verbose
    
    def on_epoch_end(self, epoch, logs=None):
        """ Log data-related information at the end of each epoch. """
        if self.verbose:
            print(f"Logging data at the end of epoch {epoch + 1}...")
        
        self._log_data(epoch)
        super().on_epoch_end(epoch, logs)
    
    def _log_data(self, epoch):
        """ Logs data distribution and other information to a file. """
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            data_info = {
                'epoch': epoch,
                'data_statistics': self._get_data_statistics(data),
            }
            with open(self.log_file, 'a') as f:
                json.dump(data_info, f)
                f.write("\n")
            if self.verbose:
                print(f"Data logged for epoch {epoch + 1}")

    def _get_data_statistics(self, data):
        """ Collects data statistics, such as feature means and distributions. """
        # Example: Get feature means (you can modify this to collect more specific stats)
        feature_means = {f"feature_{i}": np.mean(data['features'][:, i]) for i in range(data['features'].shape[1])}
        return feature_means
