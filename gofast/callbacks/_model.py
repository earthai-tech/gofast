# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:13:15 2024

@author: Daniel
"""
import os

try: 
    from tensorflow import summary
except: 
    pass 

class Callback:
    """ Base callback class for the gofast package.

    This class defines the basic structure of callbacks in the gofast package. 
    Users can inherit from this class and define their specific callback behavior 
    by overriding the appropriate methods.

    Attributes:
        model: The model object that the callback is associated with.
        history: A dictionary that records the history of metrics and other information.
    """
    
    def __init__(self, model=None):
        self.model = model
        self.history = {}

    def on_epoch_start(self, epoch, logs=None):
        """ Called at the start of each epoch """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """ Called at the end of each epoch """
        pass

    def on_batch_start(self, batch, logs=None):
        """ Called at the start of each batch """
        pass

    def on_batch_end(self, batch, logs=None):
        """ Called at the end of each batch """
        pass

    def on_train_start(self, logs=None):
        """ Called at the start of training """
        pass

    def on_train_end(self, logs=None):
        """ Called at the end of training """
        pass

class ModelCheckpoint(Callback):
    """ Callback that saves the model during training.

    This callback saves the model at the end of each epoch if the validation loss 
    improves or if it meets a certain condition. The model is saved to the file system 
    with a specified format.

    Attributes:
        filepath: Path to the file where the model should be saved.
        save_best_only: If `True`, the model will only be saved if the validation 
                        loss improves.
    """
    
    def __init__(self, model, filepath, save_best_only=False, verbose=1):
        super().__init__(model)
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss", None)
        if val_loss is not None:
            if self.save_best_only:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_model(epoch)
            else:
                self._save_model(epoch)
    
    def _save_model(self, epoch):
        """ Save the model to the specified filepath. """
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        
        model_filename = os.path.join(self.filepath, f"model_epoch_{epoch + 1}.h5")
        self.model.save(model_filename)
        if self.verbose:
            print(f"Model saved to {model_filename}")

class EarlyStopping(Callback):
    """ Callback that stops training when the monitored metric has stopped improving.

    Attributes:
        monitor: The metric to monitor (e.g., "val_loss", "accuracy").
        patience: The number of epochs with no improvement before stopping.
        min_delta: The minimum change to qualify as an improvement.
        verbose: If 1, prints a message when stopping.
    """
    
    def __init__(self, model, monitor='val_loss', patience=10, min_delta=0, verbose=1):
        super().__init__(model)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_val_metric = float('inf')  # Assuming minimizing the monitored metric
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None:
            if abs(self.best_val_metric - current) <= self.min_delta:
                self.wait += 1
            else:
                self.best_val_metric = current
                self.wait = 0

            if self.wait >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                self.model.stop_training = True
                

class LearningRateScheduler(Callback):
    """ Callback to modify the learning rate during training based on a schedule.

    Attributes:
        schedule: A function that takes the epoch index and returns the new learning rate.
        verbose: If `True`, prints the learning rate at each update.
    """
    
    def __init__(self, model, schedule, verbose=1):
        super().__init__(model)
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        """ Update learning rate at the end of each epoch. """
        new_lr = self.schedule(epoch)
        self.model.set_learning_rate(new_lr)
        if self.verbose:
            print(f"Learning rate updated to {new_lr:.6f} at epoch {epoch + 1}.")
        
            
class TensorBoardLogger(Callback):
    """ Callback to log metrics to TensorBoard during training.

    Attributes:
        log_dir: The directory where TensorBoard logs should be saved.
        histogram_freq: Frequency of histogram computation for weights.
    """
    


    def __init__(self, model, log_dir='./logs', histogram_freq=0):
        super().__init__(model)
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.writer = summary.create_file_writer(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        """ Log metrics at the end of each epoch. """
        with self.writer.as_default():
            for metric, value in logs.items():
                summary.scalar(metric, value, step=epoch)
            self.writer.flush()

if __name__=='__main__': 
    from gofast.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
    from gofast import Learner
    
    # Define your model
    model = Learner()
    
    # Define callbacks
    checkpoint_cb = ModelCheckpoint(model, filepath='./models', save_best_only=True)
    early_stopping_cb = EarlyStopping(model, patience=5, monitor='val_loss')
    lr_scheduler_cb = LearningRateScheduler(model, schedule=lambda epoch: 0.01 * (0.1 ** (epoch // 10)))
    
    # Train your model with callbacks
    X_train = ...
    y_train =...
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, 
              callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb]
              )
