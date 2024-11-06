# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:21:07 2024

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
