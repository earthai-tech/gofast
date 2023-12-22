# taylor_diagram.py

import numpy as np
import matplotlib.pyplot as plt
from taylor_diagram_cy import calculate_statistics

def plot_taylor_diagram(predictions, reference, names):
    """
    Plot a Taylor Diagram.

    Parameters
    ----------
    predictions : list of np.ndarray
        A list containing the model predictions.

    reference : np.ndarray
        The reference data array.

    names : list of str
        A list containing the names of the models.
    """
    plt.figure(figsize=(10, 7))
    # ... (Plotting code here using Matplotlib, similar to the provided example) ...
    # This part will call calculate_statistics for each model to get the stats
    for i, pred in enumerate(predictions):
        std_dev, correlation = calculate_statistics(pred, reference)
        # ... (Plotting code continues) ...

if __name__=='__main__': 
    # Example
    predictions = [np.random.rand(100), np.random.rand(100)]
    reference = np.random.rand(100)
    names = ['Model1', 'Model2']
    plot_taylor_diagram(predictions, reference, names)
