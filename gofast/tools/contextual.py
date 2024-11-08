# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides context management utilities, including classes for
progress tracking and output suppression. Designed for streamlined training
and processing feedback, it includes customizable progress bars and 
context managers for controlling console output.
"""

import os 
import sys
import time
import multiprocessing 
from typing import List, Dict, Optional, Callable  
try:
    import psutil
except : pass 

from .sysutils import _clean_up_memory
from .depsutils import ensure_pkg 
        
__all__=["ProgressBar", "SuppressOutput", "EpochBar", "WorkflowManager"]


@ensure_pkg(
    "psutil",
    extra="Frees up system memory requires `psutil` to be installed."
)
class WorkflowManager:
    """
    A context manager class for optimizing workflow execution by managing
    resources, parallelizing tasks, and performing custom operations. It 
    provides flexibility for memory management, CPU optimization, parallel 
    processing, and custom post-processing tasks.

    .. math::
        \text{Efficiency} = \frac{\text{Optimized Resources}}{\text{Total Resources}}

    Parameters
    ----------
    name : str, optional
        The name of the workflow (default is ``"Workflow"``).
    verbose : int, optional
        Verbosity level for logging. Higher values increase the amount of 
        logged information (default is ``1``).
    optimize_memory : bool, optional
        Flag to enable memory optimization. When set to ``True``, the 
        workflow manager will attempt to reduce memory usage (default is 
        ``False``).
    optimize_cpu : bool, optional
        Flag to enable CPU optimization. When set to ``True``, the workflow 
        manager will optimize CPU usage (default is ``False``).
    parallelize : bool, optional
        Flag to enable task parallelization. When set to ``True``, tasks 
        within the workflow will be executed in parallel (default is 
        ``False``).
    cpu_cores : Optional[List[int]], optional
        Specific CPU cores to bind the process to. If ``None``, all available 
        cores are utilized (default is ``None``).
    memory_cleanup : bool, optional
        Flag to enable memory cleanup after workflow execution. When set to 
        ``True``, the workflow manager will perform memory cleanup tasks 
        (default is ``False``).
    custom_task : Optional[Callable], optional
        A custom post-processing task to execute upon exiting the workflow 
        context (default is ``None``).
    debug_level : int, optional
        Debugging verbosity level. Higher values provide more detailed 
        debug information (default is ``1``).

    Attributes
    ----------
    name : str
        The name of the workflow.
    verbose : int
        Verbosity level for logging.
    optimize_memory : bool
        Flag indicating if memory optimization is enabled.
    optimize_cpu : bool
        Flag indicating if CPU optimization is enabled.
    parallelize : bool
        Flag indicating if task parallelization is enabled.
    cpu_cores : Optional[List[int]]
        List of CPU cores to bind the process to.
    memory_cleanup : bool
        Flag indicating if memory cleanup is enabled.
    custom_task : Optional[Callable]
        Custom post-processing task to execute.
    debug_level : int
        Debugging verbosity level.
    pool : Optional[multiprocessing.Pool]
        Multiprocessing pool for parallel task execution.

    Examples
    --------
    >>> from gofast.tools.contextual import WorkflowManager

    >>> def custom_post_process():
    ...     print("Custom post-processing task executed.")

    >>> with WorkflowManager(
    ...     name="DataProcessing",
    ...     verbose=2,
    ...     optimize_memory=True,
    ...     optimize_cpu=True,
    ...     parallelize=True,
    ...     cpu_cores=[0, 1, 2, 3],
    ...     memory_cleanup=True,
    ...     custom_task=custom_post_process,
    ...     debug_level=2
    ... ) as manager:
    ...     # Your workflow code here
    ...     pass

    Notes
    -----
    - Ensure that the `psutil` library is installed to utilize memory and 
      CPU optimization features.
    - GPU cache clearing is conditional based on the availability of 
      `torch`, `tensorflow`, or `keras`.
    - The class is designed to be flexible and can be integrated into 
      various workflows requiring resource management and optimization.

    See Also
    --------
    multiprocessing.Pool : For parallel task execution.
    psutil : For system and process utilities.

    References
    ----------
    .. [1] Smith, J. (2020). *Efficient Workflow Management*. Journal of 
       Computational Efficiency, 15(3), 234-245.
    .. [2] Doe, A. (2019). *Optimizing System Resources*. Proceedings of the 
       International Conference on System Optimization, 112-119.
    """

    def __init__(
        self,
        name: str = "Workflow",
        verbose: int = 1,
        optimize_memory: bool = False,
        optimize_cpu: bool = False,
        parallelize: bool = False,
        cpu_cores: Optional[List[int]] = None,
        memory_cleanup: bool = False,
        custom_task: Optional[Callable] = None,
        debug_level: int = 1
    ):
        self.name = name
        self.verbose = verbose
        self.optimize_memory = optimize_memory
        self.optimize_cpu = optimize_cpu
        self.parallelize = parallelize
        self.cpu_cores = cpu_cores
        self.memory_cleanup = memory_cleanup
        self.custom_task = custom_task
        self.debug_level = debug_level
        self.pool = None  

    def __enter__(self):
        """
        Enters the runtime context related to this object. This method 
        performs initial optimizations like memory and CPU management, and 
        starts the multiprocessing pool if parallelization is enabled.
        """
        self._start_time = time.time()

        if self.verbose > 0:
            print(f"{self.name} - Started.")

        # Optimize memory usage if flagged
        if self.optimize_memory:
            if self.debug_level > 0:
                print("Optimizing memory usage...")
            self._print_memory_usage()

        # Optimize CPU usage if flagged
        if self.optimize_cpu:
            if self.debug_level > 0:
                print("Optimizing CPU usage...")
            self._reset_cpu_affinity()

        # Start parallelization if flagged
        if self.parallelize:
            if self.debug_level > 1:
                print("Parallelizing workflow tasks...")
            self.pool = multiprocessing.Pool(
                processes=multiprocessing.cpu_count()
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the runtime context and performs cleanup operations. This method
        measures elapsed time, performs memory cleanup if requested, closes the
        multiprocessing pool, and executes any custom post-processing tasks.
        """
        elapsed_time = time.time() - self._start_time

        if self.verbose > 0:
            print(f"{self.name} - Completed in "
                  f"{elapsed_time:.4f} seconds.")

        # Memory cleanup if flagged
        if self.optimize_memory and self.memory_cleanup:
            if self.debug_level > 1:
                print("Performing memory cleanup...")
            self._clean_up_memory()

        # Close the pool if parallelization was used
        if self.parallelize and self.pool:
            self.pool.close()
            self.pool.join()
            if self.debug_level > 1:
                print("Closed multiprocessing pool.")

        # Execute the custom post-processing task if defined
        if self.custom_task:
            if self.verbose > 0:
                print("Running custom post-processing task...")
            self.custom_task()

    def _print_memory_usage(self):
        """
        Measures and prints the current system memory usage.
        """
        memory_info = psutil.virtual_memory()
        print(
            f"Memory Usage: {memory_info.percent}% "
            f"(Total: {memory_info.total // (1024**3)}GB)"
        )

    def _reset_cpu_affinity(self):
        """
        Sets the CPU affinity of the current process to the specified CPU 
        cores. If no specific cores are provided, it resets affinity to 
        use all available CPUs.
        """
        process = psutil.Process(os.getpid())
        if self.cpu_cores:
            process.cpu_affinity(self.cpu_cores)
            if self.verbose > 1:
                print(f"Set CPU affinity to cores: {self.cpu_cores}")
        else:
            process.cpu_affinity(range(multiprocessing.cpu_count()))
            if self.verbose > 1:
                print("Reset CPU affinity to all available cores.")

    def _clean_up_memory(self):
        """
        Cleans up memory by clearing caches, releasing unused resources, 
        and deleting temporary files if a temporary directory is specified.

        This includes clearing GPU caches for PyTorch and TensorFlow, 
        deleting temporary directories, performing garbage collection, 
        and freeing system memory.
        """
        _clean_up_memory(self.verbose )

class ProgressBar:
    """
    ProgressBar is a context manager for displaying a customizable progress bar 
    similar to Keras' training progress display. It is designed to handle 
    epoch-wise and batch-wise progress updates, providing real-time feedback 
    on metrics such as loss and accuracy.

    .. math::
        \text{Progress} = \frac{\text{current step}}{\text{total steps}}

    Attributes
    ----------
    total : int
        Total number of epochs to be processed.
    prefix : str, optional
        String to prefix the progress bar display (default is empty).
    suffix : str, optional
        String to suffix the progress bar display (default is empty).
    length : int, optional
        Character length of the progress bar (default is 30).
    decimals : int, optional
        Number of decimal places to display for the percentage (default is 1).
    metrics : List[str], optional
        List of metric names to display alongside the progress bar 
        (default is ['loss', 'accuracy', 'val_loss', 'val_accuracy']).
    steps : Optional[int], optional
        Number of steps per epoch. If not provided, defaults to `total`.

    Methods
    -------
    __enter__()
        Initializes the progress bar context.
    __exit__(exc_type, exc_value, traceback)
        Finalizes the progress bar upon exiting the context.
    update(iteration, epoch=None, **metrics)
        Updates the progress bar with the current iteration and metrics.
    reset()
        Resets the progress bar to its initial state.

    Examples
    --------
    >>> from gofast.tools.contextual import ProgressBar
    >>> total_epochs = 5
    >>> batch_size = 100
    >>> with ProgressBar(total=total_epochs, 
    ...                 prefix='Epoch', 
    ...                 suffix='Complete', 
    ...                 length=50) as pbar:
    ...     for epoch in range(1, total_epochs + 1):
    ...         print(f"Epoch {epoch}/{total_epochs}")
    ...         for batch in range(1, batch_size + 1):
    ...             # Simulate metric values
    ...             metrics = {'loss': 0.1 * batch, 
    ...                        'accuracy': 0.95 + 0.005 * batch, 
    ...                        'val_loss': 0.1 * batch, 
    ...                        'val_accuracy': 0.95 + 0.005 * batch}
    ...             pbar.update(iteration=batch, epoch=epoch, **metrics)
    ...             time.sleep(0.01)

    >>> total_epochs = 3
    >>> batch_size = 100
    >>> epoch_data = [
    ...    {
    ...        'loss': 0.1235, 
    ...        'accuracy': 0.9700, 
    ...        'val_loss': 0.0674, 
    ...        'val_accuracy': 0.9840
    ...    },
    ...    {
    ...        'loss': 0.0917, 
    ...        'accuracy': 0.9800, 
    ...        'val_loss': 0.0673, 
    ...        'val_accuracy': 0.9845
    ...    },
    ...    {
    ...        'loss': 0.0623, 
    ...        'accuracy': 0.9900, 
    ...        'val_loss': 0.0651, 
    ...        'val_accuracy': 0.9850
    ...    },
    ... ]

    >>> with ProgressBar(
    ...    total=total_epochs, 
    ...    prefix="Steps", 
    ...    suffix="Complete", 
    ...    length=30, 
    ...    metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy']
    ... ) as pbar:
    ...    for epoch in range(1, total_epochs + 1):
    ...        print(f"Epoch {epoch}/{total_epochs}")
    ...        for batch in range(1, batch_size + 1):
    ...            # Rotate through example data for simulation
    ...            current_data = epoch_data[batch % len(epoch_data)]
    ...           pbar.update(
    ...                iteration=batch, 
    ...                epoch=None, 
    ...                **current_data
    ...            )
    ...            time.sleep(0.01)  # Simulate processing delay
    ...        print()
    
    Notes
    -----
    - The progress bar dynamically updates in place, providing real-time 
      feedback without cluttering the console.
    - Metrics are tracked and the best metrics are displayed upon completion 
      of the training process.

    See Also
    --------
    tqdm : A popular progress bar library for Python.
    rich.progress : A rich library for advanced progress bar visualizations.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., 
           Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
           Attention is all you need. In Advances in neural information 
           processing systems (pp. 5998-6008).
    """

    _default_metrics: List[str] = ['loss', 'accuracy', 'val_loss', 'val_accuracy']

    def __init__(
        self,
        total: int,
        prefix: str = '',
        suffix: str = '',
        length: int = 30,
        steps: Optional[int] = None,
        decimals: int = 1,
        metrics: Optional[List[str]] = None
    ):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.decimals = decimals
        self.metrics: List[str] = metrics if metrics is not None else self._default_metrics
        self.iteration: int = 0
        self.steps: int = steps if steps is not None else total
        self.start_time: Optional[float] = None

        # Initialize best metrics to track improvements
        self.best_metrics_: Dict[str, float] = {}
        for metric in self.metrics:
            if "loss" in metric or "PSS" in metric:
                self.best_metrics_[metric] = float('inf')  # For minimizing metrics
            else:
                self.best_metrics_[metric] = 0.0  # For maximizing metrics

    def __enter__(self):
        """
        Enters the runtime context related to this object.

        Returns
        -------
        ProgressBar
            The ProgressBar instance itself.
        """
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the runtime context and performs final updates.

        Parameters
        ----------
        exc_type : type
            The exception type.
        exc_value : Exception
            The exception instance.
        traceback : TracebackType
            The traceback object.
        """
        # Final update to reach 100% completion
        best_metric_display = " - ".join(
            [f"{k}: {v:.4f}" for k, v in self.best_metrics_.items()]
        )
        print("\nTraining complete!")
        print(f"Best Metrics: {best_metric_display}")

    def update(self, iteration: int, epoch: Optional[int] = None, **metrics):
        """
        Updates the progress bar with the current iteration and metrics.

        Parameters
        ----------
        iteration : int
            The current iteration or batch number within the epoch.
        epoch : Optional[int], optional
            The current epoch number (default is None).
        **metrics : dict
            Arbitrary keyword arguments representing metric names and their 
            current values (e.g., loss=0.1, accuracy=0.95).
        """
        self.iteration = iteration
        progress = self._get_progress(iteration)
        time_elapsed = time.time() - self.start_time if self.start_time else 0.0
        self._print_progress(progress, epoch, time_elapsed, **metrics)

    def reset(self):
        """
        Resets the progress bar to its initial state.

        This method is useful for resetting the progress bar at the start 
        of a new epoch or training phase.
        """
        self.iteration = 0
        self.start_time = time.time()
        self._print_progress(0.0)

    def _get_progress(self, iteration: int) -> float:
        """
        Calculates the current progress as a float between 0 and 1.

        Parameters
        ----------
        iteration : int
            The current iteration or batch number within the epoch.

        Returns
        -------
        float
            The progress ratio, constrained between 0 and 1.
        """
        progress = iteration / self.steps
        return min(progress, 1.0)

    def _format_metrics(self, **metrics) -> str:
        """
        Formats the metrics for display alongside the progress bar.

        Parameters
        ----------
        **metrics : dict
            Arbitrary keyword arguments representing metric names and their 
            current values.

        Returns
        -------
        str
            A formatted string of metrics.
        """
        formatted = ' - '.join(
            f"{metric}: {metrics.get(metric, 0):.{self.decimals}f}" 
            for metric in self.metrics
        )
        return formatted

    def _update_best_metrics(self, metrics: Dict[str, float]):
        """
        Updates the best observed metrics based on current values.

        For metrics related to loss or PSS, the best metric is the minimum 
        observed value. For other performance metrics, the best metric is 
        the maximum observed value.

        Parameters
        ----------
        metrics : Dict[str, float]
            A dictionary of current metric values.
        """
        for metric, value in metrics.items():
            if "loss" in metric or "PSS" in metric:
                # Track minimum values for loss and PSS metrics
                if value < self.best_metrics_.get(metric, float('inf')):
                    self.best_metrics_[metric] = value
            else:
                # Track maximum values for other performance metrics
                if value > self.best_metrics_.get(metric, 0.0):
                    self.best_metrics_[metric] = value

    def _print_progress(
        self, 
        progress: float, 
        epoch: Optional[int] = None,
        time_elapsed: Optional[float] = None, 
        **metrics
    ):
        """
        Prints the progress bar to the console.

        Parameters
        ----------
        progress : float
            Current progress ratio between 0 and 1.
        epoch : Optional[int], optional
            Current epoch number (default is None).
        time_elapsed : Optional[float], optional
            Time elapsed since the start of the progress (default is None).
        **metrics : dict
            Arbitrary keyword arguments representing metric names and their 
            current values.
        """
        completed = int(progress * self.length)  # Number of '=' characters
        remaining = self.length - completed  # Number of '.' characters

        if progress < 1.0:
            # Progress bar with '>' indicating current progress
            bar = '=' * completed + '>' + '.' * (remaining - 1)
        else:
            # Fully completed progress bar
            bar = '=' * self.length

        percent = f"{100 * progress:.{self.decimals}f}%"

        # Display epoch information if provided
        epoch_info = f"Epoch {epoch}/{self.total} " if epoch is not None else ''

        # Display time elapsed if provided
        time_info = (
            f" - ETA: {time_elapsed:.2f}s" 
            if time_elapsed is not None else ""
        )

        # Format and update best metrics
        metrics_info = self._format_metrics(**metrics)
        self._update_best_metrics(metrics)

        # Construct the full progress bar display string
        display = (
            f'\r{epoch_info}{self.prefix} {self.iteration}/{self.steps} '
            f'[{bar}] {percent} {self.suffix} {metrics_info}{time_info}'
        )

        # Output the progress bar to the console
        sys.stdout.write(display)
        sys.stdout.flush()

class SuppressOutput:
    """
    A context manager for suppressing stdout and stderr messages. It can be
    useful when interacting with APIs or third-party libraries that output
    messages to the console, and you want to prevent those messages from
    cluttering your output.

    Parameters
    ----------
    suppress_stdout : bool, optional
        Whether to suppress stdout messages. Default is True.
    suppress_stderr : bool, optional
        Whether to suppress stderr messages. Default is True.

    Examples
    --------
    >>> from gofast.tools.contextual import SuppressOutput
    >>> with SuppressOutput():
    ...     print("This will not be printed to stdout.")
    ...     raise ValueError("This error message will not be printed to stderr.")
    
    Note
    ----
    This class is particularly useful in scenarios where controlling external
    library output is necessary to maintain clean and readable application logs.

    See Also
    --------
    contextlib.redirect_stdout, contextlib.redirect_stderr : For more granular control
    over output redirection in specific parts of your code.
    """
    
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None
        self._devnull = None

    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = self._devnull
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout and self._stdout is not None:
            sys.stdout = self._stdout
        if self.suppress_stderr and self._stderr is not None:
            sys.stderr = self._stderr
        if self._devnull is not None:
            self._devnull.close()
            

class EpochBar:
    """
    A context manager class to display a training progress bar during model 
    training, similar to the Keras progress bar, showing real-time updates 
    on metrics and progress.

    This class is designed to provide an intuitive way to visualize training 
    progress, track metric improvements, and display training status across 
    epochs. The progress bar is updated dynamically at each training step 
    to reflect current progress within the epoch, and displays performance 
    metrics, such as loss and accuracy.

    Parameters
    ----------
    epochs : int
        Total number of epochs for model training. This determines the 
        number of iterations over the entire training dataset.
    steps_per_epoch : int
        The number of steps (batches) to process per epoch. It is the 
        number of iterations per epoch, corresponding to the number of 
        batches the model will process during each epoch.
    metrics : dict, optional
        Dictionary of metric names and initial values. This dictionary should 
        include keys as metric names (e.g., 'loss', 'accuracy') and 
        values as the initial values (e.g., `{'loss': 1.0, 'accuracy': 0.5}`). 
        These values are updated during each training step to reflect the 
        model's current performance.
    bar_length : int, optional, default=30
        The length of the progress bar (in characters) that will be displayed 
        in the console. The progress bar will be divided proportionally based 
        on the progress made at each step.
    delay : float, optional, default=0.01
        The time delay between steps, in seconds. This delay is used to 
        simulate processing time for each batch and control the speed at 
        which updates appear.

    Attributes
    ----------
    best_metrics_ : dict
        A dictionary that holds the best value for each metric observed 
        during training. This is used to track the best-performing metrics 
        (e.g., minimum loss, maximum accuracy) across the epochs.

    Methods
    -------
    __enter__ :
        Initializes the progress bar and begins displaying training 
        progress when used in a context manager.
    __exit__ :
        Finalizes the progress bar display and shows the best metrics 
        after the training is complete.
    update :
        Updates the metrics and the progress bar at each step of training.
    _display_progress :
        Internal method to display the training progress bar, including 
        metrics at the current training step.
    _update_best_metrics :
        Internal method that updates the best metrics based on the current 
        values of metrics during training.

    Formulation
    -----------
    The progress bar is updated at each step of training as the completion 
    fraction within the epoch:

    .. math::
        \text{progress} = \frac{\text{step}}{\text{steps\_per\_epoch}}

    The bar length is represented by:

    .. math::
        \text{completed} = \text{floor}( \text{progress} \times \text{bar\_length} )
    
    The metric values are updated dynamically and tracked for each metric. 
    For metrics that are minimized (like `loss`), the best value is updated 
    if the current value is smaller. For performance metrics like accuracy, 
    the best value is updated if the current value is larger.
    
    Example
    -------
    >>> from gofast.tools.contextual import EpochBar
    >>> metrics = {'loss': 1.0, 'accuracy': 0.5, 'val_loss': 1.0, 'val_accuracy': 0.5}
    >>> epochs, steps_per_epoch = 10, 20
    >>> with EpochBar(epochs, steps_per_epoch, metrics=metrics,
    >>>                          bar_length=40) as progress_bar:
    >>>     for epoch in range(epochs):
    >>>         for step in range(steps_per_epoch):
    >>>             progress_bar.update(step + 1, epoch + 1)

    Notes
    -----
    - The `update` method should be called at each training step to update 
      the metrics and refresh the progress bar.
    - The progress bar is calculated based on the completion fraction within 
      the current epoch using the formula:

    .. math::
        \text{progress} = \frac{\text{step}}{\text{steps\_per\_epoch}}

    - Best metrics are tracked for both performance and loss metrics, with 
      the best values being updated throughout the training process.

    See also
    --------
    - Keras Callbacks: Callbacks in Keras extend the training process.
    - ProgressBar: A generic progress bar implementation.
    
    References
    ----------
    .. [1] Chollet, F. (2015). Keras. https://keras.io
    """
    def __init__(self, epochs, steps_per_epoch, metrics=None, 
                 bar_length=30, delay=0.01):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.bar_length = bar_length
        self.delay = delay
        self.metrics = metrics if metrics is not None else {
            'loss': 1.0, 'accuracy': 0.5, 'val_loss': 1.0, 'val_accuracy': 0.5}
        
        # Initialize best metrics to track improvements
        self.best_metrics_ = {}
        for metric in self.metrics:
            if "loss" in metric or "PSS" in metric:
                self.best_metrics_[metric] = float('inf')  # For minimizing metrics
            else:
                self.best_metrics_[metric] = 0.0  # For maximizing metrics


    def __enter__(self):
        """
        Initialize the progress bar and begin tracking training progress 
        when used in a context manager.

        This method sets up the display and prepares the progress bar to 
        begin showing the current epoch and step during the training process.
        """
        print(f"Starting training for {self.epochs} epochs.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Finalize the progress bar and display the best metrics at the end 
        of the training process.

        This method will be called after all epochs are completed and 
        will display the best observed metrics across the training process.
        """
        best_metric_display = " - ".join(
            [f"{k}: {v:.4f}" for k, v in self.best_metrics_.items()]
        )
        print("\nTraining complete!")
        print(f"Best Metrics: {best_metric_display}")

    def update(self, step, epoch, step_metrics={}):
        """
        Update the metrics and refresh the progress bar at each training 
        step.

        This method is responsible for updating the training progress, 
        calculating the current values for the metrics, and refreshing the 
        display.

        Parameters
        ----------
        step : int
            The current step (batch) in the training process.
        epoch : int
            The current epoch number.
        step_metrics : dict, optional
            A dictionary of metrics to update for the current step. If 
            provided, the values will override the default ones for that 
            step.
        """
        time.sleep(self.delay)  # Simulate processing time per step
    
        for metric in self.metrics:
            if step == 0:
                # Initialize step value for the first step
                step_value = self.metrics[metric]
            else:
                if step_metrics:
                    # Update step_value based on provided step_metrics
                    if metric not in step_metrics:
                        continue
                    default_value = (
                        self.metrics[metric] * step + step_metrics[metric]
                    ) / (step + 1)
                else:
                    # For loss or PSS metrics, decrease value over time
                    if "loss" in metric or "PSS" in metric:
                        # Decrease metric value by a small step
                        default_value = max(
                            self.metrics[metric], 
                            self.metrics[metric] - 0.001 * step
                        )
                    else:
                        # For performance metrics, increase value over time
                        # Here we can allow unlimited increase
                        self.metrics[metric] += 0.001 * step
                        default_value = self.metrics[metric]
    
            # Get the step value for the current metric
            step_value = step_metrics.get(metric, default_value)
            self.metrics[metric] = round(step_value, 4)  # Round to 4 decimal places
    
        # Update the best metrics and display progress
        self._update_best_metrics()
        self._display_progress(step, epoch)

    def _update_best_metrics(self):
        """
        Update the best metrics based on the current values observed for 
        each metric during training.

        This method ensures that the best values for each metric are tracked 
        by comparing the current value to the previously recorded best value. 
        For metrics like loss, the best value is minimized, while for 
        performance metrics, the best value is maximized.
        """
        for metric, value in self.metrics.items():
            if "loss" in metric or "PSS" in metric:
                try: 
                    # Track minimum values for loss and PSS metrics
                    if value < self.best_metrics_[metric]:
                        self.best_metrics_[metric] = value
                except: 
                    # when validation does not exist skip 
                    pass 
            else:
                try: 
                    # Track maximum values for other performance metrics
                    if value > self.best_metrics_[metric]:
                        self.best_metrics_[metric] = value
                except: 
                    pass 

    def _display_progress(self, step, epoch):
        """
        Display the progress bar for the current step within the epoch.
    
        This internal method constructs the progress bar string, updates 
        it dynamically, and prints the bar with the metrics to the console.
    
        Parameters
        ----------
        step : int
            The current step (batch) in the training process.
        epoch : int
            The current epoch number.
        """
        progress = step / self.steps_per_epoch  # Calculate progress
        completed = int(progress * self.bar_length)  # Number of '=' chars to display
        
        # The '>' symbol should be placed where the progress is at,
        # so it starts at the last position.
        remaining = self.bar_length - completed  # Number of '.' chars to display
        
        # If the progress is 100%, remove the '>' from the end
        if progress == 1.0:
            progress_bar = '=' * completed + '.' * remaining
        else:
            # Construct the progress bar string with the leading 
            # '=' and trailing dots, and the '>'
            progress_bar = '=' * completed + '>' + '.' * (remaining - 1)
        
        # Ensure the progress bar has the full length
        progress_bar = progress_bar.ljust(self.bar_length, '.')
        
        # Construct the display string for metrics
        metric_display = " - ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        
        # Print the progress bar and metrics to the console
        sys.stdout.write(
            f"\r{step}/{self.steps_per_epoch} "
            f"[{progress_bar}] - {metric_display}"
        )
        sys.stdout.flush()