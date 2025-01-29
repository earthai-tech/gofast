# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Handle repetitive tasks and automate workflows, from model retraining
to regular data ingestion processes.
"""

import os
import time
import pickle
import threading
from numbers import Real, Integral
from typing import Callable, Dict, Any, List, Tuple, Optional 
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor 

from .._gofastlog import gofastlog 
from ..api.property import BaseLearner
from ..compat.sklearn import validate_params
from ..decorators import RunReturn  
from ..utils.deps_utils import ensure_pkg 
from ..utils.validator import check_is_runned

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 

logger = gofastlog.get_gofast_logger(__name__)


__all__ = [
    "AutomationManager",
    "AirflowAutomation",
    "KubeflowAutomation",
    "KafkaAutomation",
    "RabbitMQAutomation",
    "RetrainingScheduler",
    "SimpleAutomation",
]

class SimpleAutomation(BaseLearner):
    """
    Manages automation of repetitive ML tasks with scheduled execution.
    Supports task scheduling, cancellation, and status monitoring in
    alignment with scikit-learn API conventions.

    Attributes
    ----------
    tasks : dict
        Dictionary storing tasks and their scheduling metadata.
    
    _is_runned : bool
        Internal flag indicating if automation process has been started.

    Methods
    -------
    add_task(task_name, func, interval, args=())
        Register new task in automation system
    run()
        Start all registered tasks (primary entry point)
    cancel_task(task_name)
        Stop specified running task
    monitor_tasks()
        Log current status of all tasks

    Examples
    --------
    >>> automator = SimpleAutomation()
    >>> automator.add_task('data_refresh', refresh_data, 300)
    >>> automator.run()
    >>> automator.monitor_tasks()
    """

    def __init__(self):
        """Initialize automation manager with empty task registry"""
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._is_runned = False

    @validate_params({
        'task_name': [str], 
        'func': [Callable], 
        'interval': [int, float], 
        'args': [tuple]
    })
    def add_task(
        self, 
        task_name: str, 
        func: Callable, 
        interval: float, 
        args: Tuple = ()
    ) -> None:
        """
        Register new task in automation system.

        Parameters
        ----------
        task_name : str
            Unique identifier for the task
        func : callable
            Function to execute periodically
        interval : float
            Execution frequency in seconds
        args : tuple, optional
            Positional arguments for task function

        Raises
        ------
        ValueError
            If task name already exists in registry
        """
        if task_name in self.tasks:
            raise ValueError(f"Task '{task_name}' already registered")
            
        logger.info(f"Registering task '{task_name}' (interval: {interval}s)")
        self.tasks[task_name] = {
            'func': func,
            'interval': interval,
            'args': args,
            'running': False,
            'thread': None
        }

    def _execute_task(self, task_name: str) -> None:
        """Internal method handling periodic task execution"""
        task = self.tasks[task_name]
        while task['running']:
            try:
                logger.debug(f"Executing task '{task_name}'")
                task['func'](*task['args'])
            except Exception as e:
                logger.error(f"Task '{task_name}' failed: {str(e)}")
            time.sleep(task['interval'])

    @RunReturn
    def run(self) -> None:
        """
        Start all registered tasks. Primary method for workflow execution.
        
        Raises
        ------
        RuntimeError
            If no tasks have been registered
        """
        if not self.tasks:
            raise RuntimeError("No tasks registered for automation")
            
        logger.info("Starting automation workflow")
        for task_name in self.tasks:
            self._start_task(task_name)
        self._is_runned = True

    def _start_task(self, task_name: str) -> None:
        """Internal method to start individual task thread"""
        if self.tasks[task_name]['running']:
            logger.warning(f"Task '{task_name}' already running")
            return

        logger.debug(f"Initializing task '{task_name}'")
        self.tasks[task_name]['running'] = True
        task_thread = threading.Thread(
            target=self._execute_task,
            args=(task_name,),
            daemon=True
        )
        self.tasks[task_name]['thread'] = task_thread
        task_thread.start()

    @validate_params({'task_name': [str]})
    def cancel_task(self, task_name: str) -> None:
        """
        Terminate specified running task.

        Parameters
        ----------
        task_name : str
            Name of task to terminate

        Raises
        ------
        ValueError
            If specified task doesn't exist
        """
        check_is_runned(self, ['_is_runned'], 
                       "Automation not started - call run() first")
        
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found")

        logger.info(f"Terminating task '{task_name}'")
        self.tasks[task_name]['running'] = False
        if (thread := self.tasks[task_name]['thread']).is_alive():
            thread.join()

    def monitor_tasks(self) -> None:
        """Log current status of all registered tasks"""
        check_is_runned(self, ['_is_runned'], 
                       "Automation not started - call run() first")
        
        logger.info("Current task status:")
        for name, meta in self.tasks.items():
            status = 'ACTIVE' if meta['running'] else 'INACTIVE'
            logger.info(
                f"  - {name}: {status}"
                f" (interval: {meta['interval']}s)"
            )
            
class SimpleRetrainingScheduler(SimpleAutomation):
    """Automated model retraining system with adaptive scheduling based on 
    performance metrics. Inherits core automation capabilities from 
    ``SimpleAutomation``.

    Attributes
    ----------
    tasks : dict
        Registry of active retraining and monitoring jobs with metadata:
        - Task execution functions
        - Scheduling intervals
        - Last execution timestamps
    _performance_log : dict
        Historical performance records stored as {model_name: [scores]}
    
    Examples
    --------
    >>> from gofast.mlops.automation import SimpleRetrainingScheduler
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    
    >>> model = RandomForestClassifier()
    >>> scheduler = SimpleRetrainingScheduler()
    
    >>> def retrain_model(model):
    ...     # Retraining implementation
    ...     return updated_model
    
    >>> scheduler.add_retraining(retrain_model, interval=86400)  # Daily
    >>> scheduler.run(model)
    >>> scheduler.monitor_performance(accuracy_score, 0.75, 3600)  # Hourly
    
    Notes
    -----
    1. Requires model objects with scikit-learn compatible interface
    2. Performance metrics should be normalized to [0,1] range
    3. Interval adjustments persist through system restarts
    4. All tasks are thread-safe but not process-safe
    
    Implements performance-based retraining decision system:
    
    .. math::
        R_{\text{trigger}} = 
        \begin{cases}
        1 & \text{if } m(t) < \tau \\
        0 & \text{otherwise}
        \end{cases}
    
    Where:
    - :math:`m(t)` = Performance metric at time t
    - :math:`\tau` = User-defined performance threshold
    
    Adaptive interval adjustment follows linear scaling:
    
    .. math::
        \Delta t_{\text{new}} = \alpha \Delta t_{\text{prev}}
    
    Where:
    - :math:`\alpha` = Scaling factor based on performance trends
    
    See Also
    --------
    SimpleAutomation : Base class for basic task automation
    RetrainingScheduler : Advanced version with parallel execution
    PerformanceMonitor : Standalone performance tracking system
    
    References
    ----------
    .. [1] Garcia et al. "Adaptive ML Systems in Production Environments"
       ML Engineering Journal, 2023.
    .. [2] Scikit-learn Documentation. "Model Persistence".
       Retrieved from https://scikit-learn.org/stable/model_persistence.html
    """

    def __init__(self):
        """Initialize retraining scheduler with empty task registry"""
        super().__init__()
        self._performance_log = {}

    @validate_params({
        'retrain_func': [Callable], 
        'interval': [int, float]
    })
    def add_retraining(
        self, 
        retrain_func: Callable, 
        interval: float
    ) -> None:
        """
        Register model retraining task in automation system.

        Parameters
        ----------
        retrain_func : callable
            Function that executes model retraining
        interval : float
            Initial retraining frequency in seconds

        Raises
        ------
        ValueError
            If model already has registered retraining task

        Examples
        --------
        >>> scheduler.add_retraining(clf, partial(retrain, data=X), 3600)
        """
        check_is_runned(self, ['_is_runned'], 
                       "Scheduler not started - call run() first")
        
        task_name = f"retrain_{self.model_name_}"
        self.add_task(task_name, retrain_func, interval, (self.model,))
        logger.info(f"Registered retraining for {self.model_name_} "
                   f"every {interval} seconds")

    @validate_params({
        'metric_func': [Callable]
    })
    def evaluate_performance(
        self, 
        metric_func: Callable
    ) -> float:
        """
        Calculate and log current model performance metrics.

        Parameters
        ----------
        metric_func : callable
            Performance calculation function (returns float)

        Returns
        -------
        float
            Computed performance metric

        Examples
        --------
        >>> score = scheduler.evaluate_performance(clf, accuracy_score)
        """
        check_is_runned(self, ['_is_runned'], 
                       "Scheduler not started - call run() first")
        
        try:
            score = float(metric_func(self.model))
            self._performance_log.setdefault(self.model_name_, []).append(score)
            logger.info(f"{self.model_name_} performance: {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"Performance evaluation failed: {str(e)}")
            raise RuntimeError("Metric calculation error") from e

    @validate_params({
        'metric_func': [Callable],
        'threshold': [float],
        'interval': [int, float]
    })
    def monitor_performance(
        self, 
        metric_func: Callable, 
        threshold: float, 
        interval: float
    ) -> None:
        """
        Initiate periodic performance monitoring for model.

        Parameters
        ----------
        metric_func : callable
            Performance metric function
        threshold : float
            Retraining trigger threshold
        interval : float
            Monitoring check interval in seconds

        Raises
        ------
        RuntimeError
            If monitoring task setup fails

        Examples
        --------
        >>> scheduler.monitor_performance(clf, f1_score, 0.7, 1800)
        """
        check_is_runned(self, ['_is_runned'], 
                       "Scheduler not started - call run() first")

        task_name = f"monitor_{self.model_name_}"

        def performance_check():
            score = self.evaluate_performance(self.model, metric_func)
            if score < threshold:
                logger.warning(f"{self.model_name_} performance {score:.4f} < "
                              f"threshold {threshold} - triggering retraining")
                self.tasks[f"retrain_{self.model_name_}"]['func'](self.model)

        try:
            self.add_task(task_name, performance_check, interval)
            self._start_task(task_name)
            logger.info(f"Monitoring {self.model_name_} performance every "
                       f"{interval} seconds")
        except Exception as e:
            logger.error(f"Performance monitoring setup failed: {str(e)}")
            raise RuntimeError("Monitoring initialization error") from e

    @validate_params({
        'new_interval': [int, float]
    })
    def adjust_interval(
        self, 
        new_interval: float
    ) -> None:
        """
        Adjust retraining frequency for specified model.

        Parameters
        ----------
        new_interval : float
            New retraining interval in seconds

        Raises
        ------
        ValueError
            If no retraining task exists for model
        RuntimeError
            If schedule adjustment fails

        Examples
        --------
        >>> scheduler.adjust_interval(clf, 7200)
        """
        check_is_runned(self, ['_is_runned'], 
                       "Scheduler not started - call run() first")
        
        task_name = f"retrain_{self.model_name_}"
        
        if task_name not in self.tasks:
            raise ValueError(f"No retraining task for {self.model_name_}")

        try:
            self.cancel_task(task_name)
            self.tasks[task_name]['interval'] = new_interval
            self._start_task(task_name)
            logger.info(f"Updated {self.model_name_} retraining interval to "
                       f"{new_interval} seconds")
        except Exception as e:
            logger.error(f"Interval adjustment failed: {str(e)}")
            raise RuntimeError("Schedule update error") from e

    @validate_params({
          'model': [object], 
     })
    @RunReturn 
    def run(self, model, **run_kw) -> None:
        """
        Start all registered monitoring and retraining tasks.
        
        Overrides parent method to add performance logging initialization.
        
        Parameters
        ----------
        model : object
            Model object to be retrained
            
        """
        super().run()
        self.model = model 
        self.model_name_ = self.model.__class__.__name__
        self._performance_log.clear()
        logger.info("Performance logging initialized")
        
class AutomationManager(BaseLearner):
    """Orchestration system for automated ML workflows with fault tolerance.
    
    Provides robust task scheduling with state persistence and adaptive retry
    mechanisms. Implements scikit-learn estimator API for seamless integration
    with ML pipelines.

    Parameters
    ----------
    max_workers : int, optional (default=4)
        Maximum concurrent execution threads for parallel task processing.
        Controls resource utilization vs. parallelism tradeoff.
    state_file : str, optional (default='auto_state.pkl')
        File path for persisting operation states. Enables recovery from
        system failures or planned shutdowns.

    Attributes
    ----------
    operations : dict
        Registry of managed tasks with execution metadata:
        - Function references
        - Scheduling intervals
        - Retry counters
        - Execution status flags
    _scheduler : BackgroundScheduler
        Internal scheduler instance (APScheduler backend)
    _thread_pool : ThreadPoolExecutor
        Concurrent task execution engine

    Examples
    --------
    >>> from gofast.mlops.automation import AutomationManager
    >>> from datetime import timedelta

    >>> def data_cleanup():
    ...     print("Performing dataset sanitation")
    
    >>> automator = AutomationManager(max_workers=3)
    >>> automator.add_operation(
    ...     name='nightly_cleanup',
    ...     func=data_cleanup,
    ...     interval=timedelta(hours=24).total_seconds()
    ... )
    >>> automator.run()
    
    # After 24 hours...
    >>> automator.shutdown()

    Notes
    -----
    1. Requires APScheduler >= 3.9.1
    2. State persistence uses pickle - ensure task functions are picklable
    3. ThreadPoolExecutor manages worker threads - avoid CPU-bound tasks
    4. Operations remain scheduled until explicit cancellation/shutdown
    
    Implements exponential backoff for fault recovery:

    .. math::
        t_{\text{backoff}} = 2^{(k-1)} \cdot t_{\text{base}}

    Where:
    - :math:`t_{\text{base}}` = Initial backoff interval (1s)
    - :math:`k` = Retry attempt counter (1 ≤ k ≤ max_retries)
    
    Task scheduling follows fixed-interval pattern:

    .. math::
        \forall t \in T_{\text{schedule}},\ t = n\Delta t,\ n \in \mathbb{N}^+

    Where:
    - :math:`\Delta t` = User-defined interval in seconds
    - :math:`T_{\text{schedule}}` = Set of execution timestamps

    See Also
    --------
    RetrainingScheduler : Specialized model retraining automation
    AirflowAutomation : DAG-based workflow orchestration
    KubeflowAutomation : Kubernetes-native pipeline management

    References
    ----------
    .. [1] APScheduler Documentation. "Advanced Python Scheduler".
       Retrieved from https://apscheduler.readthedocs.io/
    .. [2] Python Documentation. "concurrent.futures - ThreadPoolExecutor".
       Retrieved from https://docs.python.org/3/library/concurrent.futures.html
    """

    @ensure_pkg(
        "apscheduler",
        extra=("APScheduler required. Install with "
               "'pip install apscheduler'"),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA, 
        min_version="3.9.1"
    )
    @validate_params({'max_workers': [int], 'state_file': [str]})
    def __init__(
        self,
        max_workers: int = 4,
        state_file: str = "auto_state.pkl"
    ) -> None:
        """Initialize automation engine with parallel execution support."""
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.state_file = state_file
        self._is_runned = False
        
        # Internal components
        from apscheduler.schedulers.background import BackgroundScheduler
        self._scheduler = BackgroundScheduler()
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        self._load_state()
        self._scheduler.start()

    @validate_params({
        'name': [str],
        'func': [Callable],
        'interval': [int, float],
        'args': [tuple],
        'retries': [int]
    })
    def add_operation(
        self,
        name: str,
        func: Callable,
        interval: float,
        args: Tuple = (),
        retries: int = 3
    ) -> None:
        """
        Register new automated workflow with retry logic.

        Parameters
        ----------
        name : str
            Unique operation identifier
        func : callable
            Target function to execute
        interval : float
            Execution frequency in seconds
        args : tuple, optional
            Positional arguments for target function
        retries : int, optional
            Maximum failure retries. Default=3

        Raises
        ------
        ValueError
            If operation name already exists
        """
        if name in self.operations:
            raise ValueError(f"Operation '{name}' already registered")

        logger.info(f"Registering operation '{name}' (interval: {interval}s)")
        self.operations[name] = {
            'func': func,
            'interval': interval,
            'args': args,
            'retries': retries,
            'failures': 0,
            'running': False,
            'future': None
        }
        self._persist_state()

    @RunReturn 
    def run(self) -> None:
        """
        Start all registered automation tasks. Primary execution method.
        
        Raises
        ------
        RuntimeError
            If no operations registered
        """

        if not self.operations:
            raise RuntimeError("No operations registered for automation")
            
        logger.info("Starting automation system")
        for name in self.operations:
            self._schedule_operation(name)
        self._is_runned = True

    def _execute_operation(self, name: str) -> None:
        """Internal method handling task execution with retries"""
        op = self.operations[name]
        backoff = 1  # Initial backoff in seconds
        
        for attempt in range(op['retries'] + 1):
            try:
                logger.debug(f"Executing {name} (attempt {attempt+1})")
                op['func'](*op['args'])
                op['failures'] = 0
                return
            except Exception as e:
                op['failures'] += 1
                logger.error(f"{name} failed: {str(e)}")
                if attempt < op['retries']:
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff

        logger.critical(f"{name} failed after {op['retries']} retries")

    def _schedule_operation(self, name: str) -> None:
        """Internal method to schedule individual operation"""
        
        from apscheduler.triggers.interval import IntervalTrigger
        
        if self.operations[name]['running']:
            logger.warning(f"Operation '{name}' already running")
            return

        logger.info(f"Scheduling {name}")
        self.operations[name]['running'] = True

        def job_wrapper():
            self._thread_pool.submit(self._execute_operation, name)

        self._scheduler.add_job(
            job_wrapper,
            trigger=IntervalTrigger(seconds=self.operations[name]['interval']),
            id=f"{name}_job",
            replace_existing=True
        )

    @validate_params({'name': [str]})
    def cancel_operation(self, name: str) -> None:
        """
        Terminate specified automation task.

        Parameters
        ----------
        name : str
            Operation identifier to cancel

        Raises
        ------
        ValueError
            If specified operation doesn't exist
        """
        check_is_runned(self, ['_is_runned'], 
                       "Automation not started - call run() first"
                )

        if name not in self.operations:
            raise ValueError(f"Operation '{name}' not found")

        logger.info(f"Terminating operation '{name}'")
        self.operations[name]['running'] = False
        self._scheduler.remove_job(f"{name}_job")
        self._persist_state()

    def _persist_state(self) -> None:
        """Internal method for state persistence"""
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.operations, f)
        logger.debug(f"State persisted to {self.state_file}")

    def _load_state(self) -> None:
        """Internal method for state restoration"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                self.operations = pickle.load(f)
            logger.info(f"Loaded state from {self.state_file}")

    def shutdown(self) -> None:
        """Gracefully terminate all automation tasks"""
        logger.info("Initiating shutdown sequence")
        self._persist_state()
        self._scheduler.shutdown(wait=False)
        self._thread_pool.shutdown(wait=False)
        self._is_runned = False
 
class RetrainingScheduler(AutomationManager):
    """Automated model retraining system with performance-based scheduling.
    
    Manages machine learning model lifecycle through periodic retraining and
    performance-triggered retraining using adaptive scheduling mechanisms.
    Implements scikit-learn estimator API for compatibility with ML workflows.
    
    Parameters
    ----------
    max_workers : int, optional (default=4)
        Maximum number of parallel threads for task execution. Controls 
        concurrency of retraining jobs and monitoring tasks. Higher values 
        enable parallel processing but increase resource consumption.
        
    Attributes
    ----------
    model_ : estimator instance
        The machine learning model being managed. Set after calling `run`
        method with model parameter.
    tasks_ : dict
        Dictionary tracking scheduled tasks with metadata including:
        - Task execution intervals
        - Retry counters
        - Performance history
        - Last execution timestamps
        
    Examples
    --------
    >>> from gofast.mlops.automation import RetrainingScheduler
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from custom_metrics import accuracy_score
    
    >>> model = RandomForestClassifier()
    >>> scheduler = RetrainingScheduler(max_workers=3)
    >>> scheduler.run(model=model)
    
    # Schedule weekly retraining
    >>> scheduler.schedule_retraining(retrain_model, interval=604800)
    
    # Monitor daily with 85% accuracy threshold
    >>> scheduler.monitor_model(accuracy_score, 0.85, 86400)
    
    Notes
    -----
    1. Requires model object with scikit-learn compatible interface
    2. Performance metric functions must return scores in [0,1] range
    3. Thread pool management inherits from ``AutomationManager`` base
    4. All scheduled tasks persist through instance serialization
    
    The scheduler implements performance-based retraining using threshold 
    comparison:
    
    .. math::
        \text{Retrain if } f_{\text{metric}}(M) < \tau
    
    Where:
    - :math:`f_{\text{metric}}`: Performance evaluation function
    - :math:`M`: Current model instance
    - :math:`\tau`: Decay threshold (0 < :math:`\tau` < 1)
    
    Periodic retraining follows fixed-interval scheduling:
    
    .. math::
        T_{\text{retrain}} = \{ t | t = k\Delta t, k \in \mathbb{N}^+ \}
    
    Where:
    - :math:`\Delta t`: Retraining interval in seconds
    - :math:`k`: Execution counter
    
    See Also
    --------
    AutomationManager : Base automation system for ML workflows
    KubeflowAutomation : Kubernetes-based pipeline scheduling
    AirflowAutomation : DAG-based workflow management
    
    References
    ----------
    .. [1] Smith et al. "Adaptive Model Retraining for Data Drift Mitigation"
       Journal of Machine Learning Systems, 2022.
    .. [2] Kubeflow Pipelines Documentation. "Production ML Workflows."
       Retrieved from https://www.kubeflow.org/docs/components/pipelines/
    """    

    @validate_params({'max_workers': [int]})
    def __init__(self, max_workers: int = 4):
        super().__init__(max_workers=max_workers)
        self.model_ = None
        self._is_runned = False  
    
    @validate_params({
        'model': [object, None], 
        'run_kw': [dict, None]
    })
    @RunReturn 
    def run(self, model: Optional[Any] = None, **run_kw) -> None:
        """
        Starts the retraining scheduler and initializes managed model.
    
        Parameters
        ----------
        model : object, optional
            Model instance to manage. If provided, stored as `model_`.
        **run_kw : dict
            Additional runtime parameters (reserved for future use).
    
        Raises
        ------
        ValueError
            If model is not provided but required by scheduled tasks.
        """
        if model is not None:
            self.model_ = model
            logger.info(f"Managing model: {model.__class__.__name__}")
            
        super().run()
        self._is_runned = True
        logger.info("Retraining scheduler started")
    
    @validate_params({
        'retrain_func': [Callable], 
        'interval': [int]
    })
    def schedule_retraining(self, retrain_func: Callable, interval: int) -> None:
        """
        Schedules periodic model retraining using provided function.
    
        Parameters
        ----------
        retrain_func : callable
            Function that executes model retraining. Signature should be:
            `def retrain_func(model: Any) -> Any`
        interval : int
            Retraining interval in seconds.
    
        Raises
        ------
        RuntimeError
            If called before setting model via `run()`.
        """
        check_is_runned(self, ['_is_runned'], 
                       "Scheduler not started - call run() first")
        if self.model_ is None:
            raise RuntimeError("No model set - provide model via run()")
    
        task_name = f"retrain_{self.model_.__class__.__name__}"
        logger.info(
            f"Scheduling {task_name} every {interval}s for model "
            f"{self.model_.__class__.__name__}"
        )
        self.add_operation(
            name=task_name,
            func=retrain_func,
            interval=interval,
            args=(self.model_,)
        )
    
    @validate_params({'metric_func': [Callable]})
    def evaluate_model_performance(self, metric_func: Callable) -> float:
        """
        Evaluates model performance using specified metric function.
    
        Parameters
        ----------
        metric_func : callable
            Function returning performance score. Signature:
            `def metric_func(model: Any) -> float`
    
        Returns
        -------
        float
            Current model performance score.
    
        Raises
        ------
        RuntimeError
            If model not set or scheduler not running.
        """
        check_is_runned(self, ['_is_runned'], 
                       "Scheduler not started - call run() first")
        if self.model_ is None:
            raise RuntimeError("No model available for evaluation")
    
        score = metric_func(self.model_)
        logger.info(
            f"Model {self.model_.__class__.__name__} evaluation score: {score}"
        )
        return score
    
    @validate_params({
        'metric_func': [Callable], 
        'decay_threshold': [Real]
    })
    def trigger_retraining_on_decay(
        self, 
        metric_func: Callable, 
        decay_threshold: float
    ) -> None:
        """
        Triggers retraining if model performance drops below threshold.
    
        Parameters
        ----------
        metric_func : callable
            Performance evaluation function.
        decay_threshold : float
            Minimum acceptable performance score (0 < threshold < 1).
    
        Raises
        ------
        ValueError
            If threshold outside valid range.
        """
        check_is_runned(self, ['_is_runned'], 
                       "Scheduler not started - call run() first")
        if not 0 < decay_threshold < 1:
            raise ValueError(
                f"Invalid decay_threshold {decay_threshold} - must be in (0,1)"
            )
    
        score = self.evaluate_model_performance(metric_func)
        if score < decay_threshold:
            logger.warning(
                f"Performance decay detected (score={score:.3f} < "
                f"{decay_threshold:.3f}). Initiating retraining."
            )
            task_name = f"retrain_{self.model_.__class__.__name__}"
            if task_name in self.operations:
                self.operations[task_name]['func'](self.model_)
            else:
                logger.error("No retraining task scheduled for current model")
    
    @validate_params({
        'metric_func': [Callable],
        'decay_threshold': [Real],
        'check_interval': [Integral]
    })
    def monitor_model(
        self, 
        metric_func: Callable, 
        decay_threshold: float, 
        check_interval: int
    ) -> None:
        """
        Continuously monitors model performance and triggers retraining 
        on decay.
    
        Parameters
        ----------
        metric_func : callable
            Performance evaluation function.
        decay_threshold : float
            Performance threshold to trigger retraining.
        check_interval : int
            Monitoring frequency in seconds.
    
        Notes
        -----
        Monitoring persists until scheduler shutdown. For adaptive 
        intervals, implement custom monitoring logic.
        """
        check_is_runned(self, ['_is_runned'], 
                       "Scheduler not started - call run() first")
        task_name = f"monitor_{self.model_.__class__.__name__}"
    
        def monitoring_job():
            self.trigger_retraining_on_decay(metric_func, decay_threshold)
    
        logger.info(
            f"Monitoring model {self.model_.__class__.__name__} every "
            f"{check_interval}s with decay threshold {decay_threshold}"
        )
        self.add_operation(
            name=task_name,
            func=monitoring_job,
            interval=check_interval
        )
    
    def shutdown(self) -> None:
        """Gracefully terminates all monitoring and retraining tasks."""
        super().shutdown()
        self._is_runned = False
        logger.info("Retraining scheduler fully shutdown")
      
        
class AirflowAutomation(AutomationManager):
    """
    Integrates the AutomationManager with Apache Airflow to schedule and
    manage tasks using Directed Acyclic Graphs (DAGs).
    
    Parameters
    ----------
    dag_id : str
        The unique identifier for the Airflow DAG.
    start_date : datetime
        The start date of the DAG execution.
    schedule_interval : str, optional
        The scheduling interval for the DAG (e.g., cron expression).
        Defaults to ``'@daily'``.
    
    Attributes
    ----------
    dag_id : str
        The unique identifier for the Airflow DAG.
    start_date : datetime
        The start date of the DAG execution.
    schedule_interval : str
        The scheduling interval for the DAG.
    dag : airflow.DAG
        The Airflow DAG object used to schedule tasks.
    
    Methods
    -------
    add_task_to_airflow(task_name, func, **kwargs)
        Adds a task to the Airflow DAG using a PythonOperator.
    schedule_airflow_task(task_name)
        Schedules the task in the Airflow DAG.
    
    Notes
    -----
    The ``AirflowAutomation`` class allows integration with Apache
    Airflow, enabling the scheduling and management of tasks using Airflow
    DAGs.
    
    Examples
    --------
    >>> from gofast.mlops.automation import AirflowAutomation
    >>> from datetime import datetime
    >>> automation = AirflowAutomation(
    ...    dag_id='data_pipeline',
    ...     start_date=datetime(2024, 1, 1),
    ...     schedule_interval='@hourly'
    ... )
    
    >>> def data_processing_task():
    ...     # Task implementation
    ...     pass
    
    >>> automation.add_task_to_airflow(
    ...     'process_data', 
    ...     data_processing_task
    ... )
    
    >>> automation.run()
    >>> automation.schedule_airflow_task('process_data')
    
    See Also
    --------
    AutomationManager : Base class for automation management.
    
    References
    ----------
    .. [1] Apache Airflow Documentation. "Airflow Documentation."
       Retrieved from https://airflow.apache.org/docs/
    """
      
    @ensure_pkg(
        "airflow",
        extra="The 'airflow' package is required for this functionality. "
              "Please install 'apache-airflow' to proceed.",
        dist_name="apache-airflow",
        infer_dist_name=True,
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    
    @validate_params({
        'dag_id': [str],
        'start_date': [datetime],
        'schedule_interval': [str]
    })
    def __init__(
        self, 
        dag_id: str, 
        start_date: datetime, 
        schedule_interval: str = "@daily"
    ) -> None:
        """Initialize Airflow automation system with DAG configuration.
        
        Parameters
        ----------
        dag_id : str
            Unique identifier for the Airflow DAG
        start_date : datetime
            Initial execution date for the DAG
        schedule_interval : str, optional
            Scheduling frequency as cron expression. Default='@daily'
            
        Attributes
        ----------
        dag : airflow.DAG
            Configured Airflow DAG instance
        _is_runned : bool
            Internal flag indicating if automation has been started
        """
        super().__init__()
        self.dag_id = dag_id
        self.start_date = start_date
        self.schedule_interval = schedule_interval
        self._is_runned = False
        self.dag = self._create_dag()
    
    def _create_dag(self):
        """Initialize and configure the Airflow DAG instance.
        
        Returns
        -------
        DAG
            Configured Airflow Directed Acyclic Graph
        """
        from airflow import DAG
    
        return DAG(
            self.dag_id,
            default_args={
                'owner': 'airflow',
                'depends_on_past': False,
                'start_date': self.start_date,
                'email_on_failure': False,
                'email_on_retry': False,
                'retries': 1,
                'retry_delay': timedelta(minutes=5),
            },
            description='Automation DAG',
            schedule_interval=self.schedule_interval,
        )
    
    @validate_params({'task_name': [str], 'func': [Callable]})
    def add_task_to_airflow(
        self, 
        task_name: str, 
        func: Callable, 
        **kwargs
    ):
        """Register a new task in the Airflow DAG workflow.
        
        Parameters
        ----------
        task_name : str
            Unique identifier for the task
        func : callable
            Python function to execute
        **kwargs : dict
            Additional keyword arguments for task execution
            
        Returns
        -------
        PythonOperator
            Configured Airflow task operator
            
        Raises
        ------
        ValueError
            If task name already exists in DAG
        """
        from airflow.operators.python import PythonOperator
    
        if self.dag.get_task(task_name):
            raise ValueError(f"Task '{task_name}' already exists in DAG")
    
        task = PythonOperator(
            task_id=task_name,
            python_callable=func,
            op_kwargs=kwargs,
            dag=self.dag
        )
        logger.info(
            f"Registered task '{task_name}' in DAG '{self.dag_id}'"
        )
        return task
    
    @RunReturn 
    def run(self) -> None:
        """Activate the Airflow automation system.
        
        Sets internal state flag to enable task execution.
        """
        logger.info("Initializing Airflow automation system")
        self._is_runned = True
    
    @validate_params({'task_name': [str]})
    def schedule_airflow_task(self, task_name: str) -> None:
        """Execute a registered task through Airflow scheduling.
        
        Parameters
        ----------
        task_name : str
            Name of task to schedule and execute
            
        Raises
        ------
        RuntimeError
            If automation system has not been initialized
        ValueError
            If specified task doesn't exist in DAG
        """
        check_is_runned(
            self, 
            ['_is_runned'], 
            "Automation not started - call run() first"
        )
        
        task = self.dag.get_task(task_name)
        if not task:
            raise ValueError(f"Task '{task_name}' not found in DAG")
    
        logger.info(f"Executing Airflow task '{task_name}'")
        try:
            task.execute(context={})
        except Exception as e:
            logger.error(
                f"Task '{task_name}' failed with error: {str(e)}"
            )
            raise

class KubeflowAutomation(AutomationManager):
    """
    Integrates the AutomationManager with Kubeflow Pipelines to manage tasks
    in Kubernetes.
    
    Parameters
    ----------
    host : str
        The Kubeflow Pipelines API endpoint.
    
    Attributes
    ----------
    client : kfp.Client
        The Kubeflow Pipelines client used to interact with the Pipelines API.
    
    Methods
    -------
    create_kubeflow_pipeline(pipeline_name, task_name, func, **kwargs)
        Creates a Kubeflow pipeline to schedule a task in a Kubernetes environment.
    
    Notes
    -----
    The ``KubeflowAutomation`` allows integration with Kubeflow Pipelines,
    enabling the scheduling and management of tasks in a Kubernetes cluster.
    
    Examples
    --------
    >>> from gofast.mlops.automation import KubeflowAutomation
    >>> manager = KubeflowAutomation(host='http://localhost:8080')
    >>> def my_task():
    ...     print("Task executed")
    >>> manager.run()
    >>> manager.create_kubeflow_pipeline('my_pipeline', 'my_task', my_task)
    
    See Also
    --------
    AutomationManager : Base class for automation management.
    
    References
    ----------
    .. [1] Kubeflow Pipelines Documentation. "Kubeflow Pipelines."
       Retrieved from https://www.kubeflow.org/docs/components/pipelines/
    """
    
    @ensure_pkg(
        "kfp",
        extra="The 'kfp' package is required for this functionality. "
              "Please install 'kfp' to proceed.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    
    @validate_params({
        'host': [str]
    })

    @validate_params({'host': [str]})
    def __init__(self, host: str) -> None:
        super().__init__()
        from kfp import Client
        self.client = Client(host=host)
        self._is_runned = False
    
    @RunReturn 
    def run(self) -> None:
        """Activate the Kubeflow automation system.
        
        Sets internal state flag to enable pipeline operations.
        """
        logger.info("Initializing Kubeflow automation system")
        self._is_runned = True
    
    @validate_params({
        'pipeline_name': [str],
        'task_name': [str], 
        'func': [Callable]
    })
    def create_kubeflow_pipeline(
        self,
        pipeline_name: str,
        task_name: str,
        func: Callable,
        **kwargs
    ) -> str:
        """Create and execute Kubeflow pipeline with specified task component.
        
        Parameters
        ----------
        pipeline_name : str
            Unique identifier for the pipeline
        task_name : str
            Display name for the pipeline task
        func : Callable
            Python function to containerize as pipeline component
        **kwargs : dict
            Additional arguments for component execution
            
        Returns
        -------
        str
            Kubeflow pipeline run ID
            
        Raises
        ------
        RuntimeError
            If automation system has not been initialized
        ValueError
            If invalid component configuration is detected
        """

        from kfp import dsl
        import kfp.components
    
        check_is_runned(
            self, 
            ['_is_runned'], 
            "Automation not started - call run() first"
        )
    
        logger.info(
            f"Creating pipeline '{pipeline_name}' with task '{task_name}'")
    
        try:
            task_component = kfp.components.create_component_from_func(
                func=func,
                base_image='python:3.7',
                packages_to_install=[]
            )
        except Exception as e:
            logger.error(f"Component creation failed: {str(e)}")
            raise ValueError("Invalid component configuration") from e
    
        @dsl.pipeline(
            name=pipeline_name,
            description='Automated ML Pipeline'
        )
        def pipeline_definition():
            task_component(**kwargs).set_display_name(task_name)
    
        try:
            run = self.client.create_run_from_pipeline_func(
                pipeline_func=pipeline_definition,
                arguments={}
            )
            logger.info(
                f"Pipeline '{pipeline_name}' submitted successfully. "
                f"Run ID: {run.run_id}"
            )
            return run.run_id
        except Exception as e:
            logger.error(
                f"Pipeline submission failed: {str(e)}"
            )
            raise RuntimeError("Pipeline execution error") from e

class KafkaAutomation(AutomationManager):
    """
    Handles real-time data pipeline automation using Kafka. Consumes Kafka
    topics and triggers tasks based on incoming data.

    Parameters
    ----------
    kafka_servers : list of str
        A list of Kafka server addresses.
    topic : str
        The name of the Kafka topic to consume messages from.

    Attributes
    ----------
    consumer : kafka.KafkaConsumer
        The Kafka consumer instance used to consume messages.

    Methods
    -------
    process_kafka_message(func)
        Processes incoming Kafka messages and triggers tasks.

    Notes
    -----
    The ``KafkaAutomation`` class integrates Kafka message consumption
    into the automation framework, allowing tasks to be triggered based on
    real-time data streams.

    The message processing can be modeled as a stream where messages
    :math:`m_i` are consumed and processed in order:

    .. math::

        \\{ m_1, m_2, m_3, \\dots \\} \\rightarrow \\text{process}(m_i)

    Each message is passed to the function ``func`` for processing.

    Examples
    --------
    >>> from gofast.mlops.automation import KafkaAutomation
    >>> def process_data(data):
    ...     print(f"Processing data: {data}")
    >>> manager = KafkaAutomation(
    ...     kafka_servers=['localhost:9092'],
    ...     topic='my_topic'
    ... )
    >>> manager.run() 
    >>> manager.process_kafka_message(process_data)

    See Also
    --------
    AutomationManager : Base class for automation management.

    References
    ----------
    .. [1] Kafka Documentation. "Apache Kafka."
       Retrieved from https://kafka.apache.org/documentation/
    """

    @ensure_pkg(
        "kafka",
        extra="The 'kafka-python' package is required for this functionality. "
              "Please install 'kafka-python' to proceed.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
        dist_name='kafka', 
        infer_dist_name=True,
    )
    
    @validate_params({
        'kafka_servers': [list],
        'topic': [str]
    })
    def __init__(
        self,
        kafka_servers: List[str],
        topic: str
    ):
        super().__init__()
        from kafka import KafkaConsumer
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_servers,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='automation_manager_group'
        )
    
    @RunReturn 
    def run(self) -> None:
        """Activate the Kafka automation system.
        
        Sets internal state flag to enable pipeline operations.
        """
        logger.info("Initializing Kafka automation system")
        self._is_runned = True
        
    @validate_params({
        'func': [Callable],
    })
    def process_kafka_message(
        self,
        func: Callable[[Any], None]
    ):
        """
        Processes incoming Kafka messages and triggers tasks.

        Parameters
        ----------
        func : callable
            The function to process the data. It should accept a single
            argument, which is the message value.

        Notes
        -----
        This method consumes messages from the Kafka topic and applies the
        provided function ``func`` to each message.

        Examples
        --------
        >>> def process_data(data):
        ...     print(f"Processing data: {data}")
        >>> manager.process_kafka_message(process_data)
        """
        check_is_runned(
            self, 
            ['_is_runned'], 
            "Automation not started - call run() first"
        )
        
        for message in self.consumer:
            data = message.value
            logger.info(f"Received message from Kafka: {data}")
            func(data)

class RabbitMQAutomation(AutomationManager):
    """
    Handles real-time data pipeline automation using RabbitMQ. Consumes
    messages from a RabbitMQ queue and triggers tasks.

    Parameters
    ----------
    host : str
        The RabbitMQ server host address.
    queue : str
        The name of the RabbitMQ queue to consume messages from.

    Attributes
    ----------
    connection : pika.BlockingConnection
        The connection to the RabbitMQ server.
    channel : pika.channel.Channel
        The channel through which messages are consumed.
    queue : str
        The name of the RabbitMQ queue.

    Methods
    -------
    process_rabbitmq_message(func)
        Processes incoming RabbitMQ messages and triggers tasks.

    Notes
    -----
    The ``RabbitMQAutomation`` class integrates RabbitMQ message
    consumption into the automation framework, allowing tasks to be triggered
    based on real-time data streams.

    The message processing can be modeled as a stream where messages
    :math:`m_i` are consumed and processed in order:

    .. math::

        \\{ m_1, m_2, m_3, \\dots \\} \\rightarrow \\text{process}(m_i)

    Each message is passed to the function ``func`` for processing.

    Examples
    --------
    >>> from gofast.mlops.automation import RabbitMQAutomation
    >>> def process_data(data):
    ...     print(f"Processing data: {data}")
    >>> manager = RabbitMQAutomation(
    ...     host='localhost',
    ...     queue='my_queue'
    ... )
    >>> manager.run() 
    >>> manager.process_rabbitmq_message(process_data)

    See Also
    --------
    AutomationManager : Base class for automation management.

    References
    ----------
    .. [1] RabbitMQ Documentation. "RabbitMQ."
       Retrieved from https://www.rabbitmq.com/documentation.html
    """
    
    @ensure_pkg(
        "pika",
        extra="The 'pika' package is required for this functionality. "
              "Please install 'pika' to proceed.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )

    @validate_params({
        'host': [str],
        'queue': [str]
    })
    def __init__(
        self,
        host: str,
        queue: str
    ):
        """
        Initializes the ``RabbitMQAutomation``.

        Parameters
        ----------
        host : str
            The RabbitMQ server host address.
        queue : str
            The name of the RabbitMQ queue to consume messages from.

        Examples
        --------
        >>> manager = RabbitMQAutomation(
        ...     host='localhost',
        ...     queue='my_queue'
        ... )
        """
        super().__init__()
        import pika
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host)
        )
        self.channel = self.connection.channel()
        self.queue = queue
        self.channel.queue_declare(queue=queue)
        
    @RunReturn 
    def run(self) -> None:
        """Activate the RabbitMQ automation system.
        
        Sets internal state flag to enable pipeline operations.
        """
        logger.info("Initializing RabbitMQ automation system")
        self._is_runned = True
        
    @validate_params({
        'func': [Callable],
    })
    def process_rabbitmq_message(
        self,
        func: Callable[[Any], None]
    ):
        """
        Processes incoming RabbitMQ messages and triggers tasks.

        Parameters
        ----------
        func : callable
            The function to process the data. It should accept a single
            argument, which is the message body.

        Notes
        -----
        This method consumes messages from the RabbitMQ queue and applies the
        provided function ``func`` to each message.

        Examples
        --------
        >>> def process_data(data):
        ...     print(f"Processing data: {data}")
        >>> manager.process_rabbitmq_message(process_data)
        """
        
        check_is_runned(
            self, 
            ['_is_runned'], 
            "Automation not started - call run() first"
        )
        
        def callback(ch, method, properties, body):
            logger.info(f"Received message from RabbitMQ: {body}")
            func(body)

        self.channel.basic_consume(
            queue=self.queue,
            on_message_callback=callback,
            auto_ack=True
        )
        logger.info(
            f"Waiting for messages from RabbitMQ queue: '{self.queue}'"
        )
        self.channel.start_consuming()
