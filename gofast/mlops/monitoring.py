# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Monitor model performance in production, track key metrics, 
and set alerts for performance degradation.

This module provides the performance of machine learning models
in production environments. It computes metrics, detects drift,
and integrates with external monitoring tools and alerting mechanisms.
"""

import time
import pickle
import smtplib
import threading
from numbers import Real,  Integral
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional
)

import numpy as np
from scipy.stats import ( 
    ks_2samp, 
    chi2_contingency
)
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
from ..compat.sklearn import( 
    validate_params, 
    Interval, 
    StrOptions
)
from ..core.handlers import columns_manager 
from ..utils.deps_utils import ensure_pkg
from ..utils.validator import (
    check_is_runned, 
    check_is_fitted, 
)
from ..api.property import BaseClass
from .._gofastlog import gofastlog
from ._config import INSTALL_DEPENDENCIES, USE_CONDA 

logger = gofastlog.get_gofast_logger(__name__)

__all__=[
    "ModelPerformanceMonitor", 
    "ModelHealthChecker", 
    "DataDriftMonitor",
    "AlertManager",
    "LatencyTracker", 
    "ErrorRateMonitor",
    "CustomMetricsLogger",
    ]

class ModelPerformanceMonitor(BaseClass):
    r"""
    Monitors model performance in production by tracking various
    metrics over a sliding window and detecting data drift or
    performance degradations.

    This class computes performance metrics such as
    :math:`\text{accuracy}`, :math:`\text{precision}`, and others
    over a recent window of predictions, allowing real-time
    evaluation of model performance. It also supports optional drift
    detection using statistical tests to identify changes in data
    distributions. When performance metrics drop below specified
    thresholds, alerts can be triggered to notify stakeholders
    promptly.

    .. math::

        \text{Performance}(t) = \frac{1}{n_t}
        \sum_{i=1}^{n_t} M(\hat{y}_i, y_i),

    where :math:`M(\hat{y}_i, y_i)` represents a selected performance
    metric function (e.g. :math:`\mathbf{1}(\hat{y}_i = y_i)` for
    accuracy), and :math:`n_t` denotes the number of samples in the
    current window at time :math:`t`.

    Parameters
    ----------
    metrics : list of str, optional
        A list of performance metrics to compute and track. Common
        values include ``'accuracy'``, ``'precision'``,
        ``'recall'``, ``'f1'``, etc. The metrics are updated with
        each new batch of predictions.

    drift_detection : bool, default=True
        Whether to enable drift detection using methods like the
        Kolmogorov-Smirnov test. If set to ``True``, the class checks
        for distribution changes in labels or predictions and raises
        alerts if drift is suspected.

    alert_thresholds : dict of str to float, optional
        Threshold values for the monitored metrics. Keys should be
        metric names (e.g. ``'accuracy'``), and values are numeric
        thresholds. If a metric's value drops below its threshold,
        an alert is triggered.

    monitoring_tools : list of str, optional
        A list of external monitoring systems (e.g. ``['prometheus']``)
        for which the class provides built-in integration. Metrics are
        published to these tools for external visualization and alerting
        workflows.

    window_size : int, default=100
        The size of the sliding window of recent samples used to
        compute metrics. Older samples beyond the window are discarded
        to ensure up-to-date performance tracking.

    verbose : int, default=0
        The verbosity level for logging:
          - ``0``: Only critical errors are logged.
          - ``1``: Logs warnings and errors.
          - ``2``: Logs informational messages, warnings, and errors.
          - ``3``: Logs debug messages, info, warnings, and errors.

    Attributes
    ----------
    performance_history_ : dict of str to list of float
        Stores historical performance values for each metric. This
        attribute updates over time, maintaining a time-series of
        metric values.

    drift_status_ : dict
        Maintains the most recent drift detection results. It may
        contain keys such as ``'ks_statistic'``, ``'p_value'``, and
        ``'drift_detected'`` to reflect the outcome of drift checks.

    Methods
    -------
    run(**run_kw)
        Prepares the monitor for usage in a production or streaming
        environment.

    update(y_true, y_pred)
        Updates the sliding window with a new batch of ground-truth
        labels and predictions, then recalculates metrics and checks
        for alerts/drift.

    get_performance_history()
        Retrieves the recorded metric values over time.

    get_drift_status()
        Returns the most recent drift detection results, indicating
        whether data drift is suspected.

    reset_monitor()
        Clears all stored performance data and drift status, allowing
        the monitor to start fresh.

    set_thresholds(alert_thresholds)
        Dynamically updates or sets thresholds for performance metrics,
        enabling flexible alert configurations.

    save_state(filepath)
        Serializes and saves the internal state (metrics, drift info,
        etc.) to a file for later restoration.

    load_state(filepath)
        Loads a previously saved state from a file, restoring
        performance metrics and drift detection status.

    Notes
    -----
    - When drift detection is enabled, a statistical test (e.g.,
      Kolmogorov-Smirnov) is applied to the distribution of labels
      or predictions. If the test indicates a significant difference
      from past data, drift is flagged [1]_.
    - Alert thresholds can be configured separately for each metric.
      If a metric's current value is below the specified threshold,
      an alert is raised, allowing quick responses to performance
      regressions.

    Examples
    --------
    >>> from gofast.mlops.monitoring import ModelPerformanceMonitor
    >>> monitor = ModelPerformanceMonitor(
    ...     metrics=['accuracy', 'f1'],
    ...     window_size=50,
    ...     drift_detection=True
    ... )
    >>> monitor.run()
    >>> monitor.update(
    ...     y_true=[0, 1, 1],
    ...     y_pred=[0, 1, 1]
    ... )
    >>> print(monitor.get_performance_history())

    See Also
    --------
    DataDriftMonitor : Detects data drift using various statistical tests.
    ErrorRateMonitor : Tracks error rate of predictions over time.

    References
    ----------
    .. [1] Gama, J. et al. *A survey on concept drift adaptation*.
           ACM Computing Surveys, 46(4), 1-37.
    """

    @validate_params(
        {
            'metrics': ['array-like'],
            'drift_detection': [bool],
            'alert_thresholds': [dict, None],
            'monitoring_tools': ['array-like', None],
            'window_size': [Interval(Integral, 1, None, closed='left')],
            'verbose': [int]
        }
    )
    def __init__(
        self,
        metrics: List[str] = ['accuracy'],
        drift_detection: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
        monitoring_tools: Optional[List[str]] = None,
        window_size: int = 100,
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)

        self.metrics = columns_manager(metrics)
        self.drift_detection = drift_detection
        self.alert_thresholds = alert_thresholds or {}
        self.monitoring_tools = monitoring_tools or []
        self.window_size = window_size

        self.performance_history_: Dict[str, List[float]] = {}
        self.drift_status_: Dict[str, Any] = {}

        self._labels_window_ = []
        self._preds_window_ = []

        self._selected_metrics_: Dict[str, Callable] = {}
        self._metric_functions_: Dict[str, Callable] = {}
        self._prometheus_metrics_: Dict[str, Any] = {}
        self._alert_configs_: Dict[str, Any] = {}
        self._email_config_: Dict[str, Any] = {}
        self._slack_config_: Dict[str, Any] = {}
        self._sms_config_: Dict[str, Any] = {}

        self._is_runned = False

        # Initialize sub-components
        self._initialize_monitoring_tools()
        self._init_performance_metrics()
        self._init_alerting()

        # Initialize drift detection if enabled
        if self.drift_detection:
            self._init_drift_detection()

    def run(
        self,
        model: Optional[Any]=None,
        **run_kw
    ) -> "ModelPerformanceMonitor":
        """
        Prepare the monitor for usage with a given model in a
        production or streaming environment.

        This method sets the internal state indicating that the
        monitor is ready to receive updates. The `model` parameter
        can be stored or validated as needed.

        Parameters
        ----------
        model : Any
            The model object to be monitored. May be used in future
            extended scenarios (e.g., predictions, metadata, etc.).
            For now it does nothing and user can bypass setting the 
            model. 
        **run_kw
            Additional parameters for run configuration.

        Returns
        -------
        self : ModelPerformanceMonitor
            Returns the instance after setting up the run state.

        Notes
        -----
        This method sets an internal flag `_is_runned` to True,
        indicating that subsequent operations can proceed.
        """
        self._model_ = model  # could be used in advanced usage
        self._is_runned = True

        if self.verbose > 1:
            logger.info(
                "ModelPerformanceMonitor is now set to run mode."
            )
        return self

    def _initialize_monitoring_tools(self) -> None:
        """
        Initialize connections or clients to external monitoring
        tools (e.g., Prometheus).
        """
        for tool in self.monitoring_tools:
            if tool.lower() == 'prometheus':
                self._init_prometheus_client()
            else:
                msg = (
                    f"Unsupported monitoring tool: {tool}."
                )
                raise ValueError(msg)

    def _init_prometheus_client(self) -> None:
        """
        Initialize Prometheus client, creating gauge metrics for
        each performance metric being monitored.
        """
        @ensure_pkg(
            "prometheus_client",
            extra=(
                "To use Prometheus integration, please install "
                "'prometheus_client'."
            ),
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA
        )
        def init_client():
            import prometheus_client
            for metric in self.metrics:
                gauge_name = f'model_{metric}'
                desc = f'Model {metric} over time'
                self._prometheus_metrics_[metric] = (
                    prometheus_client.Gauge(gauge_name, desc)
                )
        init_client()

    def _init_performance_metrics(self) -> None:
        """
        Set up supported metric functions and validate user-specified
        metrics.
        """
        self._metric_functions_ = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            'log_loss': log_loss
        }

        for metric in self.metrics:
            if metric in self._metric_functions_:
                self._selected_metrics_[metric] = (
                    self._metric_functions_[metric]
                )
            else:
                msg = (
                    f"Unsupported metric: {metric}. "
                    "Supported metrics are "
                    f"{list(self._metric_functions_.keys())}."
                )
                raise ValueError(msg)

    def _init_alerting(self) -> None:
        """
        Prepare configurations for different alerting channels 
        (email, slack, sms).
        """
        # Assume the user might set these in the instance 
        self._email_config_ = getattr(
            self, '_email_config_', {}
        )
        if (
            self._email_config_.get('enabled', False)
            and self.verbose > 1
        ):
            logger.info("Email alerting enabled.")
        else:
            if self.verbose > 1:
                logger.info("Email alerting is disabled or "
                            "not configured.")

        self._slack_config_ = getattr(
            self, '_slack_config_', {}
        )
        if (
            self._slack_config_.get('enabled', False)
            and self.verbose > 1
        ):
            logger.info("Slack alerting enabled.")
        else:
            if self.verbose > 1:
                logger.info("Slack alerting is disabled or "
                            "not configured.")

        self._sms_config_ = getattr(
            self, '_sms_config_', {}
        )
        if (
            self._sms_config_.get('enabled', False)
            and self.verbose > 1
        ):
            logger.info("SMS alerting enabled.")
        else:
            if self.verbose > 1:
                logger.info("SMS alerting is disabled or "
                            "not configured.")

    def _init_drift_detection(self) -> None:
        """
        Set up drift detection (e.g., initialize statistical tests).
        """
        # Currently only sets up KS test
        def init_drift():
            self._ks_test_ = ks_2samp
        init_drift()

    def update(
        self,
        y_true: List[Any],
        y_pred: List[Any]
    ) -> "ModelPerformanceMonitor":
        """
        Update the monitoring metrics with a new batch of data.

        Parameters
        ----------
        y_true : List[Any]
            True labels or ground truth for the batch.

        y_pred : List[Any]
            Predicted labels from the model.

        Returns
        -------
        self : ModelPerformanceMonitor
            Returns the instance after updating internal state.

        Notes
        -----
        This method updates the internal sliding windows with the
        new data, computes the selected performance metrics,
        checks for alerts, and updates any integrated monitoring
        tools (e.g., Prometheus). If drift detection is enabled,
        it also performs relevant statistical tests.

        Raises
        ------
        ValueError
            If a metric computation fails due to incompatible 
            shapes or invalid values.

        Examples
        --------
        >>> monitor.update(
        ...     y_true=[0, 1, 1], 
        ...     y_pred=[0, 0, 1]
        ... )
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg="Please call 'run' before updating metrics."
        )
        # Add new data to the sliding windows
        self._labels_window_.extend(y_true)
        self._preds_window_.extend(y_pred)

        # Keep only the last `window_size` items
        self._labels_window_ = (
            self._labels_window_[-self.window_size:]
        )
        self._preds_window_ = (
            self._preds_window_[-self.window_size:]
        )

        # Compute each metric
        current_metrics = {}
        for m_name, m_func in self._selected_metrics_.items():
            try:
                # Weighted average needed if metric is 
                # precision, recall or f1
                if m_name in ['precision', 'recall', 'f1']:
                    val = m_func(
                        self._labels_window_,
                        self._preds_window_,
                        average='weighted'
                    )
                else:
                    val = m_func(
                        self._labels_window_,
                        self._preds_window_
                    )
            except ValueError as e:
                logger.error(
                    f"Error calculating {m_name}: {e}"
                )
                val = float('nan')

            current_metrics[m_name] = val
            # Record in performance history
            self.performance_history_.setdefault(
                m_name, []
            ).append(val)

            # Alert checks
            threshold = self.alert_thresholds.get(m_name)
            if threshold is not None and val < threshold:
                self._trigger_alert(m_name, val)

            # Update external monitoring (Prometheus)
            if 'prometheus' in self.monitoring_tools:
                self._prometheus_metrics_[m_name].set(val)

        # If drift detection is enabled, run it
        if self.drift_detection:
            self._detect_drift()

        return self

    def _detect_drift(self) -> None:
        """
        Perform drift detection using statistical tests 
        over the recent data window.
        """
        # If not enough data, skip
        if (
            len(self._labels_window_) < 2
            or len(self._preds_window_) < 2
        ):
            return

        stat, pval = self._ks_test_(
            self._labels_window_,
            self._preds_window_
        )
        self.drift_status_['ks_statistic'] = stat
        self.drift_status_['p_value'] = pval

        if pval < 0.05:
            self.drift_status_['drift_detected'] = True
            logger.warning(
                f"Drift detected (p={pval:.4f})."
            )
        else:
            self.drift_status_['drift_detected'] = False

    def get_performance_history(
        self
    ) -> Dict[str, List[float]]:
        """
        Retrieve the tracked performance metrics over time.

        Returns
        -------
        performance_history_ : dict of str to list of float
            A dictionary keyed by metric names, with each value 
            being a list of metric values in chronological order.

        Examples
        --------
        >>> history = monitor.get_performance_history()
        >>> print(history)
        """
        return self.performance_history_

    def get_drift_status(
        self
    ) -> Dict[str, Any]:
        """
        Get the current drift detection status.

        Returns
        -------
        drift_status_ : dict
            Dictionary containing the last drift detection 
            statistic, p-value, and a boolean indicating 
            whether drift was detected.
        """
        return self.drift_status_

    def reset_monitor(self) -> None:
        """
        Reset the monitoring state, clearing all tracked
        metrics, drift status, and data windows.
        """
        self.performance_history_.clear()
        self.drift_status_.clear()
        self._labels_window_.clear()
        self._preds_window_.clear()
        if self.verbose > 1:
            logger.info("Monitoring state has been reset.")

    def set_thresholds(
        self, 
        alert_thresholds: Dict[str, float]
    ) -> None:
        """
        Set or update custom thresholds for performance alerts.

        Parameters
        ----------
        alert_thresholds : dict of str to float
            Dictionary keyed by metric names with float 
            threshold values. If a metric's value falls 
            below its threshold, an alert will be triggered.
        """
        self.alert_thresholds.update(alert_thresholds)
        if self.verbose > 1:
            logger.info("Alert thresholds have been updated.")

    def save_state(
        self,
        filepath: str
    ) -> None:
        """
        Save the current monitoring state to a file using 
        pickle serialization.

        Parameters
        ----------
        filepath : str
            Path to the file where the state is saved.

        Examples
        --------
        >>> monitor.save_state('monitor_state.pkl')
        """
        def save():
            state = {
                'performance_history_': self.performance_history_,
                'drift_status_': self.drift_status_,
                '_labels_window_': self._labels_window_,
                '_preds_window_': self._preds_window_
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            if self.verbose > 1:
                logger.info(f"State saved to {filepath}.")
        save()

    def load_state(
        self,
        filepath: str
    ) -> None:
        """
        Load a previously saved monitoring state from a file.

        Parameters
        ----------
        filepath : str
            Path to the pickle file containing the saved state.

        Examples
        --------
        >>> monitor.load_state('monitor_state.pkl')
        """
        def load():
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.performance_history_ = (
                state.get('performance_history_', {})
            )
            self.drift_status_ = (
                state.get('drift_status_', {})
            )
            self._labels_window_ = (
                state.get('_labels_window_', [])
            )
            self._preds_window_ = (
                state.get('_preds_window_', [])
            )
            if self.verbose > 1:
                logger.info(f"State loaded from {filepath}.")
        load()

    def _trigger_alert(
        self,
        metric_name: str,
        value: float
    ) -> None:
        """
        Trigger an alert indicating performance degradation.

        Parameters
        ----------
        metric_name : str
            Name of the metric that triggered the alert.

        value : float
            Current metric value that fell below the threshold.
        """
        alert_msg = (
            f"Performance alert: {metric_name} "
            f"={value:.4f} below threshold."
        )
        logger.warning(alert_msg)
        # Send email alert if email configuration is provided
        if 'email' in self._email_config_:
            self._send_email_alert(alert_msg, self._email_config_['email'])

    def _send_email_alert(
        self,
        alert_message: str,
        email_config: Dict[str, Any]
    ) -> None:
        """
        Send an email alert using the specified configuration.
    
        Parameters
        ----------
        alert_message : str
            The content of the alert message.
        email_config : dict
            Email settings, e.g.:
            {
                'smtp_server': ...,
                'smtp_port': ...,
                'smtp_username': ...,
                'smtp_password': ...,
                'sender_email': ...,
                'receiver_email': ...
            }
    
        Notes
        -----
        This method requires valid SMTP credentials.
        """
        smtp_server = email_config.get('smtp_server')
        smtp_port = email_config.get('smtp_port', 587)
        smtp_username = email_config.get('smtp_username')
        smtp_password = email_config.get('smtp_password')
        sender_email = email_config.get('sender_email')
        receiver_email = email_config.get('receiver_email')
    
        # Ensure all required config fields exist
        if not all([
            smtp_server, smtp_username, smtp_password,
            sender_email, receiver_email
        ]):
            logger.error("Incomplete email configuration.")
            return
    
        # Create and send the email
        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = "Model Performance Alert"
            msg.attach(MIMEText(alert_message, 'plain'))
    
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
    
            if self.verbose > 1:
                logger.info("Email alert sent successfully.")
        except Exception as exc:
            logger.exception(
                f"Failed to send email alert: {exc}"
            )

class ModelHealthChecker(BaseClass):
    """
    Monitors the operational health of a system running a machine
    learning model, measuring resource usage such as CPU, memory,
    disk, GPU, network bandwidth, and inference latency.

    This class periodically collects system resource metrics and can
    raise alerts if usage exceeds user-defined thresholds. By
    tracking various health indicators, it helps prevent performance
    bottlenecks or outages in production environments. Users can
    customize missing metrics (e.g., no GPU present) or disable
    specific checks.

    .. math::

        H(t) = \bigl(\text{cpu}_t,\,
                     \text{memory}_t,\,
                     \text{disk}_t,\,
                     \text{gpu}_t,\,
                     \text{network}_t,\,
                     \text{latency}_t \bigr),

    where each component (e.g. :math:`\text{cpu}_t`) represents the
    resource usage at time :math:`t`. If any usage surpasses its
    threshold, an alert is triggered.

    Parameters
    ----------
    alert_callback : callable or None, default=None
        A user-defined function that is invoked when a monitored
        resource crosses its threshold. It should accept two
        parameters: the resource name (e.g. ``'cpu'``) and a
        dictionary with additional details (e.g., the usage
        percentage).

    cpu_threshold : float, default=80.0
        Maximum allowable CPU usage (in percent). If CPU usage
        exceeds this value, an alert is raised. Must lie within
        [0, 100].

    memory_threshold : float, default=80.0
        Maximum allowable system memory usage (in percent). If the
        usage surpasses this threshold, an alert is triggered.
        Must lie within [0, 100].

    disk_threshold : float, default=80.0
        Limit for disk usage (in percent). If the usage on the root
        filesystem (or configured partition) exceeds this value, an
        alert is generated. Must lie within [0, 100].

    gpu_threshold : float, default=80.0
        Upper bound for GPU usage (in percent). If usage is above
        this threshold, an alert is raised. If no GPU is detected,
        usage remains 0. Must lie within [0, 100].

    network_threshold : float, default=80.0
        Bandwidth usage limit in Mbps. If usage surpasses this value,
        an alert is raised. Must be non-negative.

    latency_threshold : float, default=2.0
        Maximum acceptable inference latency (in seconds). If the
        recorded latency for an operation exceeds this threshold,
        it triggers an alert. Must be non-negative.

    alert_messages : dict of str to str, optional
        Custom messages for specific resource alerts. For example,
        a key of ``'cpu'`` can map to a string describing a CPU
        usage alert. If not provided, default messages are used.

    health_retention_period : int, default=100
        Size of the sliding window for storing historical resource
        metrics. Must be a positive integer. Allows analysis of
        trends and short-term fluctuations.

    monitor_interval : int, default=60
        Interval (in seconds) between resource checks. Must be a
        positive integer. When the health checker is running in a
        background thread, each check occurs after ``monitor_interval``
        seconds.

    verbose : int, default=0
        Verbosity level for logging:
          - ``0``: Logs only critical errors.
          - ``1``: Logs warnings and errors.
          - ``2``: Logs info, warnings, and errors.
          - ``3``: Logs debug info, as well as everything above.

    Attributes
    ----------
    latency_history_ : list of float
        Retains the latencies recorded for inference operations,
        respecting the `health_retention_period`.

    health_history_ : dict of str to list of float
        Tracks CPU, memory, disk, GPU, and network usage statistics
        over recent intervals. Each key (e.g. ``'cpu'``) maps to a
        list of usage values.

    Methods
    -------
    run(**run_kw)
        Activates health monitoring. If invoked, repeated checks
        are done in the background or by manual calls to
        `check_health`.

    check_health()
        Collects CPU, memory, disk, GPU, and network usage, then
        triggers alerts if thresholds are exceeded.

    record_latency(latency)
        Logs inference latency. If above `latency_threshold`,
        an alert is raised.

    get_health_history(metric)
        Retrieves usage history for a specified metric (e.g.
        ``'cpu'``).

    Notes
    -----
    - When GPU monitoring is enabled, :math:`\text{gpu}_t` is computed
      from the highest usage across all available GPUs. If no GPU is
      found, usage is set to 0 [1]_.
    - Network usage in Mbps is estimated by measuring bytes sent and
      received over a 1-second interval and converting to megabits.

    Examples
    --------
    >>> from gofast.mlops.monitoring import ModelHealthChecker
    >>> def my_alert_callback(metric, details):
    ...     print(f"ALERT: {metric} usage high - {details}")
    >>> health_checker = ModelHealthChecker(
    ...     alert_callback=my_alert_callback,
    ...     cpu_threshold=75.0,
    ...     memory_threshold=85.0
    ... )
    >>> health_checker.run()
    >>> health_checker.check_health()

    See Also
    --------
    ErrorRateMonitor : Monitors model prediction failures.
    LatencyTracker : Tracks the latency of inference operations.

    References
    ----------
    .. [1] "GPUtil Documentation",
           https://github.com/anderskm/gputil
    """

    @validate_params(
        {
            'alert_callback': [callable, None],
            'cpu_threshold': [
                Interval(Real, 0, 100, closed='both')
            ],
            'memory_threshold': [
                Interval(Real, 0, 100, closed='both')
            ],
            'disk_threshold': [
                Interval(Real, 0, 100, closed='both')
            ],
            'gpu_threshold': [
                Interval(Real, 0, 100, closed='both')
            ],
            'network_threshold': [
                Interval(Real, 0, None, closed='left')
            ],
            'latency_threshold': [
                Interval(Real, 0, None, closed='left')
            ],
            'alert_messages': [dict, None],
            'health_retention_period': [
                Interval(Integral, 1, None, closed='left')
            ],
            'monitor_interval': [
                Interval(Integral, 1, None, closed='left')
            ],
            'verbose': [Integral]
        }
    )
    def __init__(
        self,
        alert_callback: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 80.0,
        gpu_threshold: float = 80.0,
        network_threshold: float = 80.0,
        latency_threshold: float = 2.0,
        alert_messages: Optional[
            Dict[str, str]
        ] = None,
        health_retention_period: int = 100,
        monitor_interval: int = 60,
        verbose: int = 0
    ):
    
        super().__init__(verbose=verbose)

        self.alert_callback = alert_callback
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.gpu_threshold = gpu_threshold
        self.network_threshold = network_threshold
        self.latency_threshold = latency_threshold
        self.alert_messages = alert_messages or {}
        self.health_retention_period = (
            health_retention_period
        )
        self.monitor_interval = monitor_interval

        self._latency_history_ = []
        self._health_history_ = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'gpu': [],
            'network': []
        }
        # Indicates whether 'run' was called
        self._is_runned = False

    def run(
        self,
        **run_kw
    ) -> "ModelHealthChecker":
        """
        Prepare the health checker to begin monitoring.

        Parameters
        ----------
        **run_kw : dict
            Additional run parameters, if needed.

        Returns
        -------
        self : ModelHealthChecker
            Returns the instance after enabling run mode.
        """
        self._is_runned = True
        if self.verbose > 1:
            logger.info("Health checker is now active.")
        return self

    @ensure_pkg(
        "psutil",
        extra=(
            "The 'psutil' package is required for system monitoring,"
            " including CPU, memory, and process management."
        ),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def check_health(
        self
    ) -> None:
        """
        Monitors CPU, memory, disk, GPU, and network usage,
        triggering alerts if usage exceeds thresholds.
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg="Call 'run' before using 'check_health'."
        )
        import psutil

        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        self._log_health_metric('cpu', cpu_usage)
        self._check_and_alert('cpu', cpu_usage,
                              self.cpu_threshold)

        # Memory usage
        memory_usage = self._get_memory_usage()
        self._log_health_metric('memory', memory_usage)
        self._check_and_alert(
            'memory', memory_usage,
            self.memory_threshold
        )

        # Disk usage
        disk_usage = self._get_disk_usage()
        self._log_health_metric('disk', disk_usage)
        self._check_and_alert(
            'disk', disk_usage,
            self.disk_threshold
        )

        # GPU usage
        gpu_usage = self._get_gpu_usage()
        self._log_health_metric('gpu', gpu_usage)
        self._check_and_alert(
            'gpu', gpu_usage,
            self.gpu_threshold
        )

        # Network usage
        network_usage = self._get_network_usage()
        self._log_health_metric(
            'network',
            network_usage
        )
        self._check_and_alert(
            'network',
            network_usage,
            self.network_threshold
        )

    def record_latency(
        self,
        latency: float
    ) -> None:
        """
        Record the inference latency and trigger an alert
        if it exceeds the threshold.

        Parameters
        ----------
        latency : float
            Inference latency in seconds, must be >= 0.
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg="Call 'run' before recording latency."
        )
        if latency < 0:
            raise ValueError(
                "Latency must be non-negative."
            )

        self._latency_history_.append(latency)
        if (
            len(self._latency_history_)
            > self.health_retention_period
        ):
            self._latency_history_.pop(0)

        if latency > self.latency_threshold:
            default_msg = (
                f"Inference latency high: "
                f"{latency:.2f}s"
            )
            message = self.alert_messages.get(
                'latency', default_msg
            )
            self._trigger_alert(
                'latency',
                latency,
                message
            )

    def get_health_history(
        self,
        metric: str
    ) -> List[float]:
        """
        Return historical values for a specific metric.

        Parameters
        ----------
        metric : {'cpu','memory','disk',
                  'gpu','network'}
            The metric name to retrieve.

        Returns
        -------
        list of float
            Historical values of the metric.
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg=(
                "Call 'run' before retrieving "
                "health history."
            )
        )
        if metric not in self._health_history_:
            raise ValueError(
                f"Unknown metric '{metric}'."
            )
        return self._health_history_[metric]

    def _get_memory_usage(
        self
    ) -> float:
        """
        Get current memory usage as a percentage.
        """
        import psutil
        mem_info = psutil.virtual_memory()
        return mem_info.percent

    def _get_disk_usage(
        self
    ) -> float:
        """
        Get current disk usage (root '/') as a percentage.
        """
        import psutil
        disk_info = psutil.disk_usage('/')
        return disk_info.percent

    @ensure_pkg(
        "GPUtil",
        extra="GPUtil is required for GPU monitoring.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _get_gpu_usage(
        self
    ) -> float:
        """
        Get GPU usage as a percentage. If none,
        return 0.0.
        """
        import GPUtil
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 0.0
        # Max load among all GPUs
        return max([g.load * 100 for g in gpus])

    def _get_network_usage(
        self
    ) -> float:
        """
        Calculate network bandwidth usage in Mbps.
        """
        import psutil
        net_io_1 = psutil.net_io_counters()
        sent_1 = net_io_1.bytes_sent
        recv_1 = net_io_1.bytes_recv
        time.sleep(1)
        net_io_2 = psutil.net_io_counters()
        sent_2 = net_io_2.bytes_sent
        recv_2 = net_io_2.bytes_recv

        total_bytes = (
            (sent_2 - sent_1)
            + (recv_2 - recv_1)
        )
        total_bits = total_bytes * 8
        mbps = total_bits / 1e6
        return mbps

    def _check_and_alert(
        self,
        metric: str,
        usage: float,
        threshold: float
    ) -> None:
        """
        Compare usage to threshold; trigger alert
        if above it.
        """
        if usage > threshold:
            default_msg = (
                f"{metric.capitalize()} usage too "
                f"high: {usage:.2f}%"
            )
            message = self.alert_messages.get(
                metric, default_msg
            )
            self._trigger_alert(
                metric,
                usage,
                message
            )

    def _trigger_alert(
        self,
        metric: str,
        value: float,
        message: str
    ) -> None:
        """
        Trigger an alert with metric details and
        optional callback.
        """
        logger.warning(message)
        if self.alert_callback:
            self.alert_callback(
                metric,
                {
                    'value': value,
                    'message': message
                }
            )

    def _log_health_metric(
        self,
        metric: str,
        value: float
    ) -> None:
        """
        Log and retain health metric values within
        the retention limit.
        """
        self._health_history_[metric].append(value)
        if (
            len(self._health_history_[metric])
            > self.health_retention_period
        ):
            self._health_history_[metric].pop(0)
            
class DataDriftMonitor(BaseClass):
    """
    Detects data drift by comparing the distribution of new data
    against a baseline distribution, issuing alerts if significant
    divergence is found.

    The :class:`DataDriftMonitor` uses statistical tests (e.g.
    Kolmogorov-Smirnov, Chi-squared, Jensen-Shannon) to measure how
    incoming data deviates from baseline data. If the drift measure
    crosses a threshold, the class triggers alerts, allowing early
    interventions when data distribution changes could degrade model
    performance.

    .. math::

        \\text{Drift}(D_{\\text{baseline}}, D_{\\text{new}}) = 
        \\begin{cases}
            \\text{p-value} < \\text{threshold}, & 
            \\text{for 'ks' or 'chi2'} \\\\
            \\text{divergence} > \\text{threshold}, & 
            \\text{for 'jsd'}
        \\end{cases}

    Parameters
    ----------
    alert_callback : callable or None, optional
        A function invoked when drift is detected. It should accept
        two parameters:
        `<issue_type>` (str, e.g. ``'data_drift'``) and a dictionary
        with details. Defaults to ``None``.

    drift_thresholds : dict of str to float, optional
        A dictionary mapping each feature name to a drift threshold.
        If not specified, `<drift_threshold>` is used globally.
        Example: ``{'feature_0': 0.05, 'feature_1': 0.01}``.

    drift_threshold : float, optional
        A global threshold for drift detection. For ``'ks'`` or
        ``'chi2'``, if the p-value is below this threshold,
        drift is flagged. For ``'jsd'``, drift is flagged if
        Jensen-Shannon divergence is above this. Must lie in
        [0, 1]. Defaults to 0.05.

    drift_detection_method : {'ks', 'chi2', 'jsd'}, optional
        The statistical test for detecting drift. The default
        ``'ks'`` uses Kolmogorov-Smirnov. Set to ``'chi2'`` for
        categorical data, or ``'jsd'`` for measuring divergence
        between distributions.

    handle_missing : {'skip', 'impute'}, optional
        Strategy for missing data:
          - ``'skip'``: Omit missing values
          - ``'impute'``: Replace missing values with the mean

        Defaults to ``'skip'``.

    alert_messages : dict of str to str, optional
        Custom alert messages for drift events. For instance,
        a key ``'drift'`` can contain a message displayed upon
        detecting drift. If ``None``, default messages are used.

    verbose : int, optional
        Logging verbosity:
          - ``0``: Log critical errors only
          - ``1``: Log warnings + errors
          - ``2``: Log info + warnings + errors
          - ``3``: Log debug + info + warnings + errors
        Defaults to 0.

    Attributes
    ----------
    drift_history_ : list of dict
        Stores records of all drift checks performed. Each entry
        includes a dictionary of feature-specific p-values (or
        divergence scores) and whether drift was detected.

    Methods
    -------
    fit(baseline_data)
        Configures the monitor with baseline data. Required before
        calling `<monitor_drift>`.

    monitor_drift(new_data)
        Compares `new_data` to the baseline distribution, triggering
        an alert if drift is detected.

    set_baseline_data(baseline_data)
        Updates the baseline distribution after initial fitting.

    get_drift_history()
        Returns the collection of past drift detection results.

    Notes
    -----
    Drift detection is crucial for maintaining model reliability in
    dynamic environments [1]_. If the data used for training no
    longer represents current conditions, performance may degrade.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.mlops.monitoring import DataDriftMonitor
    >>> baseline = np.random.normal(0, 1, (1000, 3))
    >>> new_batch = np.random.normal(0.5, 1, (1000, 3))
    >>> monitor = DataDriftMonitor(
    ...     drift_detection_method='ks',
    ...     drift_threshold=0.05
    ... )
    >>> monitor.fit(baseline)
    >>> monitor.monitor_drift(new_batch)

    See Also
    --------
    ModelPerformanceMonitor : Monitors model metrics and drift.
    ErrorRateMonitor : Checks for rising prediction error rates.

    References
    ----------
    .. [1] Gama, J. et al. "A survey on concept drift adaptation".
           ACM Computing Surveys, 46(4), 1-37.
    """

    @validate_params(
        {
            'alert_callback': [callable, None],
            'drift_thresholds': [dict, None],
            'drift_threshold': [
                Interval(Real, 0, 1, closed='both')
            ],
            'drift_detection_method': [
                StrOptions({'ks', 'chi2', 'jsd'})
            ],
            'handle_missing': [
                StrOptions({'skip', 'impute'})
            ],
            'alert_messages': [dict, None],
            'verbose': [int]
        }
    )
    def __init__(
        self,
        alert_callback: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
        drift_thresholds: Optional[
            Dict[str, float]
        ] = None,
        drift_threshold: float = 0.05,
        drift_detection_method: str = 'ks',
        handle_missing: str = 'skip',
        alert_messages: Optional[
            Dict[str, str]
        ] = None,
        verbose: int = 0
    ):
        """
        Initialize DataDriftMonitor with user-defined or
        default settings.
        """

        super().__init__(verbose=verbose)

        self.alert_callback = alert_callback
        self.drift_thresholds = drift_thresholds or {}
        self.drift_threshold = drift_threshold
        self.drift_detection_method = drift_detection_method
        self.handle_missing = handle_missing
        self.alert_messages = alert_messages or {}

        self._is_fitted = False
        self._baseline_data_ = None
        self._drift_history_ = []

    def fit(
        self,
        baseline_data: np.ndarray
    ) -> "DataDriftMonitor":
        """
        Fit the monitor with baseline data for drift detection.

        Parameters
        ----------
        baseline_data : np.ndarray of shape (n_samples, n_features)
            Baseline data distribution to compare against.

        Returns
        -------
        self : DataDriftMonitor
            Returns the instance after fitting.
        """
        if baseline_data.ndim != 2:
            raise ValueError(
                "`baseline_data` must be a 2D array."
            )
        # Store baseline data internally
        self._baseline_data_ = baseline_data
        self._is_fitted = True

        if self.verbose > 1:
            logger.info(
                "DataDriftMonitor has been fitted with "
                "baseline data of shape "
                f"{baseline_data.shape}."
            )
        return self

    def monitor_drift(
        self,
        new_data: np.ndarray
    ) -> None:
        """
        Compare incoming `new_data` to the baseline and
        trigger alerts if drift is detected.
        """
        check_is_fitted(
            self, 
            attributes=["_is_fitted"],
            msg="Call 'fit' before monitoring data drift."
        )
        if self._baseline_data_.shape[1] != new_data.shape[1]:
            raise ValueError(
                "Number of features in `new_data` must "
                "match the baseline."
            )

        p_values = {}
        drift_detected = False
        num_features = self._baseline_data_.shape[1]

        for i in range(num_features):
            feature_name = f"feature_{i}"

            # Handle missing values
            baseline_col = self._handle_missing_data_(
                self._baseline_data_[:, i]
            )
            new_col = self._handle_missing_data_(
                new_data[:, i]
            )

            if self.drift_detection_method == 'ks':
                stat, pval = ks_2samp(
                    baseline_col,
                    new_col
                )
                # For KS, a low p-value => difference
                drift = pval < self._get_threshold_(
                    feature_name
                )
            elif self.drift_detection_method == 'chi2':
                # Bin data for chi-squared
                b_hist, b_edges = np.histogram(
                    baseline_col,
                    bins='auto'
                )
                n_hist, _ = np.histogram(
                    new_col,
                    bins=b_edges
                )
                cont_table = np.array([
                    b_hist, n_hist
                ])
                stat, pval = chi2_contingency(
                    cont_table
                )[:2]
                drift = pval < self._get_threshold_(
                    feature_name
                )
            elif self.drift_detection_method == 'jsd':
                # Jensen-Shannon => high => difference
                pval = self._jensen_shannon_(
                    baseline_col,
                    new_col
                )
                drift = (
                    pval > self._get_threshold_(
                        feature_name
                    )
                )
            else:
                raise ValueError(
                    f"Unknown method: "
                    f"{self.drift_detection_method}"
                )

            p_values[feature_name] = pval
            if drift:
                drift_detected = True

        # Add results to drift history
        result = {
            "p_values": p_values,
            "drift_detected": drift_detected
        }
        self._drift_history_.append(result)

        # Alert if drift
        if drift_detected:
            msg = self.alert_messages.get(
                'drift',
                f"Data drift detected: {p_values}"
            )
            if self.alert_callback:
                self.alert_callback(
                    'data_drift',
                    p_values
                )
            logger.warning(msg)

    def set_baseline_data(
        self,
        baseline_data: np.ndarray
    ) -> None:
        """
        Update the baseline data distribution after
        fitting the monitor.
        """
        check_is_fitted(
            self, 
            attributes=["_is_fitted"],
            msg="Call 'fit' before updating baseline data."
        )
        if baseline_data.ndim != 2:
            raise ValueError(
                "`baseline_data` must be 2D."
            )
        self._baseline_data_ = baseline_data

        if self.verbose > 1:
            logger.info(
                "Baseline data updated. New shape: "
                f"{baseline_data.shape}."
            )

    def get_drift_history(
        self
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the complete drift detection history.
        """
        check_is_fitted(
            self, 
            attributes=["_is_fitted"],
            msg="Call 'fit' before retrieving history."
        )
        return self._drift_history_

    def _handle_missing_data_(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Internal method to handle missing data
        ('skip' or 'impute').
        """
        if self.handle_missing == 'skip':
            return data[~np.isnan(data)]
        elif self.handle_missing == 'impute':
            mean_val = np.nanmean(data)
            return np.nan_to_num(
                data,
                nan=mean_val
            )
        else:
            raise ValueError(
                f"Unknown missing strategy: "
                f"{self.handle_missing}"
            )

    def _get_threshold_(
        self,
        feature_name: str
    ) -> float:
        """
        Return the per-feature threshold if present,
        else the global threshold.
        """
        return self.drift_thresholds.get(
            feature_name,
            self.drift_threshold
        )

    def _jensen_shannon_(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Compute Jensen-Shannon Divergence (JSD)
        between two 1D arrays.
        """
        # Create histograms for probability dists
        p_hist, b_edges = np.histogram(
            p,
            bins='auto',
            density=True
        )
        q_hist, _ = np.histogram(
            q,
            bins=b_edges,
            density=True
        )

        # Avoid zeros
        eps = 1e-10
        p_hist += eps
        q_hist += eps

        # Normalize
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()

        # JSD using base=2
        jsd = jensenshannon(
            p_hist,
            q_hist,
            base=2.0
        )
        return jsd


class LatencyTracker(BaseClass):
    """
    Tracks the latency of different operations (e.g. model
    inference), storing recent values for analysis and triggering
    alerts if thresholds are exceeded.

    The :class:`LatencyTracker` continuously logs latencies to
    identify slowdowns or performance bottlenecks. A threshold
    can be set for each operation, beyond which an alert is
    raised. This approach is crucial for meeting service-level
    agreements where prompt responses are required.

    .. math::

        \\text{LatencyStats}(op, t) =
        \\bigl\\{ L_1, L_2, ..., L_n \\bigr\\},

    where each :math:`L_i` is the latency (in seconds) for the
    operation :math:`op` over a sliding window of length :math:`n`.
    If any :math:`L_i` surpasses a user-defined limit, an alert is
    triggered.

    Parameters
    ----------
    alert_callback : callable or None, optional
        A function invoked when an operation's latency crosses
        its threshold. The callback is passed three parameters:
        `<operation>` (str), the exceeded latency (float), and
        a custom message. Defaults to ``None``.

    latency_thresholds : dict of str to float, optional
        A dictionary mapping operation names to their latency
        thresholds. If absent, `<global_latency_threshold>` is
        used for all operations.

    global_latency_threshold : float, optional
        Default maximum allowed latency in seconds for any
        operation not listed in `<latency_thresholds>`. Must be
        >= 0. Defaults to 2.0.

    retention_period : int, optional
        The number of most recent latencies retained per
        operation. Must be > 0. Defaults to 100.

    percentiles_to_track : list of float or None, optional
        A list of percentile values to record for each operation.
        For instance, ``[90.0, 95.0, 99.0]`` tracks 90th, 95th,
        and 99th percentile latencies. If ``None``, it defaults
        to ``[90.0, 95.0, 99.0]``.

    alert_messages : dict of str to str, optional
        Custom messages for latency alerts, keyed by operation
        name. If an operation's latency passes its threshold,
        the corresponding message is displayed. If missing,
        default messages are used.

    verbose : int, optional
        Controls logging verbosity:
          - ``0``: Only critical errors
          - ``1``: Warnings + errors
          - ``2``: Info + warnings + errors
          - ``3``: Debug + info + warnings + errors
        Defaults to 0.

    Attributes
    ----------
    latencies_ : dict of str to list of float
        Maintains recent latencies for each named operation.

    Methods
    -------
    run(**run_kw)
        Prepares the latency tracker for usage. After this, 
        methods like `<record_latency>` can be called safely.

    record_latency(operation, latency)
        Records a new latency value for `<operation>` and alerts
        if it exceeds the defined threshold.

    get_average_latency(operation)
        Returns the mean latency for `<operation>`, or NaN if
        no data exists.

    get_tail_latency(operation, percentile=95.0)
        Retrieves the specified percentile latency (e.g. 95th).

    get_latency_distribution(operation)
        Returns an array of recorded latencies for `<operation>`.

    set_latency_threshold(operation, threshold)
        Dynamically updates the threshold for a specific
        operation.

    get_percentile_summary(operation)
        Provides a dictionary of each tracked percentile
        (e.g. ``'90.0th_percentile': 0.8``).

    Notes
    -----
    Latency outliers can greatly impact user experience [1]_.
    Tracking long-tail latencies and summarizing them at
    percentiles gives insight into worst-case performance.

    Examples
    --------
    >>> from gofast.mlops.monitoring import LatencyTracker
    >>> def my_alert_callback(op, lat, msg):
    ...     print(f"Latency Alert for {op}: {msg}")
    >>> tracker = LatencyTracker(
    ...     alert_callback=my_alert_callback,
    ...     latency_thresholds={'inference': 1.0}
    ... )
    >>> tracker.run()
    >>> tracker.record_latency('inference', 1.2)

    See Also
    --------
    ErrorRateMonitor : Tracks the frequency of prediction
        errors.

    References
    ----------
    .. [1] Dean, J. and Barroso, L.A. "The Tail at Scale".
           Communications of the ACM, 56(2), 74–80, 2013.
    
    """

    @validate_params(
        {
            'alert_callback': [callable, None],
            'latency_thresholds': [dict, None],
            'global_latency_threshold': [
                Interval(Real, 0, None, closed='left')
            ],
            'retention_period': [
                Interval(Integral, 1, None, closed='left')
            ],
            'percentiles_to_track': [list, None],
            'alert_messages': [dict, None],
            'verbose': [Integral]
        }
    )
    def __init__(
        self,
        alert_callback: Optional[
            Callable[[str, float, str], None]
        ] = None,
        latency_thresholds: Optional[
            Dict[str, float]
        ] = None,
        global_latency_threshold: float = 2.0,
        retention_period: int = 100,
        percentiles_to_track: Optional[
            List[float]
        ] = None,
        alert_messages: Optional[
            Dict[str, str]
        ] = None,
        verbose: int = 0
    ):
        """
        Constructor for LatencyTracker. Initializes thresholds,
        callback, and other configurations for monitoring.
        """
        # Initialize BaseClass (sets self.verbose, etc.)
        super().__init__(verbose=verbose)

        # Public attributes (no trailing underscores)
        self.alert_callback = alert_callback
        self.latency_thresholds = latency_thresholds or {}
        self.global_latency_threshold = (
            global_latency_threshold
        )
        self.retention_period = retention_period
        self.percentiles_to_track = (
            percentiles_to_track or [90.0, 95.0, 99.0]
        )
        self.alert_messages = alert_messages or {}

        # Internal (non-exposed) attributes
        self._latencies_ = {}
        self._is_runned = False

    def run(
        self,
        **run_kw
    ) -> "LatencyTracker":
        """
        Prepare the latency tracker for usage.

        Parameters
        ----------
        **run_kw : dict
            Additional parameters for run configuration.

        Returns
        -------
        self : LatencyTracker
            Returns the instance after enabling run mode.
        """
        self._is_runned = True
        if self.verbose > 1:
            logger.info("LatencyTracker is now running.")
        return self

    @validate_params(
        {
            'operation': [str],
            'latency': [Interval(Real, 0, None, closed='left')]
        }
    )
    def record_latency(
        self,
        operation: str,
        latency: float
    ) -> None:
        """
        Record the latency of a specific operation.

        Parameters
        ----------
        operation : str
            Name of the operation (e.g. 'inference').

        latency : float
            Time in seconds for the operation. Must be >= 0.
        """

        if operation not in self._latencies_:
            self._latencies_[operation] = []

        # Append latency and keep only the last `retention_period`
        self._latencies_[operation].append(latency)
        if (
            len(self._latencies_[operation])
            > self.retention_period
        ):
            self._latencies_[operation].pop(0)

        # Check threshold
        threshold = self.latency_thresholds.get(
            operation,
            self.global_latency_threshold
        )
        if latency > threshold:
            msg = self.alert_messages.get(
                operation,
                f"{operation.capitalize()} "
                f"latency too high: {latency}s"
            )
            logger.warning(msg)
            if self.alert_callback:
                self.alert_callback(operation, latency, msg)


    @validate_params(
        {
            'operation': [str]
        }
    )
    def get_average_latency(
        self,
        operation: str
    ) -> float:
        """
        Return the average latency for a given operation.

        Returns NaN if no data is available.
        """
        latencies = self._latencies_.get(operation, [])
        if not latencies:
            return float('nan')
        return float(np.mean(latencies))


    @validate_params(
        {
            'operation': [str],
            'percentile': [
                Interval(Real, 0, 100, closed='both')
            ]
        }
    )
    def get_tail_latency(
        self,
        operation: str,
        percentile: float = 95.0
    ) -> float:
        """
        Return the tail latency (e.g., 95th percentile)
        for a given operation.
        """
        latencies = self._latencies_.get(operation, [])
        if not latencies:
            return float('nan')
        return float(np.percentile(latencies, percentile))


    @validate_params(
        {
            'operation': [str]
        }
    )
    def get_latency_distribution(
        self,
        operation: str
    ) -> np.ndarray:
        """
        Return the array of latency values for a given
        operation.
        """
        latencies = self._latencies_.get(operation, [])
        return np.array(latencies, dtype=float)

    @validate_params(
        {
            'operation': [str],
            'threshold': [
                Interval(Real, 0, None, closed='left')
            ]
        }
    )
    def set_latency_threshold(
        self,
        operation: str,
        threshold: float
    ) -> None:
        """
        Dynamically set a new latency threshold for a
        specific operation.
        """
        self.latency_thresholds[operation] = threshold
        if self.verbose > 1:
            logger.info(
                f"Set latency threshold for {operation} "
                f"to {threshold}s."
            )

    @validate_params(
        {
            'operation': [str]
        }
    )
    def get_percentile_summary(
        self,
        operation: str
    ) -> Dict[str, float]:
        """
        Return a dictionary with the tracked percentiles
        for the specified operation.
        """
        latencies = self._latencies_.get(operation, [])
        if not latencies:
            return {}

        summary = {}
        for p in self.percentiles_to_track:
            val = float(np.percentile(latencies, p))
            key = f"{p}th_percentile"
            summary[key] = val
        return summary

class AlertManager(BaseClass):
    """
    Centralizes alerting logic by allowing email- and webhook-based
    notifications, plus optional batching and retry mechanisms.

    The :class:`AlertManager` integrates with various channels to
    send alerts. It supports email recipients through an SMTP server
    and webhook URLs (e.g. Slack, Teams). If alerts fail to send,
    the system can automatically retry for a user-defined number of
    attempts. Optionally, multiple alerts can be batched together
    and delivered at fixed intervals to reduce spam.

    .. math::

        \\text{Alerts}(t) = 
        \\bigl\\{ A_i : i=1,2,...,N \\bigr\\},

    where each :math:`A_i` is an alert triggered by the system at
    time :math:`t`. If batched, these alerts are queued and sent
    together after a specified interval.

    Parameters
    ----------
    email_recipients : list of str or None, optional
        The list of email addresses to notify. If ``None``, no
        emails are sent.

    webhook_urls : list of str or None, optional
        The list of webhook endpoints for delivering alerts to
        external services (e.g. Slack). If ``None``, no webhooks
        are sent.

    smtp_server : str or None, optional
        The SMTP server hostname or address. Required if sending
        email alerts. If ``None``, emails cannot be dispatched.

    from_email : str or None, optional
        The sender address for emails. Required if email alerts
        are used. If ``None``, emails cannot be dispatched.

    retry_attempts : int, optional
        The maximum number of retries if an alert fails to send.
        Must be >= 0. Defaults to 3.

    batch_alerts : bool, optional
        Whether to batch multiple alerts together. If ``True``,
        alerts are stored and sent as a group after each
        `<batch_interval>` seconds. Defaults to ``False``.

    batch_interval : int, optional
        The number of seconds between batched alert sends. Must
        be a positive integer. Defaults to 60.

    verbose : int, optional
        Logging verbosity:
          - ``0``: Only critical errors
          - ``1``: Warnings and errors
          - ``2``: Info, warnings, and errors
          - ``3``: Debug, info, warnings, and errors
        Defaults to 0.

    Attributes
    ----------
    batched_alerts_ : list of tuple
        Internal queue for batched alerts. Each tuple may
        contain an alert type and additional details.

    Methods
    -------
    run(**run_kw)
        Activates the alert manager. If `<batch_alerts>` is
        ``True``, starts batching in a background thread.

    send_email_alert(subject, message, retries=0)
        Sends an email alert via SMTP, retrying on failure.

    send_webhook_alert(message, retries=0)
        Posts an alert message to each configured webhook URL,
        optionally retrying on failure.

    add_email_recipient(email)
        Adds an email address to the recipient list dynamically.

    remove_email_recipient(email)
        Removes an email address from the recipient list.

    log_alert(alert_type, details)
        Logs or batches an alert for future reference.

    Notes
    -----
    Retries are managed automatically. If an alert fails,
    :math:`\mathrm{retry\_attempts}` is decremented until it
    reaches zero, after which the failure is logged [1]_.

    Examples
    --------
    >>> from gofast.mlops.monitoring import AlertManager
    >>> manager = AlertManager(
    ...     email_recipients=['admin@example.com'],
    ...     smtp_server='smtp.example.com',
    ...     from_email='alerts@example.com'
    ... )
    >>> manager.run()
    >>> manager.send_email_alert(
    ...     'Test Alert',
    ...     'This is a test message.'
    ... )

    See Also
    --------
    ModelPerformanceMonitor : Sends performance-based alerts.
    ModelHealthChecker : Raises alerts for resource issues.

    References
    ----------
    .. [1] "Python SMTP Library". Python 3 Documentation.
    """


    @validate_params(
        {
            'email_recipients': [list, None],
            'webhook_urls': [list, None],
            'smtp_server': [str, None],
            'from_email': [str, None],
            'retry_attempts': [
                Interval(Integral, 0, None, closed='left')
            ],
            'batch_alerts': [bool],
            'batch_interval': [
                Interval(Integral, 1, None, closed='left')
            ],
            'verbose': [Integral]
        }
    )
    def __init__(
        self,
        email_recipients: Optional[List[str]] = None,
        webhook_urls: Optional[List[str]] = None,
        smtp_server: Optional[str] = None,
        from_email: Optional[str] = None,
        retry_attempts: int = 3,
        batch_alerts: bool = False,
        batch_interval: int = 60,
        verbose: int = 0
    ):

        super().__init__(verbose=verbose)

        self.email_recipients = email_recipients or []
        self.webhook_urls = webhook_urls or []
        self.smtp_server = smtp_server
        self.from_email = from_email
        self.retry_attempts = retry_attempts
        self.batch_alerts = batch_alerts
        self.batch_interval = batch_interval

        self._batched_alerts_ = []
        self._is_runned = False

    def run(
        self,
        **run_kw
    ) -> "AlertManager":
        """
        Prepare the AlertManager to send alerts. If batch_alerts=True,
        starts batching in a background thread.
        """
        # Mark manager as runned
        self._is_runned = True
        if self.batch_alerts:
            self._start_batching_()

        if self.verbose > 1:
            logger.info("AlertManager is now running.")
        return self

    def send_email_alert(
        self,
        subject: str,
        message: str,
        retries: int = 0
    ) -> None:
        """
        Send an email alert to configured recipients with
        retry logic.
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg="Call 'run' before sending an email alert."
        )
        if (not self.smtp_server) or (not self.from_email):
            raise ValueError(
                "SMTP server and 'from_email' must be set "
                "for email alerts."
            )

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = ", ".join(self.email_recipients)

        try:
            with smtplib.SMTP(self.smtp_server) as server:
                server.sendmail(
                    self.from_email,
                    self.email_recipients,
                    msg.as_string()
                )
            if self.verbose > 1:
                logger.info(f"Email alert sent: {subject}")

        except Exception as exc:
            if retries < self.retry_attempts:
                logger.error(
                    "Failed to send email alert. "
                    f"Retry {retries + 1}/"
                    f"{self.retry_attempts}. Error: {exc}"
                )
                self.send_email_alert(
                    subject,
                    message,
                    retries=retries + 1
                )
            else:
                logger.error(
                    "Failed to send email alert after "
                    f"{self.retry_attempts} attempts. "
                    f"Error: {exc}"
                )

    @ensure_pkg(
        "requests",
        extra=(
            "The 'requests' package is required "
            "for sending webhook alerts."
        ),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def send_webhook_alert(
        self,
        message: str,
        retries: int = 0
    ) -> None:
        """
        Send an alert message to configured webhook URLs
        (Slack, Teams, etc.) with retry logic.
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg="Call 'run' before sending webhook alerts."
        )
        
        import requests  # Imported after ensure_pkg checks

        for url in self.webhook_urls:
            try:
                resp = requests.post(
                    url,
                    json={'text': message}
                )
                if resp.status_code == 200:
                    if self.verbose > 1:
                        logger.info(
                            f"Webhook alert sent to {url}"
                        )
                else:
                    raise ValueError(
                        "Webhook post failed with code "
                        f"{resp.status_code}"
                    )

            except Exception as exc:
                if retries < self.retry_attempts:
                    logger.error(
                        f"Failed to send webhook to {url}. "
                        "Retry "
                        f"{retries + 1}/{self.retry_attempts}. "
                        f"Error: {exc}"
                    )
                    self.send_webhook_alert(
                        message,
                        retries=retries + 1
                    )
                else:
                    logger.error(
                        "Failed to send webhook alert "
                        f"to {url} after "
                        f"{self.retry_attempts} attempts. "
                        f"Error: {exc}"
                    )

    def add_email_recipient(
        self,
        email: str
    ) -> None:
        """
        Dynamically add an email recipient.
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg="Call 'run' before adding recipients."
        )
        if email not in self.email_recipients:
            self.email_recipients.append(email)
            if self.verbose > 1:
                logger.info(f"Added email recipient: {email}")

    def remove_email_recipient(
        self,
        email: str
    ) -> None:
        """
        Dynamically remove an email recipient.
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg="Call 'run' before removing recipients."
        )
        if email in self.email_recipients:
            self.email_recipients.remove(email)
            if self.verbose > 1:
                logger.info(f"Removed email recipient: {email}")

    def log_alert(
        self,
        alert_type: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Log an alert. If batching is enabled,
        store it for later sending.
        """
        check_is_runned(
            self, 
            attributes=["_is_runned"],
            msg="Call 'run' before logging alerts."
        )
        logger.info(
            f"Alert triggered: {alert_type} - {details}"
        )
        if self.batch_alerts:
            self._batched_alerts_.append(
                (alert_type, details)
            )

    def _start_batching_(
        self
    ) -> None:
        """
        Internal method: start a background thread to
        process batched alerts at regular intervals.
        """
        def batch_sender():
            while True:
                if self._batched_alerts_:
                    # For each stored alert, attempt
                    # to send via email + webhooks
                    for atype, dets in self._batched_alerts_:
                        subj = f"Batched {atype.capitalize()} Alert"
                        msg = str(dets)
                        self.send_email_alert(subj, msg)
                        self.send_webhook_alert(msg)
                    self._batched_alerts_.clear()

                time.sleep(self.batch_interval)

        # Start the daemon thread
        thr = threading.Thread(
            target=batch_sender,
            daemon=True
        )
        thr.start()

class ErrorRateMonitor(BaseClass):
    """
    Monitors the frequency of erroneous predictions made by a 
    machine learning model, triggering alerts if the observed 
    error rate exceeds a user-defined threshold.

    The :class:`ErrorRateMonitor` helps detect performance 
    degradation early by tracking recent successes and failures 
    in model predictions. If the calculated error rate surpasses 
    `<error_threshold>`, an optional callback can be invoked to 
    raise an alert or provide additional diagnostic details.

    .. math::

        \\text{ErrorRate}(t) = 
        1 - \\frac{\\#\\text{Successes}(t)}
               {\\#\\text{Predictions}(t)}

    where :math:`\\#\\text{Successes}(t)` is the count of 
    accurate predictions and :math:`\\#\\text{Predictions}(t)` 
    is the total predictions made in the sliding window at 
    time :math:`t`.

    Parameters
    ----------
    error_threshold : float
        The upper bound on allowable error rate before raising 
        an alert. Must be in [0, 1]. For instance, 0.05 indicates 
        a 5% threshold.

    alert_callback : callable or None, optional
        A function called when the error rate exceeds 
        `<error_threshold>`. It should accept two parameters:
        the current error rate (:math:`\\text{float}`) and a 
        dictionary of error details. Defaults to ``None``.

    retention_period : int, optional
        Number of recent prediction outcomes retained in a 
        sliding window. Must be a positive integer. Defaults 
        to 100.

    error_types : list of str or None, optional
        A list of error categories (e.g. 
        ``['prediction_failure', 'timeout']``) to track 
        separately. If ``None``, no specialized tracking is 
        done by category. Defaults to ``None``.

    verbose : int, optional
        Level of logging:
          - ``0``: Only critical errors
          - ``1``: Warnings + errors
          - ``2``: Info, warnings, errors
          - ``3``: Debug, info, warnings, errors
        Defaults to 0.

    Methods
    -------
    run(**run_kw)
        Prepares this monitor for usage. Once active, predictions 
        can be logged to track error rates.

    log_prediction(outcome, error_type=None)
        Logs a new prediction outcome. If `<outcome>` is 
        ``False`` and `<error_type>` matches a known error 
        category, its count is incremented.

    get_error_rate()
        Returns the current error rate computed over the 
        retained window.

    get_error_type_count(error_type)
        Retrieves the count of a specific error category over 
        the retained window.

    Notes
    -----
    A sliding window approach is used to track outcomes, 
    ensuring the error rate reflects recent model 
    performance [1]_.

    Examples
    --------
    >>> from gofast.mlops.monitoring import ErrorRateMonitor
    >>> def alert_cb(rate, details):
    ...     print(f"ERROR RATE ALERT: {rate:.1%}!")
    >>> mon = ErrorRateMonitor(
    ...     error_threshold=0.05,
    ...     alert_callback=alert_cb
    ... )
    >>> mon.run()
    >>> mon.log_prediction(False, error_type='prediction_failure')
    >>> print(mon.get_error_rate())

    See Also
    --------
    LatencyTracker : Focuses on latency issues.
    DataDriftMonitor : Tracks changes in data distributions.

    References
    ----------
    .. [1] "Statistical Process Control (SPC)". 
           https://en.wikipedia.org/wiki/Statistical_process_control
    """

    @validate_params(
        {
            'error_threshold': [
                Interval(Real, 0, 1, closed='both')
            ],
            'alert_callback': [callable, None],
            'retention_period': [
                Interval(Integral, 1, None, closed='left')
            ],
            'error_types': [list, None],
            'verbose': [Integral]
        }
    )
    def __init__(
        self,
        error_threshold: float,
        alert_callback: Optional[
            Callable[[float, Dict[str, Any]], None]
        ] = None,
        retention_period: int = 100,
        error_types: Optional[List[str]] = None,
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)

        self.error_threshold = error_threshold
        self.alert_callback = alert_callback
        self.retention_period = retention_period
        self.error_types = error_types or []

        self._outcomes_ = []
        self._error_log_ = {
            err_t: 0 for err_t in self.error_types
        }
        self._is_runned = False

    def run(
        self,
        **run_kw
    ) -> "ErrorRateMonitor":
        """
        Prepare the monitor to track error rates. 
        Once called, predictions can be logged.
        """
        self._is_runned = True
        if self.verbose > 1:
            logger.info("ErrorRateMonitor is now running.")
        return self

    @validate_params(
        {
            'outcome': [bool],
            'error_type': [str, None]
        }
    )
    def log_prediction(
        self,
        outcome: bool,
        error_type: Optional[str] = None
    ) -> None:
        """
        Log a prediction outcome with optional error type.
        """
        self._outcomes_.append(outcome)
        # Maintain sliding window
        if len(self._outcomes_) > self.retention_period:
            self._outcomes_.pop(0)

        # If it's an error (outcome=False) and a known error type
        if (not outcome) and (error_type in self._error_log_):
            self._error_log_[error_type] += 1

        # Compute current error rate
        error_rate = 1.0 - (
            sum(self._outcomes_) / len(self._outcomes_)
        )

        # Trigger alert if above threshold
        if error_rate > self.error_threshold:
            err_details = {
                "error_rate": error_rate,
                "total_errors": (
                    len(self._outcomes_) - sum(self._outcomes_)
                ),
                "error_log": dict(self._error_log_)
            }
            logger.warning(
                f"Error rate exceeds threshold: {error_rate:.2%}"
            )
            if self.alert_callback:
                self.alert_callback(error_rate, err_details)

    def get_error_rate(
        self
    ) -> float:
        """
        Return the current error rate from the recent window.
        """
        if not self._outcomes_:
            return float('nan')
        return 1.0 - (
            sum(self._outcomes_) / len(self._outcomes_)
        )


    @validate_params(
        {
            'error_type': [str]
        }
    )
    def get_error_type_count(
        self,
        error_type: str
    ) -> int:
        """
        Return the count of a specific error type 
        in the recent window.
        """
        return self._error_log_.get(error_type, 0)

class CustomMetricsLogger(BaseClass):
    """
    Logs and tracks arbitrary user-defined metrics, providing 
    threshold-based alerts when metrics exceed configured 
    limits.

    The :class:`CustomMetricsLogger` allows you to instrument 
    any domain-specific measure (e.g. "throughput", "queue_size")
    and retain the most recent values in a sliding window. 
    Thresholds can be specified for each metric to trigger 
    alerts when a metric's value surpasses a certain limit.

    .. math::

        \\text{MetricValue}(m, t) = v_t,

    where :math:`v_t` is the value of metric :math:`m` 
    at time :math:`t`. If :math:`v_t` exceeds the threshold 
    for :math:`m`, an alert is raised via the user-provided 
    callback.

    Parameters
    ----------
    retention_period : int, optional
        Number of most recent metric values to store for 
        each metric. Must be a positive integer. Defaults to 100.

    metric_thresholds : dict of str to float or None, optional
        Mapping of metric names to their threshold values. 
        If the logged value exceeds this threshold, an alert 
        is triggered. Defaults to ``None``.

    alert_callback : callable or None, optional
        A function invoked when a metric passes its threshold. 
        It should accept two parameters: `<metric_name>` (str) 
        and `<value>` (float). Defaults to ``None``.

    verbose : int, optional
        Controls log verbosity:
          - ``0``: Critical errors only
          - ``1``: Warnings + errors
          - ``2``: Info + warnings + errors
          - ``3``: Debug + info + warnings + errors
        Defaults to 0.

    Methods
    -------
    run(**run_kw)
        Activates the logger. Metrics can be recorded once 
        this is called.

    log_metric(metric_name, value)
        Logs a value for `<metric_name>` and raises an alert 
        if it breaches its threshold.

    get_metric_history(metric_name)
        Retrieves all retained values for `<metric_name>`.

    get_moving_average(metric_name, window)
        Computes the moving average over the last 
        `<window>` entries for `<metric_name>`.

    Notes
    -----
    Maintaining a sliding window of metric values can help 
    identify trends and anomalies [1]_. By combining custom 
    metrics with alert thresholds, you gain visibility into 
    dynamic behaviors that may otherwise go unnoticed.

    Examples
    --------
    >>> from gofast.mlops.monitoring import CustomMetricsLogger
    >>> def my_alert_callback(mname, val):
    ...     print(f"ALERT: {mname} = {val}")
    >>> logger = CustomMetricsLogger(
    ...     metric_thresholds={'throughput': 500.0},
    ...     alert_callback=my_alert_callback
    ... )
    >>> logger.run()
    >>> logger.log_metric('throughput', 550.0)

    See Also
    --------
    LatencyTracker : Focuses on operation latencies.
    ErrorRateMonitor : Tracks success vs. failure rates.

    References
    ----------
    .. [1] "Moving Average". 
           https://en.wikipedia.org/wiki/Moving_average
    """
    @validate_params(
        {
            'retention_period': [
                Interval(Integral, 1, None, closed='left')
            ],
            'metric_thresholds': [dict, None],
            'alert_callback': [callable, None],
            'verbose': [Integral]
        }
    )
    def __init__(
        self,
        retention_period: int = 100,
        metric_thresholds: Optional[
            Dict[str, float]
        ] = None,
        alert_callback: Optional[
            Callable[[str, float], None]
        ] = None,
        verbose: int = 0
    ):
        """
        Constructor for CustomMetricsLogger. Manages 
        retention and threshold logic for user-defined metrics.
        """
        super().__init__(verbose=verbose)

        self.retention_period = retention_period
        self.metric_thresholds = metric_thresholds or {}
        self.alert_callback = alert_callback

        self._metrics_history_ = {}
        self._is_runned = False

    def run(
        self,
        **run_kw
    ) -> "CustomMetricsLogger":
        """
        Prepare the logger to accept and track metrics.
        """
        self._is_runned = True
        if self.verbose > 1:
            logger.info("CustomMetricsLogger is now running.")
        return self

    @validate_params(
        {
            'metric_name': [str],
            'value': [Real]
        }
    )
    def log_metric(
        self,
        metric_name: str,
        value: float
    ) -> None:
        """
        Log a custom metric value, triggering an alert 
        if threshold exceeded.
        """
        if metric_name not in self._metrics_history_:
            self._metrics_history_[metric_name] = []

        # Add the new metric value
        self._metrics_history_[metric_name].append(value)

        # Maintain sliding window for the metric
        if (
            len(self._metrics_history_[metric_name])
            > self.retention_period
        ):
            self._metrics_history_[metric_name].pop(0)

        # Check if exceeds threshold
        threshold = self.metric_thresholds.get(metric_name)
        if (threshold is not None) and (value > threshold):
            msg = (
                f"Metric '{metric_name}' exceeds "
                f"threshold: {value}"
            )
            logger.warning(msg)
            if self.alert_callback:
                self.alert_callback(metric_name, value)


    @validate_params(
        {
            'metric_name': [str]
        }
    )
    def get_metric_history(
        self,
        metric_name: str
    ) -> List[float]:
        """
        Return the history of values for a given metric.
        """
        return self._metrics_history_.get(
            metric_name, []
        )

    @validate_params(
        {
            'metric_name': [str],
            'window': [
                Interval(Integral, 1, None, closed='left')
            ]
        }
    )
    def get_moving_average(
        self,
        metric_name: str,
        window: int
    ) -> float:
        """
        Return the moving average for a metric over a 
        specified window.
        """
        history = self._metrics_history_.get(
            metric_name, []
        )
        if not history:
            return float('nan')

        window = min(window, len(history))
        recent_values = history[-window:]
        return float(np.mean(recent_values))
