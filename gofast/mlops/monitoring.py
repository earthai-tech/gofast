# -*- coding: utf-8 -*-
"""
Monitor model performance in production, track key metrics, and set alerts 
for performance degradation.

"""

from numbers import Integral 
from sklearn.metrics import ( 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
    )
from ._config import INSTALL_DEPENDENCIES, USE_CONDA 

from ..api.property import BaseClass
from ..compat.sklearn import validate_params, Interval 
from ..tools.funcutils import ensure_pkg 
from ..tools.coreutils import is_iterable 
from .._gofastlog import gofastlog 

logger=gofastlog.get_gofast_logger(__name__)


class ModelPerformanceMonitor (BaseClass):
    """
    Class to monitor model performance in production, track key metrics,
    and set alerts for performance degradation.

    Parameters
    ----------
    metrics : list of str, default=['accuracy']
        List of performance metrics to monitor. Supported metrics include
        ``'accuracy'``, ``'precision'``, ``'recall'``, ``'f1'``, etc. These
        metrics will be calculated over a sliding window of recent predictions
        to evaluate model performance over time.

    drift_detection : bool, default=True
        Whether to enable drift detection for input data and model outputs.
        When set to ``True``, the class will perform statistical tests or
        use pre-trained drift detection algorithms to identify significant
        changes in data distribution or model behavior.

    alert_thresholds : dict, default=None
        Custom thresholds for triggering alerts when performance metrics
        degrade. The keys of the dictionary are metric names (e.g.,
        ``'accuracy'``, ``'f1'``), and the values are the threshold values.
        If a monitored metric falls below its threshold, an alert will be
        triggered.

    monitoring_tools : list of str, default=None
        List of monitoring tools to integrate with, such as ``'prometheus'``,
        ``'grafana'``. Integration allows metrics to be exported to external
        systems for visualization and alerting. Currently supported tools
        are ``'prometheus'``. Integration with ``'grafana'`` is achieved
        indirectly through data sources like Prometheus.

    window_size : int, default=100
        The number of recent samples to consider for computing the
        performance metrics. This sliding window approach helps in tracking
        the model's performance in real-time by focusing on the most
        recent data.

    Attributes
    ----------
    performance_history_ : dict
        Historical performance metrics tracked over time.

    drift_status_ : dict
        Current status of drift detection for data and model.

    See Also
    --------
    DataDriftDetector : Class for detecting data drift.
    ModelDriftDetector : Class for detecting model drift.

    Notes
    -----
    This class provides methods to monitor the performance of machine
    learning models in production environments. It calculates various
    performance metrics such as accuracy, precision, recall, and F1 score.

    The performance metrics are calculated as follows:

    - **Accuracy**:

      .. math::

         \\text{Accuracy} = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}(y_i = \\hat{y}_i)

      where :math:`n` is the number of samples, :math:`y_i` is the true label,
      :math:`\\hat{y}_i` is the predicted label, and :math:`\\mathbb{1}` is the indicator function.

    Concept drift refers to the change in the statistical properties of the
    target variable over time, which can affect the performance of machine
    learning models in production environments [1]_.

    Examples
    --------
    >>> from gofast.mlops.monitoring import ModelPerformanceMonitor
    >>> monitor = ModelPerformanceMonitor(metrics=['accuracy', 'f1'], window_size=50)
    >>> for batch in data_stream:
    ...     predictions = model.predict(batch['features'])
    ...     monitor.update(batch['labels'], predictions)

    References
    ----------
    .. [1] Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014).
           A survey on concept drift adaptation. *ACM Computing Surveys (CSUR)*, 46(4), 1-37.
    """

    @validate_params(
        {
            'metrics': [list],
            'drift_detection': [bool],
            'alert_thresholds': [dict, type(None)],
            'monitoring_tools': [list, type(None)],
            'window_size': [Interval(Integral, 1, None, closed='left')],
        }
    )
    def __init__(self, metrics='accuracy', drift_detection=True,
                 alert_thresholds=None, monitoring_tools=None,
                 window_size=100):
        self.metrics = is_iterable(metrics, exclude_string=True, transform= True)
        self.drift_detection = drift_detection
        self.alert_thresholds = alert_thresholds or {}
        self.monitoring_tools = monitoring_tools or []
        self.window_size = window_size
        self.performance_history_ = {}
        self.drift_status_ = {}
        self._initialize_monitoring_tools()
        self._init_performance_metrics()
        self._labels_window = []
        self._preds_window = []
        self._init_alerting()
        if self.drift_detection:
            self._init_drift_detection()

    def _initialize_monitoring_tools(self):
        """Initialize connections to external monitoring tools."""
        for tool in self.monitoring_tools:
            if tool.lower() == 'prometheus':
                self._init_prometheus_client()
            elif tool.lower() == 'grafana':
                pass  # Grafana integration via Prometheus
            else:
                raise ValueError(f"Unsupported monitoring tool: {tool}")

    def _init_prometheus_client(self):
        """Initialize Prometheus client."""
        @ensure_pkg(
            "prometheus_client",
            extra="To use Prometheus integration, please install `prometheus_client`.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA
        )
        def init_client():
            import prometheus_client
            self.prometheus_metrics = {}
            for metric in self.metrics:
                self.prometheus_metrics[metric] = prometheus_client.Gauge(
                    f'model_{metric}', f'Model {metric} over time'
                )
        init_client()

    def _init_performance_metrics(self):
        """Initialize performance metric functions."""
        
        self.metric_functions = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            # Additional metrics can be added here
        }
        self.selected_metrics = {metric: self.metric_functions[metric]
                                 for metric in self.metrics if metric in 
                                 self.metric_functions}

    def _init_alerting(self):
        """
        Initialize alerting mechanisms.
        
        This method sets up the logging using the gofast logging utility and
        initializes configurations for additional alerting mechanisms like
        email, Slack, and SMS if they are enabled.
        
        Returns
        -------
        None
        """
        
        
        # Initialize alert configurations
        self.alert_configs = {}
        
        # Email configuration
        email_config = getattr(self, 'email_config', None)
        if email_config and email_config.get('enabled', False):
            self.alert_configs['email'] = email_config
            logger.info("Email alerting enabled.")
        else:
            logger.info("Email alerting not enabled or not configured.")
        
        # Slack configuration
        slack_config = getattr(self, 'slack_config', None)
        if slack_config and slack_config.get('enabled', False):
            self.alert_configs['slack'] = slack_config
            logger.info("Slack alerting enabled.")
        else:
            logger.info("Slack alerting not enabled or not configured.")
        
        # SMS configuration
        sms_config = getattr(self, 'sms_config', None)
        if sms_config and sms_config.get('enabled', False):
            self.alert_configs['sms'] = sms_config
            logger.info("SMS alerting enabled.")
        else:
            logger.info("SMS alerting not enabled or not configured.")
    

    def _init_drift_detection(self):
        """Initialize drift detection mechanisms."""
        @ensure_pkg(
            "scipy",
            extra="To use drift detection, please install `scipy`.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA
        )
        def init_drift():
            from scipy.stats import ks_2samp
            self.ks_test = ks_2samp
        init_drift()

    def update(self, y_true, y_pred):
        """
        Update the monitoring metrics with a new batch of data.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_pred : array-like of shape (n_samples,)
            Predicted labels.

        Returns
        -------
        self : object
            Returns self.
        """
        # Update the windowed data
        self._labels_window.extend(y_true)
        self._preds_window.extend(y_pred)
        # Keep only the recent window_size elements
        self._labels_window = self._labels_window[-self.window_size:]
        self._preds_window = self._preds_window[-self.window_size:]

        # Calculate metrics
        current_metrics = {}
        for metric_name, metric_func in self.selected_metrics.items():
            if metric_name in ['precision', 'recall', 'f1']:
                value = metric_func(self._labels_window, self._preds_window, average='weighted')
            else:
                value = metric_func(self._labels_window, self._preds_window)
            current_metrics[metric_name] = value

            # Update performance history
            self.performance_history_.setdefault(metric_name, []).append(value)

            # Check for alerts
            if (metric_name in self.alert_thresholds and
                value < self.alert_thresholds[metric_name]):
                self._trigger_alert(metric_name, value)

            # Update monitoring tools
            if 'prometheus' in self.monitoring_tools:
                self.prometheus_metrics[metric_name].set(value)

        # Perform drift detection if enabled
        if self.drift_detection:
            self._detect_drift()

        return self

    def _detect_drift(self):
        """
        Detect data and model drift using statistical tests.

        Returns
        -------
        None
        """
        if len(self._labels_window) < 2 or len(self._preds_window) < 2:
            # Not enough data to perform drift detection
            return

        y_true = self._labels_window
        y_pred = self._preds_window

        # Perform KS test between true labels and predicted labels
        stat, p_value = self.ks_test(y_true, y_pred)

        # Update drift status
        self.drift_status_['ks_statistic'] = stat
        self.drift_status_['p_value'] = p_value

        # Check if p_value is below the significance level (e.g., 0.05)
        if p_value < 0.05:
            self.drift_status_['drift_detected'] = True
            logger.warning(f"Drift detected: KS test p-value = {p_value:.4f}")
        else:
            self.drift_status_['drift_detected'] = False

    def get_performance_history(self):
        """
        Get the historical performance metrics.

        Returns
        -------
        performance_history_ : dict
            Historical performance metrics tracked over time.
        """
        return self.performance_history_

    def get_drift_status(self):
        """
        Get the current drift detection status.

        Returns
        -------
        drift_status_ : dict
            Current status of drift detection for data and model.
        """
        return self.drift_status_

    def reset_monitor(self):
        """
        Reset the monitoring state.

        Returns
        -------
        None
        """
        self.performance_history_.clear()
        self.drift_status_.clear()
        self._labels_window.clear()
        self._preds_window.clear()
        logger.info("Monitoring state has been reset.")

    def set_thresholds(self, alert_thresholds):
        """
        Set or update custom thresholds for performance alerts.

        Parameters
        ----------
        alert_thresholds : dict
            Custom thresholds for triggering alerts when performance metrics
            degrade. The keys of the dictionary are metric names (e.g.,
            ``'accuracy'``, ``'f1'``), and the values are the threshold values.

        Returns
        -------
        None
        """
        self.alert_thresholds.update(alert_thresholds)
        logger.info("Alert thresholds have been updated.")

    def save_state(self, filepath):
        """
        Save the current monitoring state to a file.

        Parameters
        ----------
        filepath : str
            Path to the file where the state will be saved.

        Returns
        -------
        None
        """
        @ensure_pkg(
            "pickle",
            extra="To save state, please ensure `pickle` is available.",
            auto_install=False
        )
        def save():
            import pickle
            state = {
                'performance_history_': self.performance_history_,
                'drift_status_': self.drift_status_,
                '_labels_window': self._labels_window,
                '_preds_window': self._preds_window
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Monitoring state saved to {filepath}.")
        save()

    def load_state(self, filepath):
        """
        Load monitoring state from a file.

        Parameters
        ----------
        filepath : str
            Path to the file from which the state will be loaded.

        Returns
        -------
        None
        """
        @ensure_pkg(
            "pickle",
            extra="To load state, please ensure `pickle` is available.",
            auto_install=False
        )
        def load():
            import pickle
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.performance_history_ = state.get('performance_history_', {})
            self.drift_status_ = state.get('drift_status_', {})
            self._labels_window = state.get('_labels_window', [])
            self._preds_window = state.get('_preds_window', [])
            logger.info(f"Monitoring state loaded from {filepath}.")
        load()


    def _send_email_alert(self, alert_message, email_config):
        """
        Send an email alert with the specified message.
    
        Parameters
        ----------
        alert_message : str
            The alert message to be sent via email.
    
        email_config : dict
            Configuration dictionary containing email settings.
    
        Returns
        -------
        None
        """
        @ensure_pkg(
            "smtplib",
            extra="To send email alerts, please ensure `smtplib` is available.",
            auto_install=False
        )
        def send_email():
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
    
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            smtp_username = email_config.get('smtp_username')
            smtp_password = email_config.get('smtp_password')
            sender_email = email_config.get('sender_email')
            receiver_email = email_config.get('receiver_email')
    
            # Create the email
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = receiver_email
            message['Subject'] = "Model Performance Alert"
            message.attach(MIMEText(alert_message, 'plain'))
    
            # Send the email
            try:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.sendmail(sender_email, receiver_email, message.as_string())
                server.quit()
                logger.info("Email alert sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
    
        send_email()
        
    def _trigger_alert(self, metric_name, value):
        """
        Trigger an alert for performance degradation.
    
        Parameters
        ----------
        metric_name : str
            The name of the metric that triggered the alert.
    
        value : float
            The current value of the metric.
    
        Returns
        -------
        None
        """
        alert_message = (f"Performance alert: {metric_name} has dropped "
                         f"below threshold: {value:.4f}")
        logger.warning(alert_message)
    
        # Send email alert if email configuration is provided
        email_config = getattr(self, 'email_config', None)
        if email_config and email_config.get('enabled', False):
            self._send_email_alert(alert_message, email_config)
    
        # Send Slack alert if Slack configuration is provided
        slack_config = getattr(self, 'slack_config', None)
        if slack_config and slack_config.get('enabled', False):
            self._send_slack_alert(alert_message, slack_config)
    
        # Send SMS alert if SMS configuration is provided
        sms_config = getattr(self, 'sms_config', None)
        if sms_config and sms_config.get('enabled', False):
            self._send_sms_alert(alert_message, sms_config)

    def _send_slack_alert(self, alert_message, slack_config):
        """
        Send a Slack alert with the specified message.
    
        Parameters
        ----------
        alert_message : str
            The alert message to be sent to Slack.
    
        slack_config : dict
            Configuration dictionary containing Slack settings.
    
        Returns
        -------
        None
        """
        @ensure_pkg(
            "requests",
            extra="To send Slack alerts, please install `requests`.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA
        )
        def send_slack():
            import requests
    
            webhook_url = slack_config.get('webhook_url')
            if not webhook_url:
                logger.error("Slack webhook URL not provided.")
                return
    
            payload = {'text': alert_message}
            try:
                response = requests.post(webhook_url, json=payload)
                if response.status_code == 200:
                    logger.info("Slack alert sent successfully.")
                else:
                    logger.error(
                        f"Failed to send Slack alert: {response.text}")
            except Exception as e:
                logger.error(
                    f"Exception occurred while sending Slack alert: {e}")
    
        send_slack()
    
    def _send_sms_alert(self, alert_message, sms_config):
        """
        Send an SMS alert with the specified message.
    
        Parameters
        ----------
        alert_message : str
            The alert message to be sent via SMS.
    
        sms_config : dict
            Configuration dictionary containing SMS settings.
    
        Returns
        -------
        None
        """
        @ensure_pkg(
            "twilio",
            extra="To send SMS alerts, please install `twilio`.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA
        )
        def send_sms():
            from twilio.rest import Client
    
            account_sid = sms_config.get('account_sid')
            auth_token = sms_config.get('auth_token')
            from_number = sms_config.get('from_number')
            to_number = sms_config.get('to_number')
    
            if not all([account_sid, auth_token, from_number, to_number]):
                logger.error("Incomplete SMS configuration provided.")
                return
    
            client = Client(account_sid, auth_token)
            try:
                message = client.messages.create(
                    body=alert_message,
                    from_=from_number,
                    to=to_number
                )
                logger.info(f"SMS alert sent successfully. SID: {message.sid}")
            except Exception as e:
                logger.error(f"Exception occurred while sending SMS alert: {e}")
    
        send_sms()
