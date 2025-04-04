# -*- coding: utf-8 -*-

import pytest
import numpy as np

from gofast.mlops.monitoring import (
    ModelHealthChecker,
    DataDriftMonitor,
    AlertManager,
    LatencyTracker,
    ErrorRateMonitor,
    CustomMetricsLogger,
    ModelPerformanceMonitor,
)

def test_model_performance_monitor():
    """
    Basic test for ModelPerformanceMonitor instantiation
    and usage.
    """
    # Instantiate with some arbitrary parameters
    monitor = ModelPerformanceMonitor(
        metrics=['accuracy', 'f1'],
        drift_detection=True,
        window_size=5,
        verbose=2
    )

    # Must call `run` before `update`
    monitor.run()

    # Update with small batch
    y_true = [0, 1, 1]
    y_pred = [0, 0, 1]
    monitor.update(y_true, y_pred)

    # Check if performance_history_ has updated
    history = monitor.get_performance_history()
    assert 'accuracy' in history
    assert 'f1' in history
    # Should have at least 1 entry per metric now
    assert len(history['accuracy']) == 1
    assert len(history['f1']) == 1


def test_model_health_checker():
    """
    Basic test for ModelHealthChecker. This may require psutil
    and/or GPUtil if actually running health checks, so we just
    ensure it can be initialized and run.
    """
    checker = ModelHealthChecker(
        cpu_threshold=75.0,
        memory_threshold=80.0,
        disk_threshold=85.0,
        gpu_threshold=90.0,
        network_threshold=100.0,
        latency_threshold=2.0,
        verbose=2
    )

    # Start the checker
    checker.run()

    # Manually call the check_health method
    # (in real usage, might be repeated in background)
    checker.check_health()

    # Ensure we can log a latency
    checker.record_latency(1.5)

    # Retrieve some history
    cpu_history = checker.get_health_history('cpu')
    assert isinstance(cpu_history, list)


def test_data_drift_monitor():
    """
    Test DataDriftMonitor with a simple baseline and new data.
    """
    # Create synthetic baseline and new data
    baseline = np.random.normal(loc=0.0, scale=1.0, size=(100, 3))
    new_data = np.random.normal(loc=0.5, scale=1.0, size=(100, 3))

    monitor = DataDriftMonitor(
        drift_threshold=0.05,  # typical p-value
        drift_detection_method='ks',
        verbose=2
    )

    # Fit with baseline data
    monitor.fit(baseline_data=baseline)

    # Check drift with new data
    monitor.monitor_drift(new_data=new_data)

    # Inspect drift history
    history = monitor.get_drift_history()
    assert isinstance(history, list)
    # Each entry in history is a dict with p_values, etc.
    if history:
        assert 'p_values' in history[-1]
        assert 'drift_detected' in history[-1]

def test_alert_manager():
    """
    Test basic usage of AlertManager. This test does not actually
    send emails or webhooks; it checks if methods can be called
    without errors.
    """
    manager = AlertManager(
        email_recipients=['test@example.com'],
        webhook_urls=['http://dummyurl.test'],
        smtp_server='smtp.example.com',
        from_email='alert_sender@example.com',
        retry_attempts=1,
        batch_alerts=False,
        batch_interval=60,
        verbose=2
    )

    manager.run()

    # Trigger email alert (mock test; no real server)
    manager.send_email_alert(
        subject='Test Alert',
        message='This is a test alert.',
    )

    # Trigger webhook alert
    manager.send_webhook_alert(
        message='Data drift detected.'
    )

    # Log an alert
    manager.log_alert('test_alert', {'info': 'test_info'})


def test_latency_tracker():
    """
    Test LatencyTracker with basic operations and thresholds.
    """
    def custom_alert(op_name, lat, msg):
        print(f"Latency Alert for {op_name}: {msg}")

    tracker = LatencyTracker(
        alert_callback=custom_alert,
        latency_thresholds={'test_op': 0.5},
        global_latency_threshold=1.0,
        retention_period=5,
        verbose=2
    )

    tracker.run()
    tracker.record_latency('test_op', 0.4)
    tracker.record_latency('test_op', 0.6)  # Exceeds threshold
    avg = tracker.get_average_latency('test_op')
    assert 0.4 <= avg <= 0.6

    tail_95 = tracker.get_tail_latency('test_op', percentile=95.0)
    assert tail_95 >= avg

    dist = tracker.get_latency_distribution('test_op')
    assert len(dist) == 2  # we added 2 latencies


def test_error_rate_monitor():
    """
    Test ErrorRateMonitor logging and error threshold checking.
    """
    def error_callback(error_rate, details):
        print(f"Error Rate Alert: {error_rate:.2%}")

    monitor = ErrorRateMonitor(
        error_threshold=0.5,
        alert_callback=error_callback,
        retention_period=3,
        error_types=['prediction_failure', 'timeout'],
        verbose=2
    )

    monitor.run()

    # Log predictions
    monitor.log_prediction(True)   # success
    monitor.log_prediction(False, error_type='prediction_failure')
    rate = monitor.get_error_rate()
    assert 0.0 <= rate <= 1.0

    # Log another failure
    monitor.log_prediction(False, error_type='timeout')
    rate = monitor.get_error_rate()
    # With 3 outcomes, at least 2 might be failures => rate >= 66%
    assert rate >= 0.66

    # Check error type count
    fail_count = monitor.get_error_type_count('prediction_failure')
    assert fail_count >= 1


def test_custom_metrics_logger():
    """
    Test CustomMetricsLogger with threshold-based alerts.
    """
    def metric_alert(metric_name, value):
        print(f"Metric '{metric_name}' exceeded threshold: {value}")

    logger_obj = CustomMetricsLogger(
        retention_period=3,
        metric_thresholds={'throughput': 10.0},
        alert_callback=metric_alert,
        verbose=2
    )

    logger_obj.run()

    # Log some metric values
    logger_obj.log_metric('throughput', 5.0)
    logger_obj.log_metric('throughput', 12.0)  # above threshold
    hist = logger_obj.get_metric_history('throughput')
    assert len(hist) == 2

    # Check moving average
    mov_avg = logger_obj.get_moving_average('throughput', window=2)
    assert mov_avg == pytest.approx((5.0 + 12.0)/2.0, 0.0001)


if __name__=='__main__': 
    pytest.main( [__file__])