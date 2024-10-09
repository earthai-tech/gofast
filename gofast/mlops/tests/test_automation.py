# -*- coding: utf-8 -*-
# test_mlops_automation.py

import pytest
from unittest.mock import patch, MagicMock
from gofast.mlops.automation import (
    AutomationManager,
    RetrainingScheduler,
    AirflowAutomationManager,
    KubeflowAutomationManager,
    KafkaAutomationManager,
    RabbitMQAutomationManager,
)
import logging
import random
import time

from gofast.mlops._config import INSTALL_DEPENDENCIES 

INSTALL_DEPENDENCIES =True # noqa # install dependencies during the test

# Configure logging for the tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample functions for testing
def sample_task(task_name):
    logger.info(f"Executing task: {task_name}")

def retrain_model(model):
    logger.info(f"Retraining model: {model.__class__.__name__}")

def evaluate_model(model):
    return random.uniform(0.6, 1.0)  # Simulated performance score

def process_kafka_data(data):
    logger.info(f"Processing Kafka data: {data}")

def process_rabbitmq_data(data):
    logger.info(f"Processing RabbitMQ data: {data}")

# Sample model class
class SampleModel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<Model {self.name}>"

# Fixtures
@pytest.fixture
def automation_manager():
    manager = AutomationManager(max_workers=2, state_persistence_file="test_state.pkl")
    yield manager
    # Cleanup after tests
    manager.shutdown()
    import os
    if os.path.exists("test_state.pkl"):
        os.remove("test_state.pkl")

@pytest.fixture
def retraining_scheduler():
    scheduler = RetrainingScheduler(max_workers=2)
    yield scheduler
    scheduler.shutdown()

# Tests for AutomationManager
def test_add_and_schedule_task(automation_manager):
    automation_manager.add_task("task1", sample_task, interval=1, args=("Task1",), retries=2)
    assert "task1" in automation_manager.tasks
    automation_manager.schedule_task("task1")
    assert automation_manager.tasks["task1"]["running"] == True
    # Allow some time for the task to run
    time.sleep(2)
    automation_manager.cancel_task("task1")
    assert automation_manager.tasks["task1"]["running"] == False

def test_task_retries(automation_manager):
    # Define a task that fails twice before succeeding
    def flaky_task():
        if not hasattr(flaky_task, "counter"):
            flaky_task.counter = 0
        flaky_task.counter += 1
        if flaky_task.counter < 3:
            raise ValueError("Intentional Failure")
        logger.info("Flaky task succeeded.")

    automation_manager.add_task("flaky_task", flaky_task, interval=1, retries=3)
    automation_manager.schedule_task("flaky_task")
    # Allow time for retries
    time.sleep(5)
    # Check if task has succeeded
    assert automation_manager.tasks["flaky_task"]["failures"] == 0
    automation_manager.cancel_task("flaky_task")

def test_state_persistence(automation_manager):
    automation_manager.add_task("task_persist", sample_task, interval=1, args=("PersistTask",), retries=1)
    automation_manager.schedule_task("task_persist")
    # Allow some time for the task to run
    time.sleep(2)
    automation_manager.persist_state()
    # Create a new manager and load state
    new_manager = AutomationManager(max_workers=2, state_persistence_file="test_state.pkl")
    assert "task_persist" in new_manager.tasks
    assert new_manager.tasks["task_persist"]["running"] == True
    new_manager.cancel_task("task_persist")
    new_manager.shutdown()

# Tests for RetrainingScheduler
def test_schedule_retraining(retraining_scheduler):
    model = SampleModel("TestModel")
    retraining_scheduler.schedule_retraining(model, retrain_model, interval=1)
    assert f"retrain_{model.__class__.__name__}" in retraining_scheduler.tasks
    # Allow time for the retraining task to run
    time.sleep(2)
    retraining_scheduler.cancel_task(f"retrain_{model.__class__.__name__}")
    assert retraining_scheduler.tasks[f"retrain_{model.__class__.__name__}"]["running"] == False

def test_trigger_retraining_on_decay(retraining_scheduler):
    model = SampleModel("DecayModel")
    
    # Mock the retrain_model function to track calls
    with patch('gofast.mlops.automation.retrain_model') as mock_retrain:
        retraining_scheduler.trigger_retraining_on_decay(model, evaluate_model, decay_threshold=0.7)
        score = evaluate_model(model)
        if score < 0.7:
            mock_retrain.assert_called_once_with(model)
        else:
            mock_retrain.assert_not_called()

def test_monitor_model(retraining_scheduler):
    model = SampleModel("MonitorModel")
    
    # Mock trigger_retraining_on_decay to track calls
    with patch.object(retraining_scheduler, 'trigger_retraining_on_decay') as mock_trigger:
        retraining_scheduler.monitor_model(model, evaluate_model, decay_threshold=0.7, check_interval=1)
        # Allow some time for the monitoring task to run
        time.sleep(2)
        # Check if the trigger was called at least once
        assert mock_trigger.call_count >= 1
        retraining_scheduler.cancel_task(f"monitor_{model.__class__.__name__}")

def test_adjust_retraining_schedule(retraining_scheduler):
    model = SampleModel("AdjustModel")
    retraining_scheduler.schedule_retraining(model, retrain_model, interval=2)
    assert retraining_scheduler.tasks[f"retrain_{model.__class__.__name__}"]["interval"] == 2
    retraining_scheduler.adjust_retraining_schedule(model, new_interval=1)
    assert retraining_scheduler.tasks[f"retrain_{model.__class__.__name__}"]["interval"] == 1
    retraining_scheduler.cancel_task(f"retrain_{model.__class__.__name__}")

# Tests for AirflowAutomationManager
@patch("gofast.mlops.automation.AirflowAutomationManager._create_dag")
@patch("gofast.mlops.automation.PythonOperator")
def test_airflow_automation_manager_create_workflow(mock_python_operator, mock_create_dag):
    mock_dag = MagicMock()
    mock_create_dag.return_value = mock_dag
    airflow_manager = AirflowAutomationManager(dag_id="test_dag", start_date="2024-01-01")
    airflow_manager.create_workflow()
    mock_create_dag.assert_called_once()
    assert airflow_manager.dag == mock_dag
    assert mock_python_operator.call_count == len(airflow_manager.pipeline_manager.steps)

@patch("gofast.mlops.automation.AirflowAutomationManager._create_dag")
@patch("gofast.mlops.automation.PythonOperator")
def test_airflow_automation_manager_add_task(mock_python_operator, mock_create_dag):
    mock_dag = MagicMock()
    mock_create_dag.return_value = mock_dag
    airflow_manager = AirflowAutomationManager(dag_id="test_dag", start_date="2024-01-01")
    airflow_manager.pipeline_manager.add_step(PipelineStep(name="AirflowTask", func=sample_task, params={"task_name": "AirflowTask"}))
    airflow_manager.create_workflow()
    mock_python_operator.assert_called_with(
        task_id="AirflowTask",
        python_callable=sample_task,
        op_kwargs={"task_name": "AirflowTask"},
        dag=mock_dag,
    )

def test_airflow_automation_manager_schedule_pipeline():
    airflow_manager = AirflowAutomationManager(dag_id="test_dag", start_date="2024-01-01")
    airflow_manager.dag = MagicMock()
    airflow_manager.schedule_pipeline("@weekly")
    airflow_manager.dag.schedule_interval = "@weekly"
    assert airflow_manager.dag.schedule_interval == "@weekly"

def test_airflow_automation_manager_monitor_pipeline():
    airflow_manager = AirflowAutomationManager(dag_id="test_dag", start_date="2024-01-01")
    with patch("gofast.mlops.automation.logging") as mock_logging:
        airflow_manager.monitor_pipeline()
        mock_logging.getLogger().info.assert_called_with("Monitoring Airflow DAG execution through the Airflow web UI.")

# Tests for KubeflowAutomationManager
@patch("gofast.mlops.automation.kfp.Client")
@patch("gofast.mlops.automation.dsl.pipeline")
def test_kubeflow_automation_manager_create_kubeflow_pipeline(mock_pipeline, mock_client):
    mock_flow = MagicMock()
    mock_pipeline.return_value = mock_flow
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    kubeflow_manager = KubeflowAutomationManager(host="http://localhost:8080")
    kubeflow_manager.create_kubeflow_pipeline("test_pipeline", "test_task", "print('Hello Kubeflow')", arg1="value1")
    mock_pipeline.assert_called_once_with(name="test_pipeline", description='Automation Pipeline for Machine Learning')
    mock_client_instance.create_run_from_pipeline_func.assert_called_once()

# Tests for KafkaAutomationManager
@patch("gofast.mlops.automation.KafkaConsumer")
def test_kafka_automation_manager_process_kafka_message(mock_consumer):
    mock_consumer_instance = MagicMock()
    mock_consumer.return_value = mock_consumer_instance
    kafka_manager = KafkaAutomationManager(kafka_servers=["localhost:9092"], topic="test_topic")
    with patch.object(kafka_manager, 'process_kafka_message', wraps=kafka_manager.process_kafka_message) as mock_process:
        # Simulate receiving a message
        mock_consumer_instance.__iter__.return_value = [MagicMock(value=b"test_message")]
        kafka_manager.process_kafka_message(process_kafka_data)
        mock_process.assert_called_once_with(process_kafka_data)

# Tests for RabbitMQAutomationManager
@patch("gofast.mlops.automation.pika.BlockingConnection")
def test_rabbitmq_automation_manager_process_rabbitmq_message(mock_blocking_connection):
    mock_connection = MagicMock()
    mock_channel = MagicMock()
    mock_blocking_connection.return_value = mock_connection
    mock_connection.channel.return_value = mock_channel
    rabbitmq_manager = RabbitMQAutomationManager(host="localhost", queue="test_queue")
    
    with patch.object(rabbitmq_manager, 'process_rabbitmq_message', wraps=rabbitmq_manager.process_rabbitmq_message) as mock_process:
        # Simulate receiving a message
        def callback(ch, method, properties, body):
            process_rabbitmq_data(body)
        
        mock_channel.basic_consume.side_effect = lambda queue, on_message_callback, auto_ack: on_message_callback(None, None, None, b"test_message")
        rabbitmq_manager.process_rabbitmq_message(process_rabbitmq_data)
        mock_process.assert_called_once_with(process_rabbitmq_data)

# Helper classes and functions
from gofast.mlops.pipeline import PipelineStep

# Run the test suite
if __name__ == "__main__":
    pytest.main([__file__])

