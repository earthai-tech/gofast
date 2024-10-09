# -*- coding: utf-8 -*-
import pytest
from unittest.mock import patch, MagicMock
from gofast.mlops.automation import (
    AirflowAutomationManager,
    KubeflowAutomationManager,
    KafkaAutomationManager,
    RabbitMQAutomationManager,
)
from datetime import datetime

from gofast.mlops._config import INSTALL_DEPENDENCIES 

INSTALL_DEPENDENCIES =True # noqa # install dependencies during the test

# Fixtures for mocking external integrations
@pytest.fixture
def airflow_manager():
    return AirflowAutomationManager(dag_id="test_dag", start_date=datetime(2024, 1, 1))


@pytest.fixture
def kubeflow_manager():
    return KubeflowAutomationManager(host="http://localhost:8080")


@pytest.fixture
def kafka_manager():
    return KafkaAutomationManager(kafka_servers=["localhost:9092"], topic="test_topic")


@pytest.fixture
def rabbitmq_manager():
    return RabbitMQAutomationManager(host="localhost", queue="test_queue")


# AirflowAutomationManager Tests
@patch("gofast.mlops.automation.PythonOperator")
@patch("gofast.mlops.automation.DAG")
def test_airflow_add_task(mock_dag, mock_python_operator, airflow_manager):
    mock_dag.return_value = MagicMock()
    mock_python_operator.return_value = MagicMock()

    # Add a task
    def sample_task(data):
        return f"Processing {data}"

    task = airflow_manager.add_task_to_airflow(task_name="sample_task", func=sample_task, data="sample_data")
    
    # Ensure task was added to the DAG
    mock_python_operator.assert_called_once_with(
        task_id="sample_task",
        python_callable=sample_task,
        op_kwargs={"data": "sample_data"},
        dag=airflow_manager.dag,
    )


@patch("gofast.mlops.automation.PythonOperator")
@patch("gofast.mlops.automation.DAG")
def test_airflow_schedule_task(mock_dag, mock_python_operator, airflow_manager):
    mock_dag.return_value = MagicMock()
    mock_python_operator.return_value = MagicMock()

    # Add and schedule a task
    def sample_task(data):
        return f"Processing {data}"

    airflow_manager.add_task_to_airflow(task_name="sample_task", func=sample_task, data="sample_data")
    airflow_manager.schedule_airflow_task("sample_task")

    # Verify that the task is scheduled in the DAG
    task = airflow_manager.dag.get_task("sample_task")
    task.execute.assert_called_once()


# KubeflowAutomationManager Tests
@patch("gofast.mlops.automation.Client")
@patch("gofast.mlops.automation.dsl.ContainerOp")
def test_kubeflow_create_pipeline(mock_container_op, mock_client, kubeflow_manager):
    mock_client.return_value = MagicMock()
    mock_container_op.return_value = MagicMock()

    def sample_task(data):
        return f"Processing {data}"

    # Create a Kubeflow pipeline
    kubeflow_manager.create_kubeflow_pipeline(
        pipeline_name="test_pipeline",
        task_name="sample_task",
        func=sample_task,
        data="sample_data"
    )

    # Ensure pipeline was created and task was added
    mock_container_op.assert_called_once_with(
        name="sample_task",
        image="python:3.7",
        command=["python", "-c", sample_task],
        arguments=[{"data": "sample_data"}]
    )


# KafkaAutomationManager Tests
@patch("gofast.mlops.automation.KafkaConsumer")
def test_kafka_process_message(mock_kafka_consumer, kafka_manager):
    mock_kafka_consumer.return_value = MagicMock()

    def sample_task(data):
        return f"Processing {data}"

    kafka_manager.process_kafka_message(sample_task)
    # Simulate receiving a Kafka message
    message = MagicMock()
    message.value = b"test_message"
    kafka_manager.consumer.__iter__.return_value = [message]

    # Ensure the task is triggered upon receiving a Kafka message
    kafka_manager.process_kafka_message(sample_task)
    assert sample_task.called_once_with(b"test_message")


# RabbitMQAutomationManager Tests
@patch("gofast.mlops.automation.pika.BlockingConnection")
def test_rabbitmq_process_message(mock_pika_connection, rabbitmq_manager):
    mock_channel = MagicMock()
    mock_pika_connection.return_value.channel.return_value = mock_channel

    def sample_task(data):
        return f"Processing {data}"

    rabbitmq_manager.process_rabbitmq_message(sample_task)
    # Simulate receiving a RabbitMQ message
    method_frame = MagicMock()
    header_frame = MagicMock()
    body = b"test_message"
    rabbitmq_manager.channel.basic_consume.call_args[1]["on_message_callback"](
        mock_channel, method_frame, header_frame, body
    )

    # Ensure the task is triggered upon receiving a RabbitMQ message
    assert sample_task.called_once_with(b"test_message")

if __name__=='__main__': 
    pytest.main( [__file__])
