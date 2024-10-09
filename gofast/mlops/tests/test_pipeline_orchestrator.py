# -*- coding: utf-8 -*-

import pytest
from gofast.mlops.pipeline import (
    PipelineStep,
    PipelineManager,
    AirflowOrchestrator,
    PrefectOrchestrator,
)
from unittest.mock import patch, MagicMock

# Sample function for pipeline steps
def sample_task_func(data):
    return f"Processed {data}"

# Fixture to create a sample pipeline manager with steps
@pytest.fixture
def pipeline_manager():
    manager = PipelineManager()
    step1 = PipelineStep(name="Step1", func=sample_task_func, params={"data": "Step 1 data"})
    step2 = PipelineStep(name="Step2", func=sample_task_func, params={"data": "Step 2 data"}, dependencies=["Step1"])
    manager.add_step(step1)
    manager.add_step(step2)
    return manager

# AirflowOrchestrator Tests
@pytest.fixture
def airflow_orchestrator(pipeline_manager):
    return AirflowOrchestrator(pipeline_manager)

@patch("gofast.mlops.pipeline.airflow.DAG")
@patch("gofast.mlops.pipeline.airflow.operators.python_operator.PythonOperator")
def test_airflow_orchestrator_create_workflow(mock_python_operator, mock_dag, airflow_orchestrator):
    mock_dag.return_value = MagicMock()
    mock_python_operator.return_value = MagicMock()
    
    # Call create_workflow
    airflow_orchestrator.create_workflow(dag_id="test_dag", start_date="2024-01-01")
    
    # Check if DAG and PythonOperator were called
    mock_dag.assert_called_once_with(
        dag_id="test_dag",
        default_args=pytest.ANY,
        description="An ML pipeline orchestrated by Airflow",
        schedule_interval="@daily",
    )
    assert mock_python_operator.call_count == 2  # Two pipeline steps

def test_airflow_orchestrator_schedule_pipeline(airflow_orchestrator):
    # Mocking the internal dag object after workflow creation
    airflow_orchestrator.dag = MagicMock()

    # Call schedule_pipeline
    airflow_orchestrator.schedule_pipeline(schedule_interval="@weekly")
    
    # Check if the schedule interval was set correctly
    assert airflow_orchestrator.dag.schedule_interval == "@weekly"

def test_airflow_orchestrator_monitor_pipeline(airflow_orchestrator):
    # Just checking the logging and flow, no actual monitoring done here
    with patch("gofast.mlops.pipeline.logging") as mock_logging:
        airflow_orchestrator.monitor_pipeline()
        mock_logging.getLogger().info.assert_called_with(
            "Monitoring Airflow DAG execution through the Airflow web UI."
        )

# PrefectOrchestrator Tests
@pytest.fixture
def prefect_orchestrator(pipeline_manager):
    return (pipeline_manager)

@patch("gofast.mlops.pipeline.prefect.Flow")
@patch("gofast.mlops.pipeline.prefect.task")
def test_prefect_orchestrator_create_workflow(mock_task, mock_flow, prefect_orchestrator):
    mock_flow.return_value = MagicMock()
    mock_task.side_effect = lambda f, **kwargs: f(**kwargs)  # Just call the original function
    
    # Call create_workflow
    prefect_orchestrator.create_workflow(flow_name="test_flow")
    
    # Check if Flow and task were called
    mock_flow.assert_called_once_with("test_flow")
    assert mock_task.call_count == 2  # Two pipeline steps

def test_prefect_orchestrator_schedule_pipeline(prefect_orchestrator):
    # Mock the flow object and the prefect.schedule.Schedule
    prefect_orchestrator.flow = MagicMock()
    with patch("gofast.mlops.pipeline.prefect.schedules.Schedule") as mock_schedule:
        with patch("gofast.mlops.pipeline.prefect.schedules.clocks.CronClock") as mock_cron_clock:
            mock_cron_clock.return_value = MagicMock()

            # Call schedule_pipeline
            prefect_orchestrator.schedule_pipeline(schedule_interval="0 12 * * *")

            # Check if the schedule was set correctly
            assert prefect_orchestrator.flow.schedule is not None
            mock_cron_clock.assert_called_once_with("0 12 * * *")

def test_prefect_orchestrator_monitor_pipeline(prefect_orchestrator):
    # Just checking the logging and flow, no actual monitoring done here
    with patch("gofast.mlops.pipeline.logging") as mock_logging:
        prefect_orchestrator.monitor_pipeline()
        mock_logging.getLogger().info.assert_called_with(
            "Monitoring Prefect Flow execution through the Prefect UI."
        )
        
if __name__=='__main__': 
    pytest.main( [__file__])
