# -*- coding: utf-8 -*-

# test_mlops_pipeline.py

import pytest
from gofast.mlops.pipeline import (
    PipelineStep,
    PipelineManager,
    PipelineOptimizer,
    ResourceMonitor,
    ResourceManager,
    create_pipeline,
    reconfigure_pipeline_on_the_fly,
    execute_step_conditionally,
    run_parallel_subpipelines,
    split_data_for_multitask_pipeline,
    rollback_to_previous_state,
    smart_retry_with_backoff,
    Pipeline
)

import logging
import random

# Configure logging for the tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample functions for pipeline steps
def sample_load_data(data):
    logger.info("Loading data...")
    return [1, 2, 3, 4, 5]

def sample_process_data(data):
    logger.info("Processing data...")
    return [x * 2 for x in data]

def sample_train_model(data, epochs=5):
    logger.info(f"Training model for {epochs} epochs...")
    # Simulate training
    return f"Model trained with {epochs} epochs."

def sample_validate_model(model_output):
    logger.info("Validating model...")
    # Simulate validation
    return "Validation accuracy: 95%"

def sample_evaluate_condition(data):
    # Condition: Proceed only if the sum of data is greater than 10
    return sum(data) > 10

def sample_fallback_step(data):
    logger.info("Executing fallback step...")
    return "Fallback executed."

def sample_error_step(data):
    logger.info("Executing error step...")
    raise ValueError("Intentional Error for Testing")

def sample_success_step(data):
    logger.info("Executing success step...")
    return "Success"

# Fixtures
@pytest.fixture
def pipeline_steps():
    load_step = PipelineStep(name="LoadData", func=sample_load_data)
    process_step = PipelineStep(name="ProcessData", func=sample_process_data, dependencies=["LoadData"])
    train_step = PipelineStep(name="TrainModel", func=sample_train_model, dependencies=["ProcessData"])
    validate_step = PipelineStep(name="ValidateModel", func=sample_validate_model, dependencies=["TrainModel"])
    return [load_step, process_step, train_step, validate_step]

@pytest.fixture
def pipeline_manager(pipeline_steps):
    manager = PipelineManager()
    for step in pipeline_steps:
        manager.add_step(step)
    return manager

@pytest.fixture
def optimizer(pipeline_manager):
    return PipelineOptimizer(pipeline_manager)

# Tests for PipelineStep
def test_pipeline_step_execution():
    step = PipelineStep(name="TestStep", func=sample_process_data)
    output = step.execute([1, 2, 3])
    assert output == [2, 4, 6], "PipelineStep execution failed."

# Tests for PipelineManager
def test_pipeline_manager_execution(pipeline_manager):
    final_output = pipeline_manager.execute(initial_data=None)
    expected_output = "Validation accuracy: 95%"
    assert final_output == expected_output, "PipelineManager execution failed."

def test_pipeline_manager_dependency_handling():
    manager = PipelineManager()
    step_a = PipelineStep(name="StepA", func=sample_load_data)
    step_b = PipelineStep(name="StepB", func=sample_process_data, dependencies=["StepA"])
    manager.add_step(step_b)
    manager.add_step(step_a)
    execution_order = manager._determine_execution_order()
    assert execution_order == ["StepA", "StepB"], "Dependency handling failed."

# Tests for PipelineOptimizer
def test_pipeline_optimizer_hyperparameter_tuning(optimizer):
    param_grid = {
        "epochs": [5, 10],
    }
    best_params = optimizer.tune_hyperparameters("TrainModel", param_grid, n_trials=2)
    assert "epochs" in best_params, "Hyperparameter tuning failed."

def test_pipeline_optimizer_resource_allocation(optimizer):
    optimizer.allocate_resources("TrainModel", {"CPU": 2, "Memory": 2 * 1024 ** 3})
    resources = optimizer.resource_manager.get_system_resources()
    assert resources["available_cpu_cores"] >= 0, "Resource allocation failed."

# Tests for ResourceMonitor
def test_resource_monitor():
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    monitor.record_usage()
    monitor.stop_monitoring()
    assert len(monitor.cpu_usage) > 0, "Resource monitoring failed."

# Tests for ResourceManager
def test_resource_manager_allocation():
    manager = ResourceManager()
    initial_cpu = manager.available_cpu_cores
    initial_memory = manager.available_memory
    cpu_allocated = manager.allocate_cpu(1)
    memory_allocated = manager.allocate_memory(1 * 1024 ** 3)
    assert cpu_allocated and memory_allocated, "Resource allocation failed."
    manager.release_resources(1, 1 * 1024 ** 3)
    assert manager.available_cpu_cores == initial_cpu, "Resource release failed."
    assert manager.available_memory == initial_memory, "Resource release failed."

# Tests for create_pipeline
def test_create_pipeline(pipeline_steps):
    pipeline = create_pipeline(steps=pipeline_steps, parallel=False)
    assert isinstance(pipeline, Pipeline), "Pipeline creation failed."

# Tests for reconfigure_pipeline_on_the_fly
def test_reconfigure_pipeline_on_the_fly(pipeline_manager):
    def new_train_model(data, epochs=3):
        logger.info(f"Reconfigured training model for {epochs} epochs...")
        return f"Model re-trained with {epochs} epochs."

    reconfigure_pipeline_on_the_fly(pipeline_manager, "TrainModel", new_train_model, {"epochs": 3})
    step = pipeline_manager.get_step("TrainModel")
    assert step.params["epochs"] == 3, "Pipeline reconfiguration failed."

# Tests for execute_step_conditionally
def test_execute_step_conditionally(pipeline_manager):
    # Add a conditional step
    conditional_step = PipelineStep(name="ConditionalStep", func=sample_success_step)
    pipeline_manager.add_step(conditional_step)
    # Execute conditionally
    execute_step_conditionally(
        pipeline_manager,
        "ConditionalStep",
        condition_func=lambda output: True,
        fallback_step=None,
    )
    status = pipeline_manager.step_metadata["ConditionalStep"]["status"]
    assert status == "success", "Conditional step execution failed."

# Tests for run_parallel_subpipelines
def test_run_parallel_subpipelines():
    # Define sub-pipelines
    sub_pipeline_steps_1 = [
        PipelineStep(name="SubLoadData1", func=sample_load_data),
        PipelineStep(name="SubProcessData1", func=sample_process_data, dependencies=["SubLoadData1"]),
    ]
    sub_pipeline_steps_2 = [
        PipelineStep(name="SubLoadData2", func=sample_load_data),
        PipelineStep(name="SubProcessData2", func=sample_process_data, dependencies=["SubLoadData2"]),
    ]
    pipeline_manager = PipelineManager()
    run_parallel_subpipelines(pipeline_manager, [sub_pipeline_steps_1, sub_pipeline_steps_2])
    # Since we don't have outputs, we just ensure no exceptions occurred

# Tests for split_data_for_multitask_pipeline
def test_split_data_for_multitask_pipeline(pipeline_manager):
    data = list(range(100))
    split_ratios = [0.5, 0.5]
    tasks = ["TrainModel", "ValidateModel"]
    split_data_for_multitask_pipeline(data, split_ratios, tasks, pipeline_manager)
    train_step = pipeline_manager.get_step("TrainModel")
    validate_step = pipeline_manager.get_step("ValidateModel")
    assert len(train_step.params["data"]) == 50, "Data splitting failed for TrainModel."
    assert len(validate_step.params["data"]) == 50, "Data splitting failed for ValidateModel."

# Tests for rollback_to_previous_state
def test_rollback_to_previous_state(pipeline_manager):
    # Simulate a failure
    pipeline_manager.step_metadata["ProcessData"]["status"] = "failed"
    # Rollback to LoadData
    rollback_to_previous_state(pipeline_manager, "LoadData")
    status = pipeline_manager.step_metadata["ProcessData"]["status"]
    assert status == "rolled_back", "Rollback mechanism failed."

# Tests for smart_retry_with_backoff
def test_smart_retry_with_backoff(pipeline_manager):
    # Add a step that fails
    error_step = PipelineStep(name="ErrorStep", func=sample_error_step)
    pipeline_manager.add_step(error_step)
    # Attempt smart retry
    smart_retry_with_backoff(pipeline_manager, "ErrorStep", max_retries=2, initial_delay=0.1)
    status = pipeline_manager.step_metadata["ErrorStep"]["status"]
    assert status == "failed", "Smart retry did not handle failures correctly."


# Run the test suite
if __name__ == "__main__":
    pytest.main([__file__])

