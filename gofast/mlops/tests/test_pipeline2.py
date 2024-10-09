# -*- coding: utf-8 -*-
import pytest
from gofast.mlops.pipeline import (
    PipelineStep, PipelineManager, PipelineOptimizer, ResourceMonitor, ResourceManager,
    create_pipeline, reconfigure_pipeline_on_the_fly, execute_step_conditionally, 
    run_parallel_subpipelines, split_data_for_multitask_pipeline, rollback_to_previous_state, 
    smart_retry_with_backoff
)

# Sample functions for testing
def dummy_preprocess(data, scale=True):
    return [x / max(data) for x in data] if scale else data

def dummy_train(data, epochs=10):
    return f"Model trained with {epochs} epochs"

def dummy_validation(model, validation_data):
    return f"Validated with {len(validation_data)} samples"

def dummy_condition(data):
    return sum(data) > 10

# Test the PipelineStep class
def test_pipeline_step():
    step = PipelineStep(name="preprocessing", func=dummy_preprocess, params={"scale": True})
    assert step.name == "preprocessing"
    output = step.execute([10, 20, 30])
    assert output == [0.3333333333333333, 0.6666666666666666, 1.0]

# Test the PipelineManager class
def test_pipeline_manager():
    pipeline_manager = PipelineManager()
    
    step1 = PipelineStep(name="Preprocessing", func=dummy_preprocess, params={"scale": True})
    step2 = PipelineStep(name="Training", func=dummy_train, params={"epochs": 5})

    pipeline_manager.add_step(step1)
    pipeline_manager.add_step(step2)

    assert len(pipeline_manager.steps) == 2
    output = pipeline_manager.execute([10, 20, 30])
    assert output == "Model trained with 5 epochs"

def test_pipeline_manager_with_retry():
    pipeline_manager = PipelineManager(retry_failed_steps=True)
    
    def failing_step(data):
        raise ValueError("Intentional Failure")

    step1 = PipelineStep(name="Preprocessing", func=dummy_preprocess, params={"scale": True})
    step2 = PipelineStep(name="Failing Step", func=failing_step)
    
    pipeline_manager.add_step(step1)
    pipeline_manager.add_step(step2)

    with pytest.raises(ValueError):
        pipeline_manager.execute([10, 20, 30])

# Test the ResourceManager class
def test_resource_manager():
    resource_manager = ResourceManager()
    assert resource_manager.available_cpu_cores > 0
    assert resource_manager.available_memory > 0

    assert resource_manager.allocate_cpu(1) is True
    assert resource_manager.allocate_memory(1 * 1024**3) is True

    resource_manager.release_resources(cpu_cores=1, memory=1 * 1024**3)

# Test the PipelineOptimizer class
def test_pipeline_optimizer():
    pipeline_manager = PipelineManager()
    optimizer = PipelineOptimizer(pipeline_manager)

    step = PipelineStep(name="Training", func=dummy_train, params={"epochs": 5})
    pipeline_manager.add_step(step)

    best_params = optimizer.tune_hyperparameters(step_name="Training", param_grid={"epochs": [5, 10, 15]}, n_trials=3)
    assert best_params in [{"epochs": 5}, {"epochs": 10}, {"epochs": 15}]

# Test create_pipeline function
def test_create_pipeline():
    step1 = PipelineStep(name="Preprocessing", func=dummy_preprocess, params={"scale": True})
    step2 = PipelineStep(name="Training", func=dummy_train, params={"epochs": 5})

    pipeline = create_pipeline(steps=[step1, step2])
    assert isinstance(pipeline, PipelineManager)

    output = pipeline.execute([10, 20, 30])
    assert output == "Model trained with 5 epochs"

# Test reconfigure_pipeline_on_the_fly function
def test_reconfigure_pipeline_on_the_fly():
    pipeline_manager = PipelineManager()
    step = PipelineStep(name="Preprocessing", func=dummy_preprocess, params={"scale": True})
    pipeline_manager.add_step(step)

    def new_preprocess(data, scale=False):
        return data

    reconfigure_pipeline_on_the_fly(pipeline_manager, "Preprocessing", new_step_func=new_preprocess)
    assert pipeline_manager.get_step("Preprocessing").func == new_preprocess

# Test execute_step_conditionally function
def test_execute_step_conditionally():
    pipeline_manager = PipelineManager()
    
    step1 = PipelineStep(name="Preprocessing", func=dummy_preprocess, params={"scale": True})
    step2 = PipelineStep(name="Training", func=dummy_train, params={"epochs": 5})
    
    pipeline_manager.add_step(step1)
    pipeline_manager.add_step(step2)
    
    execute_step_conditionally(pipeline_manager, "Training", condition_func=dummy_condition)
    
    assert pipeline_manager.get_step("Training").name == "Training"

# Test run_parallel_subpipelines function
def test_run_parallel_subpipelines():
    sub_pipeline_1 = [
        PipelineStep(name="Preprocessing", func=dummy_preprocess, params={"scale": True}),
        PipelineStep(name="Training", func=dummy_train, params={"epochs": 5}),
    ]
    
    sub_pipeline_2 = [
        PipelineStep(name="Validation", func=dummy_validation, params={"validation_data": [1, 2, 3]}),
    ]
    
    pipeline_manager = PipelineManager()
    run_parallel_subpipelines(pipeline_manager, [sub_pipeline_1, sub_pipeline_2])

# Test split_data_for_multitask_pipeline function
def test_split_data_for_multitask_pipeline():
    data = [i for i in range(100)]
    tasks = ["Task1", "Task2", "Task3"]
    split_ratios = [0.5, 0.3, 0.2]
    
    pipeline_manager = PipelineManager()
    step1 = PipelineStep(name="Task1", func=dummy_preprocess)
    step2 = PipelineStep(name="Task2", func=dummy_preprocess)
    step3 = PipelineStep(name="Task3", func=dummy_preprocess)
    
    pipeline_manager.add_step(step1)
    pipeline_manager.add_step(step2)
    pipeline_manager.add_step(step3)
    
    split_data_for_multitask_pipeline(data, split_ratios, tasks, pipeline_manager)
    
    assert len(pipeline_manager.get_step("Task1").params["data"]) == 50
    assert len(pipeline_manager.get_step("Task2").params["data"]) == 30
    assert len(pipeline_manager.get_step("Task3").params["data"]) == 20

# Test rollback_to_previous_state function
def test_rollback_to_previous_state():
    pipeline_manager = PipelineManager()
    
    step1 = PipelineStep(name="Preprocessing", func=dummy_preprocess, params={"scale": True})
    step2 = PipelineStep(name="Training", func=dummy_train, params={"epochs": 5})
    
    pipeline_manager.add_step(step1)
    pipeline_manager.add_step(step2)
    
    pipeline_manager.execute([10, 20, 30])
    
    rollback_to_previous_state(pipeline_manager, rollback_step="Preprocessing")
    assert pipeline_manager.step_metadata["Training"]["status"] == "rolled_back"

# Test smart_retry_with_backoff function
def test_smart_retry_with_backoff():
    pipeline_manager = PipelineManager()
    
    def failing_step(data):
        raise ValueError("Intentional Failure")
    
    step = PipelineStep(name="Failing Step", func=failing_step)
    pipeline_manager.add_step(step)
    
    with pytest.raises(ValueError):
        smart_retry_with_backoff(pipeline_manager, step_name="Failing Step", max_retries=2)

if __name__=='__main__': 
    pytest.main( [__file__])