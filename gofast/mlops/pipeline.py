# -*- coding: utf-8 -*-
"""
Handle the automation of end-to-end workflows. Users can build modular 
pipelines for preprocessing, training, evaluation, and deployment.
"""

# Key Features:
# Pipeline creation and management
# Reusable pipeline steps (e.g., loading data, training, validation)
# Parallel pipeline execution
# Pipeline orchestration tools (Airflow, Prefect integration)


from typing import Callable, List, Dict, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor

# Optional logging for the pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineStep:
    """
    Represents a single step in the pipeline.
    Each step should be a callable function that takes inputs and returns outputs.
    """
    def __init__(self, name: str, func: Callable, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.func = func
        self.params = params or {}

    def execute(self, data: Any) -> Any:
        """
        Executes the pipeline step with the provided data.
        """
        logger.info(f"Executing step: {self.name}")
        return self.func(data, **self.params)

class Pipeline:
    """
    Represents a machine learning pipeline.
    This allows chaining of different steps for preprocessing, model training, validation, etc.
    """
    def __init__(self, steps: Optional[List[PipelineStep]] = None, parallel: bool = False):
        """
        Initialize the pipeline with an optional list of steps.
        
        Args:
            steps (List[PipelineStep]): List of pipeline steps to execute.
            parallel (bool): Whether to run steps in parallel (if possible).
        """
        self.steps = steps or []
        self.parallel = parallel

    def add_step(self, step: PipelineStep):
        """
        Adds a new step to the pipeline.
        
        Args:
            step (PipelineStep): A single step to add to the pipeline.
        """
        logger.info(f"Adding step: {step.name}")
        self.steps.append(step)

    def execute(self, initial_data: Any) -> Any:
        """
        Executes the pipeline from start to finish.
        
        Args:
            initial_data: The input data that is passed through the pipeline.
        
        Returns:
            The final output after all pipeline steps.
        """
        data = initial_data
        if self.parallel:
            # Parallel execution
            with ThreadPoolExecutor() as executor:
                future_to_step = {executor.submit(step.execute, data): step for step in self.steps}
                results = []
                for future in future_to_step:
                    step = future_to_step[future]
                    try:
                        result = future.result()
                        logger.info(f"Step {step.name} completed successfully.")
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Step {step.name} failed with error: {str(e)}")
                        raise e
            return results[-1] if results else None
        else:
            # Sequential execution
            for step in self.steps:
                try:
                    data = step.execute(data)
                    logger.info(f"Step {step.name} completed.")
                except Exception as e:
                    logger.error(f"Pipeline step {step.name} failed: {str(e)}")
                    raise e
            return data


def create_pipeline(steps: Optional[List[PipelineStep]] = None, parallel: bool = False) -> Pipeline:
    """
    Creates a machine learning pipeline by chaining together pipeline steps.
    
    Args:
        steps (List[PipelineStep]): List of steps (functions) to execute in the pipeline.
        parallel (bool): Whether to execute the steps in parallel.
        
    Returns:
        Pipeline: A Pipeline object that can be executed.
    """
    logger.info(f"Creating a pipeline with {'parallel' if parallel else 'sequential'} execution.")
    return Pipeline(steps, parallel)


# Example pipeline step functions
def preprocess_data(data: Any, scale: bool = True) -> Any:
    """A dummy function to simulate data preprocessing."""
    logger.info("Preprocessing data...")
    if scale:
        data = [x / max(data) for x in data]  # Example scaling operation
    return data

def train_model(data: Any, epochs: int = 10) -> str:
    """A dummy function to simulate model training."""
    logger.info(f"Training model for {epochs} epochs...")
    # Simulate model training
    return f"Model trained on data with {epochs} epochs."

def validate_model(model: str, validation_data: Any) -> str:
    """A dummy function to simulate model validation."""
    logger.info("Validating model...")
    return f"{model} validated on {len(validation_data)} validation samples."


# Example usage
if __name__ == "__main__":
    # Define pipeline steps
    preprocessing_step = PipelineStep(name="Preprocessing", func=preprocess_data, params={"scale": True})
    training_step = PipelineStep(name="Training", func=train_model, params={"epochs": 5})
    validation_step = PipelineStep(name="Validation", func=validate_model, params={"validation_data": [1, 2, 3]})

    # Create the pipeline
    pipeline = create_pipeline(steps=[preprocessing_step, training_step, validation_step], parallel=False)

    # Execute the pipeline
    final_output = pipeline.execute(initial_data=[10, 20, 30])
    print(final_output)
