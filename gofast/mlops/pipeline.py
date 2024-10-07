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


import psutil
import time
from typing import Callable, List, Dict, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

# Optional logging for the pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "PipelineStep", 
    "PipelineManager", 
    "PipelineOptimizer", 
    "ResourceMonitor", 
    "ResourceManager",
    "create_pipeline", 
    "reconfigure_pipeline_on_the_fly", 
    "execute_step_conditionally", 
    "run_parallel_subpipelines", 
    "split_data_for_multitask_pipeline", 
    "rollback_to_previous_state", 
    "smart_retry_with_backoff",
]


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


class PipelineStep:
    """
    A sophisticated pipeline step that supports flexible configurations and dependencies.
    Each step can be defined with custom parameters, input-output relationships, and metadata tracking.
    """
    def __init__(self, name: str, func: Callable, params: Optional[Dict[str, Any]] = None, dependencies: Optional[List[str]] = None):
        self.name = name
        self.func = func
        self.params = params or {}
        self.dependencies = dependencies or []
        self.outputs = None

    def execute(self, data: Any) -> Any:
        """Executes the pipeline step and returns the output."""
        logger.info(f"Executing step: {self.name}")
        self.outputs = self.func(data, **self.params)
        return self.outputs

    def get_dependencies(self) -> List[str]:
        """Returns the list of dependencies for this step."""
        return self.dependencies


class PipelineManager:
    """
    Manages the creation, execution, and tracking of pipelines.
    Offers support for step dependencies, error recovery, and metadata tracking.
    """
    def __init__(self, retry_failed_steps: bool = False):
        self.steps: OrderedDict[str, PipelineStep] = OrderedDict()
        self.step_metadata: Dict[str, Any] = {}
        self.retry_failed_steps = retry_failed_steps
        self.failed_steps: List[str] = []

    def add_step(self, step: PipelineStep):
        """
        Adds a step to the pipeline with metadata tracking.
        Steps are stored in an ordered fashion to maintain execution sequence.
        """
        if step.name in self.steps:
            raise ValueError(f"Step with name {step.name} already exists.")
        logger.info(f"Adding step: {step.name} with dependencies: {step.dependencies}")
        self.steps[step.name] = step
        self.step_metadata[step.name] = {"status": "pending", "output": None}

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Retrieves a step by name."""
        return self.steps.get(name, None)

    def execute(self, initial_data: Any) -> Any:
        """
        Executes the pipeline in the correct order, respecting dependencies.
        Allows retrying failed steps if enabled.
        
        Args:
            initial_data: The input data to pass through the first pipeline step.
        
        Returns:
            Final output after executing the pipeline.
        """
        execution_sequence = self._determine_execution_order()
        data = initial_data

        for step_name in execution_sequence:
            step = self.steps[step_name]
            # Execute the step only if all dependencies have successfully executed
            if self._can_execute(step):
                try:
                    logger.info(f"Executing {step_name} with input: {data}")
                    output = step.execute(data)
                    self._update_step_metadata(step_name, "success", output)
                    data = output
                except Exception as e:
                    logger.error(f"Step {step_name} failed with error: {str(e)}")
                    self._update_step_metadata(step_name, "failed", None)
                    if not self.retry_failed_steps:
                        raise e
                    else:
                        logger.info(f"Retrying step {step_name} as retry is enabled.")
                        self.failed_steps.append(step_name)
            else:
                logger.info(f"Skipping {step_name} due to unmet dependencies.")
        return data

    def _determine_execution_order(self) -> List[str]:
        """
        Determines the order in which the steps should be executed,
        based on the dependencies defined for each step.
        """
        execution_order = []
        for step_name, step in self.steps.items():
            if not step.get_dependencies():
                execution_order.append(step_name)
            else:
                for dep in step.get_dependencies():
                    if dep not in execution_order:
                        execution_order.append(dep)
                execution_order.append(step_name)
        logger.info(f"Determined execution order: {execution_order}")
        return execution_order

    def _can_execute(self, step: PipelineStep) -> bool:
        """
        Checks if a pipeline step can be executed by verifying that all dependencies have succeeded.
        """
        for dep in step.get_dependencies():
            if self.step_metadata[dep]["status"] != "success":
                return False
        return True

    def _update_step_metadata(self, step_name: str, status: str, output: Any):
        """
        Updates the metadata for a specific step in the pipeline.
        """
        self.step_metadata[step_name]["status"] = status
        self.step_metadata[step_name]["output"] = output

    def retry_failed(self):
        """
        Retries failed steps if the retry_failed_steps flag is set.
        """
        if not self.failed_steps:
            logger.info("No failed steps to retry.")
            return

        logger.info("Retrying failed steps...")
        for step_name in self.failed_steps:
            step = self.steps[step_name]
            try:
                output = step.execute(self.step_metadata[step.get_dependencies()[0]]["output"])  # Reuse previous step output
                self._update_step_metadata(step_name, "success", output)
            except Exception as e:
                logger.error(f"Retry failed for step {step_name} with error: {str(e)}")

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata for the pipeline, including step statuses and outputs."""
        return self.step_metadata

class ResourceManager:
    """
    ResourceManager class handles the actual allocation of resources for the pipeline steps.
    This includes checking available resources and assigning them to specific steps.
    """
    def __init__(self):
        self.available_cpu_cores = psutil.cpu_count(logical=False)  # Number of physical cores
        self.available_memory = psutil.virtual_memory().total  # Total system memory

    def allocate_cpu(self, requested_cores: int) -> bool:
        """Allocates CPU cores to a step, returns True if successful, False otherwise."""
        if requested_cores <= self.available_cpu_cores:
            logger.info(f"Allocated {requested_cores} CPU cores.")
            self.available_cpu_cores -= requested_cores
            return True
        else:
            logger.warning(f"Not enough CPU cores. Requested: {requested_cores}, Available: {self.available_cpu_cores}")
            return False

    def allocate_memory(self, requested_memory: int) -> bool:
        """Allocates memory to a step, returns True if successful, False otherwise."""
        if requested_memory <= self.available_memory:
            logger.info(f"Allocated {requested_memory / (1024**3):.2f} GB memory.")
            self.available_memory -= requested_memory
            return True
        else:
            logger.warning(f"Not enough memory. Requested: {requested_memory / (1024**3):.2f} GB, Available: {self.available_memory / (1024**3):.2f} GB")
            return False

    def release_resources(self, cpu_cores: int, memory: int):
        """Releases resources after a pipeline step has been executed."""
        self.available_cpu_cores += cpu_cores
        self.available_memory += memory
        logger.info(f"Released {cpu_cores} CPU cores and {memory / (1024**3):.2f} GB memory.")

    def get_system_resources(self):
        """Returns a dictionary of the current available system resources."""
        return {
            "available_cpu_cores": self.available_cpu_cores,
            "available_memory_gb": self.available_memory / (1024**3),  # Convert to GB
        }

class ResourceMonitor:
    """
    ResourceMonitor class to monitor system resource usage during the execution of pipeline steps.
    It tracks CPU and memory usage.
    """
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
    
    def start_monitoring(self):
        """Starts monitoring system resources."""
        logger.info("Starting resource monitoring...")
        self.cpu_usage.clear()
        self.memory_usage.clear()

    def stop_monitoring(self):
        """Stops monitoring and logs the final resource usage stats."""
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        logger.info(f"Average CPU Usage: {avg_cpu}%")
        logger.info(f"Average Memory Usage: {avg_memory / (1024 ** 3):.2f} GB")

    def record_usage(self):
        """Records current CPU and memory usage."""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().used
        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)
        logger.info(f"Current CPU Usage: {cpu}%, Memory Usage: {memory / (1024 ** 3):.2f} GB")

class PipelineOptimizer:
    def __init__(self, pipeline_manager: 'PipelineManager'):
        self.pipeline_manager = pipeline_manager
        self.resource_manager = ResourceManager()  # Initialize resource manager
        self.resource_monitor = ResourceMonitor()  # Initialize resource monitor
    
    def allocate_resources(self, step_name: str, resources: Dict[str, Any]) -> None:
        """
        Allocates resources (e.g., memory, CPU) to a specific pipeline step.
        """
        logger.info(f"Allocating resources for step: {step_name} -> {resources}")
        step = self.pipeline_manager.get_step(step_name)

        cpu_allocated = False
        memory_allocated = False

        if "CPU" in resources:
            requested_cores = resources["CPU"]
            cpu_allocated = self.resource_manager.allocate_cpu(requested_cores)

        if "Memory" in resources:
            requested_memory = resources["Memory"]
            memory_allocated = self.resource_manager.allocate_memory(requested_memory)

        if cpu_allocated and memory_allocated:
            logger.info(f"Resources allocated successfully for step: {step_name}.")
        else:
            logger.error(f"Failed to allocate resources for step: {step_name}. Check resource availability.")

    def release_resources_after_step(self, step_name: str, resources: Dict[str, Any]) -> None:
        """
        Releases allocated resources after a pipeline step is completed.
        """
        logger.info(f"Releasing resources for step: {step_name} -> {resources}")

        if "CPU" in resources:
            self.resource_manager.release_resources(resources["CPU"], 0)

        if "Memory" in resources:
            self.resource_manager.release_resources(0, resources["Memory"])

    def monitor_resources_for_step(self, step_name: str, duration: int = 5) -> None:
        """
        Monitors the resource usage (CPU and Memory) for a given step over a specified duration.
        
        Args:
            step_name (str): The name of the step to monitor.
            duration (int): The duration to monitor resources (in seconds).
        """
        logger.info(f"Monitoring resources for step: {step_name} for {duration} seconds.")
        self.resource_monitor.start_monitoring()

        for _ in range(duration):
            self.resource_monitor.record_usage()
            time.sleep(1)

        self.resource_monitor.stop_monitoring()

    def tune_resources_based_on_usage(self, step_name: str, threshold: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tunes resource allocation for a specific step based on observed resource usage.
        
        Args:
            step_name (str): The name of the step to tune.
            threshold (Dict[str, Any]): Threshold values for resource usage.
            Example: {"CPU": 75, "Memory": 4 * 1024**3}  # 75% CPU and 4 GB memory threshold
        
        Returns:
            Dict[str, Any]: New recommended resources based on usage.
        """
        logger.info(f"Tuning resources for step: {step_name} based on usage.")

        avg_cpu = sum(self.resource_monitor.cpu_usage) / len(self.resource_monitor.cpu_usage)
        avg_memory = sum(self.resource_monitor.memory_usage) / len(self.resource_monitor.memory_usage)

        recommended_resources = {}

        # Adjust CPU allocation if average usage exceeds threshold
        if avg_cpu > threshold.get("CPU", 100):
            recommended_cores = min(self.resource_manager.available_cpu_cores + 1, psutil.cpu_count(logical=False))
            logger.info(f"Increasing CPU allocation to {recommended_cores} cores.")
            recommended_resources["CPU"] = recommended_cores
        else:
            recommended_resources["CPU"] = threshold.get("CPU", 1)

        # Adjust memory allocation if average usage exceeds threshold
        if avg_memory > threshold.get("Memory", self.resource_manager.available_memory):
            recommended_memory = min(self.resource_manager.available_memory + (1 * 1024 ** 3), psutil.virtual_memory().total)
            logger.info(f"Increasing memory allocation to {recommended_memory / (1024 ** 3):.2f} GB.")
            recommended_resources["Memory"] = recommended_memory
        else:
            recommended_resources["Memory"] = threshold.get("Memory", 1 * 1024 ** 3)

        return recommended_resources

def reconfigure_pipeline_on_the_fly(pipeline_manager: 'PipelineManager', step_name: str, new_step_func: Callable, new_params: Optional[Dict[str, Any]] = None):
    """
    Dynamically reconfigures a specific step in the pipeline while it is being executed.
    Allows users to swap out or adjust the function and parameters during pipeline execution.
    
    Args:
        pipeline_manager (PipelineManager): The pipeline manager containing the steps.
        step_name (str): The name of the step to reconfigure.
        new_step_func (Callable): The new function to replace the current step's function.
        new_params (Optional[Dict[str, Any]]): New parameters to use for the updated step.
        
    Returns:
        None
    """
    logger.info(f"Reconfiguring step: {step_name} with new function and parameters.")

    # Retrieve the step and update its function and parameters
    step = pipeline_manager.get_step(step_name)
    if step is not None:
        step.func = new_step_func
        if new_params:
            step.params.update(new_params)
        logger.info(f"Step {step_name} has been reconfigured with new function and parameters.")
    else:
        logger.error(f"Step {step_name} does not exist in the pipeline.")

def execute_step_conditionally(pipeline_manager: 'PipelineManager', step_name: str, condition_func: Callable[[Any], bool], fallback_step: Optional[str] = None):
    """
    Executes a specific pipeline step conditionally based on the result of a previous step.
    If the condition is not met, an alternative (fallback) step can be executed instead.
    
    Args:
        pipeline_manager (PipelineManager): The pipeline manager containing the steps.
        step_name (str): The name of the step to conditionally execute.
        condition_func (Callable[[Any], bool]): A function that takes the previous step's output and returns a boolean to decide execution.
        fallback_step (Optional[str]): The name of a fallback step to execute if the condition is not met.
    
    Returns:
        None
    """
    logger.info(f"Checking condition for step: {step_name}")

    # Get the output of the last executed step
    previous_step_output = pipeline_manager.step_metadata[pipeline_manager.steps[-1].name]["output"]
    
    # Execute step if condition is met
    if condition_func(previous_step_output):
        logger.info(f"Condition met. Executing step: {step_name}")
        pipeline_manager.execute_step(step_name)
    else:
        logger.info(f"Condition not met for step: {step_name}.")
        if fallback_step:
            logger.info(f"Executing fallback step: {fallback_step}")
            pipeline_manager.execute_step(fallback_step)
        else:
            logger.info(f"No fallback step specified. Skipping step: {step_name}")


def run_parallel_subpipelines(pipeline_manager: 'PipelineManager', sub_pipelines: List[List[PipelineStep]]):
    """
    Executes multiple sub-pipelines in parallel, allowing parts of the pipeline to be parallelized.
    
    Args:
        pipeline_manager (PipelineManager): The main pipeline manager.
        sub_pipelines (List[List[PipelineStep]]): A list of sub-pipelines, each represented as a list of PipelineStep objects.
        
    Returns:
        None
    """
    logger.info("Starting parallel execution of sub-pipelines...")
    
    # Create a thread pool to run sub-pipelines concurrently
    with ThreadPoolExecutor() as executor:
        futures = []
        for sub_pipeline_steps in sub_pipelines:
            sub_pipeline_manager = PipelineManager()
            for step in sub_pipeline_steps:
                sub_pipeline_manager.add_step(step)
            # Execute the sub-pipeline asynchronously
            futures.append(executor.submit(sub_pipeline_manager.execute, initial_data=None))
        
        # Wait for all sub-pipelines to complete
        for future in futures:
            try:
                future.result()  # Raise any exceptions that occurred during execution
                logger.info("Sub-pipeline completed successfully.")
            except Exception as e:
                logger.error(f"Error in sub-pipeline: {str(e)}")

def split_data_for_multitask_pipeline(data: Any, split_ratios: List[float], tasks: List[str], pipeline_manager: 'PipelineManager'):
    """
    Splits the input data into multiple parts for multitask pipelines and assigns them to different tasks.
    
    Args:
        data (Any): The input data to be split.
        split_ratios (List[float]): List of ratios to split the data (e.g., [0.5, 0.3, 0.2] for 50%, 30%, 20% split).
        tasks (List[str]): List of task names, corresponding to the split data.
        pipeline_manager (PipelineManager): The pipeline manager to handle the tasks.
        
    Returns:
        None
    """
    assert len(split_ratios) == len(tasks), "Number of split ratios must match the number of tasks."
    
    # Split the data based on the ratios
    total_data_points = len(data)
    split_indices = [int(total_data_points * ratio) for ratio in split_ratios]
    split_data = []
    start_idx = 0
    
    for idx in split_indices:
        end_idx = start_idx + idx
        split_data.append(data[start_idx:end_idx])
        start_idx = end_idx

    # Assign split data to the corresponding task's pipeline step
    for i, task in enumerate(tasks):
        step = pipeline_manager.get_step(task)
        if step:
            step.params.update({"data": split_data[i]})
            logger.info(f"Assigned split data to task: {task}")
        else:
            logger.error(f"Task {task} not found in the pipeline.")

def rollback_to_previous_state(pipeline_manager: 'PipelineManager', rollback_step: str):
    """
    Rolls back the pipeline to a previous stable state if a failure occurs.
    
    Args:
        pipeline_manager (PipelineManager): The pipeline manager handling the steps.
        rollback_step (str): The name of the step to roll back to.
        
    Returns:
        None
    """
    logger.info(f"Rolling back to step: {rollback_step}")

    # Ensure that the rollback step exists
    step = pipeline_manager.get_step(rollback_step)
    if step:
        # Reset the pipeline state to the specified step
        for step_name in list(pipeline_manager.steps.keys()):
            if step_name == rollback_step:
                break
            pipeline_manager.steps[step_name].outputs = None
            pipeline_manager.step_metadata[step_name]["status"] = "rolled_back"
            logger.info(f"Rolled back step: {step_name}")
    else:
        logger.error(f"Step {rollback_step} does not exist in the pipeline.")



def smart_retry_with_backoff(pipeline_manager: 'PipelineManager', step_name: str, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Retries a failed step using exponential backoff if it encounters errors during execution.
    
    Args:
        pipeline_manager (PipelineManager): The pipeline manager containing the steps.
        step_name (str): The name of the step to retry.
        max_retries (int): Maximum number of retries allowed.
        initial_delay (float): Initial delay before retrying, with backoff applied for subsequent retries.
        
    Returns:
        None
    """
    step = pipeline_manager.get_step(step_name)
    if not step:
        logger.error(f"Step {step_name} not found in the pipeline.")
        return
    
    retry_count = 0
    delay = initial_delay
    
    while retry_count < max_retries:
        try:
            logger.info(f"Retrying step {step_name} (Attempt {retry_count + 1})...")
            pipeline_manager.execute_step(step_name)
            logger.info(f"Step {step_name} executed successfully on retry {retry_count + 1}.")
            break
        except Exception as e:
            logger.error(f"Step {step_name} failed on retry {retry_count + 1}. Error: {str(e)}")
            retry_count += 1
            time.sleep(delay)
            delay *= 2  # Exponential backoff

    if retry_count == max_retries:
        logger.error(f"Step {step_name} failed after {max_retries} retries.")


class PipelineOrchestrator:
    """
    Base class for pipeline orchestration integration with tools like Airflow and Prefect.
    This class defines the basic interface for creating, scheduling, and monitoring pipelines.
    """
    
    def __init__(self, pipeline_manager: 'PipelineManager'):
        self.pipeline_manager = pipeline_manager

    def create_workflow(self):
        """
        Abstract method to create a workflow (DAG for Airflow, Flow for Prefect).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def schedule_pipeline(self, schedule_interval: str):
        """
        Abstract method to schedule a pipeline. Must be implemented by subclasses.
        Args:
            schedule_interval (str): The scheduling interval (e.g., cron expression).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def monitor_pipeline(self):
        """
        Abstract method to monitor the status of pipeline execution. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class AirflowOrchestrator(PipelineOrchestrator):
    """
    AirflowOrchestrator integrates the pipeline with Airflow, allowing for DAG creation and scheduling.
    """

    def __init__(self, pipeline_manager: 'PipelineManager'):
        super().__init__(pipeline_manager)
        self.dag = None

    def create_workflow(self, dag_id: str, start_date: Any, schedule_interval: Optional[str] = "@daily"):
        """
        Creates an Airflow DAG that runs the pipeline.
        
        Args:
            dag_id (str): The unique identifier for the DAG.
            start_date (Any): The start date of the DAG execution.
            schedule_interval (Optional[str]): The scheduling interval for the DAG (e.g., cron expression).
        """
        from airflow import DAG
        from airflow.operators.python_operator import PythonOperator
        from datetime import timedelta

        logger.info(f"Creating Airflow DAG with ID: {dag_id}")

        # Define the DAG
        default_args = {
            'owner': 'airflow',
            'depends_on_past': False,
            'start_date': start_date,
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }

        dag = DAG(
            dag_id=dag_id,
            default_args=default_args,
            description='An ML pipeline orchestrated by Airflow',
            schedule_interval=schedule_interval,
        )

        # Add pipeline steps to the DAG
        previous_task = None
        for step_name, step in self.pipeline_manager.steps.items():
            task = PythonOperator(
                task_id=step_name,
                python_callable=step.func,
                op_kwargs=step.params,
                dag=dag,
            )
            if previous_task:
                previous_task >> task
            previous_task = task

        self.dag = dag
        logger.info(f"DAG {dag_id} created successfully.")

    def schedule_pipeline(self, schedule_interval: str):
        """
        Schedules the pipeline using a cron-like schedule.
        
        Args:
            schedule_interval (str): Cron expression or Airflow predefined schedule interval (e.g., "@daily").
        """
        if self.dag is None:
            logger.error("DAG is not created. Run create_workflow first.")
            return

        logger.info(f"Scheduling DAG with interval: {schedule_interval}")
        self.dag.schedule_interval = schedule_interval

    def monitor_pipeline(self):
        """
        Monitors the pipeline execution via Airflow's monitoring UI.
        This is primarily handled by Airflow's web interface, but we log the status here.
        """
        logger.info("Monitoring Airflow DAG execution through the Airflow web UI.")

class PrefectOrchestrator(PipelineOrchestrator):
    """
    PrefectOrchestrator integrates the pipeline with Prefect, allowing for Flow creation and scheduling.
    """

    def __init__(self, pipeline_manager: 'PipelineManager'):
        super().__init__(pipeline_manager)
        self.flow = None

    def create_workflow(self, flow_name: str):
        """
        Creates a Prefect Flow that runs the pipeline.
        
        Args:
            flow_name (str): The name of the Prefect flow.
        """
        from prefect import Flow, task

        logger.info(f"Creating Prefect Flow with name: {flow_name}")

        # Create Prefect tasks for each pipeline step
        @task
        def task_wrapper(func: Callable, **kwargs):
            return func(**kwargs)

        with Flow(flow_name) as flow:
            previous_task = None
            for step_name, step in self.pipeline_manager.steps.items():
                current_task = task_wrapper(step.func, **step.params)
                if previous_task:
                    previous_task.set_downstream(current_task)
                previous_task = current_task

        self.flow = flow
        logger.info(f"Prefect Flow {flow_name} created successfully.")

    def schedule_pipeline(self, schedule_interval: str):
        """
        Schedules the pipeline (Flow) in Prefect Cloud or Prefect Server using a schedule.
        
        Args:
            schedule_interval (str): Schedule interval (cron-like expression) or a specific Prefect schedule (e.g., IntervalSchedule).
        """
        if self.flow is None:
            logger.error("Flow is not created. Run create_workflow first.")
            return

        logger.info(f"Scheduling Prefect Flow with interval: {schedule_interval}")
        from prefect.schedules import Schedule
        from prefect.schedules.clocks import CronClock

        # Add a schedule to the flow (using CronClock for cron-like schedules)
        schedule = Schedule(clocks=[CronClock(schedule_interval)])
        self.flow.schedule = schedule

    def monitor_pipeline(self):
        """
        Monitors the pipeline execution via Prefect's monitoring UI or Prefect Cloud.
        This is primarily handled by Prefect's web interface, but we log the status here.
        """
        logger.info("Monitoring Prefect Flow execution through the Prefect UI.")

# Example Usage
if __name__ == "__main__":
    # Initialize the pipeline manager and add steps
    pipeline_manager = PipelineManager(retry_failed_steps=True)
    
    # Initialize optimizer with resource monitoring and adaptive tuning
    optimizer = PipelineOptimizer(pipeline_manager)

    # Monitor resources for the "Train Model" step
    optimizer.monitor_resources_for_step(step_name="Train Model", duration=10)

    # Tune resources based on observed usage and thresholds
    new_resources = optimizer.tune_resources_based_on_usage(
        step_name="Train Model", 
        threshold={"CPU": 75, "Memory": 4 * 1024**3}  # 75% CPU usage, 4GB memory threshold
    )
    logger.info(f"Recommended resources: {new_resources}")



    # Define pipeline steps
    preprocessing_step = PipelineStep(name="Preprocessing", func=preprocess_data, params={"scale": True})
    training_step = PipelineStep(name="Training", func=train_model, params={"epochs": 5})
    validation_step = PipelineStep(name="Validation", func=validate_model, params={"validation_data": [1, 2, 3]})

    # Create the pipeline
    pipeline = create_pipeline(steps=[preprocessing_step, training_step, validation_step], parallel=False)

    # Execute the pipeline
    final_output = pipeline.execute(initial_data=[10, 20, 30])
    print(final_output)
    
    # Example Usage
    if __name__ == "__main__":
        # Example functions for pipeline steps
        def load_data(data: Any) -> List[int]:
            logger.info("Loading data...")
            return [10, 20, 30, 40]

        def process_data(data: List[int]) -> List[int]:
            logger.info("Processing data...")
            return [x * 2 for x in data]

        def train_model(data: List[int]) -> str:
            logger.info("Training model...")
            return f"Model trained on {len(data)} samples."

        # Define pipeline steps
        load_step = PipelineStep(name="Load Data", func=load_data)
        process_step = PipelineStep(name="Process Data", func=process_data, dependencies=["Load Data"])
        train_step = PipelineStep(name="Train Model", func=train_model, dependencies=["Process Data"])

        # Create and manage pipeline
        pipeline_manager = PipelineManager(retry_failed_steps=True)
        pipeline_manager.add_step(load_step)
        pipeline_manager.add_step(process_step)
        pipeline_manager.add_step(train_step)

        # Execute the pipeline
        final_output = pipeline_manager.execute(initial_data=None)
        print("Final output:", final_output)

        # Print step metadata
        print("Pipeline metadata:", pipeline_manager.get_metadata())
        
    from datetime import datetime
    
    # Example usage for Airflow orchestration
    pipeline_manager = PipelineManager()
    # Add steps to pipeline_manager...
    
    airflow_orchestrator = AirflowOrchestrator(pipeline_manager)
    airflow_orchestrator.create_workflow(dag_id="ml_pipeline", start_date=datetime(2024, 1, 1))
    airflow_orchestrator.schedule_pipeline("@daily")
    airflow_orchestrator.monitor_pipeline()
    
    # Example usage for Prefect orchestration
    pipeline_manager = PipelineManager()
    # Add steps to pipeline_manager...
    
    prefect_orchestrator = PrefectOrchestrator(pipeline_manager)
    prefect_orchestrator.create_workflow(flow_name="ml_pipeline_flow")
    prefect_orchestrator.schedule_pipeline("0 12 * * *")  # Schedule to run at noon daily
    prefect_orchestrator.monitor_pipeline()

