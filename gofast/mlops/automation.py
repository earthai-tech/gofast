# -*- coding: utf-8 -*-
"""
10. mlops.automation

Purpose: To handle repetitive tasks and automate workflows, from model retraining
to regular data ingestion processes.

Key Features:
Scheduled training jobs and data updates
Model retraining automation (based on triggers such as performance decay)
Integration with automation tools like Airflow or Kubeflow
Real-time data pipeline automation (using Kafka, RabbitMQ, etc.)

+-----------------------------------------+
|        AutomationManager                |
|-----------------------------------------|
| + add_task(func: Callable, interval: int)|
| + run_all_tasks()                       |
| + schedule_task(task_name: str)         |
| + cancel_task(task_name: str)           |
| + monitor_tasks()                       |
+-----------------------------------------+
                  |
                  v
+-----------------------------------------+
|        RetrainingScheduler              |
|-----------------------------------------|
| + schedule_retraining(model, metric)    |
| + trigger_retraining_on_decay()         |
| + evaluate_model_performance()          |
| + adjust_retraining_schedule()          |
| + monitor_model()                       |
+-----------------------------------------+


"""

import time
import random
import threading
import logging
from typing import Callable, Dict, Any, List

from concurrent.futures import ThreadPoolExecutor #, as_completed
import pickle
import os
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    import pika
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator
    from datetime import datetime, timedelta
    from kfp import Client
    import kfp.dsl as dsl
    from kafka import KafkaConsumer
except: pass 


logger = logging.getLogger(__name__)

__all__ = [
    "AutomationManager",
    "AirflowAutomationManager",
    "KubeflowAutomationManager",
    "KafkaAutomationManager",
    "RabbitMQAutomationManager",
    "RetrainingScheduler",
]


class SimpleAutomationManager:
    """
    Manages the automation of repetitive tasks such as scheduled model training
    and data ingestion. It supports adding tasks, scheduling them, and monitoring task status.
    """
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}  # Stores tasks and their scheduling metadata

    def add_task(self, task_name: str, func: Callable, interval: int, args: tuple = ()):
        """
        Adds a new task to the automation system.
        
        Args:
            task_name (str): A unique name for the task.
            func (Callable): The function to execute.
            interval (int): The interval (in seconds) between task executions.
            args (tuple): Arguments to pass to the task function.
        """
        logger.info(f"Adding task {task_name} with interval {interval} seconds.")
        if task_name in self.tasks:
            raise ValueError(f"Task {task_name} already exists.")
        
        self.tasks[task_name] = {
            "func": func,
            "interval": interval,
            "args": args,
            "running": False,
            "thread": None,
        }

    def _task_runner(self, task_name: str):
        """Internal method to run tasks repeatedly based on the interval."""
        task = self.tasks[task_name]
        while task["running"]:
            logger.info(f"Running task: {task_name}")
            task["func"](*task["args"])
            time.sleep(task["interval"])

    def schedule_task(self, task_name: str):
        """
        Schedules the execution of a task at the specified interval.
        
        Args:
            task_name (str): The name of the task to schedule.
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} does not exist.")
        
        if self.tasks[task_name]["running"]:
            logger.warning(f"Task {task_name} is already running.")
            return

        logger.info(f"Scheduling task: {task_name}")
        self.tasks[task_name]["running"] = True
        task_thread = threading.Thread(target=self._task_runner, args=(task_name,))
        self.tasks[task_name]["thread"] = task_thread
        task_thread.start()

    def cancel_task(self, task_name: str):
        """
        Cancels the execution of a scheduled task.
        
        Args:
            task_name (str): The name of the task to cancel.
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} does not exist.")
        
        logger.info(f"Cancelling task: {task_name}")
        self.tasks[task_name]["running"] = False
        task_thread = self.tasks[task_name]["thread"]
        if task_thread and task_thread.is_alive():
            task_thread.join()

    def monitor_tasks(self):
        """Monitors and logs the status of all tasks."""
        logger.info("Monitoring all tasks...")
        for task_name, task_info in self.tasks.items():
            status = "running" if task_info["running"] else "stopped"
            logger.info(f"Task {task_name} is currently {status}.")

    def run_all_tasks(self):
        """Starts all the tasks in the automation manager."""
        logger.info("Running all tasks in the Automation Manager.")
        for task_name in self.tasks.keys():
            self.schedule_task(task_name)

class SimpleRetrainingScheduler(SimpleAutomationManager):
    """
    Handles the automation of model retraining workflows based on model performance decay.
    This class schedules regular model retraining and adjusts intervals based on model performance.
    """

    def __init__(self):
        super().__init__()

    def schedule_retraining(self, model: Any, retrain_func: Callable, interval: int):
        """
        Schedules the retraining of a model at regular intervals.
        
        Args:
            model (Any): The model to be retrained.
            retrain_func (Callable): The function to retrain the model.
            interval (int): The interval (in seconds) to check for retraining.
        """
        self.add_task(
            task_name=f"retrain_{model.__class__.__name__}",
            func=retrain_func,
            interval=interval,
            args=(model,),
        )
        self.schedule_task(f"retrain_{model.__class__.__name__}")

    def evaluate_model_performance(self, model: Any, metric_func: Callable) -> float:
        """
        Evaluates the model's performance using the given metric function.
        
        Args:
            model (Any): The model to evaluate.
            metric_func (Callable): The function to calculate model performance.
            
        Returns:
            float: The calculated model performance score.
        """
        logger.info(f"Evaluating performance of model {model.__class__.__name__}")
        score = metric_func(model)
        logger.info(f"Model performance score: {score}")
        return score

    def trigger_retraining_on_decay(self, model: Any, metric_func: Callable, decay_threshold: float):
        """
        Triggers model retraining if performance decay is detected.
        
        Args:
            model (Any): The model to evaluate.
            metric_func (Callable): Function to evaluate the model's performance.
            decay_threshold (float): The threshold below which retraining is triggered.
        """
        score = self.evaluate_model_performance(model, metric_func)
        if score < decay_threshold:
            logger.warning(f"Performance decay detected (score: {score}). Triggering retraining.")
            self.tasks[f"retrain_{model.__class__.__name__}"]["func"](model)

    def adjust_retraining_schedule(self, model: Any, new_interval: int):
        """
        Adjusts the retraining schedule dynamically based on model performance.
        
        Args:
            model (Any): The model whose retraining schedule will be adjusted.
            new_interval (int): The new interval for retraining (in seconds).
        """
        task_name = f"retrain_{model.__class__.__name__}"
        if task_name in self.tasks:
            logger.info(f"Adjusting retraining schedule for model {model.__class__.__name__} to {new_interval} seconds.")
            self.cancel_task(task_name)
            self.tasks[task_name]["interval"] = new_interval
            self.schedule_task(task_name)

    def monitor_model(self, model: Any, metric_func: Callable, decay_threshold: float, check_interval: int):
        """
        Monitors the model's performance at regular intervals and triggers retraining if necessary.
        
        Args:
            model (Any): The model to monitor.
            metric_func (Callable): The function to evaluate model performance.
            decay_threshold (float): The threshold to trigger retraining.
            check_interval (int): How often to check the model's performance (in seconds).
        """
        logger.info(f"Monitoring model {model.__class__.__name__} performance for decay.")
        
        def check_model():
            self.trigger_retraining_on_decay(model, metric_func, decay_threshold)
        
        self.add_task(
            task_name=f"monitor_{model.__class__.__name__}",
            func=check_model,
            interval=check_interval,
        )
        self.schedule_task(f"monitor_{model.__class__.__name__}")
        

class AutomationManager:
    """
    Manages the automation of repetitive tasks, with support for task scheduling, retries, 
    parallel execution, fault tolerance, and state persistence.
    """
    
    def __init__(self, max_workers: int = 4, state_persistence_file: str = "automation_state.pkl"):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.scheduler = BackgroundScheduler()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.state_persistence_file = state_persistence_file
        self.load_state()  # Load previous task states (if any)
        self.scheduler.start()

    def add_task(self, task_name: str, func: Callable, interval: int, args: tuple = (), retries: int = 3):
        """
        Adds a new task with automatic retry logic and state persistence.
        
        Args:
            task_name (str): A unique name for the task.
            func (Callable): The function to execute.
            interval (int): The interval (in seconds) between task executions.
            args (tuple): Arguments to pass to the task function.
            retries (int): Number of retries for task execution in case of failure.
        """
        logger.info(f"Adding task {task_name} with interval {interval} seconds and retries {retries}.")
        if task_name in self.tasks:
            raise ValueError(f"Task {task_name} already exists.")
        
        self.tasks[task_name] = {
            "func": func,
            "interval": interval,
            "args": args,
            "retries": retries,
            "failures": 0,
            "running": False,
            "future": None,
        }

    def _run_task_with_retries(self, task_name: str):
        """Internal method to run a task with automatic retries and exponential backoff."""
        task = self.tasks[task_name]
        retry_count = 0
        backoff_time = 1  # Start with 1 second
        
        while retry_count <= task["retries"]:
            try:
                logger.info(f"Running task {task_name} (Attempt {retry_count + 1})...")
                task["func"](*task["args"])
                task["failures"] = 0
                logger.info(f"Task {task_name} completed successfully.")
                break
            except Exception as e:
                retry_count += 1
                task["failures"] += 1
                logger.error(f"Task {task_name} failed with error: {e}. Retrying in {backoff_time} seconds.")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
        
        if retry_count > task["retries"]:
            logger.error(f"Task {task_name} failed after {task['retries']} retries.")
            task["failures"] = retry_count

    def schedule_task(self, task_name: str):
        """
        Schedules the execution of a task at the specified interval.
        
        Args:
            task_name (str): The name of the task to schedule.
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} does not exist.")
        
        task = self.tasks[task_name]
        if task["running"]:
            logger.warning(f"Task {task_name} is already running.")
            return

        logger.info(f"Scheduling task: {task_name} with interval {task['interval']} seconds.")
        task["running"] = True
        self.scheduler.add_job(lambda: self.thread_pool.submit(self._run_task_with_retries, task_name),
                               "interval", seconds=task["interval"])

    def cancel_task(self, task_name: str):
        """
        Cancels the execution of a scheduled task.
        
        Args:
            task_name (str): The name of the task to cancel.
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} does not exist.")
        
        logger.info(f"Cancelling task: {task_name}")
        task = self.tasks[task_name]
        task["running"] = False
        self.scheduler.remove_all_jobs(jobstore=task_name)

    def run_all_tasks(self):
        """Schedules all tasks for execution."""
        for task_name in self.tasks.keys():
            self.schedule_task(task_name)

    def persist_state(self):
        """Saves the current task state to disk."""
        with open(self.state_persistence_file, 'wb') as f:
            pickle.dump(self.tasks, f)
        logger.info(f"Task state persisted to {self.state_persistence_file}.")

    def load_state(self):
        """Loads the task state from a file, if it exists."""
        if os.path.exists(self.state_persistence_file):
            with open(self.state_persistence_file, 'rb') as f:
                self.tasks = pickle.load(f)
            logger.info(f"Task state loaded from {self.state_persistence_file}.")

    def shutdown(self):
        """Shuts down the automation manager and saves the task state."""
        logger.info("Shutting down Automation Manager and saving state.")
        self.persist_state()
        self.scheduler.shutdown(wait=False)
        self.thread_pool.shutdown(wait=False)

class RetrainingScheduler(AutomationManager):
    """
    Extends AutomationManager to manage model retraining workflows. 
    Handles model performance monitoring, retraining, and adaptive scheduling based on performance decay.
    """

    def __init__(self, max_workers: int = 4):
        super().__init__(max_workers=max_workers)

    def schedule_retraining(self, model: Any, retrain_func: Callable, interval: int):
        """
        Schedules regular model retraining.
        
        Args:
            model (Any): The model to be retrained.
            retrain_func (Callable): The function to retrain the model.
            interval (int): The interval (in seconds) to schedule retraining.
        """
        task_name = f"retrain_{model.__class__.__name__}"
        logger.info(f"Scheduling retraining for model {model.__class__.__name__} every {interval} seconds.")
        self.add_task(task_name=task_name, func=retrain_func, interval=interval, args=(model,))
        self.schedule_task(task_name)

    def evaluate_model_performance(self, model: Any, metric_func: Callable) -> float:
        """
        Evaluates the model's performance.
        
        Args:
            model (Any): The model to evaluate.
            metric_func (Callable): A function that returns the performance score of the model.
            
        Returns:
            float: The model performance score.
        """
        score = metric_func(model)
        logger.info(f"Evaluated model {model.__class__.__name__}: Performance score = {score}")
        return score

    def trigger_retraining_on_decay(self, model: Any, metric_func: Callable, decay_threshold: float):
        """
        Triggers model retraining if performance decay is detected.
        
        Args:
            model (Any): The model to evaluate.
            metric_func (Callable): The function to evaluate model performance.
            decay_threshold (float): The threshold below which retraining is triggered.
        """
        score = self.evaluate_model_performance(model, metric_func)
        if score < decay_threshold:
            logger.warning(f"Performance decay detected for model {model.__class__.__name__} (score: {score}). Triggering retraining.")
            self.tasks[f"retrain_{model.__class__.__name__}"]["func"](model)

    def monitor_model(self, model: Any, metric_func: Callable, decay_threshold: float, check_interval: int):
        """
        Monitors the model's performance at regular intervals and triggers retraining if necessary.
        
        Args:
            model (Any): The model to monitor.
            metric_func (Callable): Function to evaluate model performance.
            decay_threshold (float): The threshold to trigger retraining.
            check_interval (int): How often to check model performance (in seconds).
        """
        def check_and_retrain():
            self.trigger_retraining_on_decay(model, metric_func, decay_threshold)
        
        logger.info(f"Monitoring model {model.__class__.__name__} performance with interval {check_interval} seconds.")
        task_name = f"monitor_{model.__class__.__name__}"
        self.add_task(task_name=task_name, func=check_and_retrain, interval=check_interval)
        self.schedule_task(task_name)



class AirflowAutomationManager(AutomationManager):
    """
    Integrates the AutomationManager with Airflow to schedule and manage tasks using DAGs.
    """

    def __init__(self, dag_id: str, start_date: datetime, schedule_interval: str = "@daily"):
        super().__init__()
        self.dag_id = dag_id
        self.start_date = start_date
        self.schedule_interval = schedule_interval
        self.dag = self._create_dag()

    def _create_dag(self):
        """Creates a DAG for scheduling tasks."""
        default_args = {
            'owner': 'airflow',
            'depends_on_past': False,
            'start_date': self.start_date,
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }

        dag = DAG(
            self.dag_id,
            default_args=default_args,
            description='Automation DAG',
            schedule_interval=self.schedule_interval,
        )
        return dag

    def add_task_to_airflow(self, task_name: str, func: Callable, **kwargs):
        """
        Adds a task to the Airflow DAG using a PythonOperator.
        
        Args:
            task_name (str): The name of the task in Airflow.
            func (Callable): The Python function to be executed.
            kwargs (dict): Additional keyword arguments for the function.
        """
        task = PythonOperator(
            task_id=task_name,
            python_callable=func,
            op_kwargs=kwargs,
            dag=self.dag
        )
        logger.info(f"Task {task_name} added to Airflow DAG {self.dag_id}.")
        return task

    def schedule_airflow_task(self, task_name: str):
        """Schedules the task in the Airflow DAG."""
        logger.info(f"Scheduling task {task_name} in Airflow.")
        self.dag.get_task(task_name).execute(context={})  # Simulate task execution in Airflow's context

class KubeflowAutomationManager(AutomationManager):
    """
    Integrates the AutomationManager with Kubeflow Pipelines to manage tasks in Kubernetes.
    """

    def __init__(self, host: str):
        super().__init__()
        self.client = Client(host=host)

    def create_kubeflow_pipeline(self, pipeline_name: str, task_name: str, func: Callable, **kwargs):
        """
        Creates a Kubeflow pipeline to schedule a task in a Kubernetes environment.
        
        Args:
            pipeline_name (str): The name of the Kubeflow pipeline.
            task_name (str): The task name within the pipeline.
            func (Callable): The function to run in the pipeline.
            kwargs (dict): Additional keyword arguments for the function.
        """
        @dsl.pipeline(
            name=pipeline_name,
            description='Automation Pipeline for Machine Learning'
        )
        def pipeline():
            dsl.ContainerOp(
                name=task_name,
                image='python:3.7',
                command=['python', '-c', func],
                arguments=[kwargs],
            )

        pipeline_id = self.client.create_run_from_pipeline_func(pipeline, arguments={})
        logger.info(f"Kubeflow pipeline {pipeline_name} created with ID {pipeline_id}.")
        return pipeline_id
    
    
class KafkaAutomationManager(AutomationManager):
    """
    Handles real-time data pipeline automation using Kafka.
    Consumes Kafka topics and triggers tasks based on incoming data.
    """

    def __init__(self, kafka_servers: List[str], topic: str):
        super().__init__()
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_servers,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='automation_manager_group'
        )

    def process_kafka_message(self, func: Callable):
        """
        Processes incoming Kafka messages and triggers tasks.
        
        Args:
            func (Callable): The function to process the data.
        """
        for message in self.consumer:
            data = message.value
            logger.info(f"Received message from Kafka: {data}")
            func(data)


class RabbitMQAutomationManager(AutomationManager):
    """
    Handles real-time data pipeline automation using RabbitMQ.
    Consumes messages from a RabbitMQ queue and triggers tasks.
    """

    def __init__(self, host: str, queue: str):
        super().__init__()
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        self.queue = queue
        self.channel.queue_declare(queue=queue)

    def process_rabbitmq_message(self, func: Callable):
        """
        Processes incoming RabbitMQ messages and triggers tasks.
        
        Args:
            func (Callable): The function to process the data.
        """
        def callback(ch, method, properties, body):
            logger.info(f"Received message from RabbitMQ: {body}")
            func(body)

        self.channel.basic_consume(queue=self.queue, on_message_callback=callback, auto_ack=True)
        logger.info(f"Waiting for messages from RabbitMQ queue: {self.queue}")
        self.channel.start_consuming()

if __name__=='__main__':

    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Example task function to simulate some work
    def example_task(task_name):
        logger.info(f"Executing task: {task_name}")
    
    # Example retraining function for a model
    def retrain_model(model):
        logger.info(f"Retraining model: {model.__class__.__name__}")
    
    # Example model evaluation function
    def evaluate_model(model):
        return random.uniform(0.6, 1.0)  # Simulated performance score
    
    # Example model class
    class SimpleModel:
        def __init__(self, name):
            self.name = name
    
        def __repr__(self):
            return f"<Model {self.name}>"
    
    
    # Initialize the AutomationManager and RetrainingScheduler
    scheduler = RetrainingScheduler()

    # Add a repetitive task (e.g., data ingestion task)
    scheduler.add_task(task_name="data_ingestion", func=example_task, interval=10, args=("Data Ingestion",))
    scheduler.schedule_task("data_ingestion")  # Schedule the task to run every 10 seconds

    # Create a model and schedule retraining
    model = SimpleModel("MyMLModel")
    scheduler.schedule_retraining(model, retrain_model, interval=3600)  # Retrain every hour

    # Monitor model performance and trigger retraining if performance decays below 0.7
    scheduler.monitor_model(model, metric_func=evaluate_model, decay_threshold=0.7, check_interval=600)  # Check every 10 minutes

    # Run all tasks in parallel
    scheduler.run_all_tasks()

    # Example: Adjust retraining schedule dynamically if needed
    scheduler.adjust_retraining_schedule(model, new_interval=1800)  # Adjust to retrain every 30 minutes

    # Keep the scheduler running
    try:
        while True:
            pass  # Simulate a long-running process
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()  # Gracefully shutdown the scheduler and persist state

    # Airflow Example
    airflow_manager = AirflowAutomationManager(dag_id="ml_pipeline", start_date=datetime(2024, 1, 1))
    
    def retrain_model(data):
        logger.info(f"Retraining model with data: {data}")
    
    # Add task to Airflow and schedule it
    airflow_manager.add_task_to_airflow("retrain_task", retrain_model, data="Training Data")
    airflow_manager.schedule_airflow_task("retrain_task")

    # Kafka Example
    kafka_manager = KafkaAutomationManager(kafka_servers=["localhost:9092"], topic="ml_topic")
    
    def handle_kafka_data(data):
        logger.info(f"Processing Kafka data: {data}")
    
    kafka_manager.process_kafka_message(handle_kafka_data)

    # RabbitMQ Example
    rabbitmq_manager = RabbitMQAutomationManager(host="localhost", queue="ml_queue")
    
    def handle_rabbitmq_data(data):
        logger.info(f"Processing RabbitMQ data: {data}")
    
    rabbitmq_manager.process_rabbitmq_message(handle_rabbitmq_data)
