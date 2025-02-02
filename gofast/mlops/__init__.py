# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Machine Learning Operations (MLOps). 

The `gofast.mlops` module provides a comprehensive suite of tools and
estimators to facilitate Machine Learning Operations (MLOps). This
module encompasses functionalities spanning automation, deployment,
inference, metadata management, monitoring, pipeline orchestration,
scaling, security, testing, utilities, and versioning. By adhering
to the scikit-learn API standards, these tools ensure compatibility
and seamless integration into existing machine learning workflows.

Available Components
--------------------
The `gofast.mlops` module is organized into several submodules, each
catering to specific aspects of MLOps:

1. **Automation**:
    - `AutomationManager`: Manages automated workflows and processes.
    - `AirflowAutomation`: Integrates with Apache Airflow for workflow automation.
    - `KubeflowAutomation`: Facilitates automation within Kubeflow pipelines.
    - `KafkaAutomation`: Manages automation tasks using Apache Kafka.
    - `RabbitMQAutomation`: Handles automation using RabbitMQ message broker.
    - `RetrainingScheduler`: Schedules model retraining tasks.
    - `SimpleAutomation`: Provides basic automation functionalities.

2. **Deployment**:
    - `ModelExporter`: Exports trained models for deployment.
    - `APIDeployment`: Deploys models as RESTful APIs.
    - `CloudDeployment`: Facilitates deployment on cloud platforms.
    - `ABTesting`: Implements A/B testing strategies for model evaluation.

3. **Inference**:
    - `BatchInference`: Handles batch processing of inference tasks.
    - `StreamingInference`: Manages real-time streaming inference.
    - `MultiModelServing`: Serves multiple models simultaneously.
    - `InferenceParallelizer`: Parallelizes inference tasks for efficiency.
    - `InferenceCacheManager`: Manages caching mechanisms for inference.

4. **Metadata**:
    - `MetadataManager`: Manages metadata associated with models and datasets.
    - `LineageTracker`: Tracks the lineage of data and models.
    - `AuditLogger`: Logs audit trails for compliance and monitoring.
    - `ReproducibilityEnsurer`: Ensures reproducibility of experiments.
    - `PerformanceTracker`: Tracks performance metrics of models.
    - `log_metadata`: Function to log metadata.
    - `retrieve`: Function to retrieve metadata.
    - `compare`: Function to compare different metadata entities.
    - `audit`: Function to perform audits on metadata.
    - `sync_with_cloud`: Syncs metadata with cloud storage.
    - `validate_schema`: Validates metadata schemas.
    - `track_experiment`: Tracks experiments for reproducibility.
    - `prune_old`: Prunes outdated metadata entries.

5. **Monitoring**:
    - `ModelPerformanceMonitor`: Monitors model performance in production.
    - `ModelHealthChecker`: Checks the health status of deployed models.
    - `DataDriftMonitor`: Monitors data drift over time.
    - `AlertManager`: Manages alerts based on monitoring metrics.
    - `LatencyTracker`: Tracks inference latency.
    - `ErrorRateMonitor`: Monitors error rates in predictions.
    - `CustomMetricsLogger`: Logs custom performance metrics.

6. **Pipeline**:
    - `Pipeline`: Represents a machine learning pipeline.
    - `PipelineStep`: Defines individual steps within a pipeline.
    - `PipelineManager`: Manages the orchestration of pipelines.
    - `PipelineOptimizer`: Optimizes pipeline configurations.
    - `ResourceMonitor`: Monitors resources used by pipelines.
    - `ResourceManager`: Manages computational resources for pipelines.
    - `create_pipeline`: Function to create a new pipeline.
    - `reconfigure_pipeline_on_the_fly`: Reconfigures pipelines dynamically.
    - `execute_step_conditionally`: Executes pipeline steps based on conditions.
    - `run_parallel_subpipelines`: Runs multiple sub-pipelines in parallel.
    - `split_data_for_multitask_pipeline`: Splits data for multi-task learning pipelines.
    - `rollback_to_previous_state`: Rolls back pipelines to a previous state.
    - `smart_retry_with_backoff`: Implements smart retry mechanisms with backoff strategies.

7. **Scaling**:
    - `ScalingManager`: Manages scaling of resources based on workload.
    - `DataPipelineScaler`: Scales data pipelines according to data volume.
    - `ElasticScaler`: Provides elastic scaling capabilities.
    - `partition_data_pipeline`: Partitions data pipelines for scalability.
    - `get_system_workload`: Retrieves current system workload metrics.
    - `elastic_scale_logic`: Implements logic for elastic scaling.

8. **Security**:
    - `BaseSecurity`: Base class for security implementations.
    - `DataEncryption`: Handles encryption of data.
    - `ModelProtection`: Protects models from unauthorized access.
    - `SecureDeployment`: Ensures secure deployment of models.
    - `AuditTrail`: Maintains audit trails for security compliance.
    - `AccessControl`: Manages access control policies.

9. **Testing**:
    - `PipelineTest`: Tests the integrity of pipelines.
    - `ModelQuality`: Assesses the quality of trained models.
    - `OverfittingDetection`: Detects overfitting in models.
    - `DataIntegrity`: Ensures data integrity throughout the pipeline.
    - `BiasDetection`: Detects biases in data and models.
    - `ModelVersionCompliance`: Ensures model versions comply with standards.
    - `PerformanceRegression`: Detects performance regressions in models.
    - `CIIntegration`: Integrates MLOps processes with Continuous Integration systems.

10. **Utilities**:
    - `ConfigManager`: Manages configuration settings.
    - `CrossValidator`: Provides cross-validation functionalities.
    - `DataVersioning`: Handles versioning of datasets.
    - `EarlyStopping`: Implements early stopping criteria for training.
    - `ExperimentTracker`: Tracks experiments and their results.
    - `ParameterGrid`: Creates grids of parameters for model tuning.
    - `PipelineBuilder`: Builds pipelines from components.
    - `Timer`: Measures execution time of code blocks.
    - `TrainTestSplitter`: Splits data into training and testing sets.
    - `calculate_metrics`: Calculates performance metrics.
    - `get_model_metadata`: Retrieves metadata of models.
    - `load_model`: Loads saved models from storage.
    - `load_pipeline`: Loads saved pipelines from storage.
    - `log_model_summary`: Logs summaries of models.
    - `save_model`: Saves models to storage.
    - `save_pipeline`: Saves pipelines to storage.
    - `set_random_seed`: Sets random seed for reproducibility.
    - `setup_logging`: Configures logging settings.

11. **Versioning**:
    - `ModelVersionControl`: Controls versions of machine learning models.
    - `DatasetVersioning`: Manages versions of datasets.
    - `PipelineVersioning`: Handles versions of pipelines.
    - `VersionComparison`: Compares different versions of models, datasets, or pipelines.


Example
-------
Below is an example demonstrating the usage of the `MajorityVoteClassifier`:

    >>> from gofast.mlops.scaling import ScalingManager
    >>> scaling_manager = ScalingManager(framework='pytorch')
    >>> scaling_manager.run(model, data)
    
    >>> from gofast.mlops.inference import BatchInference
    >>> model = MyModel()  # A model should implement a predict method
    >>> data = [{'input': x} for x in range(1000)]
    >>> batch_inference = BatchInference(
    ...     model, batch_size=64, gpu_enabled=True
    ... )
    >>> results = batch_inference.run(data)
    >>> print(results)
    
    >>> from gofast.mlops.security import DataEncryption
    >>> encryptor = DataEncryption(
    ...     encryption_algorithm='aes',
    ...     compression=True,
    ...     encryption_key='my_secret_key_1234567890123456'
    ... )
    >>> encryptor.run()
    >>> encrypted_data = encryptor.encrypt_data(b"Secret data")
    >>> decrypted_data = encryptor.decrypt_data(encrypted_data)
    >>> print(decrypted_data)
    b'Secret data'
    
    >>> from gofast.mlops.versioning import DatasetVersioning
    >>> dataset_version = DatasetVersioning(version='v1.0',
    ...                                     dataset_name='my_dataset')
    >>> X, y = load_my_data()  # Example of loading data
    >>> dataset_version.fit(X, y)
    >>> metadata = dataset_version.get_metadata()
    >>> history = dataset_version.get_version_history()
    

Additional Functionalities
--------------------------
- **Comprehensive Automation**: Automate complex workflows using
  `AutomationManager` and integrations with platforms like Airflow and
  Kubeflow.
- **Flexible Deployment Options**: Deploy models across various environments
  including APIs, cloud platforms, and perform A/B testing for model
  evaluation.
- **Robust Inference Mechanisms**: Handle both batch and streaming
  inference tasks with support for multi-model serving and parallelization.
- **Advanced Metadata Management**: Track and manage metadata to ensure
  reproducibility, compliance, and performance tracking.
- **Continuous Monitoring**: Monitor model performance, data drift,
  and system health with tools like `ModelPerformanceMonitor` and
  `DataDriftMonitor`.
- **Efficient Pipeline Orchestration**: Build, optimize, and manage
  pipelines with tools for dynamic reconfiguration and smart retries.
- **Scalable Resource Management**: Automatically scale resources based on
  workload with `ScalingManager` and `ElasticScaler`.
- **Enhanced Security**: Secure your MLOps workflows with encryption,
  access control, and audit trails.
- **Comprehensive Testing**: Ensure pipeline integrity and model quality
  with extensive testing tools.
- **Version Control**: Manage versions of models, datasets, and pipelines
  to maintain consistency and track changes over time.
- **Utility Tools**: A suite of utility functions to aid in configuration,
  validation, and experiment tracking.

References
----------
.. [1] Widrow, B., & Hoff, M. E. (1960). Adaptive Switching Circuits.
      1960 IRE WESCON Convention Record, 4, 96–104.
"""

from gofast.mlops.automation import (
    AutomationManager,
    AirflowAutomation,
    KubeflowAutomation,
    KafkaAutomation,
    RabbitMQAutomation,
    RetrainingScheduler,
    SimpleAutomation,
)

from gofast.mlops.deployment import (
    ModelExporter,
    APIDeployment,
    CloudDeployment,
    ABTesting,
)

from gofast.mlops.inference import (
    BatchInference,
    StreamingInference,
    MultiModelServing,
    InferenceParallelizer,
    InferenceCacheManager,
)

from gofast.mlops.metadata import (
    MetadataManager,
    MetadataManagerIn, 
    LineageTracker,
    AuditLogger,
    ReproducibilityEnsurer,
    PerformanceTracker,
    SchemaValidator, 
    ExperimentTracker, 
    log_metadata,
    retrieve,
    compare,
    audit,
    sync_with_cloud,
    validate_schema,
    track_experiment,
    prune_old,
)

from gofast.mlops.monitoring import (
    ModelPerformanceMonitor,
    ModelHealthChecker,
    DataDriftMonitor,
    AlertManager,
    LatencyTracker,
    ErrorRateMonitor,
    CustomMetricsLogger,
)

from gofast.mlops.pipeline import (
    Pipeline,
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
)

from gofast.mlops.scaling import (
    ScalingManager,
    DataPipelineScaler,
    ElasticScaler,
    partition_data_pipeline,
    get_system_workload,
    elastic_scale_logic,
)

from gofast.mlops.security import (
    BaseSecurity,
    DataEncryption,
    ModelProtection,
    SecureDeployment,
    AuditTrail,
    AccessControl,
)

from gofast.mlops.testing import (
    PipelineTest,
    ModelQuality,
    OverfittingDetection,
    DataIntegrity,
    BiasDetection,
    ModelVersionCompliance,
    PerformanceRegression,
    CIIntegration,
)

from gofast.mlops.utils import (
    ConfigManager,
    CrossValidator,
    DataVersioning,
    ParameterGrid,
    PipelineBuilder,
    ExperimentLogger, 
    Timer,
    TrainTestSplitter,
    calculate_metrics,
    get_model_metadata,
    load_model,
    load_pipeline,
    log_model_summary,
    save_model,
    save_pipeline,
    set_random_seed,
    setup_logging,
)

from gofast.mlops.versioning import (
    ModelVersionControl,
    DatasetVersioning,
    PipelineVersioning,
    VersionComparison,
)

__all__ = [
    # Automation
    "AutomationManager",
    "AirflowAutomation",
    "KubeflowAutomation",
    "KafkaAutomation",
    "RabbitMQAutomation",
    "RetrainingScheduler",
    "SimpleAutomation",
    
    # Deployment
    "ModelExporter",
    "APIDeployment",
    "CloudDeployment",
    "ABTesting",
    
    # Inference
    "BatchInference",
    "StreamingInference",
    "MultiModelServing",
    "InferenceParallelizer",
    "InferenceCacheManager",
    
    # Metadata
    "MetadataManager",
    "MetadataManagerIn",
    "LineageTracker",
    "AuditLogger",
    "ReproducibilityEnsurer",
    "PerformanceTracker",
    "SchemaValidator", 
    "ExperimentTracker", 
    "log_metadata",
    "retrieve",
    "compare",
    "audit",
    "sync_with_cloud",
    "validate_schema",
    "track_experiment",
    "prune_old",
    
    # Monitoring
    "ModelPerformanceMonitor",
    "ModelHealthChecker",
    "DataDriftMonitor",
    "AlertManager",
    "LatencyTracker",
    "ErrorRateMonitor",
    "CustomMetricsLogger",
    
    # Pipeline
    "Pipeline",
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
    
    # Scaling
    "ScalingManager",
    "DataPipelineScaler",
    "ElasticScaler",
    "partition_data_pipeline",
    "get_system_workload",
    "elastic_scale_logic",
    
    # Security
    "BaseSecurity",
    "DataEncryption",
    "ModelProtection",
    "SecureDeployment",
    "AuditTrail",
    "AccessControl",
    
    # Testing
    "PipelineTest",
    "ModelQuality",
    "OverfittingDetection",
    "DataIntegrity",
    "BiasDetection",
    "ModelVersionCompliance",
    "PerformanceRegression",
    "CIIntegration",
    
    # Utilities
    "ConfigManager",
    "CrossValidator",
    "DataVersioning",
    "EarlyStopping",
    "ExperimentLogger",
    "ParameterGrid",
    "PipelineBuilder",
    "Timer",
    "TrainTestSplitter",
    "calculate_metrics",
    "get_model_metadata",
    "load_model",
    "load_pipeline",
    "log_model_summary",
    "save_model",
    "save_pipeline",
    "set_random_seed",
    "setup_logging",
    
    # Versioning
    "ModelVersionControl",
    "DatasetVersioning",
    "PipelineVersioning",
    "VersionComparison",
]

import warnings
# Issuing a warning message when importing the `gofast.mlops` package or any of its modules
# Lazy loader function for triggering the warning
def __getattr__(name):
    """
    Dynamically import submodules of mlops and issue warnings.
    
    :param name: str, Name of the attribute being accessed.
    :return module: The requested submodule if available.
    :raise: AttributeError, If the submodule does not exist within 'gofast.mlops'.
    """
    # Check if the requested submodule exists
    try:
        # Attempt to load the submodule (e.g., automation, inference)
        module = __import__(f"gofast.mlops.{name}", globals(), locals(), [name])
        # Issue a warning about setting up a dedicated environment
        warning_message = (
            "Warning: You are accessing the MLOps subpackage of gofast. "
            "Please ensure that you have set up a dedicated environment for "
            "this subpackage, as it requires specific external dependencies. "
            "It is recommended to create a virtual environment with the "
            "necessary packages for MLOps tasks."
        )
        warnings.warn(warning_message, UserWarning, stacklevel=2)
        return module
    except ImportError:
        # Raise an AttributeError if the submodule does not exist
        raise AttributeError(f"Module '{name}' not found in 'gofast.mlops'.")



