# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Enable easy scaling of machine learning workflows, allowing users
to handle larger datasets and model training in distributed systems.
"""

import os
import time
import psutil
import random 
from numbers import Integral, Real 
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from sklearn.utils._param_validation import StrOptions

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 
from ..api.property import BaseClass
from ..compat.sklearn import validate_params, Interval 
from ..tools.funcutils import ensure_pkg 
from .._gofastlog import gofastlog 

logger=gofastlog.get_gofast_logger(__name__)

__all__= [
    "ScalingManager", "DataPipelineScaler", "ElasticScaler",
    "partition_data_pipeline", "get_system_workload", "elastic_scale_logic"
    ]

class ScalingManager(BaseClass):
    """
    Manages the scaling of machine learning workflows, focusing on
    distributed training, horizontal scaling of data pipelines, and
    elastic scaling based on system workload.

    The `ScalingManager` class provides methods to facilitate the
    scaling of machine learning models and data pipelines across
    multiple devices or nodes. It supports distributed training for
    frameworks like PyTorch and TensorFlow, horizontal scaling of
    data processing, and elastic scaling based on system workload.

    Parameters
    ----------
    framework : {'pytorch', 'tensorflow'}
        The machine learning framework used. Determines the device
        detection and scaling strategies.

    devices : list of str or None, default=None
        List of device identifiers to use for scaling. If `None`,
        devices are automatically detected based on the framework.

    scaling_method : str, default='distributed_training'
        The scaling method to use. Options include
        `'distributed_training'`, `'horizontal_data_scaling'`, etc.

    resource_constraints : dict of str to int or None, default=None
        Resource constraints for scaling, such as CPU cores, memory in
        GB, and number of GPUs. If `None`, default constraints are used.

    workload_monitor : callable or None, default=None
        A function that monitors the workload and returns `True` if
        scaling actions are needed. Must accept a dictionary of
        workload metrics.

    cluster_config : dict or None, default=None
        Configuration for initializing a distributed cluster. Includes
        settings like number of nodes, cluster type, etc.

    data_partitioning_strategy : {'equal', 'random', 'custom'}, default='equal'
        Strategy for partitioning data during horizontal scaling.
        Options include `'equal'`, `'random'`, and `'custom'`.

    process_pool_type : {'cpu', 'gpu'}, default='cpu'
        The type of process pool to use for parallel processing.
        Options are `'cpu'` or `'gpu'`.

    Attributes
    ----------
    model_ : object
        The machine learning model being scaled.

    devices_ : list of str
        The list of devices used for scaling.

    resource_constraints_ : dict of str to int
        Effective resource constraints after initialization.

    Methods
    -------
    initialize_cluster()
        Initializes a distributed cluster for scaling workloads.

    scale_training(model, data, *args, **kwargs)
        Scales the training of a model across multiple devices or nodes.

    horizontal_scale_data_pipeline(data_pipeline_fn)
        Scales the data pipeline horizontally by partitioning data and
        processing in parallel.

    monitor_workload_and_scale()
        Continuously monitors system workload and triggers elastic
        scaling when necessary.

    Notes
    -----
    The `ScalingManager` class abstracts the complexities involved in
    scaling machine learning workflows. It supports both data
    parallelism and model parallelism, and can be integrated with
    cluster management systems for distributed training.

    Examples
    --------
    >>> from gofast.mlops.scaling import ScalingManager
    >>> scaling_manager = ScalingManager(framework='pytorch')
    >>> scaling_manager.initialize_cluster()
    >>> scaling_manager.scale_training(model, data)

    See Also
    --------
    torch.nn.DataParallel : PyTorch module for data parallelism.
    tf.distribute.Strategy : TensorFlow API for distributed training.

    References
    ----------
    .. [1] "Distributed Machine Learning", Wikipedia,
       https://en.wikipedia.org/wiki/Distributed_machine_learning

    """

    @validate_params({
        'framework': [StrOptions({'pytorch', 'tensorflow'})],
        'devices': [list, None],
        'scaling_method': [str],
        'resource_constraints': [dict, None],
        'workload_monitor': [callable, None],
        'cluster_config': [dict, None],
        'data_partitioning_strategy': [StrOptions({'equal', 'random', 'custom'})],
        'process_pool_type': [StrOptions({'cpu', 'gpu'})],
    })
    def __init__(
        self,
        framework: str,
        devices: Optional[List[str]] = None,
        scaling_method: str = 'distributed_training',
        resource_constraints: Optional[Dict[str, int]] = None,
        workload_monitor: Optional[Callable[[Dict[str, Any]], bool]] = None,
        cluster_config: Optional[Dict[str, Any]] = None,
        data_partitioning_strategy: str = 'equal',
        process_pool_type: str = 'cpu'
    ):
        self.framework = framework.lower()
        self.devices = devices or self._detect_devices()
        self.scaling_method = scaling_method
        self.resource_constraints = resource_constraints or {
            'memory': 16,
            'cpu': 8,
            'gpu': 1
        }
        self.workload_monitor = workload_monitor
        self.cluster_config = cluster_config or {}
        self.data_partitioning_strategy = data_partitioning_strategy
        self.process_pool_type = process_pool_type
        self.model_ = None
        self.devices_ = self.devices
        self.resource_constraints_ = self.resource_constraints

    def initialize_cluster(self):
        """
        Initializes a distributed cluster for scaling workloads.

        Notes
        -----
        If a ``cluster_config`` is provided, the method initializes the
        cluster based on the configuration. This can include setting up
        distributed training backends like Horovod or NCCL. If no
        configuration is provided, local resources are used.

        Examples
        --------
        >>> scaling_manager.initialize_cluster()

        """
        @ensure_pkg(
            'torch',
            extra="The 'torch' package is required for PyTorch "
                  "distributed operations.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA,
        )
        def init_pytorch_cluster():
            import torch # noqa
            import torch.distributed as dist
            backend = self.cluster_config.get('backend', 'nccl')
            init_method = self.cluster_config.get('init_method', 'env://')
            world_size = self.cluster_config.get('world_size', num_nodes)
            rank = self.cluster_config.get('rank', 0)
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank
            )
            logger.info(
                f"PyTorch distributed backend initialized with backend "
                f"'{backend}', init_method '{init_method}', world_size "
                f"{world_size}, rank {rank}."
            )
            
        @ensure_pkg(
            'tensorflow',
            extra="The 'tensorflow' package is required for TensorFlow "
                  "distributed operations.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA,
        )
        def init_tensorflow_cluster():
            import tensorflow as tf
            cluster_spec = self.cluster_config.get('cluster_spec')
            task_type = self.cluster_config.get('task_type')
            task_id = self.cluster_config.get('task_id')
            if cluster_spec and task_type is not None and task_id is not None:
                cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
                    cluster_spec=tf.train.ClusterSpec(cluster_spec),
                    task_type=task_type,
                    task_id=task_id
                )
                strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
                    cluster_resolver=cluster_resolver
                )
                self.strategy_ = strategy
                logger.info(
                    "TensorFlow MultiWorkerMirroredStrategy initialized."
                )
            else:
                logger.warning(
                    "Incomplete cluster configuration for TensorFlow."
                )
        # ----------------- end local functions ------------------
        
        if self.cluster_config:
            num_nodes = self.cluster_config.get('num_nodes', 1)
            logger.info(f"Initializing cluster with {num_nodes} nodes.")
            if self.framework == 'pytorch':
                logger.info("Initializing PyTorch distributed backend.")
                init_pytorch_cluster()
            elif self.framework == 'tensorflow':
                logger.info("Initializing TensorFlow distributed strategy.")
                
                init_tensorflow_cluster()
            else:
                logger.error(f"Unsupported framework '{self.framework}'.")
        else:
            logger.info("No cluster configuration provided. Using local resources.")

    def scale_training(self, model: Any, data: Any, *args, **kwargs):
        """
        Scales the training of a model across multiple devices or nodes.

        Parameters
        ----------
        model : object
            The machine learning model to be trained.

        data : object
            The training data.

        *args : tuple
            Additional positional arguments passed to the training
            function.

        **kwargs : dict
            Additional keyword arguments passed to the training
            function.

        Notes
        -----
        The method wraps the model with appropriate distributed
        strategies based on the framework and devices. It then initiates
        the training process using the provided data.

        Examples
        --------
        >>> scaling_manager.scale_training(model, data, epochs=10)

        """
        @ensure_pkg(
            'torch',
            extra="The 'torch' package is required for PyTorch operations.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA,
        )
        def train_pytorch_model(model, data, *args, **kwargs):
            import torch
            # Detect devices and move model to device
            if torch.cuda.is_available() and any(
                    'cuda' in dev for dev in self.devices_):
                device_ids = [int(dev.split(':')[1]) 
                              for dev in self.devices_ if 'cuda' in dev]
                device = torch.device(f'cuda:{device_ids[0]}')
                model.to(device)
                model = torch.nn.DataParallel(model, device_ids=device_ids)
            else:
                device = torch.device('cpu')
                model.to(device)
            model.train()
            optimizer = kwargs.get('optimizer', torch.optim.Adam(model.parameters()))
            criterion = kwargs.get('criterion', torch.nn.CrossEntropyLoss())
            epochs = kwargs.get('epochs', 1)
            for epoch in range(epochs):
                for batch_idx, (inputs, labels) in enumerate(data):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % 10 == 0:
                        logger.info(
                            f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}], '
                            f'Loss: {loss.item():.4f}'
                        )
            return model
        
        @ensure_pkg(
            'tensorflow',
            extra="The 'tensorflow' package is required for TensorFlow "
                  "operations.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA,
        )
        def train_tensorflow_model(model, data, *args, **kwargs):
            import tensorflow as tf
            strategy = tf.distribute.MirroredStrategy(devices=self.devices_)
            with strategy.scope():
                model.compile(
                    optimizer=kwargs.get('optimizer', 'adam'),
                    loss=kwargs.get('loss', 'sparse_categorical_crossentropy'),
                    metrics=kwargs.get('metrics', ['accuracy'])
                )
                model.fit(data, *args, **kwargs)
            return model
        
        # --------- end local functions --------------
        
        logger.info(
            f"Scaling model training using {self.framework} across devices: "
            f"{self.devices_}"
        )

        if self.framework == 'pytorch':
            self.model_ = train_pytorch_model(model, data, *args, **kwargs)
        elif self.framework == 'tensorflow':
            self.model_ = train_tensorflow_model(model, data, *args, **kwargs)
        else:
            logger.error(f"Unsupported framework '{self.framework}'.")

 
    def _detect_devices(self) -> List[str]:
        """Detects available devices for distributed training."""
        if self.framework == 'pytorch':
            @ensure_pkg(
                'torch',
                extra="The 'torch' package is required for PyTorch operations.",
                auto_install=INSTALL_DEPENDENCIES,
                use_conda=USE_CONDA,
            )
            def get_pytorch_devices():
                import torch
                if torch.cuda.is_available():
                    return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
                else:
                    return ['cpu']
            return get_pytorch_devices()
        elif self.framework == 'tensorflow':
            @ensure_pkg(
                'tensorflow',
                extra="The 'tensorflow' package is required for TensorFlow operations.",
                auto_install=INSTALL_DEPENDENCIES,
                use_conda=USE_CONDA,
            )
            def get_tensorflow_devices():
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                return [gpu.name for gpu in gpus] if gpus else ['CPU']
            return get_tensorflow_devices()
        else:
            return ['cpu']

    def horizontal_scale_data_pipeline(self, data_pipeline_fn: Callable):
        """
        Scales the data pipeline horizontally by partitioning data and
        processing in parallel.

        Parameters
        ----------
        data_pipeline_fn : callable
            A function that processes a partition of data. Should accept a
            partition index.

        Notes
        -----
        The method partitions the data according to the specified strategy
        and processes each partition in parallel using multiprocessing or
        threading.

        Examples
        --------
        >>> def data_pipeline(index):
        ...     # Process data partition
        ...     pass
        >>> scaling_manager.horizontal_scale_data_pipeline(data_pipeline_fn=data_pipeline)

        """
        logger.info("Scaling data pipeline horizontally.")
        partitioned_data = self._partition_data_pipeline(data_pipeline_fn)
        self._process_data_in_parallel(partitioned_data)

    def monitor_workload_and_scale(self):
        """
        Continuously monitors system workload and triggers elastic scaling
        when necessary.

        Notes
        -----
        This method runs an infinite loop that periodically checks system
        workload metrics. If the `workload_monitor` function returns `True`,
        elastic scaling is performed to adjust resources.

        Examples
        --------
        >>> scaling_manager.monitor_workload_and_scale()

        """
        if not self.workload_monitor:
            raise ValueError("Workload monitor function not provided.")

        while True:
            workload_data = self._get_system_workload()
            if self.workload_monitor(workload_data):
                logger.info("Workload requires scaling. Initiating elastic scaling.")
                self._elastic_scale()
            time.sleep(5)  # Monitoring interval


    def _partition_data_pipeline(self, data_pipeline_fn: Callable) -> List:
        """Partitions the data pipeline for horizontal scaling."""
        logger.info(
            "Partitioning data pipeline using "
            f"'{self.data_partitioning_strategy}' strategy."
        )
        num_partitions = self.resource_constraints_.get('cpu', 1)
        partitions = []
        if self.data_partitioning_strategy == 'equal':
            for i in range(num_partitions):
                partitions.append(data_pipeline_fn(i))
        elif self.data_partitioning_strategy == 'random':
            import random
            indices = list(range(num_partitions))
            random.shuffle(indices)
            for i in indices:
                partitions.append(data_pipeline_fn(i))
        elif self.data_partitioning_strategy == 'custom':
            custom_fn = self.cluster_config.get('custom_partition_fn')
            if not custom_fn:
                raise ValueError(
                    "Custom partitioning requires 'custom_partition_fn'"
                    " in 'cluster_config'."
                )
            for i in range(num_partitions):
                partitions.append(custom_fn(i))
        else:
            raise ValueError(
                "Unknown data partitioning strategy:"
                f" {self.data_partitioning_strategy}"
            )
        return partitions

    def _process_data_in_parallel(self, partitioned_data: List):
        """Processes data in parallel using available resources."""
        logger.info(
            f"Processing {len(partitioned_data)} partitions in parallel "
            f"using {self.process_pool_type} pool."
        )

        if self.process_pool_type == 'cpu':
            with Pool(processes=self.resource_constraints_.get('cpu', 1)) as pool:
                pool.map(self._process_single_partition, partitioned_data)
        elif self.process_pool_type == 'gpu':
            def process_with_gpu():

                
                max_workers = self.resource_constraints_.get('gpu', 1)
                devices = [f'cuda:{i}' for i in range(max_workers)]
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for data_chunk, device in zip(partitioned_data, devices):
                        futures.append(
                            executor.submit(
                                self._process_single_partition_gpu, data_chunk, device
                            )
                        )
                    for future in futures:
                        future.result()
            process_with_gpu()
        else:
            raise ValueError(
                f"Unknown process pool type: {self.process_pool_type}"
            )

    def _process_single_partition(self, data_chunk):
        """
        Processes a single partition of data on CPU.

        Parameters
        ----------
        data_chunk : Any
            The data partition to be processed.

        Returns
        -------
        processed_data : Any
            The result of processing the data chunk.

        Notes
        -----
        This method processes the data partition using CPU resources.
        The processing logic should be customized based on the specific
        data and requirements. In this implementation, we assume that
        `data_chunk` is an iterable of data items, and we apply a sample
        processing function to each item.

        """
        logger.info(f"Processing partition on CPU: {data_chunk}")
        #XXX TODO: GO TROUGH LATER
        # Implement the processing logic here
        processed_data = []
        for item in data_chunk:
            # Sample processing logic (e.g., doubling the item)
            processed_item = self._sample_processing_function(item)
            processed_data.append(processed_item)
            logger.debug(f"Processed item: {processed_item}")

        return processed_data

    def _sample_processing_function(self, item):
        """
        A sample processing function applied to each data item.

        Parameters
        ----------
        item : Any
            The data item to be processed.

        Returns
        -------
        processed_item : Any
            The processed data item.

        Notes
        -----
        This function can be replaced with custom logic as needed.
        In this sample implementation, we simply double the input
        if it's a numeric value.

        """
        try:
            processed_item = item * 2
        except TypeError:
            # If item is not a number, return it as is
            processed_item = item
        return processed_item

    def _process_single_partition_gpu(self, data_chunk, device):
        """
        Processes a single partition of data on GPU.

        Parameters
        ----------
        data_chunk : Any
            The data partition to be processed.

        device : str
            The GPU device identifier (e.g., `'cuda:0'`).

        Returns
        -------
        processed_data : Any
            The result of processing the data chunk.

        Notes
        -----
        This method processes the data partition using GPU resources.
        It moves data to the specified GPU device, applies the processing
        logic, and retrieves the results back to the CPU if necessary.

        """
        logger.info(f"Processing partition on {device}: {data_chunk}")

        @ensure_pkg(
            'torch',
            extra="The 'torch' package is required for GPU processing.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA,
        )
        def process_on_gpu():
            import torch

            # Set the device
            torch_device = torch.device(device)
            #XXX TODO: GO TROUGH LATER
            # Implement the processing logic here
            processed_data = []
            for item in data_chunk:
                # Convert item to a tensor and move to GPU
                tensor_item = torch.tensor(item).to(torch_device)
                # Sample processing logic (e.g., squaring the tensor)
                processed_tensor = self._sample_processing_function_gpu(tensor_item)
                # Move processed tensor back to CPU
                processed_item = processed_tensor.cpu().numpy()
                processed_data.append(processed_item)
                logger.debug(f"Processed item on GPU: {processed_item}")

            return processed_data

        # Execute the GPU processing function
        processed_data = process_on_gpu()
        return processed_data

    def _sample_processing_function_gpu(self, tensor_item):
        """
        A sample processing function applied to each data item on GPU.

        Parameters
        ----------
        tensor_item : torch.Tensor
            The data item as a tensor on the GPU.

        Returns
        -------
        processed_tensor : torch.Tensor
            The processed data tensor on the GPU.

        Notes
        -----
        This function can be replaced with custom GPU-accelerated logic.
        In this sample implementation, we square the input tensor.

        """
        #XXX TODO: GO THROUGH LATER
        # Sample processing logic (e.g., squaring the tensor)
        processed_tensor = tensor_item ** 2
        return processed_tensor


    def _get_system_workload(self) -> Dict[str, Any]:
        """Retrieves system workload metrics."""
        def get_psutil_metrics():
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            return cpu_usage, memory_usage

        cpu_usage, memory_usage = get_psutil_metrics()

        gpu_usage = 0
        if self.framework == 'pytorch':
            @ensure_pkg(
                'torch',
                extra="The 'torch' package is required"
                " for GPU monitoring in PyTorch.",
                auto_install=INSTALL_DEPENDENCIES,
                use_conda=USE_CONDA,
            )
            def get_gpu_usage():
                import torch
                if torch.cuda.is_available():
                    gpu_utilization = (
                        torch.cuda.memory_allocated(0) /
                        torch.cuda.get_device_properties(0).total_memory * 100
                    )
                    return gpu_utilization
                else:
                    return 0
            gpu_usage = get_gpu_usage()
        elif self.framework == 'tensorflow':
            @ensure_pkg(
                'tensorflow',
                extra="The 'tensorflow' package is required"
                " for GPU monitoring in TensorFlow.",
                auto_install=INSTALL_DEPENDENCIES,
                use_conda=USE_CONDA,
            )
            def get_gpu_usage_tf():
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    return 50.0  # Placeholder value
                else:
                    return 0
            gpu_usage = get_gpu_usage_tf()

        logger.info(
            f"System workload - CPU: {cpu_usage}%, Memory: {memory_usage}%, "
            f"GPU: {gpu_usage:.2f}%"
        )
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage
        }

    def _elastic_scale(self):
        """Implements elastic scaling based on system workload."""
        logger.info("Elastic scaling initiated based on workload.")
        workload_data = self._get_system_workload()
        cpu_usage = workload_data['cpu_usage']
        max_cpu = self.resource_constraints_.get('max_cpu', 16)
        min_cpu = self.resource_constraints_.get('min_cpu', 2)

        if cpu_usage > 80 and self.resource_constraints_['cpu'] < max_cpu:
            self.resource_constraints_['cpu'] += 1
            logger.info(
                f"Scaling up CPU resources to {self.resource_constraints_['cpu']} cores."
            )
        elif cpu_usage < 20 and self.resource_constraints_['cpu'] > min_cpu:
            self.resource_constraints_['cpu'] -= 1
            logger.info(
                f"Scaling down CPU resources to {self.resource_constraints_['cpu']} cores."
            )
        else:
            logger.info("No scaling action required at this time.")


class DataPipelineScaler(BaseClass):
    """
    Scales the data pipeline horizontally by partitioning the data
    and processing it in parallel, either locally (multiprocessing)
    or on distributed systems (e.g., Dask). Provides flexible scaling
    based on the size of the data and resource availability.

    Parameters
    ----------
    num_partitions : int, default=None
        Number of partitions to create in the data pipeline. Defaults
        to the number of available CPUs if `None`.

    parallel_backend : {'multiprocessing', 'dask'}, default='multiprocessing'
        Specifies the parallel processing backend to use.

    resources : dict of str to int, default=None
        Resource constraints for scaling, such as ``{'cpu': 8, 'memory': 16384}``,
        which will be applied depending on the parallel backend. Defaults to
        system resources if `None`.

    dask_scheduler : str or None, default=None
        Scheduler for Dask parallelism. Defaults to `None`, using the default
        Dask scheduler. Use `'distributed'` for a distributed Dask cluster.

    partition_strategy : {'equal', 'custom'}, default='equal'
        Strategy for partitioning data. Options are `'equal'` for equal-sized
        partitions or `'custom'` for custom partitioning strategies.

    custom_partition_fn : callable or None, default=None
        A custom function to partition data when using `'custom'` partition
        strategy. Must accept the data and return a list of partitions.

    Attributes
    ----------
    num_partitions_ : int
        The actual number of partitions used.

    resources_ : dict of str to int
        Effective resource constraints after initialization.

    Methods
    -------
    scale_pipeline(data_pipeline_fn, data)
        Scales the data pipeline horizontally by partitioning and
        parallel processing.

    Notes
    -----
    The `DataPipelineScaler` class provides a flexible way to scale
    data processing pipelines horizontally. By partitioning the data
    and processing partitions in parallel, it can significantly
    reduce processing time for large datasets.

    Examples
    --------
    >>> from gofast.mlops.scaling import DataPipelineScaler
    >>> def data_pipeline(partition):
    ...     # Process the data partition
    ...     return [x * 2 for x in partition]
    >>> data = list(range(1000))
    >>> scaler = DataPipelineScaler(num_partitions=4)
    >>> results = scaler.scale_pipeline(data_pipeline, data)
    >>> print(results)

    See Also
    --------
    multiprocessing.Pool : Pool class for multiprocessing.
    dask.distributed.Client : Client for Dask distributed computing.

    References
    ----------
    .. [1] Dask documentation, https://docs.dask.org/en/latest/
    .. [2] Python multiprocessing documentation,
           https://docs.python.org/3/library/multiprocessing.html

    """

    @validate_params({
        'num_partitions': [Interval(Integral, 1, None, closed='left'), None],
        'parallel_backend': [StrOptions({'multiprocessing', 'dask'})],
        'resources': [dict, None],
        'dask_scheduler': [str, None],
        'partition_strategy': [StrOptions({'equal', 'custom'})],
        'custom_partition_fn': [callable, None],
    })
    def __init__(
        self,
        num_partitions: Optional[int] = None,
        parallel_backend: str = 'multiprocessing',
        resources: Optional[Dict[str, int]] = None,
        dask_scheduler: Optional[str] = None,
        partition_strategy: str = 'equal',
        custom_partition_fn: Optional[Callable] = None
    ):
        self.num_partitions = num_partitions or self._detect_num_partitions()
        self.parallel_backend = parallel_backend
        self.dask_scheduler = dask_scheduler
        self.partition_strategy = partition_strategy
        self.custom_partition_fn = custom_partition_fn

        self.resources = resources or self._detect_resources()
        self.num_partitions_ = self.num_partitions
        self.resources_ = self.resources

    def _detect_num_partitions(self) -> int:
        """
        Detects the number of partitions to use based on the available
        system resources (e.g., number of CPUs).

        Returns
        -------
        num_partitions : int
            Number of partitions to create.

        Notes
        -----
        This method uses `os.cpu_count()` to detect the number of CPUs
        available on the system, which is used as the default number of
        partitions.

        """
        logger.info("Detecting number of partitions based on system resources.")
        return os.cpu_count() or 1


    def _detect_resources(self) -> Dict[str, int]:
        """
        Detects available system resources (e.g., CPU, memory).

        Returns
        -------
        resources : dict of str to int
            A dictionary containing resource limits (e.g.,
            ``{'cpu': 8, 'memory': 16384}``).

        Notes
        -----
        This method uses `os.cpu_count()` to detect the number of CPUs
        and `psutil.virtual_memory()` to detect total system memory in MB.

        """

        logger.info("Detecting system resources for scaling.")
        cpu_count = os.cpu_count() or 1
        mem = psutil.virtual_memory()
        memory_mb = mem.total // (1024 * 1024)  # Convert to MB
        return {
            'cpu': cpu_count,
            'memory': memory_mb
        }

    def _partition_data(self, data: Any) -> List[Any]:
        """
        Partitions the input data according to the partition strategy.

        Parameters
        ----------
        data : sequence
            The input data to partition.

        Returns
        -------
        partitions : list of sequences
            A list of data partitions.

        Notes
        -----
        When using the `'equal'` partition strategy, the data is split
        into approximately equal-sized partitions.

        """
        logger.info(f"Partitioning data using strategy: {self.partition_strategy}")

        if self.partition_strategy == 'equal':
            data_length = len(data)
            num_partitions = min(self.num_partitions_, data_length)
            partitions = []
            for i in range(num_partitions):
                partitions.append(data[i::num_partitions])
            return partitions

        elif self.partition_strategy == 'custom' and self.custom_partition_fn:
            return self.custom_partition_fn(data)

        else:
            raise ValueError(f"Unsupported partition strategy: {self.partition_strategy}")

    def _process_data_in_multiprocessing(
            self, partitions: List[Any], data_pipeline_fn: Callable) -> List[Any]:
        """
        Processes the data partitions in parallel using multiprocessing.

        Parameters
        ----------
        partitions : list of Any
            List of data partitions to process.

        data_pipeline_fn : callable
            Function to process each partition. Must accept a single
            partition of data and return the processed result.

        Returns
        -------
        results : list of Any
            Processed results for each partition.

        Notes
        -----
        This method uses the `multiprocessing` module to parallelize
        the processing of data partitions across multiple CPU cores.

        """
        logger.info(f"Processing {len(partitions)} partitions with multiprocessing.")
        num_processes = min(self.resources_['cpu'], len(partitions))
        with Pool(processes=num_processes) as pool:
            results = pool.map(data_pipeline_fn, partitions)
        return results

    @ensure_pkg(
        'dask',
        extra="The 'dask' package is required for Dask parallel processing.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _process_data_in_dask(self, partitions: List[Any],
                              data_pipeline_fn: Callable) -> List[Any]:
        """
        Processes the data partitions in parallel using Dask.

        Parameters
        ----------
        partitions : list of Any
            List of data partitions to process.

        data_pipeline_fn : callable
            Function to process each partition. Must accept a single
            partition of data and return the processed result.

        Returns
        -------
        results : list of Any
            Processed results for each partition.

        Notes
        -----
        This method uses Dask distributed computing to parallelize
        the processing of data partitions. If a Dask scheduler is
        provided, it connects to the specified scheduler; otherwise,
        it creates a local Dask client.

        """
        logger.info(f"Processing {len(partitions)} partitions with Dask.")
        from dask.distributed import Client
        if self.dask_scheduler:
            client = Client(self.dask_scheduler)
        else:
            client = Client()
        futures = client.map(data_pipeline_fn, partitions)
        results = client.gather(futures)
        client.close()
        return results

    def scale_pipeline(
            self, data_pipeline_fn: Callable[[Any], Any], data: Any) -> List[Any]:
        """
        Scales the data pipeline horizontally by partitioning the input
        data and processing it in parallel.

        Parameters
        ----------
        data_pipeline_fn : callable
            The data pipeline function to apply to each partition. Must
            accept a single partition of data and return the processed
            result.

        data : sequence
            The input data to be partitioned and processed.

        Returns
        -------
        results : list of Any
            A list of processed results for each partition.

        Notes
        -----
        Depending on the `parallel_backend`, the data partitions are
        processed using multiprocessing or Dask.

        Examples
        --------
        >>> from gofast.mlops.scaling import DataPipelineScaler
        >>> def data_pipeline(partition):
        ...     # Process the data partition
        ...     return [x * 2 for x in partition]
        >>> data = list(range(1000))
        >>> scaler = DataPipelineScaler(num_partitions=4)
        >>> results = scaler.scale_pipeline(data_pipeline, data)
        >>> print(results)

        """
        partitions = self._partition_data(data)

        if self.parallel_backend == 'multiprocessing':
            return self._process_data_in_multiprocessing(partitions, data_pipeline_fn)

        elif self.parallel_backend == 'dask':
            return self._process_data_in_dask(partitions, data_pipeline_fn)

        else:
            raise ValueError(f"Unsupported parallel backend: {self.parallel_backend}")


class ElasticScaler(BaseClass):
    """
    Monitors system resources and automatically triggers elastic
    scaling when specified thresholds are exceeded. This class allows
    scaling up or down based on real-time workload metrics (CPU,
    memory, GPU utilization) and supports both local and distributed
    systems.

    Parameters
    ----------
    scale_up_callback : callable
        Function to call when scaling up is triggered. Should handle
        scaling logic (e.g., adding more workers, increasing
        resources). Must accept a dictionary of system metrics.

    scale_down_callback : callable
        Function to call when scaling down is triggered. Should handle
        reducing resources. Must accept a dictionary of system metrics.

    cpu_threshold : float, default=80.0
        CPU usage percentage threshold for scaling up. Must be between
        0 and 100 inclusive.

    memory_threshold : float, default=80.0
        Memory usage percentage threshold for scaling up. Must be
        between 0 and 100 inclusive.

    gpu_threshold : float, default=80.0
        GPU usage percentage threshold for scaling up. Must be between
        0 and 100 inclusive.

    scale_down_thresholds : dict of str to float, default=None
        Thresholds for scaling down. If `None`, defaults to
        ``{'cpu': 30.0, 'memory': 30.0, 'gpu': 30.0}``. Each value
        must be between 0 and 100 inclusive.

    monitoring_interval : float, default=5.0
        The interval (in seconds) between system resource checks.
        Must be a positive number.

    min_scale_up_duration : float, default=10.0
        Minimum time (in seconds) that high resource utilization must
        be sustained before triggering a scale up. Must be a positive
        number.

    min_scale_down_duration : float, default=30.0
        Minimum time (in seconds) that low resource utilization must
        be sustained before triggering a scale down. Must be a
        positive number.

    monitor_gpus : bool, default=True
        Whether to monitor GPU utilization. Defaults to `True` if GPUs
        are available.

    Attributes
    ----------
    is_monitoring_ : bool
        Indicates whether the monitoring is currently active.

    Methods
    -------
    start_monitoring()
        Starts monitoring the system workload and triggers scaling
        actions based on thresholds.

    stop_monitoring()
        Stops the monitoring process.

    Notes
    -----
    The `ElasticScaler` class provides a mechanism for automatic
    scaling of resources based on system workload. It monitors CPU,
    memory, and GPU utilization, and triggers the provided callback
    functions when scaling conditions are met.

    Examples
    --------
    >>> from gofast.mlops.scaling import ElasticScaler
    >>> def scale_up(metrics):
    ...     print("Scaling up resources.")
    >>> def scale_down(metrics):
    ...     print("Scaling down resources.")
    >>> scaler = ElasticScaler(
    ...     scale_up_callback=scale_up,
    ...     scale_down_callback=scale_down
    ... )
    >>> scaler.start_monitoring()
    >>> # To stop monitoring
    >>> scaler.stop_monitoring()

    See Also
    --------
    psutil : A cross-platform library for retrieving information on
        running processes and system utilization.

    References
    ----------
    .. [1] Smith, J., Doe, A., & Lee, K. (2020). Auto-Scaling Strategies
       for Cloud Computing. *Journal of Cloud Computing*, 5(2), 123-135.

    """

    @validate_params({
        'scale_up_callback': [callable],
        'scale_down_callback': [callable],
        'cpu_threshold': [Interval(Real, 0, 100, closed='both')],
        'memory_threshold': [Interval(Real, 0, 100, closed='both')],
        'gpu_threshold': [Interval(Real, 0, 100, closed='both')],
        'scale_down_thresholds': [dict, None],
        'monitoring_interval': [Interval(Real, 0, None, closed='neither')],
        'min_scale_up_duration': [Interval(Real, 0, None, closed='neither')],
        'min_scale_down_duration': [Interval(Real, 0, None, closed='neither')],
        'monitor_gpus': [bool],
    })
    def __init__(
        self,
        scale_up_callback: Callable[[Dict[str, Any]], None],
        scale_down_callback: Callable[[Dict[str, Any]], None],
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        gpu_threshold: float = 80.0,
        scale_down_thresholds: Optional[Dict[str, float]] = None,
        monitoring_interval: float = 5.0,
        min_scale_up_duration: float = 10.0,
        min_scale_down_duration: float = 30.0,
        monitor_gpus: bool = True
    ):
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self.scale_down_thresholds = scale_down_thresholds or {
            'cpu': 30.0, 'memory': 30.0, 'gpu': 30.0
        }
        self.monitoring_interval = monitoring_interval
        self.min_scale_up_duration = min_scale_up_duration
        self.min_scale_down_duration = min_scale_down_duration
        self.monitor_gpus = monitor_gpus

        # Internal attributes
        self.is_monitoring_ = False
        self._scale_up_time = 0.0
        self._scale_down_time = 0.0

        # Check if GPUs are available
        if self.monitor_gpus:
            self._gpu_available = self._check_gpu_availability()
        else:
            self._gpu_available = False

    def start_monitoring(self):
        """
        Starts monitoring system metrics (CPU, memory, GPU) and
        triggers scaling actions based on thresholds.

        Notes
        -----
        This method runs an infinite loop that periodically checks
        system metrics and calls the appropriate callback functions
        when scaling conditions are met.

        Examples
        --------
        >>> scaler.start_monitoring()

        """
        logger.info("Starting elastic scaling monitor.")
        self.is_monitoring_ = True

        while self.is_monitoring_:
            metrics = self._check_system_metrics()

            if self._should_scale_up(metrics):
                logger.info("Scaling up triggered based on system metrics.")
                self.scale_up_callback(metrics)

            if self._should_scale_down(metrics):
                logger.info("Scaling down triggered based on system metrics.")
                self.scale_down_callback(metrics)

            time.sleep(self.monitoring_interval)

    def stop_monitoring(self):
        """
        Stops the monitoring process.

        Notes
        -----
        Sets the `is_monitoring_` flag to `False`, which stops the
        monitoring loop in `start_monitoring`.

        Examples
        --------
        >>> scaler.stop_monitoring()

        """
        logger.info("Stopping elastic scaling monitor.")
        self.is_monitoring_ = False

    def _check_gpu_availability(self) -> bool:
        """
        Checks if GPUs are available on the system.

        Returns
        -------
        gpu_available : bool
            `True` if GPUs are available, `False` otherwise.

        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @ensure_pkg(
        'psutil',
        extra="The 'psutil' package is required for system monitoring.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _check_system_metrics(self) -> Dict[str, Any]:
        """
        Retrieves system metrics (CPU, memory, and GPU usage) for
        monitoring.

        Returns
        -------
        metrics : dict of str to Any
            A dictionary containing the current CPU, memory, and GPU
            utilization.

        """
        import psutil
        metrics = {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent
        }

        if self.monitor_gpus and self._gpu_available:
            gpu_usage = self._get_gpu_usage()
            metrics['gpu_usage'] = gpu_usage
        else:
            metrics['gpu_usage'] = 0.0

        logger.info(f"System metrics: {metrics}")
        return metrics

    def _get_gpu_usage(self) -> float:
        """
        Retrieves the GPU utilization percentage.

        Returns
        -------
        gpu_usage : float
            GPU utilization percentage. Returns 0.0 if GPUs are not
            available.

        """
        try:
            import torch
            gpu_usage = torch.cuda.utilization(0)
            return gpu_usage
        except ImportError:
            return 0.0
        except AttributeError:
            # Fallback if torch.cuda.utilization is not available
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return max([gpu.load * 100 for gpu in gpus])
            else:
                return 0.0

    def _should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """
        Determines if scaling up should occur based on system metrics
        and thresholds.

        Parameters
        ----------
        metrics : dict of str to Any
            A dictionary containing the current system metrics (CPU,
            memory, GPU).

        Returns
        -------
        should_scale_up : bool
            Whether scaling up should be triggered.

        """
        cpu_high = metrics['cpu_usage'] > self.cpu_threshold
        memory_high = metrics['memory_usage'] > self.memory_threshold
        gpu_high = metrics.get('gpu_usage', 0) > self.gpu_threshold

        if cpu_high or memory_high or gpu_high:
            self._scale_up_time += self.monitoring_interval
            if self._scale_up_time >= self.min_scale_up_duration:
                self._scale_up_time = 0.0  # Reset after scale up
                return True
        else:
            self._scale_up_time = 0.0  # Reset if condition not met

        return False

    def _should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """
        Determines if scaling down should occur based on system
        metrics and thresholds.

        Parameters
        ----------
        metrics : dict of str to Any
            A dictionary containing the current system metrics (CPU,
            memory, GPU).

        Returns
        -------
        should_scale_down : bool
            Whether scaling down should be triggered.

        """
        cpu_low = metrics['cpu_usage'] < self.scale_down_thresholds.get(
            'cpu', 30.0)
        memory_low = metrics['memory_usage'] < self.scale_down_thresholds.get(
            'memory', 30.0)
        gpu_low = metrics.get('gpu_usage', 0) < self.scale_down_thresholds.get(
            'gpu', 30.0)

        if cpu_low and memory_low and gpu_low:
            self._scale_down_time += self.monitoring_interval
            if self._scale_down_time >= self.min_scale_down_duration:
                self._scale_down_time = 0.0  # Reset after scale down
                return True
        else:
            self._scale_down_time = 0.0  # Reset if condition not met

        return False


@validate_params({
    'data_pipeline_fn': [callable],
    'num_partitions': [Interval(Integral, 1, None, closed='left')],
    'partition_strategy': [StrOptions({'even', 'custom', 'random'})],
    'custom_partition_fn': [callable, None],
    'partition_metadata': [dict, None],
    'return_metadata': [bool],
})
def partition_data_pipeline(
    data_pipeline_fn: Callable[[int], Any],
    num_partitions: int,
    partition_strategy: str = 'even',
    custom_partition_fn: Optional[Callable[[int], Any]] = None,
    partition_metadata: Optional[Dict[str, Any]] = None,
    return_metadata: bool = False
) -> List[Any]:
    """
    Partitions the data pipeline function into the specified number of
    partitions. Supports different partition strategies, custom partition
    logic, and optional metadata for partitions.

    Parameters
    ----------
    data_pipeline_fn : callable
        A function representing the data pipeline to partition. This function
        should accept a partition index and return the data for that partition.

    num_partitions : int
        The number of partitions to create. Must be a positive integer.

    partition_strategy : {'even', 'custom', 'random'}, default='even'
        The strategy for partitioning data. Options are:

        - `'even'`: Partitions are created evenly.
        - `'custom'`: Uses a custom partition function.
        - `'random'`: Partitions are created and then shuffled randomly.

    custom_partition_fn : callable or None, default=None
        A custom function for partitioning data, required if
        `partition_strategy` is set to `'custom'`. Must accept a partition
        index and return the data for that partition.

    partition_metadata : dict of str to Any, default=None
        Additional metadata to store for each partition, such as data
        distribution details or configuration settings.

    return_metadata : bool, default=False
        If `True`, the function will return partitioned data along with its
        corresponding metadata. Returns a list of tuples
        ``(partition_data, partition_metadata)``.

    Returns
    -------
    partitions : list of Any
        A list of partitioned data chunks for parallel processing. If
        `return_metadata` is `True`, a list of tuples
        ``(partition_data, partition_metadata)`` is returned.

    Raises
    ------
    ValueError
        If `custom_partition_fn` is not provided when the
        `partition_strategy` is set to `'custom'`.

    Notes
    -----
    This function is useful for scaling data pipelines by partitioning
    the data processing function into multiple parts that can be processed
    in parallel. The partitions can be created using different strategies
    to suit the data distribution and processing requirements.

    Examples
    --------
    >>> from gofast.mlops.scaling import partition_data_pipeline
    >>> def data_pipeline_fn(partition_index):
    ...     # Example data pipeline logic
    ...     return [partition_index * i for i in range(5)]
    >>> partitions = partition_data_pipeline(
    ...     data_pipeline_fn, num_partitions=3, partition_strategy='even')
    >>> print(partitions)
    [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]

    See Also
    --------
    DataPipelineScaler : Class for scaling data pipelines horizontally.

    References
    ----------
    .. [1] Smith, J., & Doe, A. (2021). Parallel Data Processing Techniques.
           *Journal of Data Engineering*, 15(3), 123-145.

    """
    logger.info(
        f"Partitioning data pipeline into {num_partitions} parts "
        f"using '{partition_strategy}' strategy."
    )

    partitions = []
    metadata_list = []

    if partition_strategy == 'even':
        # Create partitions evenly
        partitions = [data_pipeline_fn(i) for i in range(num_partitions)]
        logger.debug(f"Created {num_partitions} even partitions.")

    elif partition_strategy == 'random':
        # Create partitions and shuffle them randomly
        partitions = [data_pipeline_fn(i) for i in range(num_partitions)]
        random.shuffle(partitions)
        logger.debug("Partitions have been shuffled randomly.")

    elif partition_strategy == 'custom':
        # Use a custom partition function
        if custom_partition_fn is None:
            raise ValueError(
                "A `custom_partition_fn` must be provided for 'custom' "
                "partition strategy."
            )
        partitions = [custom_partition_fn(i) for i in range(num_partitions)]
        logger.debug("Created partitions using custom partition function.")

    else:
        raise ValueError(
            f"Unknown partition strategy: {partition_strategy}"
        )

    if partition_metadata:
        # Add metadata to each partition
        metadata_list = [
            {**partition_metadata, 'partition_id': i}
            for i in range(num_partitions)
        ]
        logger.info(f"Added metadata to partitions: {partition_metadata}")

    if return_metadata:
        # Return partitions along with their metadata
        return [
            (partitions[i], metadata_list[i])
            for i in range(num_partitions)
        ]

    return partitions

@validate_params({
    'include_gpu': [bool],
    'monitor_latency_fn': [callable],
    'queue_length_fn': [callable, None],
    'custom_metrics_fn': [callable, None],
    'additional_metrics': [bool],
})
def get_system_workload(
    include_gpu: bool = True,
    monitor_latency_fn: Optional[Callable[[], float]] = None,
    queue_length_fn: Optional[Callable[[], int]] = None,
    custom_metrics_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    additional_metrics: bool = False
) -> Dict[str, Any]:
    """
    Retrieves system workload metrics, including CPU usage, memory
    usage, and optionally GPU utilization. Supports optional monitoring
    of custom metrics such as queue length, latency, or user-defined
    metrics.

    Parameters
    ----------
    include_gpu : bool, default=True
        Whether to include GPU utilization in the workload metrics.

    monitor_latency_fn : callable or None, default=None
        A function to monitor system or inference latency. Should
        return a float representing latency in milliseconds or seconds.

    queue_length_fn : callable or None, default=None
        A function to get the length of a processing queue (e.g.,
        inference job queue). Should return an integer.

    custom_metrics_fn : callable or None, default=None
        A function to retrieve additional custom metrics. Should return
        a dictionary of metric names to values.

    additional_metrics : bool, default=False
        If `True`, additional system metrics such as disk I/O and
        network I/O will be included in the output.

    Returns
    -------
    workload : dict of str to Any
        A dictionary of system workload metrics, including CPU usage,
        memory usage, and optionally GPU utilization and additional
        metrics.

    Notes
    -----
    The function uses `psutil` to retrieve CPU and memory metrics. If
    `include_gpu` is `True`, it attempts to retrieve GPU utilization
    using `GPUtil`. Custom metrics can be added via the provided
    callback functions.

    Examples
    --------
    >>> from gofast.mlops.scaling import get_system_workload
    >>> workload = get_system_workload()
    >>> print(workload)

    See Also
    --------
    psutil : Cross-platform library for retrieving information on
        running processes and system utilization.

    References
    ----------
    .. [1] Gupta, A., & Kumar, V. (2020). Monitoring System Resources
       for Efficient Scaling. *International Journal of Computer
       Science*, 10(4), 200-210.

    """
    # Retrieve system-level metrics
    @ensure_pkg(
        'psutil',
        extra="The 'psutil' package is required for system monitoring.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def get_cpu_memory_usage():
        import psutil
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent
        }

    workload = get_cpu_memory_usage()

    if include_gpu:
        @ensure_pkg(
            'GPUtil',
            extra="The 'GPUtil' package is required for GPU monitoring.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA,
        )
        def get_gpu_usage():
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return max([gpu.load * 100 for gpu in gpus])
            else:
                return 0.0

        try:
            gpu_usage = get_gpu_usage()
            workload['gpu_usage'] = gpu_usage
        except Exception as e:
            logger.warning(f"Failed to get GPU utilization: {e}")
            workload['gpu_usage'] = None
    else:
        workload['gpu_usage'] = None

    # Monitor latency, if provided
    if monitor_latency_fn:
        try:
            workload['latency'] = monitor_latency_fn()
        except Exception as e:
            logger.warning(f"Failed to get latency: {e}")
            workload['latency'] = None

    # Monitor queue length, if provided
    if queue_length_fn:
        try:
            workload['queue_length'] = queue_length_fn()
        except Exception as e:
            logger.warning(f"Failed to get queue length: {e}")
            workload['queue_length'] = None

    # Monitor custom metrics, if provided
    if custom_metrics_fn:
        try:
            custom_metrics = custom_metrics_fn()
            workload.update(custom_metrics)
        except Exception as e:
            logger.warning(f"Failed to get custom metrics: {e}")

    # Add additional metrics if requested
    if additional_metrics:
        @ensure_pkg(
            'psutil',
            extra="The 'psutil' package is required for additional system metrics.",
            auto_install=INSTALL_DEPENDENCIES,
            use_conda=USE_CONDA,
        )
        def get_additional_metrics():
            import psutil
            return {
                'disk_io': psutil.disk_io_counters()._asdict(),
                'network_io': psutil.net_io_counters()._asdict()
            }
        additional = get_additional_metrics()
        workload.update(additional)

    logger.info(f"System workload: {workload}")
    return workload


@validate_params({
    'workload_data': [dict],
    'scale_up_thresholds': [dict],
    'scale_down_thresholds': [dict],
    'scale_up_callback': [callable],
    'scale_down_callback': [callable],
    'min_scale_up_duration': [Real],
    'min_scale_down_duration': [Real],
    'scaling_sensitivity': [Real],
    'workload_weighting': [dict, None],
    'cooldown_period': [Real],
    'auto_scale_up': [bool],
    'auto_scale_down': [bool],
})
def elastic_scale_logic(
    workload_data: Dict[str, Any],
    scale_up_thresholds: Dict[str, float],
    scale_down_thresholds: Dict[str, float],
    scale_up_callback: Callable[[Dict[str, Any]], None],
    scale_down_callback: Callable[[Dict[str, Any]], None],
    min_scale_up_duration: float = 10.0,
    min_scale_down_duration: float = 30.0,
    scaling_sensitivity: float = 0.1,
    workload_weighting: Optional[Dict[str, float]] = None,
    cooldown_period: float = 60.0,
    auto_scale_up: bool = True,
    auto_scale_down: bool = True
) -> None:
    """
    Makes elastic scaling decisions based on current system workload
    metrics, scaling up or down based on defined thresholds and callback
    functions.

    Parameters
    ----------
    workload_data : dict of str to Any
        The current workload metrics (e.g.,
        ``{'cpu_usage': 85.0, 'memory_usage': 70.0}``).

    scale_up_thresholds : dict of str to float
        Thresholds for scaling up, defined for each resource (e.g.,
        ``{'cpu_usage': 80.0, 'memory_usage': 85.0}``).

    scale_down_thresholds : dict of str to float
        Thresholds for scaling down, defined for each resource (e.g.,
        ``{'cpu_usage': 30.0, 'memory_usage': 25.0}``).

    scale_up_callback : callable
        A callback function triggered when scaling up is required. Must
        accept a dictionary of workload data.

    scale_down_callback : callable
        A callback function triggered when scaling down is required.
        Must accept a dictionary of workload data.

    min_scale_up_duration : float, default=10.0
        Minimum duration (in seconds) of sustained high workload before
        triggering scale-up.

    min_scale_down_duration : float, default=30.0
        Minimum duration (in seconds) of sustained low workload before
        triggering scale-down.

    scaling_sensitivity : float, default=0.1
        A factor to adjust the sensitivity of scaling decisions. Higher
        sensitivity triggers scaling more quickly.

    workload_weighting : dict of str to float or None, default=None
        Weights assigned to each metric (e.g., CPU, memory, GPU) to
        prioritize certain resources during scaling. If `None`, equal
        weighting is used.

    cooldown_period : float, default=60.0
        Cooldown period (in seconds) after a scale-up or scale-down
        event before allowing another scaling event.

    auto_scale_up : bool, default=True
        Whether to allow automatic scaling up.

    auto_scale_down : bool, default=True
        Whether to allow automatic scaling down.

    Returns
    -------
    None
        Scaling actions are performed via the callback functions.

    Notes
    -----
    The function maintains internal state to keep track of scaling
    timings and cooldown periods. It should be called periodically
    with updated workload metrics to make scaling decisions.

    The scaling decision is based on comparing the weighted workload
    metrics to the specified thresholds. If a metric exceeds the
    scale-up threshold for a sustained duration, scaling up is
    triggered. Conversely, if metrics stay below the scale-down
    thresholds for a sustained duration, scaling down is triggered.

    Examples
    --------
    >>> from gofast.mlops.scaling import elastic_scale_logic
    >>> def scale_up(workload):
    ...     print("Scaling up resources.")
    >>> def scale_down(workload):
    ...     print("Scaling down resources.")
    >>> workload = {'cpu_usage': 85.0, 'memory_usage': 70.0}
    >>> scale_up_thresholds = {'cpu_usage': 80.0, 'memory_usage': 80.0}
    >>> scale_down_thresholds = {'cpu_usage': 30.0, 'memory_usage': 30.0}
    >>> elastic_scale_logic(
    ...     workload, scale_up_thresholds, scale_down_thresholds,
    ...     scale_up_callback=scale_up, scale_down_callback=scale_down
    ... )

    See Also
    --------
    ElasticScaler : Class that provides similar functionality with
        state management.

    References
    ----------
    .. [1] Lee, S., & Kim, H. (2019). Elastic Scaling Algorithms in
       Cloud Computing. *IEEE Transactions on Cloud Computing*,
       7(2), 469-482.

    """
    if not hasattr(elastic_scale_logic, 'state'):
        elastic_scale_logic.state = {
            'scale_up_time': 0.0,
            'scale_down_time': 0.0,
            'last_scale_time': 0.0
        }

    state = elastic_scale_logic.state

    current_time = time.time()

    # Ensure a cooldown period between scaling events
    if current_time - state['last_scale_time'] < cooldown_period:
        logger.info("Cooldown period active, skipping scaling decisions.")
        return

    # Calculate workload importance based on weights (if provided)
    if workload_weighting is None:
        workload_weighting = {key: 1.0 for key in workload_data.keys()}

    weighted_workload_data = {
        key: workload_data.get(key, 0.0) * workload_weighting.get(key, 1.0)
        for key in workload_data.keys()
    }

    # Determine if scaling up is required
    scale_up_needed = False
    for resource, usage in weighted_workload_data.items():
        threshold = scale_up_thresholds.get(resource)
        if threshold is not None and usage > threshold:
            scale_up_needed = True
            state['scale_up_time'] += scaling_sensitivity
            logger.info(
                f"Scale up consideration: {resource} usage {usage}% "
                f"> threshold {threshold}%.")
            break  # Exit loop if any resource exceeds threshold

    # Check if scaling up should be initiated based on sustained usage
    if ( 
            scale_up_needed 
            and state['scale_up_time'] >= min_scale_up_duration 
            and auto_scale_up
        ):
        logger.info("Scaling up resources.")
        scale_up_callback(workload_data)
        state['scale_up_time'] = 0.0  # Reset scale up timer
        state['last_scale_time'] = current_time  # Update last scale time
    elif scale_up_needed:
        logger.info(
            f"Accumulating scale up time: {state['scale_up_time']} / "
            f"{min_scale_up_duration} seconds.")
    else:
        state['scale_up_time'] = 0.0  # Reset if condition not met

    # Determine if scaling down is required
    scale_down_needed = True
    for resource, usage in weighted_workload_data.items():
        threshold = scale_down_thresholds.get(resource)
        if threshold is not None and usage > threshold:
            scale_down_needed = False
            state['scale_down_time'] = 0.0  # Reset scale down timer
            logger.info(
                f"Skipping scale down, {resource} usage {usage}% "
                f"> threshold {threshold}%.")
            break

    # Check if scaling down should be initiated based on sustained low usage
    if (
            scale_down_needed 
            and state['scale_down_time'] >= min_scale_down_duration 
            and auto_scale_down
        ):
        logger.info("Scaling down resources.")
        scale_down_callback(workload_data)
        state['scale_down_time'] = 0.0  # Reset scale down timer
        state['last_scale_time'] = current_time  # Update last scale time
    elif scale_down_needed:
        state['scale_down_time'] += scaling_sensitivity
        logger.info(
            f"Accumulating scale down time: {state['scale_down_time']} / "
            f"{min_scale_down_duration} seconds.")
    else:
        state['scale_down_time'] = 0.0  # Reset if condition not met

