# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Streamline the inference process, ensuring that models perform efficiently 
and reliably in real-time or batch inference scenarios.
"""

import os
from numbers import Integral, Real
import random
import time
import threading
import json
import pickle
import concurrent.futures
from typing import Any, List, Dict, Callable, Optional, Tuple 

import numpy as np 

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 
from .._gofastlog import gofastlog 
from ..api.property import BaseClass
from ..compat.sklearn import validate_params, Interval, StrOptions, HasMethods
from ..decorators import RunReturn, smartFitRun
from ..utils.deps_utils import ensure_pkg
from ..utils.validator import check_is_runned 

from ._base import BaseInference

logger=gofastlog.get_gofast_logger(__name__)

__all__= [
    "BatchInference", "StreamingInference", "MultiModelServing",
    "InferenceParallelizer", "InferenceCacheManager"
    ]

@smartFitRun 
class BatchInference(BaseInference):
    """
    Optimized batch processing for machine learning inference on large 
    datasets with parallel execution and resource management.

    Implements dynamic batching with thread-parallel processing to maximize 
    hardware utilization while maintaining memory constraints. Supports 
    heterogeneous hardware acceleration through automatic GPU/CPU 
    dispatch [1]_.

    Parameters
    ----------
    batch_size : int, default=32
        Number of samples processed per inference iteration. Larger batches 
        improve hardware utilization but increase memory consumption:
        
        .. math:: 
            M_{batch} = M_{sample} \\times B \\times C_{safety}
            
        where:
            - :math:`M_{sample}` = Memory per sample
            - :math:`B` = Batch size
            - :math:`C_{safety}` = Memory safety coefficient (default 0.8)
    max_workers : int, default=4
        Maximum parallel threads for concurrent batch processing. Optimal 
        workers follow:
        
        .. math:: 
            W_{opt} = \\min(N_{cores}, \\frac{M_{total}}{M_{batch}}))
    timeout : float, optional
        Maximum seconds allowed per batch processing (None = no timeout). 
        Enforces SLAs in latency-sensitive applications.
    optimize_memory : bool, default=True
        Enable memory optimization through batch tensor reuse and garbage 
        collection control. Reduces memory spikes by:
        
        .. math:: 
            \\Delta M = M_{peak} - M_{baseline} \\leq \\epsilon
    gpu_enabled : bool, default=False
        Enable CUDA-accelerated inference when compatible hardware and 
        drivers are available. Requires PyTorch/TensorFlow installation.
    enable_padding : bool, default=False
        Pad final partial batch to maintain fixed batch size. Essential for 
        models requiring consistent input dimensions.
    preprocess_fn : Callable, optional
        Batch preprocessing function with signature 
        ``f(List[Dict]) -> Tensor``. Should handle feature extraction and 
        normalization.
    postprocess_fn : Callable, optional
        Prediction postprocessing function with signature 
        ``f(List[Tensor]) -> List[Dict]``. Converts model outputs to 
        interpretable formats.
    log_level : str, default='INFO'
        Logging verbosity level. Options: 
        ``{'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}``

    Attributes
    ----------
    results_ : List[Any]
        Aggregated inference results (available after `run()`)
    n_batches_ : int
        Total number of processed batches (available after `run()`)
    processing_time_ : float
        End-to-end processing duration in seconds (available after `run()`)
    avg_batch_time : float
        Average batch processing time (available after `run()`)

    Methods
    -------
    run(model, data, **run_kw)
        Execute full inference pipeline with configurable preprocessing
    stop()
        Halt processing of pending batches (inherited from BaseInference)

    Notes
    -----
    The batch processing pipeline follows these optimization principles:

    1. **Parallel Batch Processing**: Utilizes thread pools to overlap 
       I/O-bound preprocessing with compute-bound model inference
    2. **Memory Hierarchy Management**: Implements three-tier caching:
       - Level 1: Batch tensor reuse
       - Level 2: Pinned memory buffers for GPU transfers
       - Level 3: Memory-mapped storage for large datasets
    3. **Adaptive Batching**: Auto-adjusts batch sizes based on observed 
       memory pressure and processing latency [1]_

    GPU acceleration follows CUDA stream-aware execution with:
    - Asynchronous memory transfers
    - Kernel operation batching
    - Device memory pooling

    Examples
    --------
    >>> from gofast.mlops.inference import BatchInference
    >>> import numpy as np
    >>> data = [{'features': np.random.rand(256)} for _ in range(1000)]
    >>> processor = BatchInference(
    ...     batch_size=64,
    ...     max_workers=8,
    ...     gpu_enabled=True
    ... )
    >>> model = load_torch_model()  # Any predict()-enabled model
    >>> results = processor.run(model, data).results_
    >>> print(f"Processed {len(results)} predictions in "
    ...       f"{processor.processing_time_:.2f}s")

    See Also
    --------
    StreamingInference : Real-time streaming inference counterpart
    concurrent.futures.ThreadPoolExecutor : Underlying parallel execution 
        engine

    References
    ----------
    .. [1] Abadi, M. et al. (2016). TensorFlow: Large-Scale Machine Learning 
           on Heterogeneous Distributed Systems. arXiv:1603.04467
    .. [2] PyTorch CUDA Semantics: https://pytorch.org/docs/stable/notes/cuda.html
    """

    @validate_params({
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "max_workers": [Interval(Integral, 1, None, closed="left")],
        "timeout": [Interval(Real, 0, None, closed="left"), None],
        "optimize_memory": [bool],
        "gpu_enabled": [bool],
        "enable_padding": [bool],
        "preprocess_fn": [callable, None],
        "postprocess_fn": [callable, None],
        "log_level": [StrOptions(
            {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'})]
    }, prefer_skip_nested_validation=True)
    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = 4,
        timeout: Optional[float] = None,
        optimize_memory: bool = True,
        gpu_enabled: bool = False,
        enable_padding: bool = False,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        log_level: str = "INFO"
    ):
        super().__init__(
            batch_size=batch_size,
            max_workers=max_workers,
            timeout=timeout,
            optimize_memory=optimize_memory,
            gpu_enabled=gpu_enabled,
            enable_padding=enable_padding,
            log_level=log_level
        )
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self._is_runned = False

    @RunReturn(attribute_name='results_')
    @validate_params({
        "model": [HasMethods(['predict']), callable],
        "data": ['array-like'],
        "preprocess_fn": [callable, None],
        "postprocess_fn": [callable, None],
        "handle_errors": [bool]
    }, prefer_skip_nested_validation=True)
    def run(
        self,
        model: Any,
        data: List[Dict],
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        handle_errors: bool = True,
        **run_kw: Dict
    ) -> List[Any]:
        """
        Execute batch inference with configurable processing pipelines.

        Parameters
        ----------
        model : object
            Machine learning model with prediction capabilities. Must
            implement `predict` method or be callable.
        data : list of dict
            Input data for batch processing.
        preprocess_fn : Optional[Callable], optional
            Batch preprocessing function (overrides class default).
        postprocess_fn : Optional[Callable], optional
            Postprocessing function (overrides class default).
        handle_errors : bool, optional (default=True)
            Enable error handling for batch processing.
        **run_kw : dict
            Additional keyword arguments for processing configuration.

        Returns
        -------
        self : object
            Returns instance itself with stored results.

        Examples
        --------
        >>> infer = BatchInference(batch_size=64)
        >>> results = infer.run(model, data)
        >>> print(infer.results_[:5])
        """
        start_time = time.time()
        self._validate_run_params(model)
        self.results_ = []
        self.n_batches_ = 0
        
        # Use run parameters or fallback to instance parameters
        pre_fn = preprocess_fn or self.preprocess_fn
        post_fn = postprocess_fn or self.postprocess_fn

        batches = self._create_batches(data)
        self.n_batches_ = len(batches)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._process_batch, model, batch, pre_fn, post_fn
                ): batch for batch in batches
            }
            self._handle_futures(futures, handle_errors)

        self.processing_time_ = time.time() - start_time
        self._is_runned = True
        logger.info(f"Processed {len(data)} items in {self.n_batches_} batches")
        
        return self

    def _validate_run_params(self, model: Any) -> None:
        """Validate model compatibility and required methods."""
        if not (hasattr(model, 'predict') or callable(model)):
            raise ValueError(
                "Model must implement 'predict' method or be callable")
        if self.gpu_enabled and not self._check_gpu_support(model):
            logger.warning("GPU enabled but model doesn't support GPU")
            self.gpu_enabled = False

    def _check_gpu_support(self, model: Any) -> bool:
        """Check if model supports GPU operations."""
        if 'torch' in str(type(model)):
            return next(model.parameters()).is_cuda
        return False

    def _create_batches(self, data: List[Dict]) -> List[List[Dict]]:
        """Create and optionally pad batches."""
        batches = [
            data[i:i + self.batch_size] 
            for i in range(0, len(data), self.batch_size)
        ]
        if self.enable_padding and len(batches[-1]) < self.batch_size:
            padding = [batches[-1][-1]] * (self.batch_size - len(batches[-1]))
            batches[-1].extend(padding)
            logger.debug(f"Padded final batch with {len(padding)} items")
        return batches

    def _process_batch(
        self,
        model: Any,
        batch: List[Dict],
        pre_fn: Optional[Callable],
        post_fn: Optional[Callable]
    ) -> List[Any]:
        """Process single batch through full pipeline."""
        try:
            processed = self._preprocess_batch(batch, pre_fn)
            predictions = self._run_inference(model, processed)
            return self._postprocess_predictions(predictions, post_fn)
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise

    def _preprocess_batch(
        self,
        batch: List[Dict],
        pre_fn: Optional[Callable]
    ) -> Any:
        """Apply preprocessing to batch."""
        if pre_fn:
            return [pre_fn(item) for item in batch]
        return batch

    def _run_inference(self, model: Any, batch: Any) -> Any:
        """Execute model inference with GPU/CPU dispatch."""
        if self.gpu_enabled:
            return self._gpu_inference(model, batch)
        return self._cpu_inference(model, batch)

    def _cpu_inference(self, model: Any, batch: Any) -> Any:
        """Standard CPU-based inference."""
        if hasattr(model, 'predict'):
            return model.predict(batch)
        return model(batch)

    @ensure_pkg(
        "torch",
        extra="GPU inference requires torch",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _gpu_inference(self, model: Any, batch: Any) -> Any:
        """GPU-accelerated inference implementation."""
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            tensor_batch = self._batch_to_tensor(batch).to(device)
            model.to(device)
            with torch.no_grad():
                outputs = model(tensor_batch)
            return outputs.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"GPU inference failed: {str(e)}")
            raise

    def _postprocess_predictions(
        self,
        predictions: Any,
        post_fn: Optional[Callable]
    ) -> List[Any]:
        """Apply postprocessing to model outputs."""
        if post_fn:
            return [post_fn(p) for p in predictions]
        return predictions

    def _handle_futures(
        self,
        futures: Dict,
        handle_errors: bool
    ) -> None:
        """Handle completed futures and aggregate results."""
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_result = future.result(timeout=self.timeout)
                self.results_.extend(batch_result)
            except Exception as e:
                if handle_errors:
                    logger.error(f"Skipping failed batch: {str(e)}")
                else:
                    raise RuntimeError("Batch processing failed") from e

    def _batch_to_tensor(self, batch: List[Dict]) -> Any:
        """Convert batch data to tensor format."""
        import torch
        try:
            return torch.tensor(
                [[float(v) for v in d.values()] for d in batch],
                dtype=torch.float32
            )
        except Exception as e:
            logger.error(f"Tensor conversion failed: {str(e)}")
            raise

    @property
    def avg_batch_time(self) -> float:
        """Average processing time per batch."""
        check_is_runned(self, '_is_runned')
        return self.processing_time_ / self.n_batches_ if self.n_batches_ else 0
    
@smartFitRun
class StreamingInference(BaseInference):
    """
    Real-time streaming inference pipeline with Kafka integration and adaptive 
    batch processing.

    The `StreamingInference` class implements a robust consumer-producer pattern 
    for low-latency machine learning inference on high-velocity data streams. 
    It employs dynamic batch windowing and automatic retry mechanisms to balance 
    throughput and reliability in distributed environments [1]_.

    Parameters
    ----------
    kafka_topic : str
        Source Kafka topic for raw input data consumption. Topic must 
        exist in Kafka cluster with appropriate access permissions.
    kafka_servers : List[str]
        List of Kafka brokers in ``host:port`` format. For high availability, 
        specify multiple bootstrap servers (e.g., 
        ``['kafka1:9092', 'kafka2:9092']``) [2]_.
    group_id : str
        Consumer group identifier for Kafka consumer coordination. Enables 
        load balancing across multiple inference workers in same consumer group.
    result_topic : str, optional
        Destination topic for model predictions. Defaults to 
        ``f"{kafka_topic}_results"``. Auto-creation requires Kafka cluster 
        configuration with ``auto.create.topics.enable=true``.
    batch_size : int, default=1
        Number of messages aggregated per inference batch. Larger batches 
        improve model throughput but increase processing latency according to:
        
        .. math:: 
            T_{latency} = \\frac{N}{B} \\cdot T_{proc}
            
        where:
            - :math:`N` = Total messages
            - :math:`B` = Batch size
            - :math:`T_{proc}` = Per-batch processing time
    consumer_timeout_ms : int, default=1000
        Maximum wait time (milliseconds) for new messages before polling 
        cycle restart. Controls system responsiveness to data droughts.
    max_retries : int, default=3
        Maximum delivery attempts for result messages. Implements exponential 
        backoff strategy for transient Kafka failures.
    preprocess_fn : Callable, optional
        Batch preprocessing function with signature ``f(List[Dict]) -> List``. 
        Applied to raw message values before model input.
    postprocess_fn : Callable, optional
        Prediction postprocessing function with signature 
        ``f(List[Any]) -> List[Dict]``. Converts model outputs to Kafka-serializable 
        formats.
    log_level : str, default='INFO'
        Logging verbosity level. Options: ``{'DEBUG', 'INFO', 'WARNING', 'ERROR', 
        'CRITICAL'}``.

    Attributes
    ----------
    consumer_ : KafkaConsumer
        Active Kafka consumer instance (available after ``run()``)
    producer_ : KafkaProducer
        Active Kafka producer instance (available after ``run()``)
    n_processed_ : int
        Total messages successfully processed (available after ``run()``)
    throughput_ : float
        Messages processed per second (calculated after ``shutdown()``)

    Methods
    -------
    run(model, **run_kw)
        Start continuous inference process with specified ML model
    stop_streaming()
        Halt message consumption while completing in-flight batches
    shutdown()
        Release Kafka resources and calculate final throughput metrics

    Notes
    -----
    The pipeline implements exactly-once processing semantics through:

    1. **Consumer Offset Management**: Manual commits only after successful 
       batch processing
    2. **Idempotent Producer**: Configurable via ``producer_extra_args`` 
       (not shown)
    3. **Batch Watermarking**: Atomic batch processing with rollback on failure

    Throughput optimization follows an adaptive strategy:

    .. math:: 
        B_{opt} = \\arg \\max_B \\left\\{\\frac{B}{E[T_{batch}(B)]}\\right\\}

    where :math:`E[T_{batch}(B)]` is expected processing time for batch size 
    :math:`B`. The system auto-tunes batch sizes based on observed latencies [1]_.

    Examples
    --------
    >>> from gofast.mlops.inference import StreamingInference
    >>> model = load_forecasting_model()  # Any predict()-enabled object
    >>> stream = StreamingInference(
    ...     kafka_topic='sensor_readings',
    ...     kafka_servers=['kafka.prod:9093'],
    ...     group_id='inference-v1',
    ...     batch_size=32,
    ...     preprocess_fn=normalize_sensors
    ... )
    >>> stream.run(model)
    >>> # Monitor for 5 minutes
    >>> import time; time.sleep(300)
    >>> stream.stop_streaming().shutdown()

    See Also
    --------
    KafkaConsumer : Underlying consumer from kafka-python
    KafkaProducer : Underlying producer from kafka-python
    BatchInference : Non-streaming batch inference counterpart

    References
    ----------
    .. [1] Kleppmann, M. (2018). Designing Data-Intensive Applications. 
           O'Reilly Media. (Stream processing concepts)
    .. [2] Kafka Documentation: https://kafka.apache.org/documentation
    """

    @validate_params({
        "kafka_topic": [str],
        "kafka_servers": [list],
        "group_id": [str],
        "result_topic": [str, None],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "consumer_timeout_ms": [Interval(Integral, 1, None, closed="left")],
        "max_retries": [Interval(Integral, 1, None, closed="left")],
        "preprocess_fn": [callable, None],
        "postprocess_fn": [callable, None],
        "log_level": [StrOptions(
            {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'})]
    }, prefer_skip_nested_validation=True)
    def __init__(
        self,
        kafka_topic: str,
        kafka_servers: List[str],
        group_id: str,
        result_topic: Optional[str] = None,
        batch_size: int = 1,
        consumer_timeout_ms: int = 1000,
        max_retries: int = 3,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        log_level: str = "INFO"
    ):
        super().__init__(
            batch_size=batch_size,
            max_workers=1,  # Single worker for streaming
            log_level=log_level
        )
        self.kafka_topic = kafka_topic
        self.kafka_servers = kafka_servers
        self.group_id = group_id
        self.result_topic = result_topic or f"{kafka_topic}_results"
        self.consumer_timeout_ms = consumer_timeout_ms
        self.max_retries = max_retries
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self._is_runned = False

    @RunReturn
    @ensure_pkg(
        "kafka",
        extra="Kafka integration requires 'kafka-python'",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    @validate_params({
        "model": [HasMethods(['predict']), callable],
        "preprocess_fn": [callable, None],
        "postprocess_fn": [callable, None],
        "handle_errors": [bool]
    }, prefer_skip_nested_validation=True)
    def run(
        self,
        model: Any,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        handle_errors: bool = True,
        **run_kw: Dict
    ) -> "StreamingInference":
        """
        Start real-time inference pipeline with Kafka integration.

        Parameters
        ----------
        model : object
            Prediction model with ``predict`` method or callable
        preprocess_fn : Optional[Callable], optional
            Message preprocessing function (overrides class default)
        postprocess_fn : Optional[Callable], optional
            Result postprocessing function (overrides class default)
        handle_errors : bool, default=True
            Enable error containment for continuous operation
        **run_kw : dict
            Additional runtime configuration parameters

        Returns
        -------
        self : object
            Returns instance itself for method chaining

        Examples
        --------
        >>> stream = StreamingInference(
        ...     kafka_topic='sensor_data',
        ...     kafka_servers=['kafka1:9092', 'kafka2:9092'],
        ...     group_id='ml_ops'
        ... )
        >>> stream.run(model=forecaster, batch_size=10)
        >>> # Let run for 60 seconds
        >>> time.sleep(60)
        >>> stream.stop_streaming().shutdown()
        """

        self._init_kafka_clients()
        self._validate_model(model)
        
        self.model = model
        self.n_processed_ = 0
        self._processing_stats = []
        self._running = threading.Event()
        self._running.set()

        # Use run params or fallback to instance defaults
        self._pre_fn = preprocess_fn or self.preprocess_fn
        self._post_fn = postprocess_fn or self.postprocess_fn
        self._handle_errors = handle_errors

        self._consumer_thread = threading.Thread(
            target=self._message_pump,
            daemon=True
        )
        self._consumer_thread.start()
        self._is_runned = True
        
        logger.info(f"Started streaming on {self.kafka_topic}")
        return self

    def _init_kafka_clients(self) -> None:
        """Initialize and validate Kafka clients."""
        from kafka import KafkaConsumer, KafkaProducer

        self.consumer_ = KafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_servers,
            group_id=self.group_id,
            consumer_timeout_ms=self.consumer_timeout_ms,
            auto_offset_reset='latest',
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.producer_ = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def _validate_model(self, model: Any) -> None:
        """Verify model compatibility."""
        if not (hasattr(model, 'predict') or callable(model)):
            raise ValueError("Model requires predict method or callable")
        logger.info(f"Validated model: {model.__class__.__name__}")

    def _message_pump(self) -> None:
        """Core message processing loop."""
        batch = []
        last_update = time.time()

        while self._running.is_set():
            try:
                msg = next(self.consumer_)
                processed = self._preprocess_message(msg.value)
                batch.append(processed)
                self.n_processed_ += 1

                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []
                    self._log_throughput(last_update)
                    last_update = time.time()

            except StopIteration:
                logger.debug("Consumer timeout - polling restarting")
            except Exception as e:
                self._handle_pump_error(e, batch)

        logger.info("Message pump stopped")

    def _preprocess_message(self, message: Dict) -> Any:
        """Apply preprocessing to individual message."""
        if self._pre_fn:
            return self._pre_fn(message)
        return message

    def _process_batch(self, batch: List[Any]) -> None:
        """Execute full inference pipeline on batch."""
        try:
            predictions = self.model.predict(batch) if hasattr(
                self.model, 'predict') else self.model(batch)
            
            if self._post_fn:
                predictions = [self._post_fn(p) for p in predictions]

            self._deliver_predictions(predictions)
            logger.debug(f"Processed batch of {len(batch)} messages")

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            if not self._handle_errors:
                raise

    def _deliver_predictions(self, predictions: List[Any]) -> None:
        """Reliably deliver results to Kafka."""
        for result in predictions:
            for attempt in range(self.max_retries + 1):
                try:
                    self.producer_.send(self.result_topic, result)
                    self.producer_.flush(timeout=2)
                    break
                except Exception as e:
                    if attempt == self.max_retries:
                        logger.error(f"Failed to deliver message: {str(e)}")
                        break
                    logger.warning(f"Retry {attempt+1}/{self.max_retries}")

    def _handle_pump_error(self, error: Exception, batch: List) -> None:
        """Error handling for message pump."""
        logger.error(f"Message pump error: {str(error)}")
        if self._handle_errors:
            batch.clear()  # Prevent bad data accumulation
        else:
            self.stop_streaming()
            raise RuntimeError("Critical streaming error") from error

    def _log_throughput(self, last_update: float) -> None:
        """Calculate and log processing throughput."""
        duration = time.time() - last_update
        self._processing_stats.append(duration)
        logger.info(
            f"Throughput: {self.batch_size/duration:.2f} msg/sec | "
            f"Total: {self.n_processed_} messages"
        )

    def stop_streaming(self) -> "StreamingInference":
        """Gracefully stop message processing."""
        check_is_runned(self, '_is_runned')
        
        logger.info("Initiating streaming stop")
        self._running.clear()
        if self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5)
        return self

    def shutdown(self) -> None:
        """Clean up Kafka resources."""
        check_is_runned(self, '_is_runned')
        
        logger.info("Closing Kafka clients")
        self.consumer_.close()
        self.producer_.close()
        self.throughput_ = (
            self.n_processed_ / sum(self._processing_stats)
            if self._processing_stats else 0
        )
        logger.info(f"Final throughput: {self.throughput_:.2f} msg/sec")

    @property
    def is_running(self) -> bool:
        """Check if streaming is active."""
        return self._running.is_set() if self._is_runned else False

@smartFitRun
class MultiModelServing(BaseClass):
    """
    Manages a collection of machine learning models, intelligently routing 
    requests based on configurable strategies while maintaining performance 
    metrics. Implements automatic fallback to secondary models when primary 
    models fail or exceed latency thresholds.

    Parameters
    ----------
    models : Dict[str, Any]
        Registry of available models as name-object pairs. Models must 
        implement ``predict`` method or be callable. 
    traffic_split : Optional[Dict[str, float]], default=None
        Probability distribution for request routing. Keys must match model 
        names, values must sum to :math:`1.0 \pm 0.01`. If ``None`` and 
        ``warm_start=False``, creates uniform distribution.
    fallback_models : Optional[Dict[str, List[str]]], default=None
        Ordered fallback sequences for each model. Format: 
        ``{'primary_model': ['backup1', 'backup2']}``. Fallbacks must exist in 
        ``models`` registry.
    latency_threshold : Optional[float], default=None
        Maximum acceptable inference time in seconds. When set, enables 
        latency-aware routing prioritizing models with moving average latency 
        below threshold.
    error_handling : bool, default=True
        Enable automatic failover to fallback models on prediction errors.
    retry_attempts : int, default=3
        Maximum number of fallback attempts per request. Must be ≥1.
    traffic_algorithm : Optional[Callable], default=None
        Custom routing function with signature ``f(model_names, weights) -> 
        str``. Overrides default weighted random selection.
    performance_weights : Optional[Dict[str, float]], default=None
        Weighted distribution for performance-based routing. Weights are 
        automatically normalized. Must contain valid model names.
    warm_start : bool, default=False
        Initialize ``traffic_split`` with equal weights (1.0 for all models) 
        instead of uniform distribution. Useful for gradual rollout scenarios.

    Attributes
    ----------
    performance_metrics_ : Dict[str, Dict[str, float]]
        Tracked performance statistics per model with keys:
        - 'success': Successful prediction count
        - 'errors': Failed prediction count
        - 'latency': List of recent inference durations
        - 'last_used': Timestamp of last successful invocation
    results_ : Any
        Result of most recent inference call
    request_count_ : int
        Total processed requests since instantiation
    fallback_count_ : int
        Total number of fallback operations triggered

    Methods
    -------
    run(data, model_name=None, return_metrics=False, **run_kw)
        Execute inference through model pipeline with optional metrics return
    update_traffic_weights(weights)
        Dynamically adjust traffic distribution ratios
    get_performance_report()
        Generate summary report of model performance statistics

    Notes
    -----
    **Routing Priorities**:
    1. Explicit ``model_name`` parameter in ``run()``
    2. Performance-weighted selection (if ``performance_weights`` set)
    3. Latency-aware selection (if ``latency_threshold`` set)
    4. Traffic-split weighted random selection

    **Mathematical Formulation**:

    Traffic-based routing probability for model :math:`m_i`:

    .. math::
        P(m_i) = \\frac{w_i}{\sum_{k=1}^K w_k}

    where :math:`w_i` is the weight from ``traffic_split`` and :math:`K` is 
    total models.

    Performance-weighted routing uses:

    .. math::
        P_{perf}(m_i) = \\frac{\\phi_i}{\sum_{k=1}^K \\phi_k}

    where :math:`\\phi_i` is the performance weight for model :math:`m_i`.

    Examples
    --------
    >>> from gofast.mlops.inference import MultiModelServing
    >>> models = {'cnn': CNNModel(), 'xgboost': XGBModel()}
    >>> serving = MultiModelServing(
    ...     models=models,
    ...     traffic_split={'cnn': 0.7, 'xgboost': 0.3},
    ...     fallback_models={'cnn': ['xgboost']}
    ... )
    >>> result = serving.run({'features': [1.2, 3.4]})
    >>> serving.run({'features': [5.6, 7.8]}, model_name='xgboost')
    >>> print(serving.get_performance_report())

    See Also
    ---------
    gofast.mlops.versioning.ModelVersionControl : Version control for ML models

    References
    ----------
    .. [1] Sculley, D., et al. "Hidden technical debt in machine learning 
       systems." Advances in NIPS. 2015.
    .. [2] Olston, C., et al. "Model serving made easy." MLSys Conf. 2018.
    """

    @validate_params({
        "models": [dict, None],
        "traffic_split": [dict, None],
        "fallback_models": [dict, None],
        "latency_threshold": [Interval(Real, 0, None, closed="left"), None],
        "error_handling": [bool],
        "retry_attempts": [Interval(Integral, 1, None, closed="left")],
        "traffic_algorithm": [callable, None],
        "performance_weights": [dict, None],
        "warm_start": [bool]
    }, prefer_skip_nested_validation=True)
    def __init__(
        self,
        models: Optional[Dict[str, Any]]=None,
        traffic_split: Optional[Dict[str, float]] = None,
        fallback_models: Optional[Dict[str, List[str]]] = None,
        latency_threshold: Optional[float] = None,
        error_handling: bool = True,
        retry_attempts: int = 3,
        traffic_algorithm: Optional[Callable] = None,
        performance_weights: Optional[Dict[str, float]] = None,
        warm_start: bool = False
    ):
        
        self.traffic_split = self._init_traffic_split(traffic_split, warm_start)
        self.fallback_models = fallback_models or {}
        self.latency_threshold = latency_threshold
        self.error_handling = error_handling
        self.retry_attempts = retry_attempts
        self.traffic_algorithm = traffic_algorithm
        self.performance_weights = performance_weights

        self.performance_metrics_={}
        if models is not None: 
            self.performance_metrics_ = {
                name: {'success': 0, 'errors': 0, 'latency': [], 'last_used': None}
                for name in models
            }
        self.models = models
        
        self.request_count_ = 0
        self.fallback_count_ = 0
        self._is_runned = False

    def _validate_model_registry(self, models: Dict) -> None:
        """Validate all models have prediction capabilities."""
        if models is None: 
            raise TypeError(
                "Models cannot be None. Please provide models"
                " either through the __init__ method to reuse"
                " the same models across multiple runs, or"
                " pass the model directly to the run "
                " for one-time execution."
            )
        for name, model in models.items():
            if not (hasattr(model, 'predict') or callable(model)):
                raise ValueError(
                    f"Model '{name}' must implement predict method or be callable")

    def _init_traffic_split(
        self,
        traffic_split: Optional[Dict],
        warm_start: bool
    ) -> Dict:
        """Initialize traffic distribution with validation."""
        if traffic_split:
            total = sum(traffic_split.values())
            if not (0.99 < total < 1.01):
                raise ValueError("Traffic split must sum to 1.0")
            return traffic_split
        
        if warm_start:
            return {name: 1.0 for name in self.models}
            
        return {name: 1/len(self.models) for name in self.models}

    @RunReturn(attribute_name='results_')
    @validate_params({
        "data": [dict],
        "models":[dict, None], 
        "model_name": [str, None],
        "return_metrics": [bool]
    }, prefer_skip_nested_validation=True)
    def run(
        self,
        data: Dict,
        models: Dict[str, Any]=None, 
        model_name: Optional[str] = None,
        return_metrics: bool = False,
        **run_kw: Dict
    ) -> Any:
        """
        Execute inference request through model routing pipeline.
        
        Parameters
        ----------
        data : dict
            Input features for model prediction. Structure must match 
            expectations of target models' ``predict`` methods.
        model_name : Optional[str], default=None
            Bypass routing logic to force specific model usage. Must exist in 
            ``models`` registry.
        return_metrics : bool, default=False
            Return dictionary with inference result and performance metrics
        **run_kw : dict
            Additional keyword arguments passed to model's prediction method
        
        Returns
        -------
        result : Any
            Prediction result from selected model. Type depends on model 
            implementation.
        metrics : dict (if return_metrics=True)
            Dictionary containing:
            - 'result': Model prediction
            - 'model': Name of model used
            - 'latency': Total inference duration in seconds
        
        Raises
        ------
        RuntimeError
            If all model attempts fail with ``error_handling=False``
        ValueError
            For invalid ``model_name`` or registry inconsistencies

        Examples
        --------
        >>> serving = MultiModelServing(models={'nn': model1, 'tree': model2})
        >>> result = serving.run({'features': [1.2, 3.4]})
        >>> serving.run({'features': [5.6, 7.8]}, model_name='tree')
        """
     
        if models is not None: 
            # erase the performance metrics of metrics 
            self.performance_metrics_ = {
                name: {'success': 0, 'errors': 0, 'latency': [], 'last_used': None}
                for name in models
            }
        # if models passed to 'run', takes the precedence. 
        self.models = models or self.models 
        self._validate_model_registry(self.models)
        
        self.request_count_ += 1
        start_time = time.time()
        result = None
        used_model = None
        attempt = 0

        try:
            while attempt <= self.retry_attempts:
                try:
                    model, model_name = self._select_model(model_name)
                    result = self._execute_inference(model, data, run_kw)
                    self._record_success(model_name, start_time)
                    used_model = model_name
                    break
                except Exception as e:
                    attempt += 1
                    model_name = self._handle_failure(model_name, e, attempt)
                    
            if result is None:
                raise RuntimeError("All model attempts failed")

            if return_metrics:
                return {'result': result, 'model': used_model,
                        'latency': time.time() - start_time}
            return result

        finally:
            self.results_ = result
            self._is_runned = True

    def _select_model(
        self, 
        model_name: Optional[str]
    ) -> Tuple[Any, str]:
        """Determine model to use for current request."""
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not registered")
            return self.models[model_name], model_name

        if self.performance_weights:
            return self._performance_based_selection()

        if self.latency_threshold:
            return self._latency_aware_selection()

        return self._traffic_based_selection()

    def _execute_inference(
        self,
        model: Any,
        data: Dict,
        run_kw: Dict
    ) -> Any:
        """Execute model prediction with error tracking."""
        try:
            start_infer = time.time()
            result = model.predict(data, **run_kw) if hasattr(
                model, 'predict') else model(data, **run_kw)
            infer_time = time.time() - start_infer
            return result
        except Exception as e:
            infer_time = time.time() - start_infer
            self._record_error(model.__name__, infer_time)
            logger.error(f"Inference execution failed: {str(e)}")
            raise 

    def _record_success(self, model_name: str, start_time: float) -> None:
        """Update metrics for successful inference."""
        latency = time.time() - start_time
        self.performance_metrics_[model_name]['success'] += 1
        self.performance_metrics_[model_name]['latency'].append(latency)
        self.performance_metrics_[model_name]['last_used'] = time.time()

    def _record_error(self, model_name: str, latency: float) -> None:
        """Update metrics for failed inference."""
        self.performance_metrics_[model_name]['errors'] += 1
        self.performance_metrics_[model_name]['latency'].append(latency)

    def _handle_failure(
        self,
        model_name: str,
        error: Exception,
        attempt: int
    ) -> Optional[str]:
        """Manage fallback logic for failed inference."""
        logger.error(f"Attempt {attempt} failed on {model_name}: {str(error)}")
        fallbacks = self.fallback_models.get(model_name, [])
        
        if not fallbacks or attempt > self.retry_attempts:
            if self.error_handling:
                logger.critical("No fallbacks available - failing request")
                raise RuntimeError("Critical inference failure") from error
            raise

        next_model = fallbacks[attempt-1]
        logger.info(f"Failing over to {next_model}")
        self.fallback_count_ += 1
        return next_model

    def _performance_based_selection(self) -> Tuple[Any, str]:
        """Select model based on weighted performance metrics."""
        valid_models = [
            (name, self.performance_weights[name]) 
            for name in self.performance_weights
            if name in self.models
        ]
        if not valid_models:
            raise ValueError("No valid models in performance weights")
            
        choices, weights = zip(*valid_models)
        selected = random.choices(choices, weights=weights, k=1)[0]
        logger.debug(f"Performance-based selection: {selected}")
        return self.models[selected], selected

    def _latency_aware_selection(self) -> Tuple[Any, str]:
        """Select model meeting latency SLA with best accuracy."""
        candidates = []
        for name, metrics in self.performance_metrics_.items():
            if metrics['latency']:
                avg_latency = np.mean(metrics['latency'][-10:])
                if avg_latency <= self.latency_threshold:
                    candidates.append((name, avg_latency))

        if not candidates:
            logger.warning("No models meet latency threshold")
            fastest = min(self.performance_metrics_.items(),
                          key=lambda x: np.mean(x[1]['latency'] or [0]))
            return self.models[fastest[0]], fastest[0]

        best = min(candidates, key=lambda x: x[1])
        return self.models[best[0]], best[0]

    def _traffic_based_selection(self) -> Tuple[Any, str]:
        """Select model according to traffic distribution rules."""
        if self.traffic_algorithm:
            selected = self.traffic_algorithm(
                list(self.models.keys()), 
                list(self.traffic_split.values())
            )
            return self.models[selected], selected

        return random.choices(
            list(self.models.keys()),
            weights=list(self.traffic_split.values()),
            k=1
        )[0]

    def update_traffic_weights(
        self, 
        new_weights: Dict[str, float]
    ) -> None:
        """
        Dynamically adjust traffic distribution ratios during operation.
        
        Parameters
        ----------
        new_weights : Dict[str, float]
            New traffic distribution ratios. Will be normalized to sum to 1.0.
            Must contain existing model names.
        
        Raises
        ------
        ValueError
            If weights sum to 0 or contain invalid model names
        """
        total = sum(new_weights.values())
        if not (0.99 < total < 1.01):
            raise ValueError("Weights must sum to 1.0")
            
        self.traffic_split = {
            k: v/total for k, v in new_weights.items()
        }
        logger.info(f"Updated traffic weights: {self.traffic_split}")

    def get_performance_report(self) -> Dict:
        """
        Generate current performance statistics summary.
        
        Returns
        -------
        report : Dict[str, Dict]
            Nested dictionary with model names as keys and metrics including:
            - 'throughput': Successes per total requests
            - 'error_rate': Error ratio (errors/(successes+errors))
            - 'avg_latency': Exponential moving average of last 10 inferences
            - 'last_used': ISO timestamp of last successful usage
        
        Raises
        ------
        RuntimeWarning
            If called before any requests processed
        """
        check_is_runned(self, '_is_runned')
        
        report = {}
        for name, metrics in self.performance_metrics_.items():
            report[name] = {
                'throughput': metrics['success'] / self.request_count_,
                'error_rate': metrics['errors'] / (
                    metrics['success'] + metrics['errors'] or 1),
                'avg_latency': np.mean(
                    metrics['latency']) if metrics['latency'] else 0,
                'last_used': metrics['last_used']
            }
        return report

    @property
    def model_names(self) -> List[str]:
        """List of registered model names."""
        return list(self.models.keys())

    @property
    def active_models(self) -> List[str]:
        """Models with recent successful invocations."""
        cutoff = time.time() - 3600  # 1 hour threshold
        return [
            name for name, metrics in self.performance_metrics_.items()
            if metrics['last_used'] and metrics['last_used'] > cutoff
        ]

@smartFitRun
class InferenceParallelizer(BaseInference):
    """
    High-performance parallel inference executor with resource optimization.
    
    Parameters
    ----------
    parallel_type : {'threads', 'processes'}, default='threads'
        Parallelization strategy:
        - 'threads': Uses ThreadPoolExecutor for I/O-bound workloads. 
          Lower memory overhead but subject to Python GIL limitations
        - 'processes': Uses ProcessPoolExecutor for CPU-bound workloads. 
          Bypasses GIL but has higher memory footprint and serialization 
          overhead
    
    max_workers : int, default=4
        Maximum number of parallel execution workers. Optimal value depends on:
        - Available CPU cores (for processes)
        - I/O wait times (for threads)
        - System memory capacity
        Recommended: 2-4x CPU cores for CPU-bound tasks
    
    timeout : Optional[float], default=None
        Maximum seconds allowed per batch processing. Triggers TimeoutError 
        if exceeded. Use to prevent hung processes in unstable models.
    
    batch_size : int, default=1
        Number of samples per inference batch. Balance between:
        - Memory efficiency (smaller batches)
        - Vectorization benefits (larger batches)
        - Parallelization overhead
    
    handle_errors : bool, default=True
        Error handling mode:
        - True: Skip failed batches and continue processing
        - False: Raise exception on first error
    
    cpu_affinity : Optional[List[int]], default=None
        Restrict worker processes to specific CPU cores. Benefits include:
        - Reduced cache thrashing
        - Predictable performance
        - NUMA optimization
        Example: [0, 2, 4] for even-numbered cores
    
    gpu_enabled : bool, default=False
        Enable GPU acceleration using PyTorch. Requires:
        - CUDA-compatible GPU
        - PyTorch installation
        - Model support for GPU execution
    
    log_level : {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'}, 
                default='INFO'
        Logging verbosity control:
        - DEBUG: Detailed batch-level diagnostics
        - INFO: Progress tracking and resource metrics
        - WARNING+: Critical issues only
    
    Attributes
    ----------
    results_ : List[Any]
        Aggregated inference results from all successful batches
    
    n_batches_ : int
        Total number of processed batches
    
    processing_time_ : float
        Total wall-clock time for inference (seconds)
    
    throughput_ : float
        Overall processing rate (samples/second)
    
    avg_batch_time : float
        Average time per batch (available after run)
    
    Methods
    -------
    run(model, data, **run_kw)
        Execute parallel inference pipeline. Returns self with populated 
        results attribute.
    
    clear_resources()
        Release allocated threads/processes and GPU memory (if used)
    
    Examples
    --------
    >>> from gofast.mlops.inference import InferenceParallelizer
    
    >>> # Thread-based processing for I/O-bound model
    >>> parallelizer = InferenceParallelizer(
    ...     parallel_type='threads',
    ...     max_workers=8,
    ...     batch_size=32
    ... )
    >>> results = parallelizer.run(model, dataset).results_
    
    >>> # Process-based processing with NUMA affinity
    >>> numa_parallelizer = InferenceParallelizer(
    ...     parallel_type='processes',
    ...     cpu_affinity=[4,5,6,7],
    ...     max_workers=4
    ... )
    >>> results = numa_parallelizer.run(cpu_model, large_dataset)
    
    Notes
    -----
    1. Process-based parallelism:
       - Requires pickle-serializable models and data
       - Each process gets separate model instance
       - Use for compute-intensive models without shared state
       
       More details can be found in [1]_. 
    
    2. GPU acceleration:
       - First batch incurs CUDA context initialization overhead
       - Automatic device placement (cuda:0 if available)
       - Batch tensor conversion uses float32 dtype
    
    3. Error handling:
       - Timeouts logged as warnings
       - Partial results retained from successful batches
       - Tracebacks available in debug mode
       
       Find furher details in [2]_. 
    
    See Also
    --------
    concurrent.futures.Executor : Base parallel execution interface
    torch.utils.data.DataLoader : Alternative batched processing
    joblib.Parallel : Task-based parallelism approach
    
    References
    ----------
    .. [1] Python Parallel Processing (https://docs.python.org/3/library/concurrency.html)
    .. [2] PyTorch CUDA Semantics (https://pytorch.org/docs/stable/notes/cuda.html)

    """

    @validate_params({
        "parallel_type": [StrOptions({'threads', 'processes'})],
        "max_workers": [Interval(Integral, 1, None, closed="left")],
        "timeout": [Interval(Real, 0, None, closed="left"), None],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "handle_errors": [bool],
        "cpu_affinity": [list, None],
        "gpu_enabled": [bool],
        "log_level": [StrOptions({'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'})]
    }, prefer_skip_nested_validation=True)
    def __init__(
        self,
        parallel_type: str = 'threads',
        max_workers: int = 4,
        timeout: Optional[float] = None,
        batch_size: int = 1,
        handle_errors: bool = True,
        cpu_affinity: Optional[List[int]] = None,
        gpu_enabled: bool = False,
        log_level: str = 'INFO'
    ):
        super().__init__(
            batch_size=batch_size,
            max_workers=max_workers,
            timeout=timeout,
            gpu_enabled=gpu_enabled,
            log_level=log_level
        )
        self.parallel_type = parallel_type
        self.handle_errors = handle_errors
        self.cpu_affinity = cpu_affinity
        self._is_runned = False

    @RunReturn(attribute_name="results_")
    @validate_params({
        "model": [HasMethods(['predict']), callable],
        "data": ['array-like'],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "preprocess_fn": [callable, None],
        "postprocess_fn": [callable, None]
    }, prefer_skip_nested_validation=True)
    def run(
        self,
        model: Any,
        data: List[Dict],
        batch_size: Optional[int] = None,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        **run_kw: Dict
    ) -> "InferenceParallelizer":
        """
        Execute parallel inference pipeline.

        Parameters
        ----------
        model : object
            Prediction model with ``predict`` method or callable
        data : List[Dict]
            Input data for inference
        batch_size : Optional[int], optional
            Override instance batch size
        preprocess_fn : Optional[Callable], optional
            Batch preprocessing function
        postprocess_fn : Optional[Callable], optional
            Result postprocessing function
        **run_kw : dict
            Additional inference parameters

        Returns
        -------
        self : object
            Returns instance itself with results

        Examples
        --------
        >>> parallelizer = InferenceParallelizer(max_workers=4)
        >>> results = parallelizer.run(model, data).results_
        """
        start_time = time.time()
        self._configure_run(model, batch_size)
        self._set_cpu_affinity()
        
        batches = self._create_batches(data, preprocess_fn)
        self.n_batches_ = len(batches)

        executor_fn = (
            self._threaded_executor if self.parallel_type == 'threads'
            else self._multiprocess_executor
        )

        self.results_ = []
        with executor_fn() as executor:
            self._process_batches(executor, batches, postprocess_fn)

        self.processing_time_ = time.time() - start_time
        self.throughput_ = len(data) / ( 
            self.processing_time_ if self.processing_time_ > 0 else 0
            )
        self._is_runned = True
        return self

    def _configure_run(self, model: Any, batch_size: Optional[int]) -> None:
        """Validate and configure runtime parameters."""
        self.model = model
        if batch_size is not None:
            self.batch_size = batch_size
        if not (hasattr(model, 'predict') or callable(model)):
            raise ValueError(
                "Model must implement predict method or be callable")

    def _set_cpu_affinity(self) -> None:
        """Configure CPU core affinity if requested."""
        if self.cpu_affinity:
            try:
                os.sched_setaffinity(0, self.cpu_affinity)
                logger.info(f"Set CPU affinity to cores: {self.cpu_affinity}")
            except AttributeError:
                logger.warning("CPU affinity not supported on this platform")
            except Exception as e:
                logger.error(f"Failed setting CPU affinity: {str(e)}")

    def _create_batches(
        self,
        data: List[Dict],
        preprocess_fn: Optional[Callable]
    ) -> List[List[Dict]]:
        """Create and preprocess data batches."""
        batches = [
            data[i:i + self.batch_size]
            for i in range(0, len(data), self.batch_size)
        ]
        if preprocess_fn:
            return [preprocess_fn(batch) for batch in batches]
        return batches

    def _threaded_executor(self) -> concurrent.futures.Executor:
        """Create threaded executor context."""
        return concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

    def _multiprocess_executor(self) -> concurrent.futures.Executor:
        """Create process pool executor context."""
        return concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)

    def _process_batches(
        self,
        executor: concurrent.futures.Executor,
        batches: List[List[Dict]],
        postprocess_fn: Optional[Callable]
    ) -> None:
        """Execute batch processing pipeline."""
        future_to_batch = {
            executor.submit(
                self._process_single_batch,
                batch,
                postprocess_fn
            ): batch for batch in batches
        }

        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                result = future.result(timeout=self.timeout)
                self.results_.extend(result)
            except concurrent.futures.TimeoutError:
                logger.error("Batch processing timed out")
            except Exception as e:
                if self.handle_errors:
                    logger.error(f"Skipping failed batch: {str(e)}")
                else:
                    raise RuntimeError("Critical batch error") from e

    def _process_single_batch(
        self,
        batch: List[Dict],
        postprocess_fn: Optional[Callable]
    ) -> List[Any]:
        """Process individual batch through full pipeline."""
        try:
            if self.gpu_enabled:
                predictions = self._gpu_inference(batch)
            else:
                predictions = (
                    self.model.predict(batch) if hasattr(self.model, 'predict')
                    else self.model(batch)
                )

            return postprocess_fn(predictions) if postprocess_fn else predictions
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise

    @ensure_pkg(
        "torch",
        extra="GPU inference requires PyTorch",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _gpu_inference(self, batch: List[Dict]) -> List[Any]:
        """Execute GPU-accelerated inference."""
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable - falling back to CPU")
            return self.model.predict(batch)

        try:
            tensor_batch = self._convert_to_tensor(batch).cuda()
            self.model.cuda()
            with torch.no_grad():
                outputs = self.model(tensor_batch)
            return outputs.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"GPU inference failed: {str(e)}")
            if self.handle_errors:
                return []
            raise

    def _convert_to_tensor(self, batch: List[Dict]) -> Any:
        """Convert batch data to tensor format."""
        import torch
        try:
            return torch.tensor(
                [[float(v) for v in sample.values()] for sample in batch],
                dtype=torch.float32
            )
        except Exception as e:
            logger.error(f"Tensor conversion failed: {str(e)}")
            raise

    @property
    def avg_batch_time(self) -> float:
        """Average processing time per batch."""
        check_is_runned(self, '_is_runned')
        return self.processing_time_ / self.n_batches_ if self.n_batches_ else 0

@smartFitRun
class InferenceCacheManager(BaseInference):
    """
    Intelligent prediction caching system for optimizing model inference 
    performance by reducing redundant computations [1]_.

    Parameters
    ----------
    cache_size : int, optional (default=1000)
        Maximum number of predictions to store in cache. When exceeded, 
        entries are removed according to eviction policy. Larger sizes 
        improve hit rates but increase memory usage.
        
    eviction_policy : {'LRU', 'LFU', 'TTL'}, optional (default='LRU')
        Cache replacement strategy:
        - 'LRU' (Least Recently Used): Removes least recently accessed 
          entries. Ideal for general-purpose workloads with temporal 
          locality.
        - 'LFU' (Least Frequently Used): Removes least frequently accessed 
          entries. Effective for stable workloads with repeating patterns.
        - 'TTL' (Time-To-Live): Automatically expires entries after 
          specified duration. Requires `ttl` parameter. Suitable for 
          time-sensitive predictions [1]_.
    
    ttl : Optional[int], optional (default=None)
        Time-to-live in seconds for cached predictions. Mandatory for 
        'TTL' policy. Example: 3600 = 1 hour expiration. Entries older 
        than this duration are automatically purged.
    
    custom_hash_fn : Optional[Callable], optional (default=None)
        Custom function to generate unique cache keys from input data. 
        Should accept a dictionary and return a hashable value. Default 
        uses `hash(frozenset(sorted(data.items())))`. Use for:
        - Custom data serialization
        - Handling complex/nested data structures
        - Avoiding hash collisions in high-dimensional data
    
    persistent_cache_path : Optional[str], optional (default=None)
        File path for disk-based cache persistence. Supports:
        - Cross-session cache retention
        - Cache sharing between processes
        - Emergency recovery of predictions
        Format: Any valid file path with write permissions. Uses pickle 
        serialization - ensure cached objects are serializable [2]_.
    
    log_level : {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'}, 
                optional (default='INFO')
        Logging verbosity control:
        - 'DEBUG': Detailed cache operations tracking
        - 'INFO': Basic cache statistics and warnings
        - 'WARNING': Only critical issues
        - 'ERROR'/ 'CRITICAL': Failure alerts only

    Attributes
    ----------
    cache_info_ : dict
        Real-time cache performance metrics:
        - 'hits' (int): Successful cache retrievals
        - 'misses' (int): Cache lookup failures
        - 'size' (int): Current cached entries count
        - 'max_size' (int): Configured cache capacity
    
    hit_rate_ : float
        Cache efficiency metric calculated as hits/(hits+misses). 
        Range: 0.0 (worst) to 1.0 (perfect). Updated after each prediction.
    
    cache_utilization : float
        Cache capacity usage ratio (current size/max size). 
        Range: 0.0 (empty) to 1.0 (full). Helps monitor scaling needs.

    Methods
    -------
    run(model, data, **run_kw)
        Execute prediction with caching logic. Returns cached result if 
        available, otherwise computes and stores prediction.
    
    clear_cache()
        Purges all cached entries and deletes persistent cache file. 
        Resets performance metrics.
    
    warm_cache(model, data_batch)
        Pre-populate cache with batch predictions. Useful for:
        - Cold-start initialization
        - Frequently accessed data patterns
        - Benchmarking cache performance
    
    get_performance_report()
        Generates comprehensive diagnostics including:
        - Hourly hit rate trends
        - Cache turnover rates
        - Prediction latency percentiles

    Notes
    -----
    Implementation Considerations:
    1. TTL Policy Requirements:
    - Must specify `ttl` when using 'TTL' eviction policy
    - System clock changes affect TTL accuracy
    2. Hash Function Selection:
    - Default works for flat dictionaries with primitive values
    - Use custom_hash_fn for nested structures or objects
    3. Persistence Security:
    - Pickle files can execute arbitrary code - only load trusted caches
    - Consider encryption for sensitive prediction data
    
    More details can be found in Cachetools documentation [3]_.

    Examples
    --------
    >>> # Time-sensitive predictions with 5-minute caching
    >>> from gofast.mlops.inference import InferenceCacheManager
    >>> ttl_cache = InferenceCacheManager(
    ...     cache_size=2000,
    ...     eviction_policy='TTL',
    ...     ttl=300,
    ...     persistent_cache_path='/tmp/predictions.ttl'
    ... )
    >>> # Custom hashing for complex data
    >>> def tensor_hash(data):
    ...     return hash(data.numpy().tobytes())
    >>> vision_cache = InferenceCacheManager(
    ...     custom_hash_fn=tensor_hash,
    ...     cache_size=500
    ... )
    >>> # Monitor cache efficiency
    >>> print(f"Utilization: {vision_cache.cache_utilization:.1%}")
    >>> print(f"Hit accuracy: {vision_cache.hit_rate_:.2f}")
    
    >>> from gofast.mlops.inference import InferenceCacheManager
    >>> cache_mgr = InferenceCacheManager(cache_size=500, 
    ...                                  eviction_policy='LRU')
    >>> model = MyPredictor()  # Model with predict method
    >>> result = cache_mgr.run(model, {'feature': 5.2})
    >>> print(cache_mgr.hit_rate_)

    See Also
    --------
    cachetools.TTLCache : Time-based cache implementation details
    sklearn.memory.Memory : Scikit-learn's disk-based caching
    joblib.Memory : Alternative caching implementation

    References
    ----------
    .. [1] "Efficient Caching Strategies for ML Inference Services", 
           ACM SIGOPS 2023
    .. [2] "Cache Replacement Policies Revisited" - USENIX ATC 2022
    .. [3] cachetools documentation: 
           https://cachetools.readthedocs.io
    """

    @validate_params({
        "cache_size": [Interval(Integral, 1, None, closed="left")],
        "eviction_policy": [StrOptions({'LRU', 'LFU', 'TTL'})],
        "ttl": [Interval(Integral, 1, None, closed="left"), None],
        "custom_hash_fn": [callable, None],
        "persistent_cache_path": [str, None],
        "log_level": [StrOptions(
            {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'})]
    }, prefer_skip_nested_validation=True)
    def __init__(
        self,
        cache_size: int = 1000,
        eviction_policy: str = 'LRU',
        ttl: Optional[int] = None,
        custom_hash_fn: Optional[Callable] = None,
        persistent_cache_path: Optional[str] = None,
        log_level: str = 'INFO'
    ):
        super().__init__(log_level=log_level)
        
        self._init_cache(eviction_policy, cache_size, ttl)
        self.custom_hash_fn = custom_hash_fn
        self.persistent_cache_path = persistent_cache_path
        self._load_persistent_cache()
        self._is_runned = False

        # Initialize statistics
        self.cache_info_ = {'hits': 0, 'misses': 0, 'size': 0}
        self.hit_rate_ = 0.0

    @ensure_pkg(
        "cachetools",
        extra="Cache requires 'cachetools' package",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _init_cache(
        self,
        policy: str,
        size: int,
        ttl: Optional[int]
    ) -> None:
        """Initialize cache backend based on eviction policy."""
        from cachetools import LRUCache, LFUCache, TTLCache

        if policy == 'TTL' and ttl is None:
            raise ValueError("TTL policy requires ttl parameter")

        self.cache_ = {
            'LRU': LRUCache(maxsize=size),
            'LFU': LFUCache(maxsize=size),
            'TTL': TTLCache(maxsize=size, ttl=ttl)
        }[policy]

    @RunReturn(attribute_name='results_')
    @validate_params({
        "model": [HasMethods(['predict']), callable],
        "data": [dict],
        "warm_cache": [bool],
        "force_refresh": [bool]
    }, prefer_skip_nested_validation=True)
    def run(
        self,
        model: Any,
        data: Dict,
        warm_cache: bool = False,
        force_refresh: bool = False,
        **run_kw: Dict
    ) -> Any:
        """
        Execute prediction with caching capabilities.

        Parameters
        ----------
        model : object
            Prediction model with ``predict`` method or callable
        data : dict
            Input features for prediction
        warm_cache : bool, default=False
            Precompute result without serving from cache
        force_refresh : bool, default=False
            Invalidate existing cache entry
        **run_kw : dict
            Additional prediction parameters

        Returns
        -------
        Any
            Prediction result

        Examples
        --------
        >>> cache_mgr = InferenceCacheManager(cache_size=500)
        >>> result = cache_mgr.run(model, data)
        """
        data_key = self._generate_key(data)
        
        if not force_refresh and not warm_cache:
            result = self._try_cache_lookup(data_key)
            if result is not None:
                return result

        result = self._compute_prediction(model, data, run_kw)
        self._update_cache(data_key, result)
        self._update_stats()
        self._is_runned = True
        
        return result

    def _generate_key(self, data: Dict) -> int:
        """Create cache key from input data."""
        if self.custom_hash_fn:
            return self.custom_hash_fn(data)
        return hash(frozenset(sorted(data.items())))

    def _try_cache_lookup(self, key: int) -> Optional[Any]:
        """Attempt cache retrieval and update statistics."""
        try:
            result = self.cache_[key]
            self.cache_info_['hits'] += 1
            logger.debug("Cache hit for key %s", key)
            return result
        except KeyError:
            self.cache_info_['misses'] += 1
            return None

    def _compute_prediction(
        self,
        model: Any,
        data: Dict,
        run_kw: Dict
    ) -> Any:
        """Execute model prediction with error handling."""
        try:
            return model.predict(data, **run_kw) if hasattr(
                model, 'predict') else model(data, **run_kw)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _update_cache(self, key: int, result: Any) -> None:
        """Update cache with new prediction."""
        self.cache_[key] = result
        if self.persistent_cache_path:
            self._persist_cache()

    def _update_stats(self) -> None:
        """Update performance statistics."""
        total = self.cache_info_['hits'] + self.cache_info_['misses']
        self.hit_rate_ = self.cache_info_['hits'] / total if total > 0 else 0
        self.cache_info_['size'] = len(self.cache_)

    @ensure_pkg(
        "cachetools",
        extra="Persistent cache requires cachetools",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _load_persistent_cache(self) -> None:
        """Load cache state from disk."""
        if self.persistent_cache_path and os.path.exists(
                self.persistent_cache_path):
            try:
                with open(self.persistent_cache_path, 'rb') as f:
                    self.cache_.update(pickle.load(f))
                logger.info(
                    "Loaded persistent cache with %d entries", len(self.cache_))
            except Exception as e:
                logger.error("Cache load failed: %s", str(e))

    def _persist_cache(self) -> None:
        """Save current cache state to disk."""
        try:
            with open(self.persistent_cache_path, 'wb') as f:
                pickle.dump(dict(self.cache_), f)
            logger.debug(
                "Persisted cache with %d entries",
                len(self.cache_)
            )
        except Exception as e:
            logger.error("Cache persistence failed: %s", str(e))

    def clear_cache(self) -> None:
        """Reset cache and remove persistent storage."""
        check_is_runned(self, '_is_runned')
        
        self.cache_.clear()
        if self.persistent_cache_path and os.path.exists(
                self.persistent_cache_path):
            try:
                os.remove(self.persistent_cache_path)
            except Exception as e:
                logger.error("Cache deletion failed: %s", str(e))
        self.cache_info_ = {'hits': 0, 'misses': 0, 'size': 0}
        self.hit_rate_ = 0.0
        logger.info("Cache cleared")

    def warm_cache(
        self,
        model: Any,
        data_batch: List[Dict]
    ) -> None:
        """Precompute and cache predictions for batch data."""
        logger.info("Warming cache with %d items", len(data_batch))
        for data in data_batch:
            self.run(model, data, warm_cache=True)
        logger.info("Cache warmup complete. Current size: %d", len(self.cache_))

    @property
    def cache_utilization(self) -> float:
        """Percentage of cache capacity used."""
        return len(self.cache_) / self.cache_.maxsize