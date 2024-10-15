# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:42:28 2024

@author: Daniel
"""

# test_scaling.py
import pytest
import logging
import threading
import time
import os
from typing import Any, Dict, List

from gofast.mlops.scaling import (
    ScalingManager,
    DataPipelineScaler,
    ElasticScaler,
    partition_data_pipeline,
    get_system_workload,
    elastic_scale_logic
)

# Set up logging for the tests
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture
def sample_pytorch_model():
    """Fixture for a simple PyTorch model."""
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    return SimpleModel()


@pytest.fixture
def sample_tensorflow_model():
    """Fixture for a simple TensorFlow model."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential([Dense(1, input_shape=(10,))])
    return model


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    import torch

    data = [(torch.randn(10), torch.randn(1)) for _ in range(100)]
    return data


def test_scaling_manager_initialize_cluster_pytorch():
    """Test ScalingManager.initialize_cluster() with PyTorch."""
    scaling_manager = ScalingManager(framework='pytorch')
    scaling_manager.initialize_cluster()
    # Since cluster initialization might not have observable effects in a test environment,
    # we check that no exceptions are raised and devices are detected.
    assert scaling_manager.devices_ is not None
    assert isinstance(scaling_manager.devices_, list)
    assert len(scaling_manager.devices_) > 0
    logger.info("PyTorch cluster initialized successfully.")


def test_scaling_manager_initialize_cluster_tensorflow():
    """Test ScalingManager.initialize_cluster() with TensorFlow."""
    scaling_manager = ScalingManager(framework='tensorflow')
    scaling_manager.initialize_cluster()
    assert scaling_manager.devices_ is not None
    assert isinstance(scaling_manager.devices_, list)
    assert len(scaling_manager.devices_) > 0
    logger.info("TensorFlow cluster initialized successfully.")


def test_scaling_manager_scale_training_pytorch(sample_pytorch_model, sample_data):
    """Test ScalingManager.scale_training() with a PyTorch model."""
    scaling_manager = ScalingManager(framework='pytorch')
    scaling_manager.scale_training(
        model=sample_pytorch_model,
        data=sample_data,
        epochs=1,
        optimizer=None,
        criterion=None
    )
    assert scaling_manager.model_ is not None
    logger.info("PyTorch model training scaled successfully.")


def test_scaling_manager_scale_training_tensorflow(sample_tensorflow_model):
    """Test ScalingManager.scale_training() with a TensorFlow model."""
    import numpy as np

    data = np.random.rand(100, 10)
    labels = np.random.rand(100, 1)

    scaling_manager = ScalingManager(framework='tensorflow')
    scaling_manager.scale_training(
        model=sample_tensorflow_model,
        data=(data, labels),
        epochs=1
    )
    assert scaling_manager.model_ is not None
    logger.info("TensorFlow model training scaled successfully.")


def test_data_pipeline_scaler():
    """Test DataPipelineScaler with multiprocessing backend."""

    def data_pipeline(partition):
        # Simulate data processing
        return [x * 2 for x in partition]

    data = list(range(100))
    scaler = DataPipelineScaler(num_partitions=4)
    results = scaler.scale_pipeline(data_pipeline_fn=data_pipeline, data=data)

    # Verify that the results are as expected
    assert len(results) == 4
    total_processed_items = sum(len(res) for res in results)
    assert total_processed_items == 100
    logger.info("DataPipelineScaler processed data successfully.")


def test_data_pipeline_scaler_dask():
    """Test DataPipelineScaler with Dask backend."""

    def data_pipeline(partition):
        # Simulate data processing
        return [x * 2 for x in partition]

    data = list(range(100))
    scaler = DataPipelineScaler(num_partitions=4, parallel_backend='dask')
    results = scaler.scale_pipeline(data_pipeline_fn=data_pipeline, data=data)

    # Verify that the results are as expected
    assert len(results) == 4
    total_processed_items = sum(len(res) for res in results)
    assert total_processed_items == 100
    logger.info("DataPipelineScaler with Dask processed data successfully.")


def test_elastic_scaler():
    """Test ElasticScaler's monitoring and scaling callbacks."""

    # Define scaling callbacks
    scale_up_triggered = False
    scale_down_triggered = False

    def scale_up(metrics):
        nonlocal scale_up_triggered
        scale_up_triggered = True
        logger.info(f"Scale up triggered with metrics: {metrics}")

    def scale_down(metrics):
        nonlocal scale_down_triggered
        scale_down_triggered = True
        logger.info(f"Scale down triggered with metrics: {metrics}")

    # Create an ElasticScaler instance
    scaler = ElasticScaler(
        scale_up_callback=scale_up,
        scale_down_callback=scale_down,
        cpu_threshold=0.0,  # Force scaling actions for the test
        memory_threshold=0.0,
        min_scale_up_duration=0.0,
        min_scale_down_duration=0.0
    )

    # Start monitoring in a separate thread
    monitoring_thread = threading.Thread(target=scaler.start_monitoring)
    monitoring_thread.start()

    # Allow some time for monitoring
    time.sleep(2)

    # Stop monitoring
    scaler.stop_monitoring()
    monitoring_thread.join()

    # Check if scaling callbacks were triggered
    assert scale_up_triggered or scale_down_triggered
    logger.info("ElasticScaler scaling callbacks were triggered successfully.")


def test_partition_data_pipeline():
    """Test partition_data_pipeline function."""

    def data_pipeline_fn(partition_index):
        # Example data processing logic
        return [partition_index * i for i in range(5)]

    # Test with 'even' partition strategy
    partitions = partition_data_pipeline(
        data_pipeline_fn=data_pipeline_fn,
        num_partitions=3,
        partition_strategy='even'
    )
    assert len(partitions) == 3
    assert partitions[0] == [0, 0, 0, 0, 0]
    logger.info("partition_data_pipeline with 'even' strategy executed successfully.")

    # Test with 'random' partition strategy
    partitions_random = partition_data_pipeline(
        data_pipeline_fn=data_pipeline_fn,
        num_partitions=3,
        partition_strategy='random'
    )
    assert len(partitions_random) == 3
    logger.info("partition_data_pipeline with 'random' strategy executed successfully.")

    # Test with 'custom' partition strategy
    def custom_partition_fn(partition_index):
        return [partition_index + i for i in range(5)]

    partitions_custom = partition_data_pipeline(
        data_pipeline_fn=data_pipeline_fn,
        num_partitions=3,
        partition_strategy='custom',
        custom_partition_fn=custom_partition_fn
    )
    assert len(partitions_custom) == 3
    assert partitions_custom[0] == [0, 1, 2, 3, 4]
    logger.info("partition_data_pipeline with 'custom' strategy executed successfully.")


def test_get_system_workload():
    """Test get_system_workload function."""

    def mock_latency_fn():
        return 100.0  # Mock latency in milliseconds

    def mock_queue_length_fn():
        return 5  # Mock queue length

    def mock_custom_metrics_fn():
        return {'custom_metric': 42}

    workload = get_system_workload(
        include_gpu=False,
        monitor_latency_fn=mock_latency_fn,
        queue_length_fn=mock_queue_length_fn,
        custom_metrics_fn=mock_custom_metrics_fn,
        additional_metrics=True
    )

    assert 'cpu_usage' in workload
    assert 'memory_usage' in workload
    assert 'latency' in workload
    assert 'queue_length' in workload
    assert 'custom_metric' in workload
    assert 'disk_io' in workload
    assert 'network_io' in workload
    logger.info("get_system_workload retrieved all metrics successfully.")


def test_elastic_scale_logic():
    """Test elastic_scale_logic function with mock workload data."""

    # Define scaling callbacks
    scale_up_triggered = False
    scale_down_triggered = False

    def scale_up(workload):
        nonlocal scale_up_triggered
        scale_up_triggered = True
        logger.info(f"Scale up action performed with workload: {workload}")

    def scale_down(workload):
        nonlocal scale_down_triggered
        scale_down_triggered = True
        logger.info(f"Scale down action performed with workload: {workload}")

    # Mock workload data
    workload_data = {'cpu_usage': 85.0, 'memory_usage': 70.0}
    scale_up_thresholds = {'cpu_usage': 80.0, 'memory_usage': 75.0}
    scale_down_thresholds = {'cpu_usage': 30.0, 'memory_usage': 25.0}

    # Call elastic_scale_logic multiple times to simulate sustained high usage
    for _ in range(5):
        elastic_scale_logic(
            workload_data=workload_data,
            scale_up_thresholds=scale_up_thresholds,
            scale_down_thresholds=scale_down_thresholds,
            scale_up_callback=scale_up,
            scale_down_callback=scale_down,
            min_scale_up_duration=0.1,
            scaling_sensitivity=0.05,
            cooldown_period=0.0  # No cooldown for testing
        )
        time.sleep(0.1)

    assert scale_up_triggered
    logger.info("elastic_scale_logic triggered scale up successfully.")

    # Simulate low usage for scaling down
    scale_up_triggered = False  # Reset trigger
    workload_data = {'cpu_usage': 25.0, 'memory_usage': 20.0}

    for _ in range(10):
        elastic_scale_logic(
            workload_data=workload_data,
            scale_up_thresholds=scale_up_thresholds,
            scale_down_thresholds=scale_down_thresholds,
            scale_up_callback=scale_up,
            scale_down_callback=scale_down,
            min_scale_down_duration=0.2,
            scaling_sensitivity=0.05,
            cooldown_period=0.0
        )
        time.sleep(0.1)

    assert scale_down_triggered
    logger.info("elastic_scale_logic triggered scale down successfully.")

if __name__=='__main__': 
    pytest.main( [__file__])