# -*- coding: utf-8 -*-
# test_inference.py

import pytest
import json
import logging
import time

from unittest.mock import MagicMock, patch
import numpy as np

from gofast.utils.deps_utils import is_module_installed 
# Import the classes from the gofast.mlops.inference module
from gofast.mlops.inference import (
    BatchInference,
    StreamingInference,
    MultiModelServing,
    InferenceParallelizer,
    InferenceCacheManager
)

# Disable logging during tests to keep the output clean
logging.disable(logging.CRITICAL)

@pytest.mark.skipif(
    not is_module_installed("torch"), 
    reason ="'torch is required for the test to proceed."
 )
def test_batch_inference_initialization():
    """Test BatchInference initialization with default parameters"""

    processor = BatchInference()
    assert processor.batch_size == 32
    assert processor.max_workers == 4
    assert processor.optimize_memory is True
    assert processor.gpu_enabled is False
    assert processor.enable_padding is False

@pytest.mark.skipif(
    not is_module_installed("torch"), 
    reason ="'torch is required for the test to proceed."
 )
@pytest.mark.parametrize("batch_size, padding", [
    (10, True),
    (50, False),
    (100, True)
])
def test_batch_creation(batch_size, padding):
    """Test batch creation with different configurations"""

    data = [{"features": np.random.rand(10)} for _ in range(95)]
    processor = BatchInference(
        batch_size=batch_size,
        enable_padding=padding
    )
    
    batches = processor._create_batches(data)
    assert len(batches) == np.ceil(95 / batch_size)
    
    if padding:
        last_batch = batches[-1]
        assert len(last_batch) == batch_size
    else:
        assert len(batches[-1]) == 95 % batch_size or batch_size

@pytest.fixture
def sample_model():
    """Mock model with predict method"""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.zeros((10, 2)))
    return model

@pytest.mark.skipif(
    not is_module_installed("torch"), 
    reason ="'torch is required for the test to proceed."
 )
def test_batch_inference_run(sample_model):
    """Test complete batch processing pipeline"""

    data = [{"features": np.random.rand(10)} for _ in range(100)]
    processor = BatchInference(batch_size=10)
    
    result = processor.run(sample_model, data)
    assert len(result.results_) == 100
    assert processor.n_batches_ == 10
    assert processor.processing_time_ > 0
    assert processor.avg_batch_time > 0

@pytest.mark.skipif(
    not is_module_installed("torch"), 
    reason ="'torch is required for the test to proceed."
 )
def test_gpu_inference_handling(sample_model):
    """Test GPU support detection and handling"""

    # Test CUDA-enabled model
    sample_model.parameters = lambda: [MagicMock(is_cuda=True)]
    processor = BatchInference(gpu_enabled=True)
    processor._check_gpu_support(sample_model)
    assert processor.gpu_enabled is True

@pytest.mark.skipif(
    not is_module_installed("kafka"), 
    reason ="'kafka is required for the test to proceed."
 )
@patch('gofast.mlops.inference.ensure_pkg')
def test_streaming_kafka_clients(mock_ensure):
    """Test Kafka client initialization"""
    
    stream = StreamingInference(
        kafka_topic="test",
        kafka_servers=["localhost:9092"],
        group_id="test-group"
    )
    stream._init_kafka_clients()
    
    assert hasattr(stream, 'consumer_')
    assert hasattr(stream, 'producer_')
    assert stream.consumer_.topic == "test"

@pytest.fixture
def mock_kafka():
    """Mock Kafka consumer with test messages"""
    consumer = MagicMock()
    messages = [
        MagicMock(value=json.dumps({"sensor_id": i, "value": i*10}))
        for i in range(20)
    ]
    consumer.__iter__ = lambda self: iter(messages)
    return consumer

@pytest.mark.skipif(
    not is_module_installed("kafka"), 
    reason ="'kafka is required for the test to proceed."
 )
def test_streaming_processing(mock_kafka, sample_model):
    """Test complete streaming processing cycle"""
    
    with patch('kafka.KafkaConsumer', return_value=mock_kafka):
        stream = StreamingInference(
            kafka_topic="test",
            kafka_servers=["localhost:9092"],
            group_id="test-group",
            batch_size=10
        )
        stream.run(sample_model)
        time.sleep(0.1)
        stream.stop_streaming().shutdown()
        
        assert stream.n_processed_ == 20
        assert stream.throughput_ > 0

@pytest.mark.skipif(
    not is_module_installed("kafka"), 
    reason ="'kafka is required for the test to proceed."
 )
def test_streaming_error_handling(sample_model):
    """Test error containment in streaming pipeline"""

    from kafka.errors import KafkaError
    
    # Setup producer that fails 2 times before success
    producer = MagicMock()
    producer.send.side_effect = [KafkaError(), KafkaError(), None]
    
    with patch('kafka.KafkaProducer', return_value=producer):
        stream = StreamingInference(
            kafka_topic="test",
            kafka_servers=["localhost:9092"],
            group_id="test-group",
            max_retries=3
        )
        stream.producer_ = producer
        predictions = [{"prediction": 1.0}]
        
        stream._deliver_predictions(predictions)
        assert producer.send.call_count == 3

@pytest.mark.skipif(
    not is_module_installed("kafka"), 
    reason ="'kafka is required for the test to proceed."
 )
def test_throughput_calculation():
    """Test throughput calculation after shutdown"""

    
    stream = StreamingInference(
        kafka_topic="test",
        kafka_servers=["localhost:9092"],
        group_id="test-group"
    )
    stream.n_processed_ = 1000
    stream._processing_stats = [0.1 for _ in range(100)]
    stream.shutdown()
    
    assert stream.throughput_ == 1000 / 10.0  # 100 batches * 0.1s = 10s

@pytest.mark.skipif(
    not is_module_installed("torch"), 
    reason ="'torch is required for the test to proceed."
 )
@pytest.mark.parametrize("data_size, batch_size, expected", [
    (0, 32, 0),  # Empty data
    (1, 32, 1),  # Single item
    (100, 10, 10)  # Exact batches
])
def test_edge_cases(data_size, batch_size, expected):
    """Test edge cases in batch processing"""
    
    data = [{"features": np.zeros(10)} for _ in range(data_size)]
    processor = BatchInference(batch_size=batch_size)
    model = MagicMock()
    model.predict.return_value = np.zeros((batch_size, 1))
    
    processor.run(model, data)
    assert processor.n_batches_ == expected


# Fixtures for reusable test components
@pytest.fixture
def sample_models():
    """Mock model registry for multi-model testing"""
    return {
        'model_a': MagicMock(predict=MagicMock(return_value=0)),
        'model_b': MagicMock(predict=MagicMock(return_value=1)),
        'model_c': MagicMock(predict=MagicMock(return_value=2))
    }

@pytest.fixture
def sample_data():
    """Sample input data for inference requests"""
    return {'features': [1.2, 3.4, 5.6]}

# MultiModelServing Tests
class TestMultiModelServing:
    """Test suite for MultiModelServing functionality"""
    
    def test_initialization(self, sample_models):
        """Test basic initialization with valid models"""
        
        serving = MultiModelServing(models=sample_models)
        assert len(serving.models) == 3
        assert 'model_a' in serving.model_names
        
    def test_traffic_split_validation(self, sample_models):
        """Test traffic split validation logic"""
        
        with pytest.raises(ValueError):
            MultiModelServing(
                models=sample_models,
                traffic_split={'model_a': 0.5, 'model_b': 0.6}
            )
            
    def test_model_selection_priority(self, sample_models):
        """Test model selection priority order"""
        
        serving = MultiModelServing(
            models=sample_models,
            traffic_split={'model_a': 1.0},
            performance_weights={'model_b': 1.0},
            latency_threshold=1.0
        )
        
        # Test explicit model selection
        model, name = serving._select_model('model_c')
        assert name == 'model_c'
        
        # Test performance-based selection
        serving.performance_metrics_['model_b']['success'] = 10
        model, name = serving._select_model(None)
        assert name == 'model_b'
        
    def test_fallback_mechanism(self, sample_models, sample_data):
        """Test error handling and fallback workflow"""
        
        # Configure failing primary model
        sample_models['model_a'].predict.side_effect = Exception("Test error")
        serving = MultiModelServing(
            models=sample_models,
            fallback_models={'model_a': ['model_b', 'model_c']},
            retry_attempts=2
        )
        
        result = serving.run(sample_data)
        assert result == 1  # First fallback result
        assert serving.fallback_count_ == 1
        
    def test_performance_tracking(self, sample_models, sample_data):
        """Test metrics collection and reporting"""
        
        serving = MultiModelServing(models=sample_models)
        serving.run(sample_data)
        report = serving.get_performance_report()
        
        assert report['model_a']['success'] == 1
        assert serving.request_count_ == 1

@pytest.mark.skipif(
    not is_module_installed("torch"), 
    reason ="'torch is required for the test to proceed."
 )
# InferenceParallelizer Tests
class TestInferenceParallelizer:
    """Test suite for parallel inference execution"""
    
    @pytest.mark.parametrize("parallel_type", ['threads', 'processes'])
    def test_executor_selection(self, parallel_type):
        """Test correct executor initialization"""

        parallelizer = InferenceParallelizer(parallel_type=parallel_type)
        assert parallelizer.parallel_type == parallel_type
        
    def test_batch_processing(self):
        """Test complete parallel processing workflow"""
        
        model = MagicMock(predict=MagicMock(side_effect=lambda x: x))
        data = [{'features': i} for i in range(100)]
        parallelizer = InferenceParallelizer(batch_size=10)
        
        results = parallelizer.run(model, data).results_
        assert len(results) == 100
        assert parallelizer.n_batches_ == 10
        
    def test_gpu_execution(self):
        """Test GPU acceleration path"""
        
        with patch('torch.cuda.is_available', return_value=True):
            parallelizer = InferenceParallelizer(gpu_enabled=True)
            model = MagicMock()
            data = [{'features': [1.2, 3.4]}]
            
            parallelizer.run(model, data)
            assert model.cuda.called
            
    def test_cpu_affinity(self):
        """Test CPU core affinity configuration"""
        
        with patch('os.sched_setaffinity') as mock_affinity:
            parallelizer = InferenceParallelizer(cpu_affinity=[0,1])
            parallelizer._set_cpu_affinity()
            mock_affinity.assert_called_once_with(0, [0,1])

# InferenceCacheManager Tests
@pytest.mark.skipif(
    not is_module_installed("cachetools"), 
    reason ="'cachetools is required for the test to proceed."
 )
class TestInferenceCacheManager:
    """Test suite for prediction caching system"""
    
    @pytest.mark.parametrize("policy,ttl", [
        ('LRU', None),
        ('LFU', None),
        ('TTL', 60)
    ])
    def test_cache_policies(self, policy, ttl):
        """Test initialization of different cache policies"""
        
        cache = InferenceCacheManager(
            eviction_policy=policy,
            ttl=ttl,
            cache_size=100
        )
        assert cache.cache_info_['size'] == 0
        
    def test_cache_hit_behavior(self):
        """Test cache hit/miss tracking and result return"""
        
        model = MagicMock(predict=MagicMock(return_value=42))
        cache = InferenceCacheManager(cache_size=10)
        
        # First call - cache miss
        cache.run(model, {'features': 5})
        assert model.predict.call_count == 1
        assert cache.cache_info_['misses'] == 1
        
        # Second call - cache hit
        cache.run(model, {'features': 5})
        assert model.predict.call_count == 1
        assert cache.cache_info_['hits'] == 1
        
    def test_persistent_cache(self, tmp_path):
        """Test cache persistence to disk"""
        
        cache_file = tmp_path / "test_cache.pkl"
        model = MagicMock(predict=MagicMock(return_value=99))
        
        # Initial run and save
        cache1 = InferenceCacheManager(persistent_cache_path=str(cache_file))
        cache1.run(model, {'data': 1})
        assert cache_file.exists()
        
        # Load from persistent cache
        cache2 = InferenceCacheManager(persistent_cache_path=str(cache_file))
        result = cache2.run(model, {'data': 1})
        assert result == 99
        assert model.predict.call_count == 1  # Only called once overall
        
    def test_ttl_expiration(self):
        """Test time-based cache expiration"""
        
        with patch('time.time') as mock_time:
            cache = InferenceCacheManager(
                eviction_policy='TTL',
                ttl=10,
                cache_size=5
            )
            
            mock_time.return_value = 0
            cache.run(MagicMock(), {'data': 1})  # Stored at t=0
            
            mock_time.return_value = 9
            assert cache.cache_info_['size'] == 1  # Still valid
            
            mock_time.return_value = 11
            assert cache.cache_info_['size'] == 0  # Expired

# Shared Test Utilities
def test_throughput_calculation2():
    """Test throughput calculation logic"""
    
    parallelizer = InferenceParallelizer()
    parallelizer.processing_time_ = 2.5
    parallelizer.results_ = list(range(1000))
    
    assert parallelizer.throughput_ == 400  # 1000 / 2.5

def test_error_handling_modes():
    """Test error handling configuration impacts"""
    
    model = MagicMock(side_effect=Exception("Test error"))
    data = [{'features': i} for i in range(10)]
    
    # Test error suppression
    parallelizer = InferenceParallelizer(handle_errors=True)
    results = parallelizer.run(model, data)
    assert len(results.results_) == 0
    
    # Test error propagation
    parallelizer.handle_errors = False
    with pytest.raises(ValueError):
        parallelizer.run(model, data)
# Re-enable logging after tests
logging.disable(logging.NOTSET)

if __name__=='__main__': 
    pytest.main([__file__])

