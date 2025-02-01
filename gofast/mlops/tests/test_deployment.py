# -*- coding: utf-8 -*-

import pytest
import logging
import sys
# import numpy as np
IS_TORCH_INSTALLED=True 
IS_TF_INSTALLED=True 

try: 
    import torch
except : 
    IS_TORCH_INSTALLED=False 
    pass 
try: 
    import tensorflow as tf
except: 
    IS_TF_INSTALLED=False 
    pass 

try:
    from fastapi.testclient import TestClient
except: 
    pass 
# from flask import Flask, jsonify
from unittest.mock import patch, MagicMock

from gofast.utils.deps_utils import is_module_installed 
from gofast.mlops.deployment import (
    ModelExporter,
    APIDeployment,
    CloudDeployment,
    ABTesting
)

from gofast.mlops._config import INSTALL_DEPENDENCIES 

INSTALL_DEPENDENCIES =True # noqa # install dependencies during the test

# Disable logging during tests to keep output clean
logging.disable(sys.maxsize)


@pytest.mark.skipif (
    not IS_TORCH_INSTALLED, 
    reason="needs 'torch' for the test to proceed..."
    )
@pytest.fixture
def sample_pytorch_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    return model

@pytest.mark.skipif (
    not IS_TF_INSTALLED, 
    reason="needs 'tensorflow' for the test to proceed..."
    )
@pytest.fixture
def sample_tensorflow_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

@pytest.mark.skipif (
    not IS_TORCH_INSTALLED, 
    reason="needs 'torch' for the test to proceed..."
    )
class TestModelExporter:
    def test_onnx_export(self, sample_pytorch_model, tmp_path):
  
        
        exporter = ModelExporter().run(sample_pytorch_model)
        export_path = tmp_path / "model.onnx"
        
        exporter.export_to_onnx(str(export_path))
        assert export_path.exists()
        assert export_path.stat().st_size > 0

    
    def test_quantization(self, sample_pytorch_model):
        
        exporter = ModelExporter().run(sample_pytorch_model)
        calibration_data = torch.randn(1, 10)
        
        exporter.compress_model(
            method='static',
            calibration_data=calibration_data
        )
        assert hasattr(exporter.model, 'qconfig')


@pytest.mark.skipif (
    not is_module_installed("fastapi"), 
    reason="needs 'fastapi' for the test to proceed..."
    )
class TestAPIDeployment:
    def test_fastapi_endpoints(self, sample_pytorch_model):

        
        api = APIDeployment(api_type='FastAPI').run(sample_pytorch_model)
        api.create_api()
        
        client = TestClient(api.app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_rate_limiting(self, sample_pytorch_model):

        from fastapi import HTTPException
        
        api = APIDeployment(max_requests=2).run(sample_pytorch_model)
        api.create_api()
        
        with pytest.raises(HTTPException) as excinfo:
            for _ in range(3):
                api._check_rate_limit()
        assert excinfo.value.status_code == 429

class TestCloudDeployment:
    @patch('boto3.client')
    def test_aws_deployment(self, mock_boto, sample_pytorch_model):
 
        
        mock_sagemaker = MagicMock()
        mock_boto.return_value = mock_sagemaker
        
        deployer = CloudDeployment().run(sample_pytorch_model)
        config = {
            'model_data': 's3://bucket/model',
            'role_arn': 'arn:aws:iam::123456789012:role/sagemaker'
        }
        
        result = deployer.deploy_to_aws(config)
        mock_sagemaker.create_model.assert_called_once()
        assert 'arn' in result

    @patch('google.cloud.aiplatform.Model')
    def test_gcp_deployment(self, mock_model, sample_tensorflow_model):

        
        mock_endpoint = MagicMock()
        mock_model.return_value = mock_endpoint
        
        deployer = CloudDeployment().run(sample_tensorflow_model)
        config = {
            'project': 'test-project',
            'bucket': 'gs://test-bucket'
        }
        
        result = deployer.deploy_to_gcp(config)
        mock_model.upload.assert_called_once()
        assert 'uri' in result

class TestABTesting:
    def test_traffic_routing(self):
    
        
        model_v1 = MagicMock(predict=MagicMock(return_value=0.8))
        model_v2 = MagicMock(predict=MagicMock(return_value=0.7))
        
        ab_test = ABTesting(split_ratio=0.8).run(model_v1, model_v2)
        results = [ab_test.route_traffic({}) for _ in range(100)]
        
        v1_count = sum(1 for r in results if r == 0.8)
        assert 70 < v1_count < 90  # Allow statistical variation

    def test_performance_adjustment(self):

        
        ab_test = ABTesting(performance_threshold=0.1)
        ab_test.run(MagicMock(), MagicMock())
        
        initial_ratio = ab_test.split_ratio
        ab_test.evaluate_performance({'model_v1': 0.9, 'model_v2': 0.7})
        
        assert ab_test.split_ratio == initial_ratio + ab_test.traffic_increment

    def test_graceful_degradation(self):

        ab_test = ABTesting(graceful_degradation=True)
        ab_test.run(MagicMock(), MagicMock())
        
        ab_test.evaluate_performance({'model_v1': 0.4, 'model_v2': 0.3})
        assert ab_test.split_ratio == 0.5


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_deployment.py"])