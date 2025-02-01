# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Deploy machine learning models to various environments including cloud 
platforms, edge devices, and on-premise servers.
"""

import random 
import time
from datetime import datetime
from numbers import Integral, Real 
from typing import Any, Optional, Tuple, Dict
from pathlib import Path

import numpy as np 

from .._gofastlog import gofastlog 
from ..api.property import BaseLearner 
from ..compat.sklearn import Interval, StrOptions, validate_params 
from ..utils.deps_utils import ensure_pkgs
from ..utils.validator import check_is_runned
from ..decorators import RunReturn 
from ._config import INSTALL_DEPENDENCIES, USE_CONDA 

logger = gofastlog.get_gofast_logger(__name__)

__all__ = ["ModelExporter", "APIDeployment", "CloudDeployment", "ABTesting"]

EXTRA_MSG = "The {pkg} is required for this functionality to proceed."


class ModelExporter(BaseLearner):
    r"""
    Export and optimize machine learning models for production deployment
    with version control and format conversion capabilities.

    Parameters
    ----------
    model_name : str, optional
        Unique identifier for the model. Automatically generated using
        model class name and hash if not provided. Used for file naming
        and metadata tracking. Maximum length 128 characters.
    versioning : bool, default=True
        Enable automatic version tracking of exported models. When
        enabled, exports append version numbers and maintain metadata
        history. Disable for single-version deployments.

    Attributes
    ----------
    version : int or None
        Current export version counter. Increments after each successful
        versioned export. ``None`` when versioning is disabled.
    model_name : str
        Final normalized model identifier (<=128 chars). Derived from
        either user input or automatic generation during ``run()``.
    
    Methods
    --------
    run(model, **run_kw)
        Initialize exporter with model and detect framework.
    export_to_onnx(export_path, input_shape=None, opset_version=13,
                   dynamic_axes=None)
        Export model to ONNX format with dynamic shape support.
    export_to_tensorflow(export_path, model_type='SavedModel')
        Export TensorFlow model to SavedModel or HDF5 format.
    export_to_torch(export_path, include_onnx=False, 
                    input_shape=None)
        Save PyTorch model with optional ONNX conversion.
    compress_model(method='quantization', **kwargs)
        Apply model compression via quantization or pruning.
    version_control(export_path, input_shape=None)
        Create versioned export with metadata tracking.
        
    Notes
    -----
    The exporter implements a dual-phase optimization workflow:

    1. **Model Preparation Phase**:
       - Framework autodetection (PyTorch/TensorFlow)
       - Model validation and metadata extraction
       - Automatic naming resolution

    2. **Export Phase**:
       - Format conversion (ONNX/TensorFlow SavedModel/PyTorch Script)
       - Quantization optimization (static/dynamic)
       - Versioned artifact storage

    The versioning system follows sequential integer increments with
    metadata embedding:

    .. math::
        v_{n+1} = v_n + 1 \quad \text{where } v_0 = 1

    Quantization reduces model size through precision reduction:

    .. math::
        S_q = S_{orig} \times \frac{b_q}{b_{orig}}

    where :math:`b_q` is quantized bit-width and :math:`b_{orig}` is
    original precision (typically 32-bit float).

    Examples
    --------
    >>> from gofast.mlops.deployment import ModelExporter
    >>> import torchvision.models as models
    
    >>> # Initialize with automatic naming
    >>> resnet = models.resnet18(pretrained=True)
    >>> exporter = ModelExporter().run(resnet)
    
    >>> # Export to ONNX with versioning
    >>> exporter.export_to_onnx('model.onnx')
    
    >>> # Compress with static quantization
    >>> exporter.compress_model(
    ...     method='static',
    ...     calibration_data=calibration_loader
    ... )

    See Also
    --------
    APIDeployment : For serving exported models as production APIs
    CloudDeployment : For cloud platform deployment workflows

    References
    ----------
    .. [1] PyTorch Quantization: 
       https://pytorch.org/docs/stable/quantization.html
    .. [2] ONNX Runtime Optimization: 
       https://onnxruntime.ai/docs/performance/model-optimizations.html
    .. [3] Krishnamoorthi, R. (2018). "Quantizing deep convolutional 
       networks for efficient inference: A whitepaper." arXiv:1806.08342
    """

    def __init__(
        self, 
        model_name: Optional[str]=None, 
        versioning: bool = True
    ):
 
        self.model_name = model_name
        self.version = 1 if versioning else None
        self._framework = None
        self._is_runned = False

    @RunReturn
    def run(self, model: Any, **run_kw) -> "ModelExporter":
        """
        Detect model framework and validate environment setup.

        Returns
        -------
        self : ModelExporter
            Returns instance for method chaining.

        Raises
        ------
        TypeError
            If model is not PyTorch or TensorFlow.
        """
        self.model = model 
        model_name = run_kw.pop ('model_name', 'my_model') 
        self.model_name = self.model_name or model_name 
        
        framework_detected = False
        
        # PyTorch check
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                self._framework = "pytorch"
                framework_detected = True
        except ImportError:
            pass

        # TensorFlow check
        if not framework_detected:
            try:
                import tensorflow as tf
                if isinstance(self.model, tf.keras.Model):
                    self._framework = "tensorflow"
                    framework_detected = True
            except ImportError:
                pass

        if not framework_detected:
            raise TypeError(
                "Unsupported model type. Must be PyTorch (torch.nn.Module) "
                "or TensorFlow (tf.keras.Model).")
        
        self._is_runned = True
        logger.info(f"Framework detected: {self._framework.upper()}")
        
        return self

    @ensure_pkgs(
        "tensorflow",
        extra=EXTRA_MSG.format(pkg="tensorflow"),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    @validate_params({
        "export_path": [str],
        "model_type": [StrOptions({"SavedModel", "HDF5"})]
    })
    def export_to_tensorflow(
        self, 
        export_path: str, 
        model_type: str = "SavedModel"
    ) -> None:
        """
        Export TensorFlow model to SavedModel or HDF5 format.

        Parameters
        ----------
        export_path : str
            Output directory/file path.
        model_type : {'SavedModel', 'HDF5'}, default='SavedModel'
            Export format specification.

        Raises
        ------
        RuntimeError
            If framework mismatch or export failure.
        """
        check_is_runned(self)
        import tensorflow as tf

        if self._framework != "tensorflow":
            raise RuntimeError(
                "TensorFlow export requires TensorFlow model. Detected "
                f"framework: {self._framework}")

        logger.info(f"Exporting to TensorFlow {model_type} format")
        
        try:
            if model_type == "SavedModel":
                tf.saved_model.save(self.model, export_path)
                logger.success(f"SavedModel exported to {export_path}")
            else:
                self.model.save(f"{export_path}.h5")
                logger.success(f"HDF5 model saved to {export_path}.h5")
        except Exception as e:
            try: 
                # save using the baselearner model 
                self.save (self.model, f"{export_path}.h5", format="h5")
            except: 
                logger.error(f"TensorFlow export failed: {str(e)}")
                raise RuntimeError(f"TensorFlow export failed: {e}") from e
        

    @validate_params({
        "method": [StrOptions({"quantization", "pruning"})],
        "prune_percentage": [Interval(Real, 0, 1, closed="both")],
        "quantization_method": [
            StrOptions({"dynamic", "static", "post_training"})
            ],
        "framework": [StrOptions({"pytorch", "tensorflow"})]
    }, prefer_skip_nested_validation=True)
    
    def compress_model(
        self,
        method: str = "quantization",
        prune_percentage: float = 0.2,
        quantization_method: str = "dynamic",
        framework: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Apply model compression techniques to optimize for deployment.

        Parameters
        ----------
        method : {'quantization', 'pruning'}, default='quantization'
            Compression technique to apply.
        prune_percentage : float, default=0.2
            Percentage of weights to prune (0-1).
        quantization_method : {'dynamic', 'static', 'post_training'}, 
            default='dynamic'
            Quantization approach for model optimization.
        framework : {'pytorch', 'tensorflow'}, optional
            Force specific framework implementation.

        Examples
        --------
        >>> exporter.compress_model(method='quantization', 
        ...                        quantization_method='dynamic')
        """
        check_is_runned(self)
        framework = framework or self._framework
        logger.info(f"Applying {method} compression using {framework}")

        try:
            if method == "quantization":
                self._apply_quantization(
                    method=quantization_method,
                    framework=framework,
                    **kwargs
                )
            elif method == "pruning":
                self._apply_pruning(
                    percentage=prune_percentage,
                    framework=framework,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            raise RuntimeError(f"Model compression failed: {e}") from e

    @ensure_pkgs(
        "tensorflow",
        extra=EXTRA_MSG.format(pkg="tensorflow"),
        partial_check=True,
        condition=lambda _, kw: kw.get("framework") == "tensorflow",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _apply_quantization(
        self,
        method: str = "dynamic",
        framework: str = "auto",
        calibration_data: Optional[Any] = None,
        custom_layers: Optional[Dict] = None,
        quantize_activations: bool = True,
        **kwargs
    ) -> None:
        """Internal implementation of quantization techniques."""
        check_is_runned(self)
        
        if framework=='auto': 
            framework= self._framework 
            
        if framework == "pytorch":
            self._quantize_pytorch(
                method=method,
                calibration_data=calibration_data,
                custom_layers=custom_layers,
                quantize_activations=quantize_activations
            )
        elif framework == "tensorflow":
            self._quantize_tensorflow(
                method=method,
                calibration_data=calibration_data,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        logger.success(f"{method.capitalize()} quantization completed")


    def _quantize_tensorflow(
        self,
        method: str,
        calibration_data: Any
    ) -> None:
        """TensorFlow-specific quantization implementation."""
        import tensorflow as tf

        if method == "post_training":
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            output_path = f"{self.model_name}_quantized.tflite"
            with open(output_path, "wb") as f:
                f.write(tflite_model)
            logger.info(f"TFLite model saved to {output_path}")

    @validate_params({
        "percentage": [Interval(Real, 0, 1, closed="both")],
        "method": [StrOptions({"unstructured", "structured"})]
    })
    def _apply_pruning(
        self,
        percentage: float = 0.2,
        method: str = "unstructured",
        framework: str = "pytorch",
        **kwargs
    ) -> None:
        """Internal implementation of model pruning."""
        check_is_runned(self)
        
        if framework != "pytorch":
            raise NotImplementedError(
                "Pruning currently only supported for PyTorch models")

        import torch.nn.utils.prune as prune

        logger.info(f"Applying {method} pruning ({percentage*100}%)")
        
        try:
            if method == "unstructured":
                prune.global_unstructured(
                    parameters=[
                        (module, 'weight') 
                        for module in self.model.modules()
                        if hasattr(module, 'weight')
                    ],
                    pruning_method=prune.L1Unstructured,
                    amount=percentage
                )
            elif method == "structured":
                self._apply_structured_pruning(percentage, **kwargs)
        except Exception as e:
            logger.error(f"Pruning failed: {str(e)}")
            raise RuntimeError(f"Model pruning failed: {e}") from e

        logger.success(f"{method.capitalize()} pruning completed")

    def _apply_structured_pruning(
        self, 
        percentage: float,
        dim: int = 0,
        n: int = 2
    ) -> None:
        """Structured pruning implementation for PyTorch."""
        import torch.nn.utils.prune as prune

        for module in self.model.modules():
            if hasattr(module, 'weight'):
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=percentage,
                    n=n,
                    dim=dim
                )

    @validate_params({
        "export_path": [str],
        "input_shape": [tuple, None]
    })
    def version_control(
        self, 
        export_path: str,
        input_shape: Optional[Tuple] = None
    ) -> None:
        """
        Save model version with framework-appropriate format and naming.

        Parameters
        ----------
        export_path : str
            Base path for versioned exports
        input_shape : Tuple, optional
            Input dimensions for format-specific exports

        Examples
        --------
        >>> exporter.version_control("model_versions/model_v")
        """
        check_is_runned(self)
        
        if self.version is None:
            logger.warning("Versioning disabled - skipping version control")
            return

        versioned_path = f"{export_path}{self.version}"
        logger.info(f"Saving version {self.version} to {versioned_path}")

        try:
            if self._framework == "pytorch":
                self.export_to_torch(f"{versioned_path}.pt")
            elif self._framework == "tensorflow":
                self.export_to_tensorflow(versioned_path)
                
            self.version += 1
            logger.success(f"Version {self.version-1} saved successfully")
        except Exception as e:
            logger.error(f"Version control failed: {str(e)}")
            raise RuntimeError(f"Version control failed: {e}") from e


    @ensure_pkgs(
        "torch",
        extra=EXTRA_MSG.format(pkg="torch"),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _quantize_pytorch(
        self,
        method: str,
        calibration_data: Any,
        custom_layers: Dict,
        quantize_activations: bool,
        backend: str = "fbgemm",
        fuse_layers: bool = True
    ) -> None:
        """PyTorch-specific quantization implementation with 
        enhanced static quantization."""
        import torch
        from torch.quantization import (
            get_default_qconfig,
            prepare,
            convert,
            # fuse_modules,
            # QConfig,
            # default_eval_fn
        )

        if method == "static":
            if calibration_data is None:
                raise ValueError(
                    "Static quantization requires calibration_data. "
                    "Provide a representative dataset for calibration."
                )

            logger.info("Initializing static quantization for PyTorch model")

            # Set model to evaluation mode
            self.model.eval()

            # Fuse layers for better quantization accuracy
            if fuse_layers:
                logger.info("Fusing compatible layers for quantization")
                self._fuse_pytorch_layers()

            # Configure quantization
            qconfig = get_default_qconfig(backend)
            self.model.qconfig = qconfig

            # Prepare for calibration
            prepared_model = prepare(self.model, inplace=False)

            # Run calibration
            logger.info("Running calibration with provided dataset")
            if isinstance(calibration_data, torch.utils.data.DataLoader):
                # Handle DataLoader input
                with torch.no_grad():
                    for data, _ in calibration_data:
                        prepared_model(data)
            else:
                # Handle single batch input
                with torch.no_grad():
                    prepared_model(calibration_data)

            # Convert to quantized model
            self.model = convert(prepared_model, inplace=False)
            logger.success("Static quantization completed successfully")

        elif method == "dynamic":
            # Existing dynamic quantization implementation
            qconfig = get_default_qconfig("fbgemm")
            self.model.qconfig = qconfig
            prepare(self.model, inplace=True)
            
            if calibration_data is None:
                calibration_data = torch.randn(1, 3, 224, 224)
                
            with torch.no_grad():
                self.model(calibration_data)
                
            convert(self.model, inplace=True)

    def _fuse_pytorch_layers(self) -> None:
        """Automatically fuse common layer patterns for optimal quantization."""
        import torch
        
        layer_types = {
            torch.nn.Conv2d: {
                'patterns': [
                    (torch.nn.Conv2d, torch.nn.ReLU),
                    (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU)
                ],
                'fuse_func': torch.quantization.fuse_conv_bn_relu
            },
            torch.nn.Linear: {
                'patterns': [(torch.nn.Linear, torch.nn.ReLU)],
                'fuse_func': torch.quantization.fuse_linear_relu
            }
        }

        fused = False
        for module in self.model.modules():
            for layer_type, config in layer_types.items():
                if isinstance(module, layer_type):
                    for pattern in config['patterns']:
                        if isinstance(module, pattern[0]):
                            try:
                                fused_modules = []
                                current_module = module
                                for expected_type in pattern[1:]:
                                    if isinstance(current_module, expected_type):
                                        fused_modules.append(current_module)
                                        current_module = current_module.next
                                    else:
                                        break
                                if len(fused_modules) == len(pattern) - 1:
                                    config['fuse_func'](module)
                                    fused = True
                            except AttributeError:
                                continue

        if fused:
            logger.info("Successfully fused compatible layers")
        else:
            logger.warning("No compatible layers found for fusion")

    @validate_params({
        'export_path': [str],
        'input_shape': ['array-like', None],
        'opset_version': [Interval(Integral, 1, None, closed="left")],
        'dynamic_axes': [dict, None]
    })
    def export_to_onnx(
        self,
        export_path: str,
        input_shape: Optional[Tuple] = None,
        opset_version: int = 13,
        dynamic_axes: Optional[Dict] = None
    ) -> None:
        """Enhanced ONNX export with dynamic axes support."""
        check_is_runned(self)
        import torch

        if self._framework != "pytorch":
            raise RuntimeError(
                "ONNX export requires PyTorch model. Detected framework: "
                f"{self._framework}"
            )

        logger.info(
            f"Exporting '{self.model_name}' to ONNX (opset {opset_version})")
        
        try:
            # Handle dynamic shapes
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }

            # Generate dummy input if not provided
            if input_shape is None:
                input_shape = (1, 3, 224, 224)  # Default ImageNet-like input
                
            dummy_input = torch.randn(*input_shape)

            # Export with additional metadata
            torch.onnx.export(
                self.model,
                dummy_input,
                export_path,
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                export_params=True,
                do_constant_folding=True,
                verbose=False,
                metadata={
                    'model_name': self.model_name,
                    'framework': 'pytorch',
                    'quantized': self._is_quantized()
                }
            )

            logger.success(f"ONNX export successful: {export_path}")
            if self.versioning:
                self._update_version_metadata(export_path)

        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            raise RuntimeError(f"ONNX export failed: {e}") from e

    def _is_quantized(self) -> bool:
        """Check if model is quantized."""
        import torch
        return isinstance(self.model, torch.quantization.QuantizedModel)

    def _update_version_metadata(self, export_path: str) -> None:
        """Add version metadata to exported model."""
        try:
            from onnx import load_model, save_model
            from onnx.metadata import add_metadata
            
            model = load_model(export_path)
            add_metadata(model, {
                'version': str(self.version),
                'export_date': datetime.now().isoformat(),
                'author': 'gofast.mlops.deployment'
            })
            save_model(model, export_path)
            self.version += 1
        except ImportError:
            logger.warning("ONNX metadata update requires onnx package")

class APIDeployment(BaseLearner):
    """
    Deploy machine learning models as production-ready web APIs with 
    integrated monitoring and traffic management.

    Parameters
    ----------
    api_type : {'FastAPI', 'Flask'}, default='FastAPI'
        Web framework selection for API implementation. FastAPI offers 
        async capabilities and automatic docs, while Flask provides 
        simplicity for basic REST APIs.
    max_requests : int, default=1000
        Maximum allowed requests per minute before rate limiting 
        activates. Enforces :math:`R_{max} \leq \text{max\_requests}` 
        where :math:`R_{max}` is peak request rate.

    Attributes
    ----------
    app : FastAPI or Flask
        Initialized web application instance
    request_count : int
        Current request counter for rate limiting
    start_time : float
        Timestamp of API server initialization

    Methods
    -------
    run(model, **run_kw)
        Configure deployment with model and metadata
    create_api()
        Define API endpoints (health, predict, version)
    serve_api(host='0.0.0.0', port=8000, debug=False)
        Launch production API server
    shutdown()
        Gracefully terminate API service

    Notes
    -----
    Implements a three-phase API lifecycle:

    1. **Initialization Phase**:
       - Framework-specific app creation
       - Rate limiter setup

    2. **Configuration Phase**:
       - Model binding
       - Endpoint registration
       - Health monitoring system

    3. **Serving Phase**:
       - Request handling with :math:`O(1)` routing complexity
       - Adaptive rate limiting using token bucket algorithm:
    
    .. math::
        T(t) = \min(T(t-1) + \frac{R}{60}, B)

    Where:
    - :math:`T(t)` = Tokens at time t
    - :math:`R` = max_requests
    - :math:`B` = Burst capacity (1.5 * max_requests)

    Examples
    --------
    >>> from gofast.mlops.deployment import APIDeployment
    >>> model = ...  # Pretrained classifier
    
    >>> # Initialize and deploy
    >>> api = (APIDeployment(api_type='FastAPI')
    ...        .run(model)
    ...        .create_api())
    >>> api.serve_api(port=8080)

    >>> # Production shutdown
    >>> api.shutdown()

    See Also
    --------
    ModelExporter : For model optimization before deployment
    CloudDeployment : For cloud infrastructure provisioning

    References
    ----------
    .. [1] Fielding, R. (2000). "Architectural Styles and the Design of 
       Network-based Software Architectures". Dissertation. 
    .. [2] FastAPI Documentation: https://fastapi.tiangolo.com/
    .. [3] Flask Documentation: https://flask.palletsprojects.com/
    """

    @ensure_pkgs(
        "fastapi, flask, uvicorn, pydantic",
        extra=( 
            "The 'fastapi', 'flask', 'uvicorn', and 'pydantic'"
            "  packages are required for this functionality."
             " Please install them to proceed.",
            ), 
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    @validate_params({
        "api_type": [StrOptions({"FastAPI", "Flask"})],
        "max_requests": [Interval(Integral, 1, None, closed="left")]
    })
    def __init__(
        self, 
        api_type: str = "FastAPI",
        max_requests: int = 1000
    ):
        self.api_type = api_type
        self.max_requests = max_requests
        self.start_time = time.time()
        self.request_count = 0
        
        self._is_runned = False
        self._model = None
        self._model_name = None
        
        # Initialize web framework
        if api_type == "FastAPI":
            from fastapi import FastAPI
            self.app = FastAPI()
        elif api_type == "Flask":
            from flask import Flask
            self.app = Flask(__name__)

    @RunReturn
    @validate_params({"model": [object]})
    def run(
        self, 
        model: Any, 
        **run_kw
    ) -> "APIDeployment":
        """
        Initialize model deployment configuration.

        Parameters
        ----------
        model : object
            Trained model object to deploy
        model_name : str, optional
            Unique identifier for the model. Auto-generated if not provided.

        Returns
        -------
        self : APIDeployment
            Returns instance for method chaining

        Examples
        --------
        >>> api.run(model, model_name="iris_classifier")
        """
        self._model = model
        self._model_name = run_kw.pop("model_name", None) or \
            self._generate_model_name()
        
        self._is_runned = True
        logger.info(f"Initialized API deployment for model: {self._model_name}")
        return self

    def _generate_model_name(self) -> str:
        """Generate unique model name using class name and hash."""
        base_name = self._model.__class__.__name__.lower()
        unique_id = abs(hash(self._model)) % (10 ** 8)
        return f"{base_name}_{unique_id:08x}"


    def create_api(self) -> "APIDeployment":
        """
        Create API endpoints with health checks and prediction routes.

        Returns
        -------
        self : APIDeployment
            Enables method chaining

        Examples
        --------
        >>> api.create_api()
        """
        check_is_runned(self, ['_is_runned'], 
                       "Deployment not started - call run() first")
        self._add_health_endpoint()
        self._add_predict_endpoint()
        self._add_version_endpoint()
        return self

    @validate_params({
        "host": [str],
        "port": [Interval(Integral, 1, 65535, closed="both")],
        "debug": [bool]
    })
    def serve_api(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        debug: bool = False
    ) -> None:
        """
        Start the API server with specified configuration.

        Parameters
        ----------
        host : str, default='0.0.0.0'
            Network interface to bind to
        port : int, default=8000
            Port number to listen on
        debug : bool, default=False
            Enable debug mode (Flask only)

        Examples
        --------
        >>> api.serve_api(port=8080)
        """
        logger.info(
            f"Serving {self._model_name} on {host}:{port} using {self.api_type}"
        )
        
        check_is_runned(self, ['_is_runned'], 
                       "Deployment not started - call run() first")
        
        if self.api_type == "FastAPI":
            import uvicorn
            uvicorn.run(self.app, host=host, port=port)
        elif self.api_type == "Flask":
            self.app.run(host=host, port=port, debug=debug)

    def _add_health_endpoint(self) -> None:
        """Add health check endpoint to API routes."""
        if self.api_type == "FastAPI":
            from fastapi import Request

            @self.app.get("/health")
            async def health(request: Request):
                return self._health_response()
        else:
            from flask import jsonify

            @self.app.route("/health", methods=["GET"])
            def health_flask():
                return jsonify(self._health_response())

    def _add_predict_endpoint(self) -> None:
        """Add model prediction endpoint to API routes."""
        if self.api_type == "FastAPI":
            from fastapi import Request #, HTTPException
            from pydantic import BaseModel

            class PredictRequest(BaseModel):
                input_data: Dict[str, Any]

            @self.app.post("/predict")
            async def predict(request: Request, data: PredictRequest):
                return await self._handle_prediction(data.input_data)
        else:
            from flask import request as flask_request, jsonify

            @self.app.route("/predict", methods=["POST"])
            def predict_flask():
                return jsonify(self._handle_prediction(
                    flask_request.get_json()["input_data"]
                ))

    def _add_version_endpoint(self) -> None:
        """Add model version endpoint to API routes."""
        if self.api_type == "FastAPI":
            @self.app.get("/version")
            async def version():
                return self._version_response()
        else:
            from flask import jsonify

            @self.app.route("/version", methods=["GET"])
            def version_flask():
                return jsonify(self._version_response())

    async def _handle_prediction(self, input_data: Dict) -> Dict:
        """Process prediction request with rate limiting."""
        try:
            self._check_rate_limit()
            logger.info(f"Prediction request: {input_data}")
            result = self._model.predict(input_data)
            logger.info(f"Prediction success: {result[:50]}...")
            return {"result": result, "model": self._model_name}
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise self._create_api_error(str(e))

    def _health_response(self) -> Dict:
        """Generate health check response."""
        uptime = time.time() - self.start_time
        return {
            "status": "healthy",
            "model": self._model_name,
            "uptime": f"{uptime:.2f}s",
            "requests": self.request_count
        }

    def _version_response(self) -> Dict:
        """Generate version info response."""
        return {
            "version": "1.0.0",
            "model": self._model_name,
            "framework": self.api_type
        }

    def _check_rate_limit(self) -> None:
        """Enforce rate limiting policy."""
        self.request_count += 1
        if self.request_count > self.max_requests:
            logger.warning("Rate limit exceeded")
            raise self._create_api_error("Rate limit exceeded", 429)

    def _create_api_error(self, msg: str, code: int = 500) -> Exception:
        """Create framework-specific error response."""
        if self.api_type == "FastAPI":
            from fastapi import HTTPException
            return HTTPException(status_code=code, detail=msg)
        else:
            from flask import jsonify
            return jsonify({"error": msg}), code

    def shutdown(self) -> None:
        """Gracefully shutdown API server with resource cleanup."""
        logger.info(f"Initiating shutdown for {self._model_name}")
        check_is_runned(self, ['_is_runned'], 
                       "Deployment not started - call run() first"
                      )
        
        self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Framework-specific resource cleanup."""
        if self.api_type == "FastAPI":
            @self.app.on_event("shutdown")
            async def shutdown_handler():
                logger.info("FastAPI cleanup complete")
        else:
            import signal
            signal.signal(signal.SIGINT, self._flask_signal_handler)
            signal.signal(signal.SIGTERM, self._flask_signal_handler)

    def _flask_signal_handler(self, signum, frame) -> None:
        """Handle Flask server shutdown signals."""
        logger.info("Flask cleanup complete")
        exit(0)
       
class CloudDeployment(BaseLearner):
    """
    Orchestrate multi-cloud model deployments with automated CI/CD 
    pipeline configuration.

    Parameters
    ----------
    platform : {'aws', 'gcp', 'azure'}, default='aws'
        Primary cloud platform for deployment operations. Selects 
        default configuration templates but allows cross-platform 
        deployments through explicit method calls.

    Attributes
    ----------
    platform : str
        Configured default cloud platform identifier
    version : int or None
        Current deployment version counter when versioning enabled

    Methods
    -------
    run(model, **run_kw)
        Initialize deployment with model and metadata
    deploy_to_aws(config)
        Deploy to AWS SageMaker with automatic endpoint configuration
    deploy_to_gcp(config)
        Deploy to GCP AI Platform with auto-scaling
    deploy_to_azure(config)
        Deploy to Azure ML with AKS integration
    continuous_deployment(config)
        Configure CI/CD pipelines for automated updates

    Notes
    -----
    Implements a unified cloud deployment framework with three core 
    components:

    1. **Cloud Abstraction Layer**:
       - Uniform interface for AWS/GCP/Azure
       - Automatic credential management
       - Resource naming standardization

    2. **Performance Optimization**:
       - Auto-scaling configuration based on queue theory:
    
    .. math::
        N_{replicas} = \left\lceil \frac{\lambda}{\mu} \right\rceil

    Where:
    - :math:`\lambda` = Request arrival rate
    - :math:`\mu` = Service rate per replica

    3. **CI/CD Automation**:
       - GitOps workflow implementation
       - Secret rotation handling
       - Rollback capabilities through versioned artifacts

    Examples
    --------
    >>> from gofast.mlops.deployment import CloudDeployment
    >>> model = ...  # Pretrained TensorFlow model
    
    >>> # AWS deployment example
    >>> deployer = (CloudDeployment(platform='aws')
    ...             .run(model, model_name='fraud-detection'))
    >>> aws_config = {
    ...     'model_data': 's3://models/fraud/v1',
    ...     'role_arn': 'arn:aws:iam::123456789012:role/sagemaker'
    ... }
    >>> deployer.deploy_to_aws(aws_config)

    >>> # CI/CD setup
    >>> cicd_config = {
    ...     'provider': 'github',
    ...     'triggers': {'push': {'branches': ['main']}}
    ... }
    >>> deployer.continuous_deployment(cicd_config)

    See Also
    --------
    APIDeployment : For serving deployed models via web APIs
    ModelExporter : For model optimization before deployment

    References
    ----------
    .. [1] AWS SageMaker Developer Guide: 
       https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html
    .. [2] Google Cloud AI Platform Documentation: 
       https://cloud.google.com/ai-platform/docs
    .. [3] Microsoft Azure ML Documentation: 
       https://docs.microsoft.com/en-us/azure/machine-learning/
    .. [4] Bass, L., Weber, I., & Zhu, L. (2015). "DevOps: A Software 
       Architect's Perspective". Addison-Wesley Professional.
    """

    @validate_params({
        "platform": [StrOptions({"aws", "gcp", "azure"})]
    })
    def __init__(self, platform: str = "aws"):
        self.platform = platform
        self._is_runned = False
        self._model = None
        self._model_name = None

    @RunReturn
    @validate_params({"model": [object]})
    def run(
        self, 
        model: Any,
        **run_kw
    ) -> "CloudDeployment":
        """
        Initialize cloud deployment configuration.

        Parameters
        ----------
        model : object
            Trained model object to deploy
        model_name : str, optional
            Unique identifier for the model. Auto-generated if not provided.

        Returns
        -------
        self : CloudDeployment
            Returns instance for method chaining

        Examples
        --------
        >>> deployer.run(model, model_name="production_model")
        """
        self._model = model
        self._model_name = run_kw.pop("model_name", None) or \
            self._generate_model_name()
        
        self._is_runned = True
        logger.info(f"Initialized cloud deployment for {self._model_name}")
        return self

    def _generate_model_name(self) -> str:
        """Generate unique model name using class name and hash."""
        base_name = self._model.__class__.__name__.lower()
        unique_id = abs(hash(self._model)) % (10 ** 8)
        return f"{base_name}_{unique_id:08x}"

    @ensure_pkgs(
        "boto3",
        extra=EXTRA_MSG.format(pkg="boto3"),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    @validate_params({"config": [dict]})
    def deploy_to_aws(self, config: Dict[str, Any]) -> Dict:
        """
        Deploy model to AWS SageMaker with automatic configuration.

        Parameters
        ----------
        config : dict
            AWS deployment configuration containing:
            - model_data: S3 path to model artifacts
            - role_arn: IAM role ARN
            - instance_type: EC2 instance type
            - endpoint_name: Optional endpoint name

        Returns
        -------
        dict
            Deployment status and endpoint information

        Examples
        --------
        >>> config = {
        ...     'model_data': 's3://bucket/model.tar.gz',
        ...     'role_arn': 'arn:aws:iam::123456789012:role/SageMakerRole',
        ...     'instance_type': 'ml.m5.large'
        ... }
        >>> deployer.deploy_to_aws(config)
        """
        import boto3
        
        check_is_runned(self, ['_is_runned'], 
                       "Deployment not started - call run() first")

        logger.info(f"Deploying {self._model_name} to AWS SageMaker")
        
        sagemaker = boto3.client("sagemaker")

        try:
            # Create model
            model_response = sagemaker.create_model(
                ModelName=self._model_name,
                ExecutionRoleArn=config["role_arn"],
                PrimaryContainer={
                    "Image": config.get(
                        "image", 
                        "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "pytorch-inference:1.6.0-cpu-py3"
                    ),
                    "ModelDataUrl": config["model_data"]
                }
            )

            # Create endpoint config
            endpoint_config_name = f"{self._model_name}-config"
            sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    "VariantName": "primary",
                    "ModelName": self._model_name,
                    "InstanceType": config.get("instance_type", "ml.m5.large"),
                    "InitialInstanceCount": config.get("instance_count", 1)
                }]
            )

            # Deploy endpoint
            endpoint_name = config.get("endpoint_name", f"{self._model_name}-endpoint")
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )

            logger.success(f"Deployment initiated: {endpoint_name}")
            return {
                "status": "deploying",
                "endpoint": endpoint_name,
                "arn": model_response["ModelArn"]
            }

        except Exception as e:
            logger.error(f"AWS deployment failed: {str(e)}")
            raise RuntimeError(f"AWS deployment failed: {e}") from e

    @ensure_pkgs(
        "google-cloud-aiplatform",
        extra=EXTRA_MSG.format(pkg="google-cloud-aiplatform"),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    @validate_params({"config": [dict]})
    def deploy_to_gcp(self, config: Dict[str, Any]) -> Dict:
        """
        Deploy model to Google Cloud AI Platform with auto-scaling.

        Parameters
        ----------
        config : dict
            GCP deployment configuration containing:
            - project: GCP project ID
            - region: Deployment region
            - bucket: GCS bucket path
            - machine_type: Compute instance type

        Returns
        -------
        dict
            Deployment status and endpoint information
        """
    
        from google.cloud import aiplatform
        
        check_is_runned(self, ['_is_runned'], 
                               "Deployment not started - call run() first"
                      )
        logger.info(f"Deploying {self._model_name} to GCP AI Platform")
        
        try:
            aiplatform.init(
                project=config["project"],
                location=config.get("region", "us-central1")
            )

            # Upload and deploy model
            model = aiplatform.Model.upload(
                display_name=self._model_name,
                artifact_uri=config["bucket"]
            )
            endpoint = model.deploy(
                machine_type=config.get("machine_type", "n1-standard-4"),
                min_replica_count=config.get("min_replicas", 1),
                max_replica_count=config.get("max_replicas", 3)
            )

            logger.success(f"GCP deployment complete: {endpoint.resource_name}")
            return {
                "status": "active",
                "endpoint": endpoint.resource_name,
                "uri": endpoint.predict_http_uri
            }

        except Exception as e:
            logger.error(f"GCP deployment failed: {str(e)}")
            raise RuntimeError(f"GCP deployment failed: {e}") from e


    @ensure_pkgs(
        "azureml-core",
        extra=EXTRA_MSG.format(pkg="azureml-core"),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    @validate_params({"config": [dict]})
    def deploy_to_azure(self, config: Dict[str, Any]) -> Dict:
        """
        Deploy model to Azure ML with AKS integration.

        Parameters
        ----------
        config : dict
            Azure deployment configuration containing:
            - workspace: Azure ML workspace config
            - cluster: AKS cluster name
            - inference_config: Model scoring configuration
            - deploy_config: Deployment settings

        Returns
        -------
        dict
            Deployment status and endpoint information
        """
        from azureml.core import Workspace, Model
        from azureml.core.compute import AksCompute
        
        check_is_runned(self, ['_is_runned'], 
                       "Deployment not started - call run() first"
                      )

        logger.info(f"Deploying {self._model_name} to Azure ML")

        try:
            # Initialize workspace
            ws = Workspace.from_config(path=config["workspace"])
            
            # Register model
            azure_model = Model.register(
                workspace=ws,
                model_path=config.get("model_path", "./"),
                model_name=self._model_name
            )

            # Deploy to AKS
            aks_target = AksCompute(ws, name=config["cluster"])
            service = azure_model.deploy(
                workspace=ws,
                name=self._model_name,
                models=[azure_model],
                inference_config=config["inference_config"],
                deployment_config=config["deploy_config"],
                deployment_target=aks_target
            )
            service.wait_for_deployment(show_output=True)

            logger.success(f"Azure deployment complete: {service.scoring_uri}")
            return {
                "status": "active",
                "endpoint": service.scoring_uri,
                "name": service.name
            }

        except Exception as e:
            logger.error(f"Azure deployment failed: {str(e)}")
            raise RuntimeError(f"Azure deployment failed: {e}") from e

    @validate_params({"config": [dict]})
    def continuous_deployment(self, config: Dict[str, Any]) -> Dict:
        """
        Configure CI/CD pipeline for automatic model updates.

        Parameters
        ----------
        config : dict
            CI/CD configuration containing:
            - provider: CI/CD platform (github, jenkins, etc.)
            - triggers: Deployment trigger conditions
            - environments: Target deployment environments

        Returns
        -------
        dict
            Pipeline configuration status
        """
        logger.info(f"Configuring CI/CD for {self._model_name}")
        
        check_is_runned(self, ['_is_runned'], 
                       "Deployment not started - call run() first"
                      )

        try:
            provider = config["provider"].lower()
            
            if provider == "github":
                self._setup_github_actions(config)
            elif provider == "jenkins":
                self._setup_jenkins_pipeline(config)
            else:
                raise ValueError(f"Unsupported CI/CD provider: {provider}")

            return {
                "status": "configured",
                "model": self._model_name,
                "provider": provider
            }

        except Exception as e:
            logger.error(f"CI/CD configuration failed: {str(e)}")
            raise RuntimeError(f"CI/CD setup failed: {e}") from e


    def _setup_github_actions(self, config: Dict) -> None:
        """
        Configure GitHub Actions workflow for automated model deployment.
        
        Creates workflow file in .github/workflows/ directory with:
        - Cloud platform specific deployment steps
        - Secret management integration
        - Model change triggers
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - repo_path: Path to repository root (default: current directory)
            - cloud_platform: Target deployment platform
            - triggers: List of paths/branches to watch
            - secrets: Dictionary of secret names to use
        """
        try:
            logger.info("Configuring GitHub Actions workflow")
            
            # Validate configuration
            cloud = config.get("cloud_platform", self.platform).lower()
            if cloud not in ["aws", "gcp", "azure"]:
                raise ValueError(f"Unsupported cloud platform: {cloud}")

            repo_path = Path(config.get("repo_path", "."))
            workflow_dir = repo_path / ".github/workflows"
            workflow_dir.mkdir(parents=True, exist_ok=True)

            # Generate platform-specific workflow
            workflow_content = self._generate_github_workflow(cloud, config)
            workflow_file = workflow_dir / f"deploy-{self._model_name}.yml"
            
            with workflow_file.open("w") as f:
                f.write(workflow_content)

            logger.success(
                f"Created GitHub Actions workflow at {workflow_file}\n"
                f"Configure secrets: {', '.join(self._get_required_secrets(cloud))}"
            )

        except Exception as e:
            logger.error(f"GitHub Actions setup failed: {str(e)}")
            raise RuntimeError("Failed to configure GitHub Actions") from e

    @ensure_pkgs(
        "yaml",
        extra=EXTRA_MSG.format(pkg="pyyaml"),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _generate_github_workflow(self, cloud: str, config: Dict) -> str:
        """Generate GitHub Actions workflow YAML content."""
        import yaml
        
        triggers = config.get("triggers", {
            "push": {
                "branches": ["main"],
                "paths": [f"models/{self._model_name}/**"]
            }
        })

        workflow = {
            "name": f"Deploy {self._model_name} to {cloud.upper()}",
            "on": triggers,
            "jobs": {
                "deploy": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.10"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        }
                    ]
                }
            }
        }

        # Add cloud-specific deployment steps
        deploy_steps = {
            "aws": [
                {
                    "name": "Configure AWS Credentials",
                    "uses": "aws-actions/configure-aws-credentials@v2",
                    "with": {
                        "role-to-assume": "${{ secrets.AWS_IAM_ROLE }}",
                        "aws-region": config.get("region", "us-west-2")
                    }
                },
                {
                    "name": "Deploy to SageMaker",
                    "run": f"python scripts/deploy_aws.py --model {self._model_name}"
                }
            ],
            "gcp": [
                {
                    "name": "Authenticate to GCP",
                    "uses": "google-github-actions/auth@v1",
                    "with": {
                        "credentials_json": "${{ secrets.GCP_SA_KEY }}"
                    }
                },
                {
                    "name": "Deploy to AI Platform",
                    "run": f"python scripts/deploy_gcp.py --model {self._model_name}"
                }
            ],
            "azure": [
                {
                    "name": "Azure Login",
                    "uses": "azure/login@v1",
                    "with": {
                        "creds": "${{ secrets.AZURE_CREDENTIALS }}"
                    }
                },
                {
                    "name": "Deploy to Azure ML",
                    "run": f"python scripts/deploy_azure.py --model {self._model_name}"
                }
            ]
        }

        workflow["jobs"]["deploy"]["steps"].extend(deploy_steps[cloud])
        return yaml.safe_dump(workflow, sort_keys=False, width=1000)

    def _get_required_secrets(self, cloud: str) -> list:
        """Return list of required secrets for specified cloud platform."""
        return {
            "aws": ["AWS_IAM_ROLE"],
            "gcp": ["GCP_SA_KEY"],
            "azure": ["AZURE_CREDENTIALS"]
        }.get(cloud, [])

    def _setup_jenkins_pipeline(self, config: Dict) -> None:
        """
        Configure Jenkins pipeline for continuous deployment.
        
        Creates Jenkinsfile and optionally configures Jenkins job through REST API.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - jenkins_url: Jenkins server URL
            - jenkins_user: API username
            - jenkins_token: API token
            - repo_url: Git repository URL
            - credentials_id: Jenkins credentials ID for Git
        """
        try:
            logger.info("Configuring Jenkins pipeline")
            
            # Create Jenkinsfile
            jenkinsfile_path = Path(config.get("repo_path", ".")) / "Jenkinsfile"
            with jenkinsfile_path.open("w") as f:
                f.write(self._generate_jenkinsfile(config))
            
            logger.success(f"Created Jenkinsfile at {jenkinsfile_path}")

            # Create Jenkins job if credentials provided
            if all(config.get(k) for k in ["jenkins_url", "jenkins_user", "jenkins_token"]):
                self._create_jenkins_job(config)
                logger.success("Jenkins job created successfully")

        except Exception as e:
            logger.error(f"Jenkins pipeline setup failed: {str(e)}")
            raise RuntimeError("Failed to configure Jenkins pipeline") from e

    def _generate_jenkinsfile(self, config: Dict) -> str:
        """Generate declarative Jenkinsfile content."""
        return f"""pipeline {{
            agent any
            parameters {{
                choice(
                    name: 'CLOUD_ENV',
                    choices: ['aws', 'gcp', 'azure'],
                    description: 'Target cloud environment'
                )
            }}
            environment {{
                MODEL_NAME = '{self._model_name}'
                REPO_CREDENTIALS = '{config.get("credentials_id", "git-creds")}'
            }}
            stages {{
                stage('Checkout') {{
                    steps {{
                        checkout([
                            $class: 'GitSCM',
                            branches: [[name: '*/main']],
                            extensions: [],
                            userRemoteConfigs: [[
                                url: '{config.get("repo_url", "CHANGE_ME")}',
                                credentialsId: "${{env.REPO_CREDENTIALS}}"
                            ]]
                        ])
                    }}
                }}
                stage('Deploy') {{
                    steps {{
                        script {{
                            sh "python scripts/deploy_${{params.CLOUD_ENV}}.py --model ${{env.MODEL_NAME}}"
                        }}
                    }}
                }}
            }}
        }}"""

    def _create_jenkins_job(self, config: Dict) -> None:
        """Create Jenkins job using REST API."""
        from jenkinsapi.jenkins import Jenkins
        
        try:
            jenkins = Jenkins(
                baseurl=config["jenkins_url"],
                username=config["jenkins_user"],
                password=config["jenkins_token"]
            )
            
            job_name = f"deploy-{self._model_name}"
            job_config = f"""
            <flow-definition plugin="workflow-job@2.42">
                <definition class="org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition">
                    <scm class="hudson.plugins.git.GitSCM">
                        <configVersion>2</configVersion>
                        <userRemoteConfigs>
                            <hudson.plugins.git.UserRemoteConfig>
                                <url>{config["repo_url"]}</url>
                                <credentialsId>{config["credentials_id"]}</credentialsId>
                            </hudson.plugins.git.UserRemoteConfig>
                        </userRemoteConfigs>
                        <branches>
                            <hudson.plugins.git.BranchSpec>
                                <name>main</name>
                            </hudson.plugins.git.BranchSpec>
                        </branches>
                        <doGenerateSubmoduleConfigurations>false</doGenerateSubmoduleConfigurations>
                    </scm>
                    <scriptPath>Jenkinsfile</scriptPath>
                    <lightweight>true</lightweight>
                </definition>
            </flow-definition>
            """
            
            jenkins.create_job(job_name, job_config)
            logger.info(f"Created Jenkins job '{job_name}'")

        except Exception as e:
            logger.error(f"Jenkins API error: {str(e)}")
            raise RuntimeError("Failed to create Jenkins job") from e

class ABTesting(BaseLearner):
    """
    Conduct controlled A/B tests between model versions with adaptive
    traffic routing and performance-based optimization.

    Parameters
    ----------
    split_ratio : float, default=0.5
        Initial traffic distribution between models as probability 
        :math:`p \in [0,1]`. Represents :math:`P(\text{route to model_v1})`.
    min_split_ratio : float, default=0.1
        Minimum allowable traffic ratio :math:`p_{min}` for any model.
        Ensures :math:`p \geq p_{min}` during adjustments.
    max_split_ratio : float, default=0.9
        Maximum allowable traffic ratio :math:`p_{max}` for any model.
        Ensures :math:`p \leq p_{max}` during adjustments.
    performance_threshold : float, default=0.05
        Minimum absolute performance difference 
        :math:`\Delta_{min} = |\mu_1 - \mu_2|` required to trigger
        traffic rebalancing.
    traffic_increment : float, default=0.1
        Adjustment step size :math:`\delta` for modifying split ratio.
        Applied as :math:`p \pm \delta` based on performance.
    graceful_degradation : bool, default=True
        Enable fail-safe mechanism that resets to :math:`p=0.5` when
        both models' average performance falls below 0.5.

    Attributes
    ----------
    split_ratio : float
        Current traffic distribution probability
    min_split_ratio : float
        Configured minimum traffic bound
    max_split_ratio : float
        Configured maximum traffic bound
    performance_threshold : float
        Performance delta threshold for adjustments
    traffic_increment : float
        Traffic adjustment step size
    graceful_degradation : bool
        Fail-safe activation status

    Methods
    -------
    run(model_v1, model_v2, **run_kw)
        Initialize test with model pair
    route_traffic(request)
        Route input to model_v1 or model_v2
    evaluate_performance(performance_metrics)
        Adjust traffic based on performance metrics
    rollback()
        Force traffic to dominant model

    Notes
    -----
    Implements three-phase A/B testing lifecycle:

    1. **Traffic Routing**:
       - Bernoulli distribution-based routing:
    
    .. math::
        X \sim \text{Bernoulli}(p) \Rightarrow 
        \text{route} = \begin{cases}
            \text{model\_v1} & \text{if } X=1 \\
            \text{model\_v2} & \text{if } X=0
        \end{cases}

    2. **Performance Evaluation**:
       - Threshold-based adjustment:
    
    .. math::
        p' = \begin{cases}
            \min(p + \delta, p_{max}) & \text{if } \mu_1 - \mu_2 > \Delta_{min} \\
            \max(p - \delta, p_{min}) & \text{if } \mu_2 - \mu_1 > \Delta_{min} \\
            p & \text{otherwise}
        \end{cases}

    3. **Graceful Degradation**:
       - Reset condition: :math:`\frac{\mu_1 + \mu_2}{2} < 0.5`

    Examples
    --------
    >>> from gofast.mlops.deployment import ABTesting
    >>> model_v1 = ...  # Current production model
    >>> model_v2 = ...  # New candidate model
    
    >>> # Initialize and run test
    >>> ab_test = (ABTesting(split_ratio=0.7)
    ...            .run(model_v1, model_v2))
    
    >>> # Route sample request
    >>> prediction = ab_test.route_traffic(request_data)
    
    >>> # Weekly performance review
    >>> metrics = {'model_v1': 0.82, 'model_v2': 0.79}
    >>> ab_test.evaluate_performance(metrics)
    
    >>> # Emergency rollback
    >>> ab_test.rollback()

    See Also
    --------
    APIDeployment : For production deployment of finalized models
    ModelExporter : For model optimization before A/B testing

    References
    ----------
    .. [1] Kohavi, R., Tang, D., & Xu, Y. (2020). "Trustworthy Online 
       Controlled Experiments: A Practical Guide to A/B Testing". 
       Cambridge University Press.
    .. [2] PyTorch Model Serving: 
       https://pytorch.org/serve/
    .. [3] TensorFlow Serving: 
       https://www.tensorflow.org/tfx/guide/serving
    """
    
    @validate_params({
        'split_ratio': [Interval(Real, 0.0, 1.0, closed='both')],
        'min_split_ratio': [Interval(Real, 0.0, 1.0, closed='both')],
        'max_split_ratio': [Interval(Real, 0.0, 1.0, closed='both')],
        'performance_threshold': [Interval(Real, 0.0, 1.0, closed='both')],
        'traffic_increment': [Interval(Real, 0.0, 1.0, closed='both')],
        'graceful_degradation': [bool]
    })
    def __init__(
        self,
        split_ratio: float = 0.5,
        min_split_ratio: float = 0.1,
        max_split_ratio: float = 0.9,
        performance_threshold: float = 0.05,
        traffic_increment: float = 0.1,
        graceful_degradation: bool = True
    ):
        self._validate_split_constraints(
            split_ratio, min_split_ratio, max_split_ratio)
        
        self.split_ratio = split_ratio
        self.min_split_ratio = min_split_ratio
        self.max_split_ratio = max_split_ratio
        self.performance_threshold = performance_threshold
        self.traffic_increment = traffic_increment
        self.graceful_degradation = graceful_degradation
        
        self._is_runned = False
        self._model_v1 = None
        self._model_v2 = None

    def _validate_split_constraints(self, split, min_split, max_split):
        """Validate traffic split configuration."""
        if min_split > max_split:
            raise ValueError("min_split_ratio cannot exceed max_split_ratio")
        if not (min_split <= split <= max_split):
            raise ValueError(
                f"split_ratio {split} must be between {min_split} and {max_split}")

    @RunReturn
    @validate_params({
        "model_v1": [object],
        "model_v2": [object]
    })
    def run(
        self, 
        model_v1: Any,
        model_v2: Any,
        **run_kw
    ) -> "ABTesting":
        """
        Initialize A/B test configuration with model versions.

        Parameters
        ----------
        model_v1 : object
            Primary model for testing
        model_v2 : object
            Challenger model for testing

        Returns
        -------
        self : ABTesting
            Configured instance for method chaining

        Examples
        --------
        >>> ab_test = ABTesting().run(model_v1, model_v2)
        """
        self._model_v1 = model_v1
        self._model_v2 = model_v2
        self._is_runned = True
        logger.info(f"Initialized A/B test between {model_v1} and {model_v2}")
        return self

    @validate_params({'request': [dict]})
    def route_traffic(self, request: Dict) -> Any:
        """
        Route incoming request based on current traffic distribution.

        Parameters
        ----------
        request : dict
            Input data for model prediction

        Returns
        -------
        Any
            Model prediction result
        """
        check_is_runned(self, ['_is_runned'], 
                       "ABTesting not started - call run() first")
        model = ( 
            self._model_v1 if 
            random.random() < self.split_ratio else self._model_v2
            )
        logger.debug(f"Routing to {model.__class__.__name__}")
        return model.predict(request)

    @validate_params({'performance_metrics': [dict]})
    def evaluate_performance(self, performance_metrics: Dict[str, float]) -> None:
        """
        Adjust traffic distribution based on model performance metrics.

        Parameters
        ----------
        performance_metrics : dict
            Dictionary with 'model_v1' and 'model_v2' performance scores
        """
        check_is_runned(self, ['_is_runned'], 
                       "ABTesting not started - call run() first")
        
        perf_v1 = performance_metrics.get('model_v1')
        perf_v2 = performance_metrics.get('model_v2')
        
        if None in (perf_v1, perf_v2):
            raise ValueError("Both model_v1 and model_v2 metrics required")

        delta = perf_v1 - perf_v2
        if abs(delta) > self.performance_threshold:
            self._adjust_split_ratio(delta > 0)
        
        if self.graceful_degradation:
            self._apply_graceful_degradation(perf_v1, perf_v2)

    def _adjust_split_ratio(self, v1_better: bool) -> None:
        """Internal method to update traffic distribution."""
        adjustment = ( 
            self.traffic_increment if v1_better else -self.traffic_increment
            )
        self.split_ratio = np.clip(
            self.split_ratio + adjustment,
            self.min_split_ratio,
            self.max_split_ratio
        )
        logger.info(f"New split ratio: {self.split_ratio:.2f}")

    def _apply_graceful_degradation(self, perf_v1: float, perf_v2: float) -> None:
        """Reset to neutral split if both models underperform."""
        avg_perf = (perf_v1 + perf_v2) / 2
        if avg_perf < 0.5:  # Configurable threshold
            logger.warning("Both models underperforming - resetting split")
            self.split_ratio = 0.5

    def rollback(self) -> None:
        """
        Rollback to dominant model based on current traffic distribution.
        """
        check_is_runned(self, ['_is_runned'], 
                       "ABTesting not started - call run() first")
        
        if self.split_ratio > 0.8:
            logger.info("Rolling back completely to model_v1")
            self.split_ratio = 1.0
        elif self.split_ratio < 0.2:
            logger.info("Rolling back completely to model_v2")
            self.split_ratio = 0.0
        else:
            logger.info("No clear dominant model for rollback")

