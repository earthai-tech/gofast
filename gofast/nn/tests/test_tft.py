# -*- coding: utf-8 -*-
import pytest
import numpy as np
import warnings 
from gofast.compat.tf import HAS_TF 
from gofast.utils.deps_utils import ensure_module_installed 

if not HAS_TF: 
    try:
        HAS_TF=ensure_module_installed("tensorflow", auto_install=True)
    except  Exception as e: 
        warnings.warn(f"Fail to install `tensorflow` library: {e}")
   
if HAS_TF: 
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam

from gofast.nn.transformers import TemporalFusionTransformer
from gofast.nn._tft import ( 
    GatedResidualNetwork, VariableSelectionNetwork, 
    StaticEnrichmentLayer, TemporalAttentionLayer
)
    
#
if __name__=='__main__': 
    pytest.main( [__file__])