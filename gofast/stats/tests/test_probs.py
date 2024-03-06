# -*- coding: utf-8 -*-
"""
test_hierarchical_linear_model.py
@author: Daniel
"""


import numpy as np
import pytest
from gofast.stats.probs import hierarchical_linear_model  
from gofast.stats.probs import stochastic_volatility_model 

def is_pymc3_installed (): 
    try : 
        import  pymc3 # noqa
        return True 
    except: return False 
    
PYMC3_INSTALLED= is_pymc3_installed() 

@pytest.mark.skipif (not PYMC3_INSTALLED, reason=(
    "Need pymc3 to be installed for running 'hierarchical_linear_model'")
    )
def test_hierarchical_linear_model_basic():
    # Dummy data
    X = np.random.normal(size=(100, 3))
    y = np.random.normal(size=100)
    groups = np.random.randint(0, 5, size=100)

    # Test without sampling
    model = hierarchical_linear_model(X, y, groups, n_samples=0, 
                                      return_inference_data=False)
    assert model is not None, "Model should not be None"
    
@pytest.mark.skipif (not PYMC3_INSTALLED, reason=(
    "Need pymc3 to be installed for running 'hierarchical_linear_model'")
    )
def test_stochastic_volatility_model_basic():
    # Dummy data
    returns = np.random.normal(0, 1, 100)

    # Test without sampling
    model = stochastic_volatility_model(returns, sampling=False,
                                        return_inference_data=False)
    assert model is not None, "Model should not be None"
    
@pytest.mark.slow  # Use this decorator if you want to skip slow tests with pytest -k "not slow"    
@pytest.mark.skipif (not PYMC3_INSTALLED, reason=(
    "Need pymc3 to be installed for running 'hierarchical_linear_model'")
    )
def test_hierarchical_linear_model_with_sampling():
    # Dummy data
    import  pymc3 as pm  # 
    X = np.random.normal(size=(20, 2))  # Reduced size for speed
    y = np.random.normal(size=20)
    groups = np.random.randint(0, 3, size=20)  # Reduced number of groups for speed

    # Test with sampling
    model, trace = hierarchical_linear_model(X, y, groups, sampling=True, n_samples=50, 
                                             return_inference_data=False)
    
    assert model is not None, "Model should not be None"
    assert isinstance(trace, pm.backends.base.MultiTrace), "Trace should be a MultiTrace object"
    assert len(trace) == 50, "Trace should contain the specified number of samples"

@pytest.mark.slow    
@pytest.mark.skipif (not PYMC3_INSTALLED, reason=(
    "Need pymc3 to be installed for running 'stochastic_volatility_model'")
    )
def test_stochastic_volatility_model_with_sampling():
    import  pymc3 as pm  # 
    # Dummy data
    returns = np.random.normal(0, 1, 50)  # Reduced size for speed

    # Test with sampling
    model, trace = stochastic_volatility_model(
        returns, sampling=True, n_samples=50, return_inference_data=False)

    assert model is not None, "Model should not be None"
    assert isinstance(trace, pm.backends.base.MultiTrace), "Trace should be a MultiTrace object"
    assert len(trace) == 50, "Trace should contain the specified number of samples"
