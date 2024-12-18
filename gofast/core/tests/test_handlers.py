# -*- coding: utf-8 -*-

import pytest 
from gofast.core.handlers import param_deprecated_message 

# The param_deprecated_message decorator will handle two main cases:
# 1) Parameters that are deprecated or renamed.
#    If old parameter is given by the user, raise a warning and delegate its value 
#    to the new parameter if defined. Otherwise, use a default if provided.
#
# 2) Parameters that should meet a certain condition. If the condition is met, 
#    raise a warning and possibly override the parameter with a default value.
#
# The decorator can be used on functions, methods, or class initializers.
#
# Arguments:
#  - deprecated_params_mappings: A dictionary or list of dicts that map old param names to:
#       {
#         'old': 'old_param_name',
#         'new': 'new_param_name',       # optional if renaming
#         'message': 'warning message',  # optional custom message
#         'default': default_value,       # optional default if new not provided
#       }
#
#  - conditions_params_mappings: A dictionary or list of dicts defining conditions for parameters:
#       {
#         'param': 'param_name',
#         'condition': lambda val: True or False,
#         'message': 'warning message',
#         'default': default_value,  # override the value if condition is True
#       }
#
# - warning_category: The category of warning to raise (default FutureWarning).
#
# - default_extra_message: Extra message to append to any warning message if 
#   no custom message is provided.
#
# The decorator will inspect parameters passed to the decorated function/class __init__ 
# and:
#  - If deprecated parameter is found, warn and move/rename it to the new parameter or set default.
#  - If condition is met for a parameter, warn and possibly override with default.

# ===========================================
# Test Case 1: TemporalTransformerFusion
# ==========================================
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'quantiles',
            'condition': lambda v: v is not None,
            'message': "Current version only supports 'quantiles=None'. Resetting quantiles to None.",
            'default': None
        }
    ]
)
class TemporalTransformerFusion:
    def __init__(
        self,
        static_input_dim,
        dynamic_input_dim,
        num_static_vars,
        num_dynamic_vars,
        hidden_units,
        num_heads=4,
        dropout_rate=0.1,
        forecast_horizon=1,
        quantiles=None,  # should be None
        activation='elu',
        use_batch_norm=False,
        num_lstm_layers=1,
        lstm_units=None,
        **kwargs
    ):
        # Initialization logic...
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.num_static_vars = num_static_vars
        self.num_dynamic_vars = num_dynamic_vars
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.num_lstm_layers = num_lstm_layers
        self.lstm_units = lstm_units

# Pytest test for TemporalTransformerFusion with quantiles parameter
def test_temporal_transformer_quantiles():
    with pytest.warns(FutureWarning, match="Current version only supports 'quantiles=None'. Resetting quantiles to None."):
        tft = TemporalTransformerFusion(
            static_input_dim=10,
            dynamic_input_dim=5,
            num_static_vars=2,
            num_dynamic_vars=3,
            hidden_units=64,
            quantiles=5  # Should trigger the warning and reset to None
        )
    
    # Ensure quantiles is reset to None
    assert tft.quantiles is None

# ===========================================================================================
# Test Case 2: Deprecated Parameter Handling (nonlinear_input_estimator -> input_estimator)
# ===========================================================================================
@param_deprecated_message(
    deprecated_params_mappings=[
        {
            'old': 'nonlinear_input_estimator',
            'new': 'input_estimator',
            'message': "Parameter 'nonlinear_input_estimator' is deprecated. Use 'input_estimator' instead."
        },
        {
            'old': 'nonlinear_output_estimator',
            'new': 'output_estimator',
            'message': "Parameter 'nonlinear_output_estimator' is deprecated. Use 'output_estimator' instead."
        }
    ]
)
class EstimatorModel:
    def __init__(
        self, 
        nonlinear_input_estimator=None, 
        nonlinear_output_estimator=None, 
        input_estimator=None, 
        output_estimator=None, 
        **kwargs
        ):
        self.input_estimator = nonlinear_input_estimator
        self.output_estimator = nonlinear_output_estimator
        self.input_estimator=input_estimator 
        self.output_estimator= output_estimator 
        


# Pytest test for deprecated parameter renaming
def test_deprecated_parameter_renaming():
    with pytest.warns(FutureWarning, match="Parameter 'nonlinear_input_estimator' is deprecated. Use 'input_estimator' instead."):
        with pytest.warns(FutureWarning, match="Parameter 'nonlinear_output_estimator' is deprecated. Use 'output_estimator' instead."):
            model = EstimatorModel(
                nonlinear_input_estimator="input_model",
                nonlinear_output_estimator="output_model", 
                input_estimator=None, 
                output_estimator=None, 
            )
    
    # Assert the values are transferred to the new parameters
    assert model.input_estimator == "input_model"
    assert model.output_estimator == "output_model"


# A sample function to apply the decorator
@param_deprecated_message(
    deprecated_params_mappings=[
        {
            'old': 'old_param',
            'new': 'new_param',
            'message': 'old_param is deprecated, use new_param instead.',
            'default': 'default_value'
        }
    ]
)
def test_function(old_param=None, new_param=None):
    return new_param

# Test cases

def test_deprecated_param_warning():
    with pytest.warns(FutureWarning, match="old_param is deprecated, use new_param instead."):
        result = test_function(old_param="deprecated_value")
    
    # Assert that the deprecated parameter triggers the warning
    assert result == "deprecated_value"  # The value should be passed correctly to the new_param


# Testing conditionally validated parameters
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'some_param',
            'condition': lambda val: val < 10,
            'message': 'some_param must be greater than or equal to 10',
            'default': 10
        }
    ]
)
def condition_test_function(some_param):
    return some_param

def test_condition_param_warning():
    with pytest.warns(FutureWarning, match="some_param must be greater than or equal to 10"):
        result = condition_test_function(some_param=5)
    
    # Check if the default value is assigned when condition fails
    assert result == 10

def test_condition_param_no_warning():
    result = condition_test_function(some_param=15)
    
    # No warning and should return the original value
    assert result == 15

# Run the tests
# ============================
# Pytest Main Execution Block
# ============================


if __name__=='__main__': 
    # TemporalTransformerFusion(2, 4, 2, 4, 64, dropout_rate=0.3, quantiles = [0.1, 0.5, 0.9] )
    # HWClassifier(nonlinear_input_estimator='estimator')
    
    pytest.main([__file__])