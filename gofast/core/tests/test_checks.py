# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:04:24 2024

@author: Daniel
"""

from numbers import Integral, Real 
import pytest
import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, Interval, HasMethods, Hidden
from typing import List, Dict, Any
from gofast.core.checks import ParamsValidator, InvalidParameters 
# Assuming ParamsValidator and InvalidParameters are imported from the module
# from your_module import ParamsValidator, InvalidParameters
#%%
# Mock functions to use as test cases
@ParamsValidator(
    {
        "a": [int],
        "b": [float],
        "c": [str, None],
        "d": [bool],
    }
)
def example1(a, b, c=None, d=False):
    return a, b, c, d

@ParamsValidator(
    {
        "a": [Integral],
        "b": [Real],
        "c": [str, None],
        "d": [bool],
    }
)
def example2(a, b, c=None, d=False):
    return a, b, c, d

@ParamsValidator(
    {
        "a": ['array-like'],
        "b": [Interval(Real, 0, 1, closed='left')],
        "c": [StrOptions({'elu', 'gelu'}), None],
        "d": ['boolean'],
        "f": [HasMethods(['fit', 'predict']), None],
        # "g": [Hidden(Interval(Real, 0, 1, closed ='neither')), None],
    }
)
def example3(a, b, c=None, d=False, f=None, g=None):
    return a, b, c, d, f, g

@ParamsValidator(
    {
        "a": ['array-like:np'],
        "b": ['array-like:pd.df:transf'],
        "c": ['array-like:list', None],
        "d": ['boolean'],
        "f": [HasMethods(['fit', 'predict']), None],
        # "g": [Hidden(Real), None],
    }
)
def example4(a, b, c=None, d=False, f=None, g=None):
    return a, b, c, d, f, g

@ParamsValidator(
    {
        "a": ['array-like:np'],
        "b": ['array-like:pd.df:transf'],
        "c": [List[str], None],
        "d": ['boolean'],
        "i": [Dict[str, float], None],
    },
    skip_nested_validation=False
)
def example5(a, b, c=None, d=False, i=None):
    return a, b, c, d, i

class TestParamsValidator:
    def test_example1_valid(self):
        assert example1(1, 2.0) == (1, 2.0, None, False)
        assert example1(10, 3.14, "test", True)

    def test_example1_invalid_a(self):
        with pytest.raises(InvalidParameters):
            example1("1", 2.0)

    def test_example1_invalid_b(self):
        with pytest.raises(InvalidParameters):
            example1(1, "2.0")

    def test_example1_invalid_c(self):
        with pytest.raises(InvalidParameters):
            example1(1, 2.0, 123)

    def test_example1_invalid_d(self):
        with pytest.raises(InvalidParameters):
            example1(1, 2.0, "test", "True")

    def test_example2_valid(self):
        assert example2(5, 3.5) == (5, 3.5, None, False)
        assert example2(10, 0.99, "hello", True)

    def test_example2_invalid_a(self):
        with pytest.raises(InvalidParameters):
            example2(5.5, 3.5)

    def test_example2_invalid_b(self):
        with pytest.raises(InvalidParameters):
            example2(5, "3.5")

    def test_example3_valid(self):
        a = [1, 2, 3]
        b = 0.5
        c = 'elu'
        f = MockModel()
        g = np.array([1, 2, 3])
        result = example3(a, b, c, True, f, g)
        assert result == (a, b, c, True, f, g)

    def test_example3_invalid_a(self):
        with pytest.raises(InvalidParameters):
            example3("not array-like", 0.5)

    def test_example3_invalid_b_interval(self):
        with pytest.raises(InvalidParameters):
            example3([1, 2, 3], 1.5)

    def test_example3_invalid_c_options(self):
        with pytest.raises(InvalidParameters):
            example3([1, 2, 3], 0.5, "invalid_option")

    def test_example3_invalid_d(self):
        with pytest.raises(InvalidParameters):
            example3([1, 2, 3], 0.5, "elu", "not_bool")

    def test_example3_invalid_f_methods(self):
        with pytest.raises(InvalidParameters):
            example3([1, 2, 3], 0.5, "elu", True, f="not a model")

    def test_example4_valid_transf(self):
        a = np.array([1, 2, 3])
        b = [1, 2, 3]  # Should be transformed to pd.DataFrame
        result = example4(a, b)
        assert isinstance(result[1], pd.DataFrame)

    def test_example4_invalid_a(self):
        with pytest.raises(InvalidParameters):
            example4([1, 2, 3], pd.DataFrame({"col": [1, 2, 3]}))  # a expects np.ndarray

    def test_example4_invalid_b_transf(self):
        with pytest.raises(InvalidParameters):
            example4(np.array([1, 2, 3]), "not transformable to df")

    def test_example5_valid_skip_nested(self):
        a = np.array([1, 2, 3])
        b = pd.DataFrame({"col": [1, 2, 3]})
        c = ["str1", "str2"]
        i = {"key1": 1.0, "key2": 2.5}
        result = example5(a, b, c, True, i)
        assert isinstance (result[-1], dict)

    def test_example5_invalid_a(self):
        with pytest.raises(InvalidParameters):
            example5([1, 2, 3], pd.DataFrame({"col": [1, 2, 3]}))

    def test_example5_invalid_b_transf(self):
        with pytest.raises(InvalidParameters):
            example5(np.array([1, 2, 3]), "not transformable to df")

    def test_example5_invalid_c_type(self):
        with pytest.raises(InvalidParameters):
            example5(np.array([1, 2, 3]), pd.DataFrame({"col": [1, 2, 3]}), c=123)

    def test_example5_invalid_i_type(self):
        with pytest.raises(InvalidParameters):
            example5(
                np.array([1, 2, 3]),
                pd.DataFrame({"col": [1, 2, 3]}),
                i={"key1": "not float"}
            )

    def test_class_decorator_valid(self):
        @ParamsValidator(
            {
                "a": [int],
                "b": [str],
            }
        )
        class Example:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        obj = Example(10, "test")
        assert obj.a == 10
        assert obj.b == "test"

    def test_class_decorator_invalid(self):
        @ParamsValidator(
            {
                "a": [int],
                "b": [str],
            }
        )
        class Example:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        with pytest.raises(InvalidParameters):
            Example("not int", "test")

    def test_nested_validation_skip(self):
        @ParamsValidator(
            {
                "c": [List[str], None],
            },
            skip_nested_validation=True
        )
        def func(c=None):
            return c

        assert func(["a", "b", "c"]) == ["a", "b", "c"]
        assert func(None) is None
        # No error even if list items are not strings
        assert func([1, 2, 3]) == [1, 2, 3]

    def test_nested_validation_no_skip(self):
        @ParamsValidator(
            {
                "c": [List[str], None],
            },
            skip_nested_validation=False
        )
        def func(c=None):
            return c

        assert func(["a", "b", "c"]) == ["a", "b", "c"]
        assert func(None) is None
        with pytest.raises(InvalidParameters):
            func([1, 2, 3])

    def test_dict_validation_skip(self):
        @ParamsValidator(
            {
                "i": [Dict[str, Any], None],
            },
            skip_nested_validation=True
        )
        def func(i=None):
            return i

        assert func({"key1": 1.0, "key2": "value"}) == {"key1": 1.0, "key2": "value"}
        assert func(None) is None

    def test_dict_validation_no_skip(self):
        @ParamsValidator(
            {
                "i": [Dict[str, float], None],
            },
            skip_nested_validation=False
        )
        def func(i=None):
            return i

        assert func({"key1": 1.0, "key2": 2.5}) == {"key1": 1.0, "key2": 2.5}
        assert func(None) is None
        with pytest.raises(InvalidParameters):
            func({"key1": "not float"})

# Mock model with fit and predict methods
class MockModel:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return X

if __name__=='__main__': 
    pytest.main([__file__])
