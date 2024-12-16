# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
# gofast.nn.transformers.py

"""
Add dostring here. Come back to new lines after 65 characters 
and give an concrete example of each transformers until to predict 

# XTFT( 
#      static_input_dim: int,
#      dynamic_input_dim: int,
#      future_covariate_dim: int,
#      embed_dim: int = 32,
#      forecast_horizons: int = 1,
#      quantiles: Union[str, List[float], None] = None,
#      max_window_size: int = 10,
#      memory_size: int = 100,
#      num_heads: int = 4,
#      dropout_rate: float = 0.1,
#      output_dim: int = 1, 
#      anomaly_config: Optional[Dict[str, Any]] = None,  
#      attention_units: int = 32,
#      hidden_units: int = 64,
#      lstm_units: int = 64,
#      scales: Union[str, List[int], None] = None,
#      multi_scale_agg: Optional[str] = None, 
#      activation: str = 'relu',
#      use_residuals: bool = True,
#      use_batch_norm: bool = False,
#      final_agg: str = 'last',
#      )

# TemporalFusionTransformer (
#     static_input_dim,
#     dynamic_input_dim,
#     num_static_vars,
#     num_dynamic_vars,
#     hidden_units,
#     num_heads=4,  
#     dropout_rate=0.1,
#     forecast_horizon=1,
#     quantiles=None,
#     activation='elu',
#     use_batch_norm=False,
#     num_lstm_layers=1,
#     lstm_units=None, 
#     )


"""
from ._tft import TemporalFusionTransformer 
from ._xtft import XTFT 

__all__ =["TemporalFusionTransformer", "XTFT"]