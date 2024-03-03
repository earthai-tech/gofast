# -*- coding: utf-8 -*-
"""
The `gofast.geo` module is designed to offer advanced geospatial analysis and 
manipulation tools within the gofast framework, facilitating the handling of 
geographical data, spatial transformations, and visualization.
"""
import typing
if typing.TYPE_CHECKING:
    # Avoid errors in type checkers (e.g. mypy) for experimental modules.
    from .system import  GeoIntelligentSystem # noqa
    
__all__= ['select_base_stratum' , 'reduce_samples' , 'transmissivity', 
          'calculate_K', 'compress_aquifer_data', 'correct_data_location', 
          'make_coords', 'compute_azimuth', 'calculate_bearing', ]
def __getattr__(name):
    if name =="GeoIntelligentSystem":
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            "deprecation cycle. To use it, you need to explicitly import "
            "`enable_geo_intel_system`:\n"
            "`from gofast.experimental import enable_geo_intel_system`"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")