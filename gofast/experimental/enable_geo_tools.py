# -*- coding: utf-8 -*-
"""Enables Geosciences tools

The API and results of this estimator might change without any deprecation
cycle.

Importing this file dynamically sets :class:`~gofast.geo.Profile`
as an attribute of the impute module::

    >>> # explicitly require this experimental feature
    >>> from gofast.experimental import enable_geo_tools  # noqa
    >>> # now you can import normally from impute
    >>> from gofast.geo import Profile, Location
    
Created on Sat Feb  3 20:58:51 2024
@author: Daniel
"""

from ..geo.site import Profile, Location
from .. import geo 

# use settattr to avoid mypy errors when monkeypatching
for name, value in zip (["Profile", "Location"], [Profile, Location]):
    setattr(geo, name , value )

geo.__all__ += ["Profile", "Location"]
