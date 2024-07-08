# -*- coding: utf-8 -*-
"""
@author: LKouadio <etanoyau@gmail.com> @~Daniel
"""
import re 
import os 
import warnings 
import numpy as np 

from ..decorators import CheckGDALData

with warnings.catch_warnings():  # noqa 
    warnings.filterwarnings(action='ignore', category=UserWarning)
    HAS_GDAL = CheckGDALData(None)._gdal_data_found
    NEW_GDAL = False

HAS_PROJ=False 
if (not HAS_GDAL):
    try:
        import pyproj
    except ImportError:
        raise RuntimeError("Either GDAL or PyProj must be installed")
else:
    import osgeo
    if hasattr(osgeo, '__version__') and int(osgeo.__version__[0]) >= 3:
        NEW_GDAL = True
    HAS_PROJ=True 
# Import pyproj and set ESPG_DICT 
EPSG_DICT = {}
try:
    if HAS_PROJ: 
        epsgfn = os.path.join(pyproj.pyproj_datadir, 'epsg')
        f = open(epsgfn, 'r')
        lines = f.readlines()

    for line in lines:
        if ('#' in line): continue
        epsg_code_val = re.compile('<(\d+)>').findall(line)
        if epsg_code_val is not None and len(epsg_code_val) > 0 and \
            epsg_code_val[0].isdigit():
            epsg_code = int(epsg_code_val[0])
            epsg_string = re.compile('>(.*)<').findall(line)[0].strip()

            EPSG_DICT[epsg_code] = epsg_string
        else:
            #print("epsg_code_val NOT found for this line ", line, epsg_code_val)
            pass  
   
except Exception:
    path = os.path.dirname(os.path.abspath(__file__))
    epsg_dict_fn = os.path.join(path, 'epsg.npy')

    EPSG_DICT = np.load(epsg_dict_fn, allow_pickle=True).item()