# -*- coding: utf-8 -*-
#   Author: L.Kouadio <etanoyau@gmail.com>
#   License: BSD-3-Clause
"""
Config Dataset Module
=====================

This module sets up all datasets for efficient fetching from local or remote sources 
based on specified tags indicating the data processing level.

Author: LKouadio <etanoyau@gmail.com>
License: BSD-3-Clause
"""

import re
from warnings import warn

from .load import (load_bagoue, load_iris, load_hlogs, load_nlogs, load_mxs,
                   load_forensic, load_jrs_bet)
from ..tools.funcutils import listing_items_format
from ..exceptions import DatasetError
from .._gofastlog import gofastlog

_logger = gofastlog().get_gofast_logger(__name__)

__all__ = [
    "load_bagoue", "load_iris", "load_hlogs", "load_nlogs", "fetch_data",
    "load_mxs", "load_forensic", "load_jrs_bet", "DATASET"
]

_DTAGS = ("bagoue", "iris", "hlogs", "nlogs", "mxs", "forensic", "jrs_bet")

# Error messages for different processing stages
_ERROR_MSGS = {
    'pipe': "Failed to build default transformer pipeline.",
    'origin': "Failed to fetch original data."
}
_ERROR_MSGS.update({tag: f"Failed to fetch {tag} data." for tag in _DTAGS if tag != 'pipe'})

# Compile regex for tag matching
_TAG_REGEX = re.compile(r'|'.join(_DTAGS + ('origin',)), re.IGNORECASE)

# Attempt to import _fetch_data function
try:
    from ._config import _fetch_data
except ImportError:
    warn("'fetch_data' is unavailable. Use specific 'load_<area_name>' functions instead.")
    _fetch_data = None


def fetch_data(tag, **kwargs):
    """
    Fetch dataset based on a specified tag.

    Parameters
    ----------
    tag: str
        Name of the data processing stage or dataset area.
    **kwargs: dict
        Additional keyword arguments for the dataset loading function.

    Returns
    -------
    Varies
        The fetched dataset, formatted based on the specified tag.

    Raises
    ------
    DatasetError
        If the tag is unknown or if data loading fails.
    """
    load_funcs = {
        'bagoue': load_bagoue, 'iris': load_iris, 'hlogs': load_hlogs,
        'nlogs': load_nlogs, 'mxs': load_mxs, 'forensic': load_forensic,
        'jrs_bet': load_jrs_bet
    }
    tag = _parse_tag(tag, default='bagoue')
    if _fetch_data and callable(_fetch_data) and tag not in load_funcs.keys():
        return _fetch_data(tag, data_names = load_funcs.keys(), **kwargs)

    load_func = load_funcs.get(tag)
    if not load_func:
        raise DatasetError(f"No load function available for tag '{tag}'.")

    return load_func(**kwargs)

def _parse_tag(tag, default='bagoue'):
    """
    Sanitize and validate the dataset tag.

    Parameters
    ----------
    tag: str
        The dataset tag to be parsed.
    default: str
        Default dataset to use if the tag is not recognized.

    Returns
    -------
    str
        The sanitized and validated tag.
    """
    more= str(tag).lower().split() 
    tag_match = _TAG_REGEX.search(tag.lower())
      
    if not tag_match:
        warn(f"Unknown tag '{tag}'. Using default '{default} {tag.lower()}'.", FutureWarning)
        return default + f' {tag.lower()}'
    
    # if tag + other string, then added if tag =default otherwise skip
    if tag_match.group()==default and len(more) > 1 :
        return default + str(tag).lower().replace (default, '')
    
    return tag_match.group()

# Format dataset loading functions list
_formatted_load_funcs = ["{:<7}: {:<7}()".format(s.upper(), 'load_' + s) for s in _DTAGS]
_DATASET_LIST = listing_items_format(
    _formatted_load_funcs, "Fetch data using 'load_<type_of_data|area_name>' or 'fetch_data(<type_of_data|area_name>)'.",
    inline=True, verbose=False
)

_DATASET_DOC = """
Gofast dataset includes various data types for software implementation, such as:
- 'nlogs': Land subsidence data in Nansha district, Guangzhou, China.:doi:`https://doi.org/10.1016/j.jenvman.2024.120078`. 
- 'Bagoue': Flow rate features data. :doi:`https://doi.org/10.1029/2021wr031623` or :doi:`https://doi.org/10.1007/s11269-023-03562-5`
- 'Hlogs' and 'Mxs': Hydrogeological engineering logging data.:doi:` https://doi.org/10.1007/s12145-024-01236-3`
- 'Forensic': DNA forensic dataset from West Africa.
- 'Jrs_bet': Lottery dataset for educational purposes.
"""

DATASET = type("DATASET", (), {"KIND": _DTAGS, "HOW": _DATASET_LIST, "DOC": _DATASET_DOC})
