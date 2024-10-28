# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Network utilities module for handling network operations, URL validation, 
and data fetching.

This module includes functions for downloading data, tracking download 
progress, validating URLs, and fetching JSON data from APIs or web resources.
"""

import re
import json 
import warnings 
from six.moves import urllib 

from .depsutils import is_installing, is_module_installed

__all__= ["download_progress_hook", "url_checker", "validate_url", 
          "validate_url_by_validators", "fetch_json_data_from_url"
          ]


def url_checker (url: str , install:bool = False, 
                 raises:str ='ignore')-> bool : 
    """
    check whether the URL is reachable or not. 
    
    function uses the requests library. If not install, set the `install`  
    parameter to ``True`` to subprocess install it. 
    
    Parameters 
    ------------
    url: str, 
        link to the url for checker whether it is reachable 
    install: bool, 
        Action to install the 'requests' module if module is not install yet.
    raises: str 
        raise errors when url is not recheable rather than returning ``0``.
        if `raises` is ``ignore``, and module 'requests' is not installed, it 
        will use the django url validator. However, the latter only assert 
        whether url is right but not validate its reachability. 
              
    Returns
    --------
        ``True``{1} for reacheable and ``False``{0} otherwise. 
        
    Example
    ----------
    >>> from gofast.tools.coreutils import url_checker 
    >>> url_checker ("http://www.example.com")
    ...  0 # not reacheable 
    >>> url_checker ("https://gofast.readthedocs.io/en/latest/api/gofast.html")
    ... 1 
    
    """
    isr =0 ; success = False 
    
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        #domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
    
    try : 
        import requests 
    except ImportError: 
        if install: 
            success  = is_installing('requests', DEVNULL=True) 
        if not success: 
            if raises=='raises': 
                raise ModuleNotFoundError(
                    "auto-installation of 'requests' failed."
                    " Install it mannually.")
                
    else : success=True  
    
    if success: 
        try:
            get = requests.get(url) #Get Url
            if get.status_code == 200: # if the request succeeds 
                isr =1 # (f"{url}: is reachable")
                
            else:
                warnings.warn(
                    f"{url}: is not reachable, status_code: {get.status_code}")
                isr =0 
        
        except requests.exceptions.RequestException as e:
            if raises=='raises': 
                raise SystemExit(f"{url}: is not reachable \nErr: {e}")
            else: isr =0 
            
    if not success : 
        # use django url validation regex
        # https://github.com/django/django/blob/stable/1.3.x/django/core/validators.py#L45
        isr = 1 if re.match(regex, url) is not None else 0 
        
    return isr 

def download_progress_hook(t):
    """
    Hook to update the tqdm progress bar during a download.

    Parameters
    ----------
    t : tqdm
        An instance of tqdm to update as the download progresses.
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        Updates the progress bar.

        Parameters
        ----------
        b : int
            Number of blocks transferred so far [default: 1].
        bsize : int
            Size of each block (in tqdm-compatible units) [default: 1].
        tsize : int, optional
            Total size (in tqdm-compatible units). If None, remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to 

def fetch_json_data_from_url (url:str , todo:str ='load'): 
    """ Retrieve JSON data from url 
    :param url: Universal Resource Locator .
    :param todo:  Action to perform with JSON:
        - load: Load data from the JSON file 
        - dump: serialize data from the Python object and create a JSON file
    """
    with urllib.request.urlopen(url) as jresponse :
        source = jresponse.read()
    data = json.loads(source)
    if todo .find('load')>=0:
        todo , json_fn  ='loads', source 
        
    if todo.find('dump')>=0:  # then collect the data and dump it
        # set json default filename 
        todo, json_fn = 'dumps',  '_urlsourcejsonf.json'  
        
    return todo, json_fn, data 


def validate_url(url: str) -> bool:
    """
    Check if the provided string is a valid URL.

    Parameters
    ----------
    url : str
        The string to be checked as a URL.

    Raises
    ------
    ValueError
        If the provided string is not a valid URL.

    Returns
    -------
    bool
        True if the URL is valid, False otherwise.

    Examples
    --------
    >>> validate_url("https://www.example.com")
    True
    >>> validate_url("not_a_url")
    ValueError: The provided string is not a valid URL.
    """
    from urllib.parse import urlparse
    
    if is_module_installed("validators"): 
        return validate_url_by_validators (url)
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("The provided string is not a valid URL.")
    return True

def validate_url_by_validators(url: str):
    """
    Check if the provided string is a valid URL using `validators` packages.

    Parameters
    ----------
    url : str
        The string to be checked as a URL.

    Raises
    ------
    ValueError
        If the provided string is not a valid URL.

    Returns
    -------
    bool
        True if the URL is valid, False otherwise.

    Examples
    --------
    >>> validate_url("https://www.example.com")
    True
    >>> validate_url("not_a_url")
    ValueError: The provided string is not a valid URL.
    """
    import validators
    if not validators.url(url):
        raise ValueError("The provided string is not a valid URL.")
    return True

