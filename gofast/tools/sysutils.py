# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
System utilities module for managing system-level operations.

This module provides utilities essential for system-level tasks such as
color management, regular expression searching, and projection validation,
along with other miscellaneous system operations.
"""

import os 
import re
import platform
import socket
import subprocess 
import tempfile
import shutil
import importlib.util
import inspect 
import itertools 
from typing import Union, Tuple, Dict,Optional, List
from typing import Sequence, Any, Callable
import multiprocessing
from concurrent.futures import as_completed 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .._gofastlog import gofastlog
from ..api.summary import ReportFactory 
from ..api.util import get_table_size
from .coreutils import is_iterable
from .depsutils import import_optional_dependency, ensure_pkgs

try:
    import psutil
except : pass 
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TW = get_table_size() 

logger = gofastlog().get_gofast_logger(__name__)

__all__= [
    'check_port_in_use',
    'clean_temp_files',
    'create_temp_dir',
    'create_temp_file',
    'environment_summary',
    'find_by_regex',
    'find_similar_string',
    'get_cpu_usage',
    'get_disk_usage',
    'get_gpu_info',
    'get_installed_packages',
    'get_memory_usage',
    'get_python_version',
    'get_system_info',
    'get_uptime',
    'is_gpu_available',
    'is_package_installed',
    'is_path_accessible',
    'is_port_open',
    'manage_env_variable',
    'manage_file_lock',
    'manage_temp',
    'parallelize_jobs',
    'represent_callable',
    'run_command',
    'safe_getattr',
    'system_uptime'
]

@ensure_pkgs(
    ['psutil'], 
    extra="`get_cpu_usage` requires the `psutil` "
    "package for system resource monitoring",
    auto_install=True
)
def get_cpu_usage(per_cpu: bool = False) -> Optional[float]:
    """
    Returns the current CPU usage as a percentage, optionally providing 
    per-core usage for systems with multiple cores.
    
    Parameters
    ----------
    per_cpu : bool, default=False
        If True, returns a list with the CPU usage percentage for each core. 
        If False, returns the overall CPU usage as a single percentage.
    
    Returns
    -------
    usage : float or list of float, optional
        If `per_cpu` is False, returns the overall CPU usage as a float percentage.
        If `per_cpu` is True, returns a list with each entry corresponding to the
        usage percentage of an individual core.
    
    Notes
    -----
    This function uses the `psutil` library to retrieve CPU usage information 
    and requires an interval of 1 second to calculate the usage accurately.

    Examples
    --------
    >>> from gofast.tools.sysutils import get_cpu_usage
    >>> get_cpu_usage()
    23.5
    >>> get_cpu_usage(per_cpu=True)
    [20.4, 25.1, 21.3, 24.5]

    """
    try:
        usage = psutil.cpu_percent(interval=1, percpu=per_cpu)
        logger.debug(f"CPU usage retrieved: {usage}")
        return usage
    except Exception as e:
        logger.error(f"Failed to retrieve CPU usage: {e}")
        return None
    
@ensure_pkgs(
    ['psutil'], 
    extra="`get_memory_usage` requires the `psutil`"
    " package for system memory usage",
    auto_install=True
)
def get_memory_usage() -> Optional[Tuple[float, float, float]]:
    """
    Retrieves system memory usage statistics, providing the total, used, 
    and available memory in megabytes (MB).
    
    Returns
    -------
    memory : tuple of float
        A tuple containing:
        
        - `total_memory` : Total memory in MB.
        - `used_memory` : Used memory in MB.
        - `available_memory` : Available memory in MB.
    
    Notes
    -----
    This function leverages the `psutil` library for retrieving memory usage 
    information. The conversion to MB is performed by dividing each value 
    by 1024^2.

    Examples
    --------
    >>> from gofast.tools.sysutils import get_memory_usage
    >>> total, used, available = get_memory_usage()
    >>> print(f"Total: {total} MB, Used: {used} MB, Available: {available} MB")
    Total: 8192 MB, Used: 4096 MB, Available: 4096 MB

    """
    try:
        mem = psutil.virtual_memory()
        total_memory = mem.total / (1024 ** 2)  # Convert to MB
        used_memory = mem.used / (1024 ** 2)    # Convert to MB
        available_memory = mem.available / (1024 ** 2)  # Convert to MB
        logger.debug(f"Memory usage retrieved: Total: {total_memory} MB,"
                     " Used: {used_memory} MB, Available: {available_memory} MB")
        memory_infos ={
            "Total": f"{total_memory} MB", 
            "Used": f"{used_memory} MB", 
            "Available": f"{available_memory} MB"
            }
        summary = ReportFactory(
            "Memory usage retrieved", descriptor="memory_usage"
            ).add_mixed_types(memory_infos, table_width=TW )
        return summary
    
    except Exception as e:
        logger.error(f"Failed to retrieve memory usage: {e}")
        return None

@ensure_pkgs(
    ['psutil'], 
    extra="`get_disk_usage` requires the `psutil`"
    " package for disk usage statistics ",
    auto_install=True
)
def get_disk_usage(path: str = "/") -> Optional[Tuple[float, float, float]]:
    """
    Returns disk usage statistics for a specified filesystem path, 
    including total, used, and free disk space in gigabytes (GB).
    
    Parameters
    ----------
    path : str, default='/'
        The filesystem path for which to check disk usage statistics. 
        By default, it uses the root directory (`/`).
    
    Returns
    -------
    disk_usage : tuple of float, optional
        A tuple containing:
        
        - `total_disk` : Total disk space in GB.
        - `used_disk` : Used disk space in GB.
        - `free_disk` : Free disk space in GB.

    Notes
    -----
    Disk usage information is gathered using the `psutil` library. Disk space
    is converted to gigabytes (GB) by dividing the values by 1024^3.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist on the filesystem.
    PermissionError
        If the program does not have permission to access the specified path.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import get_disk_usage
    >>> total, used, free = get_disk_usage(path="/")
    >>> print(f"Total: {total} GB, Used: {used} GB, Free: {free} GB")
    Total: 256 GB, Used: 128 GB, Free: 128 GB

    """
    try:
        usage = psutil.disk_usage(path)
        total_disk = usage.total / (1024 ** 3)  # Convert to GB
        used_disk = usage.used / (1024 ** 3)    # Convert to GB
        free_disk = usage.free / (1024 ** 3)    # Convert to GB
        logger.debug(f"Disk usage for path '{path}': Total: {total_disk} GB,"
                     " Used: {used_disk} GB, Free: {free_disk} GB")
        disk_usage_infos ={
            "Total": f"{total_disk} GB", 
            "Used": f"{used_disk} GB", 
            "Free": f"{free_disk} GB"
            }
        summary = ReportFactory(
            "Disk usage Infos", 
            descriptor="disk_usage"
            ).add_mixed_types(disk_usage_infos, table_width=TW )
        return summary
     
    except FileNotFoundError:
        logger.error(f"Path not found: {path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied for path: {path}")
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve disk usage for path '{path}': {e}")
        return None

def is_gpu_available() -> bool:
    """
    Checks if a GPU is available for computation on the system, using the 
    PyTorch library if it is installed.
    
    Returns
    -------
    available : bool
        True if a GPU is available, False otherwise.
    
    Notes
    -----
    This function relies on the `torch` library (PyTorch) to detect GPU 
    availability. If PyTorch is not installed, it logs a warning and 
    returns False.
    
    Raises
    ------
    ImportError
        If PyTorch is not installed and thus the GPU availability 
        check cannot be performed.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import is_gpu_available
    >>> is_gpu_available()
    True

    """
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        logger.debug(f"GPU availability check: {gpu_available}")
        return gpu_available
    except ImportError:
        logger.warning(
            "PyTorch is not installed. GPU availability check cannot be performed.")
        return False
    except Exception as e:
        logger.error(f"Failed to check GPU availability: {e}")
        return False


@ensure_pkgs( "torch", "torch library is required for retrieving"
             " detailed information about available GPUs."
 )
def get_gpu_info() -> Optional[Dict[str, str]]:
    """
    Provides detailed information about available GPUs, including device name, 
    memory capacity, and CUDA version (if PyTorch is installed).
    
    Returns
    -------
    gpu_info : dict or None
        Dictionary containing GPU details, including:
        
        - `device_count` : Number of available GPU devices.
        - `device_name` : Name of the first GPU device.
        - `memory_total` : Total memory of the first GPU device in GB.
        - `cuda_version` : CUDA version, if available.

        If no GPU is available or PyTorch is not installed, returns None.
    
    Notes
    -----
    This function requires PyTorch to check for GPU availability. If PyTorch 
    is not installed, it logs a warning and returns None.

    Raises
    ------
    ImportError
        If PyTorch is not installed on the system.
    RuntimeError
        If there is an issue retrieving GPU properties.

    Examples
    --------
    >>> from gofast.tools.sysutils import get_gpu_info
    >>> gpu_info = get_gpu_info()
    >>> print(gpu_info)
    {'device_count': '1', 'device_name': 'NVIDIA Tesla T4', 
     'memory_total': '15.99 GB', 'cuda_version': '11.1'}

    """
    if not TORCH_AVAILABLE:
        logger.warning(
            "PyTorch not installed; detailed GPU information is unavailable.")
        return None

    if not torch.cuda.is_available():
        logger.info("No GPU is available on this system.")
        return None

    try:
        gpu_info = {
            "device_count": str(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": ( 
                f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
                ),
            "cuda_version": torch.version.cuda or "N/A",
        }
        logger.debug(f"GPU information: {gpu_info}")
        summary = ReportFactory(
            "GPU Infos", 
            descriptor="gpu_info"
            ).add_mixed_types(gpu_info, table_width=TW )
        
        return summary

    except Exception as e:
        logger.error(f"Error retrieving GPU information: {e}")
        return None

def system_uptime() -> str:
    """
    Retrieves the system uptime, which is the duration the system has been 
    running since the last boot, in a human-readable format.

    Returns
    -------
    uptime : str
        System uptime in the format "Xd:Yh:Zm:Ws", where X, Y, Z, and W 
        represent days, hours, minutes, and seconds, respectively.
    
    Notes
    -----
    This function is cross-platform and works on Windows, macOS (Darwin), 
    and Linux. It uses different commands to retrieve uptime based on the 
    operating system.

    Raises
    ------
    NotImplementedError
        If the function is called on an unsupported operating system.
    RuntimeError
        If an error occurs while retrieving uptime.

    Examples
    --------
    >>> from gofast.tools.sysutils import system_uptime
    >>> system_uptime()
    '2d:10h:33m:12s'
    
    """
    try:
        uptime_seconds = None
        if platform.system() == "Windows":
            uptime_seconds = int(subprocess.check_output("net stats srv", shell=True)
                                 .decode().split("since")[1].strip().split()[0])
        elif platform.system() == "Linux":
            uptime_seconds = int(float(open("/proc/uptime").readline().split()[0]))
        elif platform.system() == "Darwin":
            uptime_seconds = int(subprocess.check_output("sysctl -n kern.boottime", shell=True)
                                 .decode().split(",")[0].split(" ")[4])
        
        if uptime_seconds is not None:
            days, remainder = divmod(uptime_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{days}d:{hours}h:{minutes}m:{seconds}s"
            logger.debug(f"System uptime: {uptime_str}")
            return uptime_str
        else:
            raise NotImplementedError("Unsupported operating system.")
    except Exception as e:
        logger.error(f"Failed to retrieve system uptime: {e}")
        return "N/A"

def is_port_open(port: int) -> bool:
    """
    Checks if a specified network port is open or occupied on the local machine.
    
    Parameters
    ----------
    port : int
        The port number to check for availability.

    Returns
    -------
    bool
        Returns True if the port is open (not in use), otherwise False.
    
    Notes
    -----
    This function uses a socket connection to check if the specified port 
    is open. It is helpful in applications where network services or 
    applications need to bind to a specific port.
    
    Raises
    ------
    ValueError
        If an invalid port number is provided.

    Examples
    --------
    >>> from gofast.tools.sysutils import is_port_open
    >>> is_port_open(8080)
    False
    
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(("localhost", port))
        is_open = result != 0
        status = "open" if is_open else "occupied"
        logger.debug(f"Port {port} is {status}.")
        return is_open

@ensure_pkgs(
    ['psutil'], 
    extra="`get_disk_usage` requires the `psutil`"
    " package for summarizing of the current environment ",
    auto_install=True
)
def environment_summary() -> Dict[str, str]:
    """
    Provides a summary of the current computing environment, including 
    information on Python version, OS, CPU, memory, available GPU(s), 
    and a list of installed Python packages.

    Returns
    -------
    env_info : dict
        Dictionary containing environment details, including:
        
        - `python_version` : The version of Python in use.
        - `os` : Operating system name.
        - `os_version` : Version of the operating system.
        - `cpu_count` : Number of logical CPU cores.
        - `memory` : Total system memory in GB.
        - `device_count`, `device_name`, `memory_total`, `cuda_version` 
          (if available) : GPU details from `detailed_gpu_info`.
        - `installed_packages` : List of installed Python packages 
          (first 10) in `name==version` format.
    
    Notes
    -----
    The function attempts to load installed packages using `pkg_resources`.
    If `pkg_resources` is not available, it defaults to "N/A" for 
    installed packages.

    Raises
    ------
    ImportError
        If `pkg_resources` is not installed.
    RuntimeError
        If an error occurs while gathering environment information.

    Examples
    --------
    >>> from gofast.tools.sysutils import environment_summary
    >>> env_info = environment_summary()
    >>> print(env_info)
    {'python_version': '3.9.5', 'os': 'Linux', 'os_version': '5.4.0-80-generic',
     'cpu_count': '4', 'memory': '15.5 GB', 'device_count': '1', 
     'device_name': 'NVIDIA Tesla T4', 'memory_total': '15.99 GB', 
     'cuda_version': '11.1', 'installed_packages': 'numpy==1.21.0, pandas==1.3.0, ...'}

    """
    
    env_info = {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu_count": str(psutil.cpu_count(logical=True)),
        "memory": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
    }

    # GPU information if available
    gpu_info = get_gpu_info()
    if gpu_info:
        env_info.update(gpu_info)

    try:
        # List installed packages if possible
        import pkg_resources
        installed_packages = [
            f"{d.project_name}=={d.version}" for d in pkg_resources.working_set]
        env_info["installed_packages"] = ", ".join(
            installed_packages[:10]) + "..." if len(
                installed_packages) > 10 else ", ".join(installed_packages)
    except ImportError:
        env_info["installed_packages"] = "N/A"
    
    logger.debug(f"Environment summary: {env_info}")
    
    summary = ReportFactory(
        "Environment Summary", 
        descriptor="env_info"
        ).add_mixed_types(env_info, table_width=TW )
    
    return summary 


def manage_env_variable(
    var_name: str,
    value: Optional[str] = None,
    default: Optional[str] = None,
    action: str = "get",
    file_path: Optional[str] = None,
    overwrite: bool = False
) -> Optional[str]:
    """
    Manages environment variables, allowing retrieval, setting, or loading 
    from a `.env` file.
    
    Parameters
    ----------
    var_name : str
        The name of the environment variable to retrieve, set, or load.
    
    value : str, optional
        The value to set for the environment variable. Only used if `action`
        is `"set"`. Default is None.
    
    default : str, optional
        The default value to return if the environment variable `var_name` 
        is not found when `action` is `"get"`. If None, returns None when the 
        variable is not found. Default is None.
    
    action : str, default="get"
        The action to perform. Options are:
            - `"get"`: Retrieves the environment variable `var_name`.
            - `"set"`: Sets the environment variable `var_name` to `value`.
            - `"load"`: Loads environment variables from a `.env` file 
              specified by `file_path`.
    
    file_path : str, optional
        The path to the `.env` file to load variables from when `action`
        is `"load"`. Required if `action` is `"load"`.
    
    overwrite : bool, default=False
        If True, allows overwriting existing environment variables when 
        `action` is `"load"` or `"set"`. If False, preserves the current 
        value of existing environment variables.
    
    Returns
    -------
    result : str or None
        - If `action` is `"get"`, returns the value of the environment 
          variable `var_name` or `default` if the variable is not set.
        - If `action` is `"set"` or `"load"`, returns None.
    
    Notes
    -----
    - This function is useful for managing configuration data securely by 
      utilizing environment variables.
    - Loading from a `.env` file allows you to define multiple variables 
      in a single file, each defined in the `KEY=VALUE` format.
    
    Raises
    ------
    ValueError
        If `action` is `"set"` and `value` is not provided, or if `action`
        is `"load"` and `file_path` is not specified.
    
    FileNotFoundError
        If `action` is `"load"` and `file_path` does not exist.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import manage_env_variable
    >>> manage_env_variable('HOME', action='get')
    '/home/username'
    >>> manage_env_variable('NEW_VAR', value='new_value', action='set')
    >>> manage_env_variable('NEW_VAR', action='get')
    'new_value'
    >>> manage_env_variable('NON_EXISTENT_VAR', default='default_value', action='get')
    'default_value'
    >>> manage_env_variable(
        var_name='', action='load', file_path='/path/to/.env', overwrite=True)

    See Also
    --------
    os.getenv : Retrieves environment variables.
    os.environ : Provides access to the environment variables.
    """
    if action == "get":
        # Get environment variable or return default if not found
        return os.getenv(var_name, default)

    elif action == "set":
        if value is None:
            raise ValueError("A value must be provided when action='set'.")
        if overwrite or var_name not in os.environ:
            os.environ[var_name] = value

    elif action == "load":
        if file_path is None:
            raise ValueError("file_path must be specified when action='load'.")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{file_path}' was not found.")

        with open(file_path) as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip comments and empty lines
                if "=" not in line:
                    raise ValueError(f"Invalid line in .env file: '{line}'")
                
                file_var_name, file_value = map(str.strip, line.split("=", 1))
                if overwrite or file_var_name not in os.environ:
                    os.environ[file_var_name] = file_value
    else:
        raise ValueError(
            f"Invalid action '{action}'. Expected 'get', 'set', or 'load'.")


def is_path_accessible(path: str, permissions: str = "r") -> bool:
    """
    Checks if a specified path is accessible with the given permissions.
    
    Parameters
    ----------
    path : str
        The path to check for accessibility.
    
    permissions : str, optional
        The permission types to check: `'r'` for read, `'w'` for write, 
        `'x'` for execute. Multiple permissions can be specified, e.g., 
        `"rw"`. Default is `"r"`.
    
    Returns
    -------
    accessible : bool
        True if the path is accessible with the specified permissions, 
        otherwise False.
    
    Notes
    -----
    This function verifies file permissions in the current user context, 
    ensuring flexibility for multi-user environments.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import is_path_accessible
    >>> is_path_accessible("/path/to/file", permissions="rw")
    True

    """
    if not os.path.exists(path):
        return False

    permission_checks = {
        "r": os.R_OK,
        "w": os.W_OK,
        "x": os.X_OK,
    }
    
    # Validate the permissions argument
    for perm in permissions:
        if perm not in permission_checks:
            raise ValueError(f"Invalid permission '{perm}'. Use 'r', 'w', or 'x'.")
    
    # Check each specified permission
    return all(os.access(path, permission_checks[perm]) for perm in permissions)


@ensure_pkgs(
    names="fcntl",
    extra="`fcntl` is required for file locking/unlocking in Unix-based systems.",
    auto_install=False,
    dist_name="fcntl",
    # `fcntl` is standard on Unix;let the user know if unavailable.
    infer_dist_name=False 
)
def manage_file_lock(
    file_path: str,
    action: str = "lock",
    blocking: bool = True,
    exclusive: bool = True
) -> Optional[int]:
    """
    Manages file locking and unlocking to prevent concurrent access.
    
    This function allows both locking and unlocking actions on a file to 
    prevent or allow concurrent access. It opens the file and applies an 
    exclusive lock or shared lock, depending on the parameters specified.
    
    Parameters
    ----------
    file_path : str
        Path to the file that needs to be locked or unlocked.
    
    action : str, default="lock"
        Specifies the action to perform: `"lock"` to acquire a lock, 
        or `"unlock"` to release a previously acquired lock.
    
    blocking : bool, default=True
        If True, the lock will block until it can be acquired. If False, 
        the lock will raise an exception if it cannot be acquired immediately.
    
    exclusive : bool, default=True
        If True, an exclusive lock is applied. If False, a shared lock is 
        applied (other processes can read the file simultaneously).
    
    Returns
    -------
    file_descriptor : int or None
        If `action` is `"lock"`, returns the file descriptor on success; 
        otherwise, None if `action` is `"unlock"` or if locking fails.
    
    Notes
    -----
    This function uses the `fcntl` module for locking, which is only 
    available on Unix-based systems. The lock is maintained as long as the 
    file descriptor remains open.

    - For `"lock"`, the function opens the file and applies a lock. 
    - For `"unlock"`, it removes the lock and closes the file descriptor.

    Raises
    ------
    ValueError
        If the `action` parameter is not one of `"lock"` or `"unlock"`.
    
    OSError or IOError
        If locking or unlocking the file fails.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import manage_file_lock
    >>> fd = manage_file_lock("/path/to/file", action="lock", blocking=True)
    >>> if fd:
    ...     print("File is locked.")
    ...     manage_file_lock(fd, action="unlock")
    ...     print("File is unlocked.")
    
    See Also
    --------
    os.open : Opens a file descriptor.
    fcntl.flock : Applies or removes file locks.
    """
    import fcntl

    if action not in {"lock", "unlock"}:
        raise ValueError(f"Invalid action '{action}'. Expected 'lock' or 'unlock'.")

    # Lock action: Open and lock the file
    if action == "lock":
        try:
            fd = os.open(file_path, os.O_RDWR)
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            lock_flag = 0 if blocking else fcntl.LOCK_NB
            fcntl.flock(fd, lock_type | lock_flag)
            logger.debug(f"Locked file '{file_path}' with fd {fd}")
            return fd
        except (OSError, IOError) as e:
            logger.error(f"Failed to lock file '{file_path}': {e}")
            return None

    # Unlock action: Unlock and close the file descriptor
    elif action == "unlock":
        try:
            fcntl.flock(file_path, fcntl.LOCK_UN)
            os.close(file_path)
            logger.debug(f"Unlocked file '{file_path}' with fd {file_path}")
            return None
        except (OSError, IOError) as e:
            logger.error(f"Failed to unlock file descriptor '{file_path}': {e}")
            return None


def get_system_info() -> Dict[str, str]:
    """
    Retrieves basic system information including OS, Python version, 
    CPU details, and GPU availability.

    Returns
    -------
    system_info : dict
        A dictionary containing basic system information:
        
        - `os_name` : Name of the operating system.
        - `os_version` : Version of the operating system.
        - `python_version` : Python version.
        - `cpu_count` : Number of logical CPUs.
        - `gpu_available` : Whether a GPU is available (`True` or `False`).

    Notes
    -----
    This function checks for GPU availability via PyTorch if installed, 
    otherwise it defaults to False.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import get_system_info
    >>> get_system_info()
    {'os_name': 'Linux', 'os_version': '5.4.0-81-generic', 'python_version': '3.8.5', 
     'cpu_count': '8', 'gpu_available': 'True'}
    
    See Also
    --------
    get_python_version : Retrieves the current Python version.
    """
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    system_info = {
        "os_name": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": str(psutil.cpu_count(logical=True)),
        "gpu_available": str(gpu_available),
    }
    return system_info


def get_python_version() -> str:
    """
    Returns the version of Python being used in the current environment.

    Returns
    -------
    python_version : str
        The version of Python currently in use.

    Examples
    --------
    >>> from gofast.tools.sysutils import get_python_version
    >>> get_python_version()
    '3.8.5'

    See Also
    --------
    get_system_info : Provides broader system information, including Python version.
    """
    return platform.python_version()


def get_installed_packages() -> List[str]:
    """
    Lists all installed packages along with their versions in the current 
    Python environment.

    Returns
    -------
    installed_packages : list of str
        A list of installed packages and their versions in the format 
        `package_name==version`.

    Notes
    -----
    This function is useful for dependency management and tracking installed 
    packages, especially in data science and production environments.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import get_installed_packages
    >>> get_installed_packages()
    ['numpy==1.21.0', 'pandas==1.3.0', 'scikit-learn==0.24.2', ...]

    See Also
    --------
    environment_summary : Summarizes the environment, including installed packages.
    """
    try:
        import pkg_resources
        installed_packages = [f"{d.project_name}=={d.version}"
                              for d in pkg_resources.working_set]
        return installed_packages
    except ImportError:
        logger.warning(
            "pkg_resources is not available. Cannot list installed packages.")
        return []

def run_command(command: str, capture_output: bool = True) -> Optional[str]:
    """
    Runs a shell command and optionally captures its output.
    
    Parameters
    ----------
    command : str
        The shell command to execute.

    capture_output : bool, default=True
        If True, captures and returns the command’s output. If False, 
        runs the command without capturing output, which is useful for 
        commands that produce a large output or run interactively.
    
    Returns
    -------
    output : str or None
        Returns the command output as a string if `capture_output` is True. 
        If `capture_output` is False, returns None.
    
    Notes
    -----
    This function uses `subprocess.run` to execute shell commands, which 
    allows for error handling and logging.
    
    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with a non-zero status and `capture_output` is 
        True.

    Examples
    --------
    >>> from gofast.tools.sysutils import run_command
    >>> run_command("echo Hello World")
    'Hello World\n'
    
    """
    try:
        result = subprocess.run(
            command, shell=True, 
            capture_output=capture_output, 
            text=True, 
            check=True
        )
        return result.stdout if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")
        return None


def create_temp_file(suffix: str = "", prefix: str = "tmp") -> str:
    """
    Creates a temporary file and returns its path.
    
    Parameters
    ----------
    suffix : str, optional
        The suffix for the temporary file. Default is an empty string.

    prefix : str, optional
        The prefix for the temporary file. Default is `"tmp"`.
    
    Returns
    -------
    file_path : str
        The full path of the created temporary file.
    
    Notes
    -----
    This function is useful for handling data temporarily in applications 
    where files need to be stored and accessed for a short time.

    Examples
    --------
    >>> from gofast.tools.sysutils import create_temp_file
    >>> temp_file = create_temp_file()
    >>> print(temp_file)
    '/tmp/tmpabcd1234'
    
    See Also
    --------
    create_temp_dir : Creates a temporary directory.
    """
    temp_file = tempfile.NamedTemporaryFile(
        suffix=suffix, prefix=prefix, delete=False)
    temp_file.close()
    return temp_file.name

def create_temp_dir(prefix: str = "tmp") -> str:
    """
    Creates a temporary directory and returns its path.
    
    Parameters
    ----------
    prefix : str, optional
        The prefix for the temporary directory name. Default is `"tmp"`.
    
    Returns
    -------
    dir_path : str
        The full path of the created temporary directory.
    
    Notes
    -----
    This function is helpful for managing temporary directories in 
    applications where short-term data storage is needed.

    Examples
    --------
    >>> from gofast.tools.sysutils import create_temp_dir
    >>> temp_dir = create_temp_dir()
    >>> print(temp_dir)
    '/tmp/tmpabcd1234'
    
    See Also
    --------
    create_temp_file : Creates a temporary file.
    """
    return tempfile.mkdtemp(prefix=prefix)


def clean_temp_files(directory: Optional[str] = None) -> None:
    """
    Cleans up temporary files in a specified directory.
    
    Parameters
    ----------
    directory : str, optional
        The directory to clean up. If None, cleans the default temporary 
        directory.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function is particularly useful for freeing up disk space in 
    data-intensive applications.

    Examples
    --------
    >>> from gofast.tools.sysutils import clean_temp_files
    >>> clean_temp_files("/path/to/temp/dir")
    
    """
    dir_to_clean = directory or tempfile.gettempdir()
    for item in os.listdir(dir_to_clean):
        item_path = os.path.join(dir_to_clean, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")


def is_package_installed(package_name: str) -> bool:
    """
    Checks if a specific package is installed in the current Python 
    environment.
    
    Parameters
    ----------
    package_name : str
        The name of the package to check.
    
    Returns
    -------
    bool
        True if the package is installed, otherwise False.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import is_package_installed
    >>> is_package_installed("numpy")
    True
    
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


def manage_temp(
    suffix: str = "",
    prefix: str = "tmp",
    action: str = "create_file",
    directory: Optional[str] = None,
    clean_all: bool = False
) -> Optional[str]:
    """
    Manages temporary files and directories by creating, accessing, or 
    cleaning them as needed.
    
    Parameters
    ----------
    suffix : str, optional
        Suffix for the temporary file or directory, used only if `action`
        is `"create_file"` or `"create_dir"`. Default is an empty string.
    
    prefix : str, optional
        Prefix for the temporary file or directory, used only if `action`
        is `"create_file"` or `"create_dir"`. Default is `"tmp"`.
    
    action : str, default="create_file"
        Specifies the operation to perform. Options include:
            - `"create_file"`: Creates a temporary file and returns its path.
            - `"create_dir"`: Creates a temporary directory and returns its path.
            - `"clean"`: Cleans temporary files in the specified directory or 
              the system temp directory if none is provided.
    
    directory : str, optional
        Directory to clean when `action` is `"clean"`. If `None`, uses the 
        system’s default temporary directory. Ignored for file or directory 
        creation actions.
    
    clean_all : bool, default=False
        If `True`, removes all files and directories within the specified 
        directory. If `False`, only deletes files or directories created by 
        this process. Used only when `action` is `"clean"`.
    
    Returns
    -------
    temp_path : str or None
        - For `"create_file"` and `"create_dir"` actions, returns the path 
          of the created file or directory.
        - For `"clean"`, returns None.
    
    Raises
    ------
    ValueError
        If an invalid action is specified.

    Notes
    -----
    This function is useful for managing temporary resources in data 
    processing tasks, where files or directories need to be created and 
    cleaned up after use.

    Examples
    --------
    >>> from gofast.tools.sysutils import manage_temp
    >>> temp_file = manage_temp(action="create_file")
    >>> print(temp_file)
    '/tmp/tmpabcd1234'
    
    >>> temp_dir = manage_temp(action="create_dir", prefix="data_")
    >>> print(temp_dir)
    '/tmp/data_abcd1234'
    
    >>> manage_temp(action="clean", directory="/path/to/temp", clean_all=True)

    See Also
    --------
    tempfile : Module for creating temporary files and directories.
    shutil : High-level file operations.
    """

    # Create a temporary file and return its path
    if action == "create_file":
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, delete=False)
        temp_file.close()
        return temp_file.name

    # Create a temporary directory and return its path
    elif action == "create_dir":
        return tempfile.mkdtemp(suffix=suffix, prefix=prefix)

    # Clean up temporary files and directories
    elif action == "clean":
        dir_to_clean = directory or tempfile.gettempdir()
        for item in os.listdir(dir_to_clean):
            item_path = os.path.join(dir_to_clean, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path) and clean_all:
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")
        return None
    
    # Handle invalid action
    else:
        raise ValueError(
            f"Invalid action '{action}'. Expected 'create_file',"
            " 'create_dir', or 'clean'.")

def check_port_in_use(port: int) -> bool:
    """
    Checks if a port is currently in use, which is useful for server-based 
    applications.
    
    Parameters
    ----------
    port : int
        The port number to check.
    
    Returns
    -------
    bool
        True if the port is in use, otherwise False.
    
    Examples
    --------
    >>> from gofast.tools.sysutils import check_port_in_use
    >>> check_port_in_use(8080)
    False
    
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_uptime() -> str:
    """
    Returns the system uptime in a human-readable format.
    
    Returns
    -------
    uptime : str
        The system uptime formatted as `"Xd:Yh:Zm:Ws"`, where X, Y, Z, and W 
        are days, hours, minutes, and seconds respectively.
    
    Notes
    -----
    This function is useful for monitoring or diagnosing long-running 
    processes on the system.

    Examples
    --------
    >>> from gofast.tools.sysutils import get_uptime
    >>> get_uptime()
    '2d:5h:34m:12s'
    
    """
    uptime_seconds = int(psutil.boot_time())
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d:{hours}h:{minutes}m:{seconds}s"

# ----

def parallelize_jobs(
    function: Callable,
    tasks: Sequence[Dict[str, Any]] = (),
    n_jobs: Optional[int] = None,
    executor_type: str = 'process') -> list:
    """
    Parallelize the execution of a callable across multiple processors, 
    supporting both positional and keyword arguments.

    Parameters
    ----------
    function : Callable[..., Any]
        The function to execute in parallel. This function must be picklable 
        if using `executor_type='process'`.
    tasks : Sequence[Dict[str, Any]], optional
        A sequence of dictionaries, where each dictionary contains 
        two keys: 'args' (a tuple) for positional arguments,
        and 'kwargs' (a dict) for keyword arguments, for one execution of
        `function`. Defaults to an empty sequence.
    n_jobs : Optional[int], optional
        The number of jobs to run in parallel. `None` or `1` uses a single 
        processor, any positive integer specifies the
        exact number of processors to use, `-1` uses all available processors. 
        Default is None (1 processor).
    executor_type : str, optional
        The type of executor to use. Can be 'process' for CPU-bound tasks or
        'thread' for I/O-bound tasks. Default is 'process'.

    Returns
    -------
    list
        A list of results from the function executions.

    Raises
    ------
    ValueError
        If `function` is not picklable when using 'process' as `executor_type`.

    Examples
    --------
    >>> from gofast.tools.sysutils import parallelize_jobs
    >>> def greet(name, greeting='Hello'):
    ...     return f"{greeting}, {name}!"
    >>> tasks = [
    ...     {'args': ('John',), 'kwargs': {'greeting': 'Hi'}},
    ...     {'args': ('Jane',), 'kwargs': {}}
    ... ]
    >>> results = parallelize_jobs(greet, tasks, n_jobs=2)
    >>> print(results)
    ['Hi, John!', 'Hello, Jane!']
    """
    if executor_type == 'process':
        import_optional_dependency("cloudpickle")
        import cloudpickle
        try:
            cloudpickle.dumps(function)
        except cloudpickle.PicklingError:
            raise ValueError("The function to be parallelized must be "
                             "picklable when using 'process' executor.")

    num_workers = multiprocessing.cpu_count() if n_jobs == -1 else (
        1 if n_jobs is None else n_jobs)
    
    ExecutorClass = ProcessPoolExecutor if executor_type == 'process' \
        else ThreadPoolExecutor
    
    results = []
    with ExecutorClass(max_workers=num_workers) as executor:
        futures = [executor.submit(function, *task.get('args', ()),
                                   **task.get('kwargs', {})) for task in tasks]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results
 
def find_by_regex (o , pattern,  func = re.match, **kws ):
    """ Find pattern in object whatever an "iterable" or not. 
    
    when we talk about iterable, a string value is not included.
    
    Parameters 
    -----------
    o: str or iterable,  
        text litteral or an iterable object containing or not the specific 
        object to match. 
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
    
    func: re callable , default=re.match
        regular expression search function. Can be
        [re.match, re.findall, re.search ],or any other regular expression 
        function. 
        
        * ``re.match()``:  function  searches the regular expression pattern and 
            return the first occurrence. The Python RegEx Match method checks 
            for a match only at the beginning of the string. So, if a match is 
            found in the first line, it returns the match object. But if a match 
            is found in some other line, the Python RegEx Match function returns 
            null.
        * ``re.search()``: function will search the regular expression pattern 
            and return the first occurrence. Unlike Python re.match(), it will 
            check all lines of the input string. The Python re.search() function 
            returns a match object when the pattern is found and “null” if 
            the pattern is not found
        * ``re.findall()`` module is used to search for 'all' occurrences that 
            match a given pattern. In contrast, search() module will only 
            return the first occurrence that matches the specified pattern. 
            findall() will iterate over all the lines of the file and will 
            return all non-overlapping matches of pattern in a single step.
    kws: dict, 
        Additional keywords arguments passed to functions :func:`re.match` or 
        :func:`re.search` or :func:`re.findall`. 
        
    Returns 
    -------
    om: list 
        matched object put is the list 
        
    Example
    --------
    >>> from gofast.tools.sysutils import find_by_regex
    >>> from gofast.datasets import load_hlogs 
    >>> X0, _= load_hlogs (as_frame =True )
    >>> columns = X0.columns 
    >>> str_columns =','.join (columns) 
    >>> find_by_regex (str_columns , pattern='depth', func=re.search)
    ... ['depth']
    >>> find_by_regex(columns, pattern ='depth', func=re.search)
    ... ['depth_top', 'depth_bottom']
    
    """
    om = [] 
    if isinstance (o, str): 
        om = func ( pattern=pattern , string = o, **kws)
        if om: 
            om= om.group() 
        om =[om]
    elif is_iterable(o): 
        o = list(o) 
        for s in o : 
            z = func (pattern =pattern , string = s, **kws)
            if z : 
                om.append (s) 
                
    if func.__name__=='findall': 
        om = list(itertools.chain (*om )) 
    # keep None is nothing 
    # fit the corresponding pattern 
    if len(om) ==0 or om[0] is None: 
        om = None 
    return  om 


def find_similar_string(
        name: str,
        container: Union[List[str], Tuple[str, ...], Dict[Any, Any]],
        stripitems: Union[str, List[str], Tuple[str, ...]] = '_',
        deep: bool = False,
) -> Optional[str]:
    """
    Find the most similar string in a container to the provided name.

    This function searches for the most likely matching string in a container
    based on the provided `name`. It sanitizes the `name` by stripping specified
    characters and can perform a deep search to find partial matches.

    Parameters
    ----------
    name : str
        The string to search for in the container.

    container : list, tuple, or dict
        The container with strings to search in.

    stripitems : str or list of str, optional
        Characters or strings to strip from `name` before searching. If a string,
        multiple items can be separated by ':', ',', or ';'. Default is ``'_'``.

    deep : bool, optional
        If ``True``, performs a deeper search by checking if `name` is a substring
        of any item in the container. Default is ``False``.

    Returns
    -------
    result : str or None
        The most similar string from the container, or ``None`` if no match is found.

    Examples
    --------
    >>> from gofast.tools.sysutils import find_similar_string
    >>> container = {'dipole': 1, 'quadrupole': 2}
    >>> find_similar_string('dipole_', container)
    'dipole'
    >>> find_similar_string('dip', container, deep=True)
    'dipole'
    >>> find_similar_string('+dipole__', container, stripitems='+;__', deep=True)
    'dipole'

    Notes
    -----
    This function is useful when trying to find the closest matching string
    in a container, especially when exact matches are not guaranteed due to
    formatting inconsistencies or typos.

    See Also
    --------
    str.strip : Returns a copy of the string with leading and trailing characters removed.

    References
    ----------
    .. [1] Python documentation on string methods.
    """
    # Validate inputs
    if not isinstance(name, str):
        raise TypeError(
            "`name` must be a string, got {type(name).__name__}")
    if not isinstance(container, (list, tuple, dict)):
        raise TypeError(
            "`container` must be a list, tuple,"
            f" or dict, got {type(container).__name__}")
    if not isinstance(stripitems, (str, list, tuple)):
        raise TypeError(
            f"`stripitems` must be a string or list/tuple of strings,"
            f" got {type(stripitems).__name__}")
    
    # Process stripitems
    if isinstance(stripitems, str):
        for sep in (':', ',', ';'):
            if sep in stripitems:
                stripitems = stripitems.split(sep)
                break
        else:
            stripitems = [stripitems]
    else:
        stripitems = list(stripitems)
    
    # Sanitize name
    for s in stripitems:
        name = name.strip(s)
    
    # Prepare container keys
    if isinstance(container, dict):
        container_keys = [key.lower() for key in container.keys()]
        keys_list = list(container.keys())
    else:
        container_keys = [str(item).lower() for item in container]
        keys_list = list(container)
    
    name_lower = name.lower()
    try:
        index = container_keys.index(name_lower)
        result = keys_list[index]
        return result
    except ValueError:
        pass  # Not found, proceed

    if deep:
        for idx, item in enumerate(container_keys):
            if name_lower in item:
                result = keys_list[idx]
                return result
    return None

def represent_callable(
        obj: Callable, skip: Optional[Union[str, List[str]]] = None) -> str:
    """
    Represent callable objects by formatting their signatures.

    This function generates a string representation of a callable object's
    signature, including parameters and default values. It supports classes,
    functions, and instance methods.

    Parameters
    ----------
    obj : callable
        The callable object to format.

    skip : str or list of str, optional
        Parameter names to skip in the representation. Useful for omitting
        certain attributes.

    Returns
    -------
    representation : str
        A string representation of the callable object's signature.

    Raises
    ------
    TypeError
        If `obj` is not a callable object.

    Examples
    --------
    >>> from gofast.tools.sysutils import represent_callable
    >>> def example_function(a, b=2):
    ...     pass
    >>> represent_callable(example_function)
    'example_function(a, b=2)'
    >>> class ExampleClass:
    ...     def __init__(self, x, y=10):
    ...         self.x = x
    ...         self.y = y
    >>> represent_callable(ExampleClass)
    'ExampleClass(x, y=10)'
    >>> instance = ExampleClass(5)
    >>> represent_callable(instance)
    'ExampleClass(x=5, y=10)'

    Notes
    -----
    This function is useful for logging or displaying the parameters of
    callable objects in a readable format.

    See Also
    --------
    inspect.signature : Get a signature object for the callable.

    References
    ----------
    .. [1] Python documentation on the inspect module.
    """
    if not callable(obj) and not hasattr(obj, '__dict__'):
        raise TypeError(f"Object '{obj}' is not callable or does not have attributes.")
    
    if isinstance(skip, str):
        skip = [skip]
    elif skip is None:
        skip = []
    else:
        skip = list(skip)

    obj_name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    try:
        sig = inspect.signature(obj)
        params = [
            f"{name}={repr(param.default)}" 
            if param.default is not inspect.Parameter.empty else name
            for name, param in sig.parameters.items()
            if name not in skip
        ]
        representation = f"{obj_name}({', '.join(params)})"
    except (TypeError, ValueError):
        # If obj is an instance, get its __dict__ attributes
        attrs = {
            k: v for k, v in vars(obj).items()
            if not k.startswith('_') and k not in skip
        }
        # Limit the number of attributes displayed
        if len(attrs) > 6:
            displayed_attrs = list(attrs.items())[:3] + [
                ('...', '...')] + list(attrs.items())[-3:]
        else:
            displayed_attrs = attrs.items()
        params = [f"{k}={repr(v)}" for k, v in displayed_attrs]
        representation = f"{obj_name}({', '.join(params)})"
    return representation


def safe_getattr(obj: Any, name: str, default_value: Optional[Any] = None) -> Any:
    """
    Safely get an attribute from an object, with a helpful error message.

    This function attempts to retrieve an attribute from the given object.
    If the attribute is not found, it can return a default value or raise
    an AttributeError with a suggestion for a similar attribute.

    Parameters
    ----------
    obj : object
        The object from which to retrieve the attribute.

    name : str
        The name of the attribute to retrieve.

    default_value : any, optional
        A default value to return if the attribute is not found. If ``None``,
        an ``AttributeError`` will be raised.

    Returns
    -------
    value : any
        The value of the retrieved attribute or the default value.

    Raises
    ------
    AttributeError
        If the attribute is not found and no default value is provided.

    Examples
    --------
    >>> from gofast.tools.sysutils import safe_getattr
    >>> class MyClass:
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> obj = MyClass(1, 2)
    >>> safe_getattr(obj, 'a')
    1
    >>> safe_getattr(obj, 'c', default_value='default')
    'default'
    >>> safe_getattr(obj, 'c')
    Traceback (most recent call last):
        ...
    AttributeError: 'MyClass' object has no attribute 'c'. Did you mean 'a'?

    Notes
    -----
    This function enhances the built-in `getattr` by providing helpful
    suggestions when an attribute is not found.

    See Also
    --------
    getattr : Built-in function to get an attribute from an object.

    References
    ----------
    .. [1] Python documentation on built-in functions.
    """
    if hasattr(obj, name):
        return getattr(obj, name)
    
    if default_value is not None:
        return default_value
    
    # Attempt to find a similar attribute name
    similar_attr = find_similar_string(name, vars(obj), deep=True)
    suggestion = f". Did you mean '{similar_attr}'?" if similar_attr else ""
    
    raise AttributeError(
        f"'{obj.__class__.__name__}' object has no attribute '{name}'{suggestion}")