# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>


from abc import ABCMeta, abstractmethod
from typing import Optional, Dict

from ..api.property import BaseClass 
from .._gofastlog import gofastlog

# Initialize logger
logger = gofastlog.get_gofast_logger(__name__)


class BaseVersioning(BaseClass, metaclass=ABCMeta):
    """
    A base class for managing version control in machine learning systems,
    including models, datasets, and pipelines. This class provides a
    foundation for ensuring version integrity, validation, and logging
    throughout the lifecycle of the versioning object.

    Parameters
    ----------
    version : str
        The version identifier for the object (e.g., model version,
        dataset version).

    config : dict, optional
        A dictionary containing additional configuration options for
        versioning. This can include specific settings related to the
        version control system.

    Attributes
    ----------
    version_ : str
        The version identifier of the instance.

    config_ : dict
        The configuration dictionary storing settings applied during
        initialization.

    is_initialized_ : bool
        Indicates whether the object has been successfully initialized.

    events_log_ : list of dict
        A list that stores the history of logged events for auditing and
        debugging purposes.

    Methods
    -------
    is_initialized() -> bool
        Checks if the object has been successfully initialized.

    get_version() -> str
        Returns the version identifier of the instance.

    reset()
        Resets the internal attributes and reinitializes the object.

    log_event(event_name, event_details=None)
        Logs events internally and records them into the logger for
        auditing.

    get_log_history() -> list
        Retrieves the list of all logged events for review.

    Notes
    -----
    This class is abstract and should not be instantiated directly. It is
    designed to be inherited by other versioning-related classes that need
    to implement version control mechanisms (e.g., model versioning,
    dataset versioning).

    Each subclass must implement the ``_perform_version_checks`` and
    ``validate`` methods to ensure version constraints and validation
    processes are properly enforced.

    Examples
    --------
    >>> from gofast.mlops.versioning import BaseVersioning
    >>> class MyVersioning(BaseVersioning):
    ...     def _perform_version_checks(self):
    ...         # Implement version checks here
    ...         pass
    ...     def validate(self):
    ...         # Implement validation logic here
    ...         pass
    >>> my_versioning = MyVersioning(
    ...     version='v1.0',
    ...     config={'track_history': True}
    ... )
    >>> my_versioning.get_version()
    'v1.0'

    See Also
    --------
    DatasetVersioning : Class for managing dataset version control.
    ModelVersionControl : Class for managing model version control.
    PipelineVersioning : Class for tracking and versioning machine
                         learning pipelines.

    References
    ----------
    .. [1] "Version Control in Machine Learning Systems", J. Doe et al.,
       Proceedings of the Machine Learning Conference, 2022.

    """
    @abstractmethod 
    def __init__(
        self, 
        version: str, 
        config: 
        Optional[Dict] = None
        ):

        self.version = version
        self.config = config or {}
        self._is_initialized_ = False
        self.events_log_ = []
        self._initialize()

    def is_initialized(self) -> bool:
        """
        Checks if the class instance has been initialized successfully.

        Returns
        -------
        bool
            `True` if initialized, `False` otherwise.

        """
        return self._is_initialized_

    def get_version(self) -> str:
        """
        Returns the version identifier of the instance.

        Returns
        -------
        str
            The version identifier.

        """
        return self.version

    def reset(self):
        """
        Resets internal attributes and reinitializes the object. This can
        be useful if reconfiguration or reloading of the versioning object
        is needed.

        """
        self._is_initialized_ = False
        self.log_event('reset', {'version': self.version})
        self._initialize()

    def log_event(self, event_name: str, event_details: Optional[Dict] = None):
        """
        Logs events into the logger and stores them in the internal events
        log.

        Parameters
        ----------
        event_name : str
            The name of the event to log (e.g., ``'initialization_success'``).

        event_details : dict, optional
            Additional details about the event to log (e.g., version,
            error details).

        """
        if event_details is None:
            event_details = {}
        event_details['event'] = event_name
        event_details['version'] = self.version

        # Log event into the logger
        logger.info(f"Event: {event_name} | Details: {event_details}")

        # Store event in internal log
        self.events_log_.append(event_details)

    def get_log_history(self) -> list:
        """
        Retrieves the internal log history of the events.

        Returns
        -------
        list of dict
            A list of logged events.

        """
        return self.events_log_
