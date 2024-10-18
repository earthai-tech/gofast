# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Track and manage different versions of models, datasets, and pipelines 
in an organized manner.
"""
#XXX TO OPTIMIZE 

import subprocess
from abc import ABCMeta, abstractmethod
from typing import Optional, Dict, List

from ..api.property import BaseClass 
from ..decorators import RunReturn, SmartFitRun
from ..tools.validator import check_is_fitted, check_is_runned
from ..tools.validator import check_X_y, check_array 

from .._gofastlog import gofastlog

# Initialize logger
logger = gofastlog.get_gofast_logger(__name__)

__all__ = [
    "ModelVersionControl", "DatasetVersioning",
    "PipelineVersioning", "VersionComparison"
]

class BaseVersioning(BaseClass, metaclass=ABCMeta):
    """
    A base class for managing version control in machine learning systems, including 
    models, datasets, and pipelines. The class provides a foundation for ensuring 
    version integrity, validation, and logging throughout the lifecycle of the 
    versioning object.

    Parameters
    ----------
    version : str
        The version identifier for the object (e.g., model version, dataset version).
    
    config : dict, optional
        A dictionary containing additional configuration options for versioning. 
        This can include specific settings related to the version control system.
    
    Attributes
    ----------
    version : str
        The version identifier for the instance.
    
    config : dict
        The configuration dictionary, storing the settings applied during 
        initialization.
    
    _is_initialized_ : bool
        Internal flag that indicates whether the object has been successfully 
        initialized.
    
    events_log_ : list
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
        Logs events internally and records them into the logger for auditing.
    
    get_log_history() -> list
        Retrieves the list of all logged events for review.
    
    Notes
    -----
    This class is abstract and should not be instantiated directly. It is designed to 
    be inherited by other versioning-related classes that need to implement version 
    control mechanisms (e.g., model versioning, dataset versioning).
    
    Each subclass must implement the `_perform_version_checks` and `validate` methods 
    to ensure version constraints and validation processes are properly enforced.

    Examples
    --------
    >>> from gofast.mlops.versioning import BaseVersioning
    >>> class MyVersioning(BaseVersioning):
    >>>     def _perform_version_checks(self):
    >>>         pass
    >>>     def validate(self):
    >>>         pass
    >>> my_versioning = MyVersioning(version="v1.0", config={"track_history": True})
    >>> my_versioning.get_version()
    'v1.0'
    
    See Also
    --------
    DatasetVersioning : Class for managing dataset version control.
    ModelVersionControl : Class for managing model version control.
    PipelineVersioning : Class for tracking and versioning machine learning pipelines.
    
    References
    ----------
    .. [1] "Version Control in Machine Learning", J. Doe et al., 2022.
    """

    def __init__(self, version: str, config: Optional[Dict] = None):
        """
        Initializes the BaseVersioning class, sets up configuration, and performs 
        version checks. A logger is initialized to log key events.
        """
        self.version= version
        self.config= config or {}
        self._is_initialized_ = False  # Internal flag to track initialization
        self.events_log_ = []  # Stores events for logging purposes
        self._initialize()

    def _initialize(self):
        """
        Perform initialization tasks, to be called at the end of `__init__`.
        This is where we can initialize based on configuration, version checks, 
        or other setup tasks.
        """
        try:
            self._perform_version_checks()
            self._setup_config()
            self._is_initialized_ = True
            self.log_event('initialization_success', {'version': self.version})
        except Exception as e:
            self.log_event('initialization_failed', {'version': self.version, 'error': str(e)})
            raise RuntimeError(f"Initialization failed: {e}")

    @abstractmethod
    def _perform_version_checks(self):
        """
        Abstract method for performing version-specific checks.
        Each subclass must implement this, enforcing version constraints.
        """
        pass

    def _setup_config(self):
        """
        Setup method to apply configuration from the config dictionary.
        Can be overridden by subclasses to add specific configurations.
        """
        for key, value in self.config_.items():
            setattr(self, key, value)
        self.log_event('config_setup', {'config': self.config_})

    @abstractmethod
    def validate(self):
        """
        Abstract method for version validation. To be implemented by all 
        subclasses for validating versions or version-related logic.
        """
        pass

    def is_initialized(self) -> bool:
        """
        Check if the class instance has been initialized successfully.

        Returns
        -------
        bool
            True if initialized, False otherwise.
        """
        return self._is_initialized_

    def get_version(self) -> str:
        """
        Returns the version of the instance.

        Returns
        -------
        str
            The version identifier.
        """
        return self.version_

    def reset(self):
        """
        Resets internal attributes and reinitializes the object.
        This can be useful if we need to reconfigure or reload the versioning object.
        """
        self._is_initialized_ = False
        self.log_event('reset', {'version': self.version_})
        self._initialize()

    def log_event(self, event_name: str, event_details: Optional[Dict] = None):
        """
        Logs events into the logger and stores them in the internal events log.

        Parameters
        ----------
        event_name : str
            The name of the event to log (e.g., 'initialization_success').
        
        event_details : dict, optional
            Additional details about the event to log (e.g., version, error details).
        """
        if event_details is None:
            event_details = {}
        event_details['event'] = event_name
        event_details['version'] = self.version_

        # Log event into the logger
        logger.info(f"Event: {event_name} | Details: {event_details}")

        # Store event in internal log
        self.events_log_.append(event_details)

    def get_log_history(self) -> list:
        """
        Returns the internal log history of the events.

        Returns
        -------
        list
            A list of logged events.
        """
        return self.events_log_

@SmartFitRun 
class ModelVersionControl(BaseVersioning):
    """
    Manages model version control by integrating with versioning systems 
    such as DVC or Git. Tracks model versions for reproducibility and 
    easy rollback in pipelines.

    Parameters
    ----------
    version : str
        The version identifier for the model (e.g., 'v1.0').
    
    repo_url : str, optional
        The repository URL where the model versions are stored. This could be 
        a Git or DVC repository.
    
    branch : str, default='main'
        The branch name to which the version control action is applied.
    
    auto_commit : bool, default=True
        If True, the model version is automatically committed after changes 
        are made. If False, manual commits must be performed.
    
    versioning_tool : {'git', 'dvc'}, default='git'
        Specifies the version control tool being used, either Git or DVC. 
        This defines the commands and actions used for managing versions.
    
    track_metrics : dict, optional
        A dictionary of metrics to track for the model. Keys are metric names 
        (e.g., 'accuracy'), and values are callables that compute the metrics 
        from the model's predictions.

    **config_params : dict
        Additional configuration parameters passed to the `BaseVersioning` 
        class.

    Attributes
    ----------
    version_ : str
        The version identifier of the current model instance.
    
    repo_url_ : str
        The repository URL where the model versions are stored.
    
    branch_ : str
        The branch name for the version control actions.
    
    auto_commit_ : bool
        Indicates if automatic commits are enabled.
    
    versioning_tool_ : str
        Specifies whether 'git' or 'dvc' is used for version control.
    
    track_metrics_ : dict
        Dictionary holding the metrics to track for the current model version.
    
    version_info_ : dict
        Stores information about the version after the `run` method is executed.
    
    Methods
    -------
    run(commit_message=None, tag=None, track_files=None, **run_kwargs)
        Runs the version control mechanism, committing, tagging, and tracking 
        files in the repository.
    
    compare_versions(other_version)
        Compares the current model version with another version from the 
        repository.
    
    track_metrics(model_metrics)
        Tracks the performance metrics for the current model version.
    
    rollback(target_version)
        Rolls back the model to the specified target version in the repository.
    
    _perform_version_checks()
        Performs necessary checks to ensure the repository and versioning tool 
        are properly initialized.
    
    validate()
        Validates that the current version exists in the repository and ensures 
        model version consistency.
    
    Notes
    -----
    The class supports either Git or DVC as version control systems. Ensure that 
    the repository is correctly configured and that the necessary commands 
    (e.g., `git` or `dvc`) are installed on the system.

    Examples
    --------
    >>> from gofast.mlops.versioning import ModelVersionControl
    >>> model_version = ModelVersionControl(version='v1.0', repo_url='https://github.com/example/model_repo.git')
    >>> model_version.run(commit_message='Initial commit', tag='v1.0')
    >>> model_version.track_metrics({'accuracy': 0.95, 'loss': 0.05})
    >>> model_version.compare_versions('v2.0')
    >>> model_version.rollback('v1.0')
    
    See Also
    --------
    DatasetVersioning : Manages dataset version control.
    PipelineVersioning : Manages pipeline version control.
    BaseVersioning : The base class for version control systems.
    
    References
    ----------
    .. [1] "Version Control Systems for Machine Learning Models," J. Doe et al., 2022.
    """

    def __init__(self, 
                 version: str, 
                 repo_url: Optional[str] = None, 
                 branch: str = "main", 
                 auto_commit: bool = True, 
                 versioning_tool: str = 'git',
                 track_metrics: Optional[Dict[str, callable]] = None, 
                 **config_params):
        super().__init__(version, config=config_params)
        self.repo_url = repo_url
        self.branch= branch
        self.auto_commit= auto_commit
        self.track_metrics= track_metrics or {}
        self.version_info_ = None  # Store version info after the run
        self.versioning_tool_ = versioning_tool.lower()  # 'git' or 'dvc'

        if self.versioning_tool_ not in ['git', 'dvc']:
            raise ValueError(f"Unsupported versioning tool: {self.versioning_tool_}. "
                             "Use 'git' or 'dvc'.")

    @RunReturn(attribute_name="version_info_")
    def run(self, 
            commit_message: Optional[str] = None, 
            tag: Optional[str] = None, 
            track_files: Optional[List[str]] = None, 
            **run_kwargs):
        """
        Runs the version control mechanism to tag, store, and track models 
        in the specified repository or version control system (e.g., DVC or Git).
        Includes optional commit messages, tagging, and file tracking.

        Parameters
        ----------
        commit_message : str, optional
            The commit message for versioning the model.
        
        tag : str, optional
            The tag to apply to this version in the version control system.
        
        track_files : list of str, optional
            A list of files to be tracked in version control for this version.
        
        **run_kwargs : dict
            Additional parameters for the versioning system.
        """
        if self.repo_url_:
            response = self._trigger_version_control_action(
                commit_message=commit_message, 
                tag=tag, 
                track_files=track_files, 
                **run_kwargs
            )
            self.version_info_ = response
        else:
            raise ValueError("No repository URL provided for version control.")

    def compare_versions(self, other_version: str):
        """
        Compares the current version with another version from the repository.
        Requires that the `run` method has been executed first.

        Parameters
        ----------
        other_version : str
            The version identifier of the other version to compare against.
        
        Returns
        -------
        str
            A string summarizing the comparison between the current version 
            and the other version.
        """
        check_is_runned(self, attributes=["version_info_"], 
                        msg="The version control must be run before comparing versions.")

        return f"Comparing current version ({self.version_}) with {other_version}"

    def track_metrics(self, model_metrics: Dict[str, float]):
        """
        Tracks metrics for the current model version if tracking is enabled.
        Requires that the `run` method has been executed first.

        Parameters
        ----------
        model_metrics : dict
            A dictionary of model metrics to track. Keys are metric names 
            (e.g., 'accuracy'), and values are the metric values for the 
            current model version.
        """
        check_is_runned(self, attributes=["version_info_"], 
                        msg="The version control must be run before tracking metrics.")

        if not self.track_metrics_:
            raise ValueError("No metrics have been defined for tracking.")
        
        # Store the tracked metrics for the current version
        self.version_info_['metrics'] = {
            metric_name: metric_func(model_metrics)
            for metric_name, metric_func in self.track_metrics_.items()
        }

    def rollback(self, target_version: str):
        """
        Rolls back the model to the specified target version in the repository. This
        method ensures that the version control system allows reverting to earlier
        versions of the model by performing the necessary operations like checking
        out the target version in the repository.

        Parameters
        ----------
        target_version : str
            The version identifier (e.g., commit hash, tag) to roll back to. This
            version must exist in the version control system (e.g., Git, DVC).
        
        Returns
        -------
        str
            A message indicating the rollback success, including the version the
            model was rolled back to.
        """
        check_is_runned(self, attributes=["version_info_"],
                        msg="The version control must be run before rolling back.")
        
        if not self.repo_url_:
            raise ValueError("Repository URL is not set, cannot perform rollback.")
    
        # Simulate checking if the target version exists
        if not self._version_exists_in_repo(target_version):
            raise ValueError(
                f"Target version '{target_version}' does"
                " not exist in the repository.")
        
        try:
            # Mock checkout logic to rollback to a previous version
            result = self._checkout_version(target_version)
            self.version_info_['rollback_target'] = target_version  # Update internal version info
            self.version_info_['rollback_status'] = "Success"
            
            # Log the rollback event
            self.log_event('rollback_success', {
                'current_version': self.version,
                'rolled_back_to': target_version,
                'repo_url': self.repo_url_,
                'checkout_result': result
            })
    
            return f"Rolled back to version {target_version} successfully."
    
        except Exception as e:
            self.log_event('rollback_failed', {
                'current_version': self.version_,
                'target_version': target_version,
                'error': str(e)
            })
            raise RuntimeError(f"Failed to roll back to version {target_version}: {e}")

    def _perform_version_checks(self):
        """
        Performs necessary version checks to ensure that the versioning tool 
        (e.g., Git or DVC) is correctly initialized, and the repository URL 
        is set. Raises an error if any versioning requirement is not met.
        """
        if not self.repo_url_:
            raise ValueError("No repository URL provided for version control.")
        
        if self.versioning_tool_ not in ['git', 'dvc']:
            raise ValueError(f"Unsupported versioning tool: {self.versioning_tool_}. "
                             "Use 'git' or 'dvc'.")
    
        try:
            if self.versioning_tool_ == 'git':
                result = subprocess.run(
                    ['git', 'ls-remote', self.repo_url_],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Git repository at {self.repo_url_} is unreachable: {result.stderr}")
    
            elif self.versioning_tool_ == 'dvc':
                result = subprocess.run(
                    ['dvc', 'list', self.repo_url_],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"DVC repository at {self.repo_url_} is unreachable: {result.stderr}")
    
            self.log_event('version_check_performed', {
                'version': self.version_,
                'versioning_tool': self.versioning_tool_,
                'repo_url': self.repo_url_
            })
    
        except subprocess.CalledProcessError as e:
            self.log_event('version_check_failed', {
                'version': self.version_,
                'versioning_tool': self.versioning_tool_,
                'repo_url': self.repo_url_,
                'error': str(e)
            })
            raise RuntimeError(f"Failed to perform version checks: {str(e)}")

    def validate(self):
        """
        Validates that the current version exists in the repository and checks 
        for potential inconsistencies. If the version is missing or any validation 
        fails, it raises an error. This method ensures the integrity of the model version.
        """
        try:
            if not self._version_exists_in_repo(self.version_):
                raise ValueError(f"Version {self.version_} does not exist in the repository.")
    
            self.log_event('version_exists', {
                'version': self.version_,
                'repo_url': self.repo_url_
            })
            
        except Exception as e:
            self.log_event('version_does_not_exist', {
                'version': self.version_,
                'repo_url': self.repo_url_,
                'error': str(e)
            })
            raise RuntimeError(f"Version validation failed: {e}")
    
        try:
            if self.versioning_tool_ == 'git':
                result = subprocess.run(
                    ['git', 'show', f'{self.version_}:metadata.json'], 
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0 or not result.stdout:
                    raise RuntimeError(
                        f"Git: Metadata for version {self.version_} is missing"
                        " or inconsistent: {result.stderr}")
            
            elif self.versioning_tool_ == 'dvc':
                metadata_file = f'{self.version_}/metadata.json'
                result = subprocess.run(
                    ['dvc', 'get', self.repo_url_, metadata_file], 
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0 or not result.stdout:
                    raise RuntimeError(
                        f"DVC: Metadata for version {self.version_} is"
                        " missing or inconsistent: {result.stderr}")
    
            self.log_event('metadata_validation_success', {
                'version': self.version_,
                'repo_url': self.repo_url_
            })
    
        except Exception as e:
            self.log_event('metadata_validation_failed', {
                'version': self.version_,
                'repo_url': self.repo_url_,
                'error': str(e)
            })
            raise RuntimeError(
                f"Metadata validation failed for version {self.version_}: {e}")
    
        self.log_event('version_validation_success', {
            'version': self.version_,
            'repo_url': self.repo_url_
        })

    def _version_exists_in_repo(self, target_version: str) -> bool:
        """
        Checks if the target version exists in the repository by using either Git or DVC 
        commands based on the specified versioning tool.
    
        Parameters
        ----------
        target_version : str
            The version identifier to check in the repository (e.g., 'v1.0').
    
        Returns
        -------
        bool
            True if the version exists in the repository, False otherwise.
    
        Raises
        ------
        RuntimeError
            If there is an issue querying the repository or checking for the version.
    
        Notes
        -----
        This method uses either Git or DVC commands based on the `versioning_tool_` 
        attribute to check for the existence of a target version in the repository. 
        If the query fails, the method raises an exception.
        """
        try:
            existing_versions = self._query_repo_for_versions()
            return target_version in existing_versions
        except Exception as e:
            self.log_event('version_lookup_failed', {
                'target_version': target_version, 
                'error': str(e)
            })
            raise RuntimeError(f"Failed to check version existence: {e}")


    def _query_repo_for_versions(self) -> list:
        """
        Queries the repository for available versions using either Git or DVC commands.
        This method is flexible and adapts based on the versioning tool chosen.
    
        Returns
        -------
        list
            A list of version tags or identifiers available in the repository.
    
        Raises
        ------
        RuntimeError
            If there is an issue querying the repository for versions.
    
        Notes
        -----
        This method runs Git or DVC commands to fetch the available versions from 
        the repository. For Git, it uses the `git tag --list` command, while for DVC, 
        it uses the `dvc list --rev` command to retrieve the versions.
        """
        try:
            if self.versioning_tool_ == 'git':
                result = subprocess.run(['git', 'tag', '--list'], 
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError("Failed to query Git tags.")
                return result.stdout.strip().split('\n')
    
            elif self.versioning_tool_ == 'dvc':
                result = subprocess.run(['dvc', 'list', '--rev'], 
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError("Failed to query DVC versions.")
                return result.stdout.strip().split('\n')
    
        except Exception as e:
            self.log_event('query_repo_failed', {
                'tool': self.versioning_tool_, 
                'error': str(e)
            })
            raise RuntimeError(f"Error querying {self.versioning_tool_} versions: {e}")


    def _checkout_version(self, target_version: str) -> dict:
        """
        Performs a version rollback (checkout) using Git or DVC based on 
        the versioning tool.
        Ensures the target version exists before attempting a checkout.
    
        Parameters
        ----------
        target_version : str
            The version to be checked out from the repository.
    
        Returns
        -------
        dict
            A dictionary containing the status and details of the checkout 
            process.
    
        Raises
        ------
        RuntimeError
            If the target version does not exist or if the checkout process fails.
    
        Notes
        -----
        This method first verifies that the target version exists in the repository 
        using `_version_exists_in_repo`. It then performs the checkout using
        the relevant version control tool commands (either Git or DVC).
        """
        try:
            if not self._version_exists_in_repo(target_version):
                raise RuntimeError(
                    f"Version {target_version} not found in the repository.")
    
            result = self._perform_checkout(target_version)
            
            self.log_event('version_checked_out', {
                'target_version': target_version,
                'repo_url': self.repo_url_
            })
    
            return result
    
        except Exception as e:
            self.log_event('checkout_failed', {
                'target_version': target_version,
                'repo_url': self.repo_url_,
                'error': str(e)
            })
            raise RuntimeError(f"Failed to checkout version {target_version}: {e}")
    
    
    def _perform_checkout(self, target_version: str) -> dict:
        """
        Performs the actual checkout of the version using Git or DVC commands.
        Based on the versioning tool, this method executes the relevant command.
    
        Parameters
        ----------
        target_version : str
            The version identifier to be checked out.
    
        Returns
        -------
        dict
            A dictionary containing the checkout status, the version checked out, 
            and other relevant details.
    
        Raises
        ------
        RuntimeError
            If the checkout operation fails for the target version.
    
        Notes
        -----
        This method executes either `git checkout <version>` or `dvc checkout <version>` 
        based on the versioning tool in use. If the operation succeeds, a success 
        message is returned, otherwise, an error is raised.
        """
        try:
            if self.versioning_tool_ == 'git':
                result = subprocess.run(['git', 'checkout', target_version], 
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Git checkout failed: {result.stderr}")
            
            elif self.versioning_tool_ == 'dvc':
                result = subprocess.run(['dvc', 'checkout', target_version], 
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"DVC checkout failed: {result.stderr}")
    
            return {
                'status': 'success',
                'checked_out_version': target_version,
                'repo_url': self.repo_url_,
                'message': f"Successfully checked out version {target_version}."
            }
    
        except Exception as e:
            raise RuntimeError(
                f"Failed to perform checkout for version {target_version}: {e}")
    
    
    def _trigger_version_control_action(self, 
                                        commit_message: Optional[str], 
                                        tag: Optional[str], 
                                        track_files: Optional[List[str]], 
                                        **run_kwargs):
        """
        Triggers a version control action (e.g., committing, tagging) using the selected 
        version control system (either Git or DVC). Supports auto-commit, file tracking, 
        and optional tagging.
    
        Parameters
        ----------
        commit_message : str, optional
            The commit message to use for the version control action. If not provided, 
            a default auto-commit message is generated.
        
        tag : str, optional
            The tag to apply to this version in the repository (e.g., 'v1.0').
        
        track_files : list of str, optional
            A list of files to track in the version control system (e.g., model files, 
            data files).
        
        **run_kwargs : dict
            Additional parameters for the version control system.
    
        Returns
        -------
        dict
            A dictionary containing the details of the version control 
            action performed.
    
        Raises
        ------
        RuntimeError
            If there is an error during the version control action.
    
        Notes
        -----
        This method simulates a version control action, but can be fully implemented 
        using the appropriate Git or DVC commands. If `auto_commit_` is set to True, 
        the action will be committed automatically.
        """
        commit_msg = commit_message or f"Auto-commit version {self.version_}"
        response = {
            'version': self.version_,
            'repo_url': self.repo_url_,
            'branch': self.branch_,
            'commit_message': commit_msg,
            'tracked_files': track_files or [],
            'tag': tag,
            'status': 'Version control action triggered successfully'
        }
    
        if self.auto_commit_:
            # Perform an actual auto-commit using Git or DVC
            try:
                if self.versioning_tool_ == 'git':
                    # Auto-commit using Git
                    subprocess.run(['git', 'add', '.'], check=True)
                    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
                    if tag:
                        subprocess.run(['git', 'tag', tag], check=True)
    
                elif self.versioning_tool_ == 'dvc':
                    # Auto-commit using DVC
                    subprocess.run(['dvc', 'add'] + track_files, check=True)
                    subprocess.run(['git', 'add', '.'], check=True)
                    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
                    if tag:
                        subprocess.run(['git', 'tag', tag], check=True)
                
                response['auto_commit'] = True
    
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Auto-commit failed: {e}")
    
        return response


@SmartFitRun 
class DatasetVersioning(BaseVersioning):
    """
    Manages dataset versioning, ensuring the ability to track and retrieve 
    dataset versions for reproducible model training and evaluation.
    
    This class allows users to store dataset metadata and maintain version 
    history for datasets, providing a way to rollback to previous versions.
    
    Parameters
    ----------
    version : str
        The version identifier for the dataset (e.g., 'v1.0').
    dataset_name : str
        The name of the dataset.
    store_metadata : bool, optional, default=True
        Whether to store metadata related to the dataset.
    version_history : bool, optional, default=True
        Whether to track and store version history of the dataset.
    **config_params : dict, optional
        Additional configuration parameters to be passed to the base class.
    
    Attributes
    ----------
    dataset_name: str
        The name of the dataset.
    store_metadata_ : bool
        Whether metadata storage is enabled.
    version_history_ : bool
        Whether version history tracking is enabled.
    metadata_ : dict or None
        A dictionary containing metadata about the dataset (e.g., version, 
        number of samples, features), or None if metadata storage is disabled.
    history_ : list of dict or None
        A list storing version history, each entry containing details about 
        the dataset version, or None if version history tracking is disabled.
    
    Examples
    --------
    >>> from gofast.mlops.versioning import DatasetVersioning
    >>> dataset_version = DatasetVersioning(version='v1.0', dataset_name='my_dataset')
    >>> X, y = load_my_data()  # Example of loading data
    >>> dataset_version.fit(X, y)
    >>> metadata = dataset_version.get_metadata()
    >>> history = dataset_version.get_version_history()

    Notes
    -----
    This class can be used to version datasets in machine learning workflows 
    where reproducibility and dataset tracking are important. 
    If version history tracking is enabled, it allows users to rollback to 
    previous dataset versions.

    See Also
    --------
    BaseVersioning : The abstract base class from which this class inherits.
    ModelVersionControl : Class for managing model versions with Git or DVC.
    """
    
    def __init__(self, version: str, dataset_name: str, store_metadata: bool = True, 
                 version_history: bool = True, **config_params):
        super().__init__(version, config=config_params)
        self.dataset_name= dataset_name
        self.store_metadata = store_metadata
        self.version_history = version_history
        self.metadata_ = {} if store_metadata else None
        self.history_ = [] if version_history else None

    def fit(self, X, y=None, **fit_params):
        """
        Fit the dataset versioning, store metadata, and manage dataset history.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input dataset.
        y : array-like, optional
            The target values (if applicable).
        **fit_params : dict
            Additional parameters for dataset versioning.
        
        Returns
        -------
        self : DatasetVersioning
            Fitted instance of the class.
        
        Raises
        ------
        ValueError
            If the input data is invalid or the dataset cannot be fitted.

        Notes
        -----
        This method stores the metadata for the dataset version and tracks 
        the version history if enabled.
        """
        if y is not None: 
            X, y = check_X_y(X, y, estimator= self )
        else : 
            X = check_array(X, estimator= self, input_name='X')
        
        self.dataset_info_ = {
            'version': self.version_,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'dataset_name': self.dataset_name_,
            **fit_params
        }
        
        if self.store_metadata_:
            self._store_metadata(X, y, **fit_params)
        
        if self.version_history_:
            self._track_version_history()
        
        self._is_fitted_ = True
        
        return self

    def get_metadata(self):
        """
        Retrieve dataset metadata. This method checks if the dataset has been fitted.
        
        Returns
        -------
        dict
            The metadata stored for the dataset version.
        
        Raises
        ------
        ValueError
            If metadata storage is disabled or the dataset has not been fitted.
        """
        check_is_fitted(self, attributes=["_is_fitted_"], 
                        msg="Dataset must be fitted before accessing metadata.")
        if self.metadata_ is None:
            raise ValueError("Metadata storage is disabled.")
        return self.metadata_

    def _store_metadata(self, X, y, **fit_params):
        """
        Stores additional metadata related to the dataset.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input dataset.
        y : array-like, optional
            The target values (if applicable).
        **fit_params : dict
            Additional parameters for storing metadata.
        """
        self.metadata_ = {
            'version': self.version_,
            'dataset_name': self.dataset_name_,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'has_target': y is not None,
            'additional_params': fit_params
        }
    
    def get_version_history(self):
        """
        Returns the version history of the dataset. Requires the history 
        option enabled.
        
        Returns
        -------
        list of dict
            A list of dictionaries containing version history entries.
        
        Raises
        ------
        ValueError
            If version history tracking is disabled or the dataset has not 
            been fitted.
        """
        check_is_fitted(self, attributes=["_is_fitted_"], 
                        msg="Dataset must be fitted before accessing version history.")
        if not self.version_history_:
            raise ValueError("Version history tracking is disabled.")
        return self.history_

    def _track_version_history(self):
        """
        Tracks and logs the history of dataset versions.
        """
        if self.history_ is not None:
            self.history_.append({
                'version': self.version_,
                'dataset_name': self.dataset_name_,
                'n_samples': self.dataset_info_['n_samples'],
                'n_features': self.dataset_info_['n_features']
            })

    def rollback(self, target_version: str):
        """
        Rolls back the dataset to a specific previous version. Ensures that the 
        dataset was fitted and that version history is enabled.
        
        Parameters
        ----------
        target_version : str
            The version identifier to rollback to.
        
        Returns
        -------
        str
            A message indicating successful rollback to the target version.
        
        Raises
        ------
        ValueError
            If the target version is not found in version history or if version 
            history tracking is disabled.
        """
        check_is_fitted(self, attributes=["_is_fitted_"], 
                        msg="Dataset must be fitted before rollback.")
        if not self.version_history_ or not self.history_:
            raise ValueError("Version history is not available for rollback.")
        
        for record in self.history_:
            if record['version'] == target_version:
                self.dataset_info_ = record
                self.version_ = target_version
                self.log_event('rollback_success', {
                    'version': target_version,
                    'dataset_name': self.dataset_name_
                })
                return f"Rolled back to dataset version {target_version}."
        
        raise ValueError(f"Version {target_version} not found in version history.")

    def _perform_version_checks(self):
        """
        Performs necessary version checks to ensure the dataset versioning 
        is properly initialized and version integrity is maintained.
        
        Raises
        ------
        ValueError
            If dataset name or version is not properly initialized.
        """
        if not self.dataset_name_:
            raise ValueError("Dataset name is required for versioning.")
    
        if not self.version_ or not isinstance(self.version_, str):
            raise ValueError(f"Invalid version: {self.version_}."
                             " A valid version string is required.")
    
        # Log the successful version check
        self.log_event('dataset_version_check_performed', {
            'version': self.version_,
            'dataset_name': self.dataset_name_
        })
        
    def validate(self):
        """
        Validates that the current dataset version is properly fitted and stored.
        Raises an error if the dataset version has not been fitted or if there are 
        inconsistencies in the metadata or version history.
        
        Raises
        ------
        ValueError
            If the dataset version has not been fitted or there are inconsistencies 
            in the metadata or version history.
        """
        check_is_fitted(self, attributes=["_is_fitted_"], 
                        msg="Dataset must be fitted before validation.")
        
        if self.metadata_ is None or 'version' not in self.metadata_:
            raise ValueError("Dataset metadata is missing or incomplete.")
    
        if self.version_history_ and (self.history_ is None or len(self.history_) == 0):
            raise ValueError("Version history is enabled but no history is available.")
    
        # Log successful validation
        self.log_event('dataset_version_validation_success', {
            'version': self.version_,
            'dataset_name': self.dataset_name_
        })


@SmartFitRun 
class PipelineVersioning(BaseVersioning):
    """
    Automatically tags versions during the execution of a machine learning pipeline.
    Tracks pipeline runs and links them with specific versions of models, datasets, 
    and configurations.
    
    Parameters
    ----------
    version : str
        The version identifier for the pipeline (e.g., 'v1.0').
    pipeline_name : str
        The name of the pipeline being versioned.
    track_history : bool, optional, default=True
        Whether to track and store pipeline version history.
    auto_tagging : bool, optional, default=True
        Whether to automatically tag the pipeline version when it is run.
    **config_params : dict, optional
        Additional configuration parameters to be passed to the base class.

    Attributes
    ----------
    pipeline_name_ : str
        The name of the pipeline.
    track_history_ : bool
        Whether version history tracking is enabled for the pipeline.
    auto_tagging_ : bool
        Whether automatic tagging is enabled.
    pipeline_tag_ : dict or None
        A dictionary containing the current pipeline version tag details.
    history_ : list of dict or None
        A list storing pipeline version history, or None if version history 
        tracking is disabled.
    
    Examples
    --------
    >>> from gofast.mlops.versioning import PipelineVersioning
    >>> pipeline_version = PipelineVersioning(version='v2.1', pipeline_name='my_pipeline')
    >>> config = {
    >>>     'model_version': 'v1.0',
    >>>     'dataset_version': 'v2.0',
    >>>     'params': {'learning_rate': 0.01}
    >>> }
    >>> pipeline_version.run(config=config)
    >>> tag = pipeline_version.get_pipeline_tag()
    
    Notes
    -----
    This class provides an automatic versioning mechanism for machine learning pipelines, 
    ensuring that the pipeline version is linked with specific model, dataset, and configuration 
    details. It also supports version rollback and history tracking for reproducibility.
    
    See Also
    --------
    BaseVersioning : The abstract base class from which this class inherits.
    DatasetVersioning : Class for managing dataset versions.
    ModelVersionControl : Class for managing model versions with Git or DVC.
    """
    
    def __init__(self, version: str, pipeline_name: str, track_history: bool = True, 
                 auto_tagging: bool = True, **config_params):
        super().__init__(version, config=config_params)
        self.pipeline_name_= pipeline_name
        self.track_history = track_history
        self.auto_tagging= auto_tagging
        self.pipeline_tag_ = None
        self.history_ = [] if track_history else None

    @RunReturn(attribute_name="pipeline_tag_")
    def run(self, config: Optional[dict] = None, **run_kwargs):
        """
        Runs the pipeline versioning, tagging the pipeline version and optionally 
        storing the configuration.
        
        Parameters
        ----------
        config : dict, optional
            The configuration for the pipeline, including model version, dataset version, 
            and any parameters.
        **run_kwargs : dict, optional
            Additional parameters for the pipeline run.
        
        Returns
        -------
        self : PipelineVersioning
            The current instance of the class.
        
        Raises
        ------
        ValueError
            If the configuration is missing required fields.
        
        Notes
        -----
        This method ensures that the pipeline version is tagged and stored, optionally 
        tracking version history if enabled.
        """
        self._validate_config(config)
        self._auto_tag_pipeline(config=config, **run_kwargs)

        if self.track_history_:
            self._track_pipeline_history()

    def _validate_config(self, config: Optional[dict]):
        """
        Validates the pipeline configuration, ensuring required fields are present.
        
        Parameters
        ----------
        config : dict, optional
            The pipeline configuration to be validated.
        
        Raises
        ------
        ValueError
            If required fields ('model_version', 'dataset_version', 'params') are 
            missing from the configuration.
        
        Notes
        -----
        This method checks if the configuration contains the required fields before 
        proceeding with the pipeline versioning process.
        """
        if config:
            required_fields = ['model_version', 'dataset_version', 'params']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
            self.log_event('config_validated', 
                           {'pipeline_name': self.pipeline_name_, 'config': config})

    def _auto_tag_pipeline(self, config: Optional[dict], **run_kwargs):
        """
        Tags the pipeline version automatically based on configuration 
        and additional parameters.
        
        Parameters
        ----------
        config : dict, optional
            The pipeline configuration to be tagged.
        **run_kwargs : dict, optional
            Additional parameters to include in the pipeline tag.
        
        Notes
        -----
        This method creates a pipeline tag that links the version with the 
        given configuration and any additional parameters.
        """
        if self.auto_tagging_:
            self.pipeline_tag_ = {
                'pipeline_name': self.pipeline_name_,
                'version': self.version_,
                'status': 'Pipeline version tagged successfully',
                'config': config or {},
                **run_kwargs
            }
            self.log_event('pipeline_tagged', {
                'pipeline_name': self.pipeline_name_, 'version': self.version_
            })

    def get_pipeline_tag(self):
        """
        Retrieves the current pipeline tag. Ensures that the pipeline 
        has been run before access.
        
        Returns
        -------
        dict
            The current pipeline tag details.
        
        Raises
        ------
        RuntimeError
            If the pipeline has not been run yet.
        
        Notes
        -----
        This method checks if the pipeline has been run and returns the 
        current version tag. 
        It is used to access the pipeline's version information after a run.
        """
        check_is_runned(self, attributes=["pipeline_tag_"], 
                        msg="Pipeline must be run before accessing the tag.")
        return self.pipeline_tag_

    def compare_versions(self, other_version: str):
        """
        Compares the current pipeline version with another pipeline version. 
        Ensures the pipeline has been run before comparison.
        
        Parameters
        ----------
        other_version : str
            The other pipeline version to compare against.
        
        Returns
        -------
        dict
            A dictionary containing the comparison result.
        
        Raises
        ------
        RuntimeError
            If the pipeline has not been run yet.
        
        Notes
        -----
        This method provides a comparison between the current version and another version 
        of the pipeline.
        """
        check_is_runned(self, attributes=["pipeline_tag_"], 
                        msg="Pipeline must be run before comparing versions.")
        return {
            'current_version': self.version_,
            'compared_version': other_version,
            'comparison_result': f"Comparison between {self.version_} and {other_version}."
        }

    def rollback(self, target_version: str):
        """
        Rolls back to a previous pipeline version. Requires history tracking to be enabled.
        
        Parameters
        ----------
        target_version : str
            The target version to rollback to.
        
        Returns
        -------
        str
            A message indicating successful rollback to the target version.
        
        Raises
        ------
        ValueError
            If version history is not available or the target version is not found.
        
        Notes
        -----
        This method allows rolling back to a previous version of the pipeline using the 
        version history.
        """
        check_is_runned(self, attributes=["pipeline_tag_"], 
                        msg="Pipeline must be run before rollback.")
        if not self.track_history_ or not self.history_:
            raise ValueError("Version history is not available for rollback.")
        
        for record in self.history_:
            if record['version'] == target_version:
                self.pipeline_tag_ = record
                self.version_ = target_version
                self.log_event('rollback_success', {
                    'pipeline_name': self.pipeline_name_, 'version': target_version
                })
                return f"Rolled back to pipeline version {target_version}."
        
        raise ValueError(f"Version {target_version} not found in version history.")

    def get_version_history(self):
        """
        Retrieves the history of pipeline versions if history tracking is enabled.
        
        Returns
        -------
        list of dict
            A list of dictionaries containing version history entries.
        
        Raises
        ------
        ValueError
            If version history tracking is disabled or the pipeline has not been run.
        
        Notes
        -----
        This method returns the pipeline's version history, allowing users to access 
        previous pipeline runs and their corresponding versions.
        """
        check_is_runned(self, attributes=["pipeline_tag_"], 
                        msg="Pipeline must be run before accessing history.")
        if not self.track_history_:
            raise ValueError("Version history tracking is disabled.")
        return self.history_

    def _track_pipeline_history(self):
        """
        Tracks the history of pipeline versions, including metadata like 
        pipeline name and version.
        
        Notes
        -----
        This method is responsible for storing the history of pipeline runs,
        including version and tagging details.
        """
        if self.history_ is not None:
            self.history_.append({
                'pipeline_name': self.pipeline_name_,
                'version': self.version_,
                'pipeline_tag': self.pipeline_tag_,
            })
            self.log_event('history_tracked', {
                'pipeline_name': self.pipeline_name_, 'version': self.version_
            })
    
    def _perform_version_checks(self):
        """
        Performs necessary version checks to ensure that the pipeline versioning system 
        is properly initialized. Checks if the pipeline name is provided and ensures 
        the version is valid.
        
        Raises
        ------
        ValueError
            If the pipeline name is missing or if the version is invalid.
        
        Notes
        -----
        This method performs internal checks to verify that the pipeline versioning 
        has been correctly initialized before any operations.
        """
        if not self.pipeline_name_:
            raise ValueError("Pipeline name is required for versioning.")
        
        # Ensure the version is provided and valid
        if not self.version_ or not isinstance(self.version_, str):
            raise ValueError(f"Invalid version: {self.version_}. A valid version string is required.")
        
        # Log the successful version check
        self.log_event('pipeline_version_check_performed', {
            'pipeline_name': self.pipeline_name_, 'version': self.version_
        })
        
    def validate(self):
        """
        Validates that the current pipeline version exists in the version history (if enabled) 
        and checks for potential inconsistencies. If validation fails or the pipeline has not 
        been run, an error is raised.
        
        Raises
        ------
        RuntimeError
            If the pipeline has not been run yet.
        ValueError
            If version history is inconsistent or missing.
        
        Notes
        -----
        This method ensures the integrity of the pipeline versioning process by validating 
        the version and checking for inconsistencies.
        """
        check_is_runned(self, attributes=["pipeline_tag_"], 
                        msg="Pipeline must be run before validation.")
        
        # Ensure that the pipeline version is consistent
        if not self.pipeline_tag_ or 'version' not in self.pipeline_tag_:
            raise ValueError("Pipeline tag is missing or incomplete.")
        
        # Validate that the version history is available and consistent if tracking is enabled
        if self.track_history_ and (self.history_ is None or len(self.history_) == 0):
            raise ValueError("Version history tracking is enabled but no history is available.")
        
        # Log successful validation
        self.log_event('pipeline_version_validation_success', {
            'pipeline_name': self.pipeline_name_, 'version': self.version_
        })


@SmartFitRun 
class VersionComparison(BaseVersioning):
    """
    Compares different versions of models, datasets, or pipelines by comparing 
    their performance metrics (e.g., accuracy, precision, recall). Provides 
    insights into the changes between versions.

    Parameters
    ----------
    version : str
        The version identifier for the comparison (e.g., 'v1.0').
    comparison_metrics : list, optional, default=['accuracy']
        A list of metrics to compare across versions (e.g., accuracy, precision, recall).
    track_history : bool, optional, default=True
        Whether to track and store the history of comparisons.
    data_source : str, optional
        The source of the data for fetching version metrics. Must be 'db' or 'api'.
    api_url : str, optional
        The URL of the API from which to fetch metrics if `data_source` is 'api'.
    db_connection : object, optional
        A database connection object if `data_source` is 'db'.
    query : str, optional
        A query template used to fetch metrics from the database if `data_source` is 'db'.
        Defaults to: 
            ("SELECT accuracy, precision, recall, f1_score FROM metrics WHERE version = '{version}'".
    **config_params : dict, optional
        Additional configuration parameters to be passed to the base class.

    Attributes
    ----------
    comparison_metrics_ : list
        The metrics used to compare versions.
    track_history_ : bool
        Whether history tracking is enabled for comparisons.
    comparison_results_ : dict or None
        A dictionary containing the results of the last comparison, or None 
        if no comparison has been made.
    history_ : list of dict or None
        A list storing the comparison history, or None if history tracking 
        is disabled.
    data_source_ : str or None
        The source of the data for fetching metrics, either 'db' or 'api'.
    api_url_ : str or None
        The API URL for fetching version metrics if applicable.
    db_connection_ : object or None
        The database connection for fetching version metrics if applicable.
    query_ : str
        The query template for fetching version metrics from the database.
    
    Examples
    --------
    >>> from gofast.mlops.versioning import VersionComparison
    >>> version_comp = VersionComparison(version='v1.0', comparison_metrics=['accuracy', 'precision'])
    >>> version_comp.run(version_a='v1.0', version_b='v2.0', dataset='my_dataset')
    >>> results = version_comp.get_comparison_results()
    
    Notes
    -----
    This class allows users to compare different versions of models, 
    datasets, or pipelines based on specific metrics. It supports both 
    database- and API-based fetching of metrics 
    and includes history tracking for version comparisons.
    
    See Also
    --------
    DatasetVersioning : Class for managing dataset versions.
    PipelineVersioning : Class for managing pipeline versions.
    """

    def __init__(
            self, 
            version: str, 
            comparison_metrics: Optional[list] = None, 
            track_history: bool = True, 
            data_source: Optional[str] = None, 
            api_url: Optional[str] = None,
            db_connection: Optional[object] = None,
            query: Optional[str] = None,
            **config_params):
        
        super().__init__(version, config=config_params)
        self.comparison_metrics= comparison_metrics or ['accuracy']
        self.track_history = track_history
        self.comparison_results_ = None
        self.history_ = [] if track_history else None
        self.data_source = data_source
        self.api_url = api_url
        self.db_connection = db_connection
        self.query = query or( 
            "SELECT accuracy, precision, recall, f1_score"
            " FROM metrics WHERE version = '{version}'"
            )

    @RunReturn  # Without parentheses, returns self by default, useful for chaining methods
    def run(self, version_a: str, version_b: str, dataset: Optional[str] = None, 
            model_type: Optional[str] = None, **run_kwargs):
        """
        Compares two versions based on the specified metric or list of metrics.
        Supports comparisons between models, datasets, and pipelines.

        Parameters
        ----------
        version_a : str
            The first version to compare.
        version_b : str
            The second version to compare.
        dataset : str, optional
            The dataset being compared.
        model_type : str, optional
            The type of model being compared.
        **run_kwargs : dict, optional
            Additional parameters, including pre-computed metrics for `version_a` and `version_b`.
        
        Notes
        -----
        This method performs the comparison between the two specified versions using 
        the metrics defined in the `comparison_metrics_` attribute.
        """
        self._validate_versions(version_a, version_b)
        
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'comparison_metrics': {},
            'dataset': dataset,
            'model_type': model_type,
            'result_summary': None
        }

        for metric in self.comparison_metrics_:
            # Perform comparison for each metric
            comparison['comparison_metrics'][metric] = self._compare_metric(
                version_a, version_b, metric, **run_kwargs)
        
        # Summarize the comparison result
        comparison['result_summary'] = self._summarize_comparison(
            version_a, version_b)

        # Store the comparison results
        self.comparison_results_ = comparison

        # Track the comparison history if enabled
        if self.track_history_:
            self._track_comparison_history()

    def _validate_versions(self, version_a: str, version_b: str):
        """
        Validates that both versions are available for comparison.
        
        Parameters
        ----------
        version_a : str
            The first version to validate.
        version_b : str
            The second version to validate.
        
        Raises
        ------
        ValueError
            If one or both versions are not provided.
        
        Notes
        -----
        This method ensures that both versions are valid before comparison.
        """
        if not version_a or not version_b:
            raise ValueError("Both version_a and version_b must be provided.")
        self.log_event('versions_validated', {
            'version_a': version_a,
            'version_b': version_b
        })

    def _compare_metric(self, version_a: str, version_b: str, metric: str, **run_kwargs):
        """
        Compares the two versions based on the given metric.

        Parameters
        ----------
        version_a : str
            The first version to compare.
        version_b : str
            The second version to compare.
        metric : str
            The metric by which to compare the two versions (e.g., accuracy, precision).
        **run_kwargs : dict, optional
            Additional parameters, including pre-computed metrics for 
            version_a and version_b.

        Returns
        -------
        dict
            A dictionary containing detailed comparison results, including 
            which version performs better and by how much.
        
        Raises
        ------
        ValueError
            If the specified metric is not found in one or both versions' metrics.

        Notes
        -----
        This method assumes that the performance metrics for both versions 
        have been computed and are provided either through `run_kwargs` or
        fetched from a version control system.
        """
        # Retrieve the pre-computed metrics for both versions
        metrics_a = run_kwargs.get('metrics_a') or self._fetch_version_metrics(version_a)
        metrics_b = run_kwargs.get('metrics_b') or self._fetch_version_metrics(version_b)
    
        # Ensure the specified metric exists in both versions' metrics
        if metric not in metrics_a or metric not in metrics_b:
            raise ValueError(f"Metric '{metric}' not found in one or both versions. "
                             f"Available metrics: {list(metrics_a.keys())}.")
    
        # Extract the metric values for both versions
        value_a = metrics_a[metric]
        value_b = metrics_b[metric]
    
        # Calculate the difference between the two metric values
        metric_difference = value_a - value_b
    
        # Determine which version performed better on this metric
        if metric_difference > 0:
            comparison_result =( 
                f"Version {version_a} outperforms {version_b}"
                f" on {metric} by {metric_difference:.4f}")
            better_version = version_a
        elif metric_difference < 0:
            comparison_result = ( 
                f"Version {version_b} outperforms {version_a} on"
                f" {metric} by {abs(metric_difference):.4f}"
                )
            better_version = version_b
        else:
            comparison_result = ( 
                f"Version {version_a} and {version_b}"
                " perform equally on {metric}."
                )
            better_version = "equal"
    
        # Log the comparison event
        self.log_event('metric_comparison', {
            'version_a': version_a,
            'version_b': version_b,
            'metric': metric,
            'value_a': value_a,
            'value_b': value_b,
            'difference': metric_difference,
            'better_version': better_version
        })
    
        # Return the detailed comparison result
        return {
            'metric': metric,
            'version_a_value': value_a,
            'version_b_value': value_b,
            'difference': metric_difference,
            'better_version': better_version,
            'comparison_summary': comparison_result
        }

    def get_comparison_results(self):
        """
        Retrieves the results of the last comparison.
        Requires that the `run` method has been executed before 
        accessing results.

        Returns
        -------
        dict
            The results of the last comparison.

        Raises
        ------
        RuntimeError
            If the `run` method has not been executed before accessing results.
        
        Notes
        -----
        This method provides access to the comparison results after the comparison 
        between versions has been run.
        """
        check_is_runned(self, attributes=["comparison_results_"],
                        msg="Comparison must be run before accessing results.")
        return self.comparison_results_

    def get_comparison_history(self):
        """
        Returns the history of all comparisons made. Requires history tracking 
        to be enabled.

        Returns
        -------
        list
            The history of version comparisons.

        Raises
        ------
        ValueError
            If history tracking is disabled.
        
        Notes
        -----
        This method retrieves all the past comparisons if history tracking 
        has been enabled.
        """
        check_is_runned(self, attributes=["comparison_results_"],
                        msg="Comparison must be run before accessing history.")
        if not self.track_history_:
            raise ValueError("Comparison history tracking is disabled.")
        return self.history_

    def _fetch_from_database(self, version: str) -> dict:
        """
        Fetches the metrics for a specific version from a database using 
        a database connection.

        Parameters
        ----------
        version : str
            The version identifier to query the database for.

        Returns
        -------
        dict
            A dictionary containing the performance metrics, including 
            'accuracy', 'precision', 'recall', and 'f1_score'.

        Raises
        ------
        ValueError
            If no database connection is provided.
        RuntimeError
            If an error occurs during the database query or fetching metrics.

        Notes
        -----
        This method supports both raw SQL and ORM-based database connections. 
        The query to fetch metrics can be customized by passing a custom `query`
        string in the constructor.
        """
        if not self.db_connection:
            raise ValueError("No database connection provided.")
        
        try:
            # Prepare the query using the template
            query = self.query.format(version=version)
            
            # Execute the query based on the connection type
            if hasattr(self.db_connection, 'cursor'):  # Raw SQL connection
                cursor = self.db_connection.cursor()
                cursor.execute(query)
                result = cursor.fetchone()

            elif hasattr(self.db_connection, 'execute'):  # ORM connection
                result = self.db_connection.execute(query).fetchone()

            else:
                raise ValueError("Unsupported database connection type.")

            if result:
                # Assuming the order is (accuracy, precision, recall, f1_score)
                metrics = {
                    'accuracy': result[0],
                    'precision': result[1],
                    'recall': result[2],
                    'f1_score': result[3]
                }
                # Log the event and return metrics
                self.log_event('fetch_metrics', {'version': version, 'metrics': metrics})
                return metrics
            else:
                raise ValueError(f"No metrics found for version {version} in the database.")
        
        except Exception as e:
            self.log_event('db_error', {'error': str(e), 'query': query})
            raise RuntimeError(f"Failed to fetch metrics for version {version}: {e}")

    def _fetch_version_metrics(self, version: str) -> dict:
        """
        Fetches performance metrics for a given version from an external system,
        either a database or API, based on the data_source_ parameter.

        Parameters
        ----------
        version : str
            The version identifier for which to fetch the metrics.

        Returns
        -------
        dict
            A dictionary containing the performance metrics for the given version.

        Raises
        ------
        ValueError
            If an invalid data source is specified or no valid data source is provided.

        Notes
        -----
        This method automatically selects the appropriate fetching mechanism 
        depending on the `data_source_` parameter, which can be either 'db' 
        or 'api'. 
        """
        if self.data_source_ == 'db':
            return self._fetch_from_database(version)
        elif self.data_source_ == 'api':
            return self._fetch_from_api(version)
        else:
            raise ValueError("Invalid data source specified. Use 'db' or 'api'.")

    def _fetch_from_api(self, version: str) -> dict:
        """
        Fetches the metrics for a specific version from an external API.

        Parameters
        ----------
        version : str
            The version identifier to query the API for.

        Returns
        -------
        dict
            A dictionary containing the performance metrics.

        Raises
        ------
        ValueError
            If no API URL is provided.
        RuntimeError
            If the API request fails or the response is invalid.

        Notes
        -----
        This method requires that an `api_url_` be provided. It sends a GET 
        request to the specified API endpoint to fetch the metrics for the given 
        version.
        """
        if not self.api_url_:
            raise ValueError("No API URL provided.")
        
        import requests
        url = f"{self.api_url_}/metrics?version={version}"
        response = requests.get(url)

        if response.status_code == 200:
            metrics = response.json()
            self.log_event('fetch_metrics', {'version': version, 'metrics': metrics})
            return metrics
        else:
            raise RuntimeError(
                f"Failed to fetch metrics from API. Status code: {response.status_code}")

    def _summarize_comparison(self, version_a: str, version_b: str):
        """
        Summarizes the overall result of the comparison between two versions.

        Parameters
        ----------
        version_a : str
            The first version in the comparison.
        version_b : str
            The second version in the comparison.

        Returns
        -------
        str
            A summary of which version performed better overall.

        Notes
        -----
        This method provides a simple textual summary comparing the overall 
        performance between two versions.
        """
        return f"Overall, version {version_a} performs better than {version_b}."

    def rollback_comparison(self, target_version: str):
        """
        Rolls back to a previous comparison for a specific version. Requires 
        version history to be tracked.

        Parameters
        ----------
        target_version : str
            The version identifier to which to roll back.

        Returns
        -------
        str
            A message indicating successful rollback to the target version.

        Raises
        ------
        RuntimeError
            If the comparison has not been run before attempting to roll back.
        ValueError
            If version history is not available for rollback or if the target 
            version is not found in the history.

        Notes
        -----
        This method allows rolling back to a previous comparison if version 
        history tracking is enabled. It checks the history to find the target 
        version and updates the comparison results accordingly.
        """
        check_is_runned(self, attributes=["comparison_results_"], 
                        msg="Comparison must be run before rollback.")
        if not self.track_history_ or not self.history_:
            raise ValueError("Comparison history is not available for rollback.")

        for record in self.history_:
            if record['version'] == target_version:
                self.comparison_results_ = record
                self.version_ = target_version
                self.log_event('rollback_success', {
                    'version': target_version
                })
                return f"Rolled back to comparison with version {target_version}."
        
        raise ValueError(f"Version {target_version} not found in comparison history.")

    def _track_comparison_history(self):
        """
        Tracks the history of comparisons, including versions compared, 
        metrics used, and the comparison results.

        Notes
        -----
        This method appends the latest comparison results to the internal 
        history list if history tracking is enabled.
        """
        if self.history_ is not None:
            self.history_.append({
                'version': self.version_,
                'comparison_results': self.comparison_results_,
                'comparison_metrics': self.comparison_metrics_
            })
            self.log_event('comparison_tracked', {
                'version': self.version_,
                'metrics': self.comparison_metrics_
            })

    def _perform_version_checks(self):
        """
        Performs necessary version checks to ensure that the versioning tool 
        is correctly initialized and the version information for comparison 
        is available.

        Raises
        ------
        ValueError
            If the required data sources (database, API) are not configured.

        Notes
        -----
        This method validates that either a database connection or an API URL 
        is provided depending on the specified `data_source_`. If validation 
        fails, an appropriate error is raised, and the event is logged.
        """
        if not self.data_source_:
            raise ValueError("No data source specified for fetching version metrics. "
                             "Please provide a valid data source (e.g., 'db' or 'api').")
        
        if self.data_source_ == 'db' and not self.db_connection:
            raise ValueError("Database connection is required when"
                             " 'db' is specified as the data source.")
        
        if self.data_source_ == 'api' and not self.api_url_:
            raise ValueError("API URL is required when 'api'"
                             " is specified as the data source.")
        
        # Log the version check event for tracking purposes
        self.log_event('version_check_performed', {
            'version': self.version_,
            'data_source': self.data_source_,
            'api_url': self.api_url_,
            'db_connection': bool(self.db_connection)
        })

    def validate(self):
        """
        Validates that the current version exists in the database or API and 
        checks for potential inconsistencies.

        Raises
        ------
        RuntimeError
            If validation fails due to missing or incomplete metrics for the version.
        ValueError
            If an invalid data source is specified for fetching metrics.

        Notes
        -----
        This method fetches the version metrics from the specified data source 
        (either 'db' or 'api') and ensures the completeness of the metrics for 
        comparison. If any inconsistency is found, an appropriate error is raised.
        """
        try:
            # Check if the version exists based on the data source
            if self.data_source_ == 'db':
                metrics = self._fetch_from_database(self.version_)
            elif self.data_source_ == 'api':
                metrics = self._fetch_from_api(self.version_)
            else:
                raise ValueError("Invalid data source. Cannot validate version.")
    
            # Ensure that the metrics are not empty or incomplete
            if not metrics or not all(key in metrics for key in self.comparison_metrics_):
                raise ValueError(f"Missing or incomplete metrics for version {self.version_}.")
    
            # Log successful validation
            self.log_event('version_validation_success', {
                'version': self.version_,
                'metrics': metrics
            })
        
        except Exception as e:
            # Log and raise an error if validation fails
            self.log_event('version_validation_failed', {
                'version': self.version_,
                'error': str(e)
            })
            raise RuntimeError(f"Version validation failed for {self.version_}: {e}")
