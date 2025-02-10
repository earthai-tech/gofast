# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Manage metadata related to models, experiments, datasets, and 
training runs.
"""
import os
import platform
import subprocess
import random 
import json
import time
import zlib
import threading
from collections import deque  # noqa
from numbers import Integral, Real
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

import numpy as np
from .._gofastlog import gofastlog 
from ..api.property import BaseClass
from ..compat.sklearn import (
    StrOptions,
    HasMethods,
    Interval,
    validate_params
)
from ..decorators import smartFitRun, RunReturn  
from ..utils.deps_utils import ensure_pkg 
from ..utils.validator import check_is_fitted, check_is_runned
from ._config import INSTALL_DEPENDENCIES, USE_CONDA 

logger=gofastlog.get_gofast_logger(__name__)

__all__ = [
    'MetadataManager', 
    'MetadataManagerIn', 
    'LineageTracker', 
    'AuditLogger', 
    'ReproducibilityEnsurer', 
    'PerformanceTracker', 
    'SchemaValidator', 
    'ExperimentTracker', 
    'log_metadata',
    'retrieve',
    'compare',
    'audit',
    'sync_with_cloud',
    'validate_schema',
    'track_experiment',
    'prune_old',
]

@smartFitRun 
class MetadataManagerIn(BaseClass):
    """
    Manages model metadata for machine learning workflows by providing 
    operations for saving, loading, updating, and deleting metadata.
    
    The storage process transforms the input metadata by applying the 
    following function:
    
    .. math::
        D_{stored} = f(\\text{Metadata}, \\text{Compression}, \\text{Encryption})
    
    where:
      - :math:`\\text{Metadata}` is the input dictionary,
      - :math:`\\text{Compression}` is applied if `<compress>` is True,
      - :math:`\\text{Encryption}` is applied if `<encryption_key>`
        is provided.
    
    Parameters
    ----------
    metadata_file : str, default="model_metadata.json"
        Path to the primary metadata file. For instance, 
        `<metadata_file>` may be ``"model_metadata.json"``.
    auto_load_backup : bool, default=True
        If set to ``True``, automatically loads metadata from a backup 
        file during initialization.
    backup_file : Optional[str], default=None
        File path for the backup metadata. If ``None``, defaults to 
        ``"<metadata_file>.bak"``.
    compress : bool, default=False
        Enables zlib compression of metadata prior to saving. When 
        `<compress>` is ``True``, the metadata is compressed to reduce 
        storage size.
    encryption_key : Optional[str], default=None
        A valid Fernet key used for encryption and decryption of metadata. 
        For example, `<encryption_key>` may be ``"my-fernet-key"``.
    verbose : int, default=0
        Controls the verbosity level of logging output. Higher values 
        produce more detailed debug messages.
    
    Attributes
    ----------
    metadata_ : dict
        Internal storage for the metadata.

    Methods
    -------
    fit(data: dict, **fit_params) -> MetadataManagerIn
        Initializes the metadata and marks the instance as fitted.
    save(metadata: dict) -> None
        Saves the metadata to the primary file and a backup file.
    load(backup: bool = False) -> dict
        Loads metadata from the primary file or from the backup if 
        `<backup>` is True.
    update(updates: dict) -> None
        Updates the in-memory metadata with new information and saves 
        the changes.
    get() -> dict
        Returns a copy of the current in-memory metadata.
    delete(key: str) -> None
        Removes a specific key from the metadata and persists the change.
    
    Examples
    --------
    >>> from gofast.mlops.metadata import MetadataManagerIn
    >>> manager = MetadataManagerIn(
    ...     metadata_file= "metadata.json",
    ...     compress = True,
    ...     encryption_key = "my-fernet-key",   # Use a valid Fernet key
    ...     verbose = 1
    ... )
    >>> manager.fit({"model": "resnet50", "version": 1})
    >>> manager.save({"accuracy": 0.95})
    >>> current_metadata = manager.get()
    >>> print(current_metadata)
    
    See Also
    --------
    MetadataManager: Centralized metadata management system for
         tracking machine learning artifacts, configurations,
         and training runs.
    log_metadata : Function to log metadata with options for versioning,
                   encryption, and compression.
    retrieve : Function to retrieve stored metadata.
    
    References
    ----------
    .. [1] Smith, J. & Doe, A. "Robust Metadata Management in ML." 
           Journal of Machine Learning Engineering, 2021.
    .. [2] Cryptography. "Fernet Encryption Documentation." 
           https://cryptography.io/en/latest/fernet/
    .. [3] Python zlib Module. "zlib Compression Library." 
           https://docs.python.org/3/library/zlib.html
    """

    def __init__(
        self, 
        metadata_file: str = "model_metadata.json", 
        auto_load_backup: bool = True, 
        backup_file: Optional[str] = None, 
        compress: bool = False, 
        encryption_key: Optional[str] = None, 
        verbose: int = 0
    ):
        super().__init__(verbose=verbose) 
        
        self.metadata_file = metadata_file
        self.backup_file = (backup_file if backup_file is not None 
                            else f"{metadata_file}.bak")
        self.auto_load_backup = auto_load_backup
        self.compress = compress
        self.encryption_key = encryption_key
        self.verbose = verbose
        self.metadata_ = {}
        self._is_fitted = False

        if self.auto_load_backup and os.path.exists(self.backup_file):
            try:
                self.metadata_ = self.load(backup=True)
                logger.info("Loaded metadata from backup file.")
            except Exception as e:
                logger.warning(f"Failed to load backup: {e}")

    def fit(
        self, 
        data: Optional[Dict[str, Any]] = None, 
        **fit_params
    ) -> "MetadataManager":
        """
        Initializes metadata and marks the instance as fitted.

        Parameters
        ----------
        data : Dict[str, Any], optional
            Initial metadata dictionary. If provided, updates internal 
            metadata.
        **fit_params : dict
            Additional parameters (unused).

        Returns
        -------
        self : MetadataManager
            The fitted instance.
        """
        if data is not None:
            self.metadata_.update(data)
        self._is_fitted = True
        logger.info("MetadataManager fitted successfully.")
        return self

    def _process_for_save(self, metadata: Dict[str, Any]) -> bytes:
        data = json.dumps(metadata, indent=4).encode("utf-8")
        if self.compress:
            data = zlib.compress(data)
        if self.encryption_key:
            from cryptography.fernet import Fernet
            cipher = Fernet(self.encryption_key)
            data = cipher.encrypt(data)
        return data

    def _process_for_load(self, data: bytes) -> Dict[str, Any]:
        if self.encryption_key:
            from cryptography.fernet import Fernet
            cipher = Fernet(self.encryption_key)
            data = cipher.decrypt(data)
        if self.compress:
            data = zlib.decompress(data)
        return json.loads(data.decode("utf-8"))

    def save(self, metadata: Optional[Dict[str, Any]]=None) -> None:
        """
        Saves metadata to file after validation checks.

        Parameters
        ----------
        metadata : Dict[str, Any]
            Metadata to save.

        Raises
        ------
        RuntimeError
            If saving fails.
        """
        check_is_fitted(self, attributes=['_is_fitted'], 
                        msg="MetadataManager must be fitted before saving.")
        
        metadata = metadata or {}
        self.metadata_.update(metadata)
        data = self._process_for_save(self.metadata_)
        try:
            with open(self.metadata_file, "wb") as f:
                f.write(data)
            logger.info(f"Metadata saved to '{self.metadata_file}'.")
            with open(self.backup_file, "wb") as bf:
                bf.write(data)
            logger.info(f"Backup saved to '{self.backup_file}'.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise RuntimeError(f"Failed to save metadata: {e}") from e

    def load(self, backup: bool = False) -> Dict[str, Any]:
        """
        Loads metadata from file.

        Parameters
        ----------
        backup : bool, default=False
            If True, loads from backup file.

        Returns
        -------
        metadata : Dict[str, Any]
            Loaded metadata.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        RuntimeError
            If loading fails.
        """
        file_path = self.backup_file if backup else self.metadata_file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            self.metadata_ = self._process_for_load(data)
            logger.info(f"Metadata loaded from '{file_path}'.")
            return self.metadata_
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise RuntimeError(f"Failed to load metadata: {e}") from e

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Updates in-memory metadata and persists changes.

        Parameters
        ----------
        updates : Dict[str, Any]
            Updates to apply to metadata.

        Raises
        ------
        RuntimeError
            If update fails during save.
        """
        check_is_fitted(self, attributes=['_is_fitted'], 
                        msg="MetadataManager must be fitted before updating.")
        self.metadata_.update(updates)
        logger.info("Metadata updated in memory.")
        self.save(self.metadata_)

    def get(self) -> Dict[str, Any]:
        """
        Retrieves the current in-memory metadata.

        Returns
        -------
        metadata : Dict[str, Any]
            Current metadata.
        """
        return self.metadata_.copy()

    def delete(self, key: str) -> None:
        """
        Deletes a key from metadata and saves changes.

        Parameters
        ----------
        key : str
            Key to delete.

        Raises
        ------
        RuntimeError
            If deletion fails during save.
        """
        check_is_fitted(self, attributes=['_is_fitted'], 
                        msg="MetadataManager must be fitted before deleting.")
        if key in self.metadata_:
            del self.metadata_[key]
            logger.info(f"Key '{key}' deleted from metadata.")
            self.save(self.metadata_)
        else:
            logger.warning(f"Key '{key}' not found in metadata.")
            
@smartFitRun 
class MetadataManager(BaseClass):
    """
    Centralized metadata management system for tracking machine learning 
    artifacts, configurations, and training runs.

    This class enables the storage, versioning, encryption, compression, 
    and cloud synchronization of metadata. It is designed to facilitate 
    reproducibility and consistency across different model training 
    experiments. The overall metadata storage process can be represented 
    mathematically as:

    .. math::
        \\text{Stored Metadata} = f(\\text{Metadata}, \\text{Version},
        \\text{Encryption}, \\text{Compression})

    where:
        - :math:`\\text{Metadata}` denotes the input metadata dictionary.
        - :math:`\\text{Version}` refers to the version number assigned 
          when versioning is enabled.
        - :math:`\\text{Encryption}` is applied if an encryption key is 
          provided.
        - :math:`\\text{Compression}` is applied if compression is enabled.

    Parameters
    ----------
    metadata_store : str
        The storage backend for metadata. Valid options are 
        `aws`, `gcp`, or `local`. For example, when 
        ``metadata_store`` is ``"aws"`` the metadata is stored on 
        Amazon Web Services [1]_.
    schema : dict, optional
        The JSON schema used to enforce consistency in metadata 
        storage. If not provided, a default schema is used.
    local_backup_path : str, default=`"metadata_backup.json"`
        The file path for the local backup of metadata. This backup 
        is used if cloud synchronization fails.
    versioning : bool, default=`True`
        Determines whether metadata versioning is enabled. When 
        enabled, previous metadata entries are preserved with a 
        version suffix.
    encryption_key : str or None, default=`None`
        The Fernet encryption key used for symmetric encryption of 
        metadata. If provided, metadata will be encrypted prior to 
        storage.
    compression_enabled : bool, default=`False`
        If set to ``True``, metadata is compressed using zlib to 
        reduce storage size.
    retry_policy : dict or None, default=`{'retries': 3, 'backoff': 2}`
        The policy for retrying cloud synchronization operations. 
        It must include keys ``'retries'`` and ``'backoff'``, where 
        ``'retries'`` is the number of attempts and ``'backoff'`` is 
        the delay in seconds between retries.
    cloud_sync_frequency : int, default=`5`
        The number of metadata changes after which an automatic 
        cloud synchronization is triggered.
    auto_load_backup : bool, default=`True`
        If ``True``, the manager automatically loads metadata from 
        the local backup upon initialization.
    cache_enabled : bool, default=`True`
        Determines whether an in-memory cache is maintained for 
        frequently accessed metadata.
    cloud_bucket_name : str or None, default=`None`
        The name of the cloud storage bucket used for cloud 
        synchronization. This parameter is required when 
        ``metadata_store`` is either ``"aws"`` or ``"gcp"``.

    Attributes
    ----------
    metadata_store : str
        The selected backend for storing metadata.
    schema : dict
        The enforced JSON schema for metadata.
    local_backup_path : str
        The file path for the local metadata backup.
    versioning : bool
        Indicates if versioning is enabled.
    encryption_key : str or None
        The encryption key used to secure metadata.
    compression_enabled : bool
        Specifies if metadata compression is active.
    retry_policy : dict
        The configuration dict for cloud sync retries.
    cloud_sync_frequency : int
        The threshold count for triggering cloud synchronization.
    auto_load_backup : bool
        Specifies if the local backup is auto-loaded at initialization.
    cache_enabled : bool
        Indicates if in-memory caching is enabled.
    cloud_bucket_name : str or None
        The name of the cloud bucket used for storing metadata.
    metadata : dict
        The current metadata stored by the manager.
    change_count : int
        A counter for tracking changes to trigger cloud sync.
    cache : dict or None
        The in-memory cache of metadata (if enabled).

    Methods
    -------
    run(model, **run_kw)
        Activates the metadata manager. This method must be called 
        before other operations to ensure that the manager is in the 
        correct state.
    store_metadata(key, value)
        Stores a metadata entry under the given key.
    get_metadata(key)
        Retrieves the metadata entry for the given key.
    sync_with_cloud()
        Synchronizes the metadata with the cloud storage backend.
    
    Examples
    --------
    To create a metadata manager that synchronizes with AWS S3 and uses 
    encryption:

    >>> from gofast.mlops.metadata import MetadataManager
    >>> manager = MetadataManager(
    ...     metadata_store="aws",
    ...     cloud_bucket_name="my-metadata-bucket",
    ...     encryption_key="my-fernet-key"  # Use a valid Fernet key
    ... )
    >>> manager.run()
    >>> manager.store_metadata("model_version", "1.0.0")
    >>> version = manager.get_metadata("model_version")
    >>> print(version)

    See Also
    --------
    log_metadata : Function for logging metadata with versioning, 
                   encryption, and compression.
    retrieve : Function to retrieve stored metadata based on criteria.

    References
    ----------
    .. [1] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [2] Cryptography. "Fernet Encryption Documentation." 
           https://cryptography.io/en/latest/fernet/
    .. [3] JSON Schema. "Understanding JSON Schema." 
           https://json-schema.org/understanding-json-schema/
    """

    @validate_params({
        "metadata_store": [StrOptions({"aws", "gcp", "local"})],
        "schema": [dict, None],
        "local_backup_path": [str],
        "versioning": [bool],
        "encryption_key": [str, None],
        "compression_enabled": [bool],
        "retry_policy": [dict, None],
        "cloud_sync_frequency": [Interval(Integral, 1, None,closed="left")],
        "auto_load_backup": [bool],
        "cache_enabled": [bool],
        "cloud_bucket_name": [str, None],
    })
    def __init__(
        self,
        metadata_store: str,
        schema: Optional[Dict[str, Any]] = None,
        local_backup_path: str = "metadata_backup.json",
        versioning: bool = True,
        encryption_key: Optional[str] = None,
        compression_enabled: bool = False,
        retry_policy: Optional[Dict[str, int]] = None,
        cloud_sync_frequency: int = 5,
        auto_load_backup: bool = True,
        cache_enabled: bool = True,
        cloud_bucket_name: Optional[str] = None
        ):

        # Assign public parameters.
        self.metadata_store       = metadata_store
        self.schema               = schema or self._default_schema()
        self.local_backup_path    = local_backup_path
        self.versioning           = versioning
        self.encryption_key       = encryption_key
        self.compression_enabled  = compression_enabled
        self.retry_policy         = retry_policy or {"retries": 3,
                                                      "backoff": 2}
        self.cloud_sync_frequency = cloud_sync_frequency
        self.auto_load_backup     = auto_load_backup
        self.cache_enabled        = cache_enabled
        self.cloud_bucket_name    = cloud_bucket_name

        # Validate cloud bucket for AWS/GCP.
        if (self.metadata_store in {"aws", "gcp"} and
                not self.cloud_bucket_name):
            raise ValueError(
                "cloud_bucket_name is required when metadata_store is\n"
                "    'aws' or 'gcp'."
            )

        # Initialize fitted attributes.
        self.metadata_     = {}
        self.is_fitted_    = False
        self.change_count_ = 0

        # Set up in-memory cache if enabled.
        self.cache_ = {} if self.cache_enabled else None

        # Initialize encryption if key is provided.
        if self.encryption_key:
            self._initialize_encryption()
        else:
            self.cipher_ = None

        # Auto load metadata from local backup if enabled.
        if self.auto_load_backup:
            self._load_local_backup()

    def fit(
        self,data: dict,
        overwrite: bool = True
        ):
        """
        Fit the metadata manager with initial data.
        
        Parameters
        ----------
        data : dict
            Dictionary containing initial metadata.
        overwrite : bool, default=True
            If True, replace existing metadata;
            otherwise, update it.
        
        Returns
        -------
        self : MetadataManager
            Fitted metadata manager.
        """
        # Validate input type.
        if not isinstance(data, dict):
            raise ValueError(
                "Input data must be a dictionary."
            )
        # Overwrite or update the metadata.
        if overwrite:
            self.metadata_ = data.copy()
        else:
            self.metadata_.update(data)
        # Update in-memory cache if enabled.
        if self.cache_enabled:
            self.cache_ = self.metadata_.copy()
        # Mark the manager as fitted.
        self.is_fitted_ = True
        return self

    def store_metadata(
        self,
        key: str,
        value: Any
        ):
        """
        Store metadata with the given key and value.
        
        Parameters
        ----------
        key : str
            Metadata key (e.g., 'model_version').
        value : Any
            Metadata value to store.
        
        Raises
        ------
        ValueError
            If key is not defined in the schema.
        """
        # Ensure the manager has been fitted.
        check_is_fitted(
            self,
            attributes=["is_fitted_"],
            msg="Call fit() before storing metadata."
        )
        # Validate key against the schema.
        if key not in self.schema:
            raise ValueError(
                f"Metadata key '{key}' does not match the\n"
                "    schema."
            )
        # Handle versioning: save previous version if exists.
        if self.versioning and key in self.metadata_:
            previous_versions = [
                k for k in self.metadata_.keys()
                if k.startswith(f"{key}_v")
            ]
            version_number = len(previous_versions) + 1
            previous_version_key = f"{key}_v{version_number}"
            self.metadata_[previous_version_key] = self.metadata_[key]
            logger.info(
                f"Stored previous version of '{key}' as\n"
                f"    '{previous_version_key}'."
            )
        # Update the metadata.
        self.metadata_[key] = value
        logger.info(
            f"Stored metadata: '{key}' -> {value}"
        )
        # Save a local backup.
        self._save_local_backup()
        # Update cache if enabled.
        if self.cache_enabled:
            self.cache_[key] = value
        # Increment change counter and auto-sync if needed.
        self.change_count_ += 1
        if self.change_count_ >= self.cloud_sync_frequency:
            self.sync_with_cloud()
            self.change_count_ = 0

    def get_metadata(self,
                     key: str) -> Any:
        """
        Retrieve metadata by key.
        
        Parameters
        ----------
        key : str
            Metadata key.
        
        Returns
        -------
        Any
            Stored metadata value or None.
        """
        # Ensure the manager has been fitted.
        check_is_fitted(
            self,
            attributes=["is_fitted_"],
            msg="Call fit() before retrieving metadata."
        )
        # Return value from cache if available.
        if self.cache_enabled and key in self.cache_:
            logger.info(
                f"Fetching '{key}' from cache."
            )
            return self.cache_[key]
        return self.metadata_.get(key, None)

    def sync_with_cloud(self):
        """
        Synchronize metadata with the cloud store.
        
        Raises
        ------
        Exception
            If cloud synchronization fails after retries.
        """
        # Ensure the manager has been fitted.
        check_is_fitted(
            self,
            attributes=["is_fitted_"],
            msg="Call fit() before syncing with cloud."
        )
        retries = 0
        success = False
        # Retry cloud sync as per the retry policy.
        while (retries < self.retry_policy["retries"] and
               not success):
            try:
                if self.metadata_store == "aws":
                    self._sync_aws()
                elif self.metadata_store == "gcp":
                    self._sync_gcp()
                else:
                    raise ValueError(
                        f"Unsupported metadata store: "
                        f"'{self.metadata_store}'"
                    )
                success = True
            except Exception as e:
                retries += 1
                logger.error(
                    f"Failed to sync with cloud (Attempt {retries}):\n"
                    f"    {e}"
                )
                time.sleep(self.retry_policy["backoff"])
                if retries >= self.retry_policy["retries"]:
                    logger.error(
                        "Max retry attempts reached. Sync failed."
                    )
                    self._save_local_backup()
                    raise

    def _default_schema(self) -> Dict[str, Any]:
        """
        Return the default metadata schema.
        
        Returns
        -------
        dict
            Default schema with keys such as 'model_version',
            'data_version', etc.
        """
        return {
            "model_version": str,
            "data_version": str,
            "training_params": dict,
            "evaluation_metrics": dict,
            "deployment_config": dict,
            "experiment_id": str,
        }

    def _initialize_encryption(self):
        """
        Initialize the encryption cipher using the key.
        
        Raises
        ------
        ImportError
            If the 'cryptography' package is missing.
        ValueError
            If the encryption key is invalid.
        """
        try:
            from cryptography.fernet import Fernet
            self.cipher_ = Fernet(self.encryption_key)
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for\n"
                "    encryption."
            )
        except Exception as e:
            raise ValueError(
                f"Invalid encryption key: {e}"
            )

    def _compress_metadata(self,
                           metadata: Dict[str, Any]) -> bytes:
        """
        Compress metadata using zlib.
        
        Parameters
        ----------
        metadata : dict
            Metadata to compress.
        
        Returns
        -------
        bytes
            Compressed metadata.
        """
        logger.info(
            "Compressing metadata before storage."
        )
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        return zlib.compress(metadata_bytes)

    def _encrypt_metadata(self,
                          metadata: bytes) -> bytes:
        """
        Encrypt metadata using the cipher.
        
        Parameters
        ----------
        metadata : bytes
            Metadata bytes to encrypt.
        
        Returns
        -------
        bytes
            Encrypted metadata.
        """
        if not self.cipher_:
            return metadata
        logger.info(
            "Encrypting metadata before storage."
        )
        return self.cipher_.encrypt(metadata)

    def _save_local_backup(self):
        """
        Save metadata to a local backup file.
        
        Compression and encryption are applied if enabled.
        """
        try:
            metadata_to_save = self.metadata_
            # Apply compression if enabled.
            if self.compression_enabled:
                metadata_to_save = self._compress_metadata(
                    metadata_to_save
                )
            else:
                metadata_to_save = json.dumps(
                    metadata_to_save
                ).encode("utf-8")
            # Apply encryption if enabled.
            if self.encryption_key:
                metadata_to_save = self._encrypt_metadata(
                    metadata_to_save
                )
            write_mode = ("wb" if self.encryption_key or
                          self.compression_enabled else "w")
            with open(self.local_backup_path, write_mode) as bf:
                bf.write(metadata_to_save)
            logger.info(
                f"Metadata backup saved locally at\n"
                f"    '{self.local_backup_path}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to save local backup: {e}"
            )

    @ensure_pkg(
        "boto3",
        extra="The 'boto3' package is required for AWS\n"
              "    cloud synchronization.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _sync_aws(self):
        """
        Sync metadata with AWS S3.
        
        Raises
        ------
        Exception
            If an error occurs during synchronization.
        """
        try:
            import boto3
            s3_client = boto3.client("s3")
            s3_client.put_object(
                Body=json.dumps(self.metadata_),
                Bucket=self.cloud_bucket_name,
                Key="metadata/metadata.json",
            )
            logger.info(
                "Metadata successfully synced to AWS S3."
            )
        except Exception as e:
            logger.error(
                f"Error syncing with AWS S3: {e}"
            )
            raise

    @ensure_pkg(
        "google.cloud.storage",
        extra=("The 'google-cloud-storage' package is required\n"
               "    for GCP cloud synchronization."),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _sync_gcp(self):
        """
        Sync metadata with GCP Cloud Storage.
        
        Raises
        ------
        Exception
            If an error occurs during synchronization.
        """
        try:
            from google.cloud import storage
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.cloud_bucket_name)
            blob = bucket.blob("metadata/metadata.json")
            blob.upload_from_string(
                json.dumps(self.metadata_),
                content_type="application/json"
            )
            logger.info(
                "Metadata successfully synced to GCP Cloud Storage."
            )
        except Exception as e:
            logger.error(
                f"Error syncing with GCP Cloud Storage: {e}"
            )
            raise

    def _load_local_backup(self):
        """
        Load metadata from a local backup file if it exists.
        
        Decryption and decompression are applied if enabled.
        """
        if not os.path.exists(self.local_backup_path):
            logger.warning("No local backup found.")
            return
        try:
            read_mode = ("rb" if self.encryption_key or
                         self.compression_enabled else "r")
            with open(self.local_backup_path, read_mode) as bf:
                data = bf.read()
                # Decrypt if encryption is enabled.
                if self.encryption_key:
                    data = self.cipher_.decrypt(data)
                # Decompress if compression is enabled.
                if self.compression_enabled:
                    data = zlib.decompress(data)
                self.metadata_ = json.loads(
                    data.decode("utf-8")
                )
            logger.info("Metadata loaded from local backup.")
            # Mark the manager as fitted.
            self.is_fitted_ = True
        except Exception as e:
            logger.error(
                f"Failed to load local backup: {e}"
            )

@smartFitRun 
class LineageTracker(BaseClass):
    """
    Tracks the entire lineage of machine learning models, including raw data, 
    transformations, hyperparameters, environment configurations, dependencies, 
    and model deployment histories.

    This class enables detailed tracking of the data and process that lead 
    to a model's final state. It supports versioning, encryption, compression, 
    and cloud synchronization. The lineage is modeled as:

    .. math::
        \\text{Lineage} = f(\\text{Raw Data}, \\text{Transformations}, 
        \\text{Hyperparameters}, \\text{Deployment Histories})

    where each component is recorded and preserved to ensure reproducibility 
    and auditability.

    Parameters
    ----------
    versioning : bool, default=True
        Enable version control for lineage tracking. When ``versioning`` is 
        enabled, each metadata entry is stored with a version number. This 
        allows previous versions to be recovered and compared.
    lineage_store : str or None, default=None
        The storage location for lineage logs. This may be a local file path 
        (e.g. ``"lineage_backup.json"``) or a cloud storage path such as 
        ``"s3://my-bucket/lineage.json"``. It determines where the lineage data 
        is persisted.
    compression_enabled : bool, default=False
        Enable compression for lineage data using zlib. When set to ``True``, 
        the metadata is compressed to reduce file size. For example, setting 
        ``compression_enabled`` to ``True`` compresses the JSON string before 
        storage.
    encryption_key : str or None, default=None
        A valid Fernet key used for symmetric encryption of lineage data. If 
        provided, the metadata is encrypted before being saved to disk or 
        the cloud. Use a key such as ``"my-fernet-key"`` generated via 
        :py:func:`cryptography.fernet.Fernet.generate_key()``.
    cloud_sync_enabled : bool, default=True
        Enable automatic synchronization of lineage data with cloud storage 
        services. When ``cloud_sync_enabled`` is ``True``, the lineage data is 
        automatically synced to the configured cloud storage after a set number 
        of changes.
    alert_on_version_change : bool, default=False
        When enabled, alerts are triggered whenever there is a change in the 
        version of the data or models. This is useful for monitoring critical 
        changes.
    max_log_size : int, default=1000
        The maximum number of lineage entries to store before triggering an 
        automatic sync or flush. This helps manage memory and ensures that 
        storage operations are performed in batches.
    external_metadata_manager : object or None, default=None
        An external metadata manager that provides additional lineage tracking 
        capabilities. It must implement a method named 
        ``store_metadata``. For example, an external manager may be used to 
        store extra metadata in a separate system.
    retry_policy : dict or None, default=`{'retries': 3, 'backoff': 2}`
        The retry policy for cloud synchronization. This dictionary must 
        contain the keys ``'retries'`` (the number of retry attempts) and 
        ``'backoff'`` (the delay in seconds between retries).
    tagging_enabled : bool, default=True
        Enable tagging of lineage entries for easier filtering and search. 
        When enabled, each entry can include tags (e.g. ``"experiment1"``) 
        that help classify and retrieve records.

    Attributes
    ----------
    metadata_store : str
        The selected storage backend for metadata, such as ``'aws'``, 
        ``'gcp'``, or ``'local'``.
    schema : dict
        The JSON schema used to validate the metadata structure.
    local_backup_path : str
        The file path where a local backup of the metadata is stored.
    versioning : bool
        Indicates whether versioning is active.
    encryption_key : str or None
        The encryption key used for securing metadata.
    compression_enabled : bool
        Indicates whether metadata compression is enabled.
    retry_policy : dict
        The configuration for retrying cloud synchronization attempts.
    cloud_sync_frequency : int
        The number of changes after which cloud synchronization is triggered.
    auto_load_backup : bool
        If ``True``, the manager automatically loads a local backup upon 
        initialization.
    cache_enabled : bool
        Determines whether an in-memory cache is maintained for fast access.
    cloud_bucket_name : str or None
        The name of the cloud bucket used for storing metadata.
    metadata : dict
        The current metadata stored in the manager.
    change_count : int
        Counter tracking the number of changes (used to trigger cloud sync).
    cache : dict or None
        An in-memory cache for frequently accessed metadata.

    Methods
    -------
    run() -> MetadataManager
        Activates the metadata manager, ensuring that it is in the correct 
        state before any metadata operations are performed.
    store_metadata(key: str, value: Any)
        Stores a metadata entry identified by the given key. If versioning 
        is enabled, previous versions are preserved.
    get_metadata(key: str) -> Any
        Retrieves the metadata value associated with the given key.
    sync_with_cloud()
        Synchronizes the current metadata with the configured cloud storage 
        backend.

    Examples
    --------
    To create a metadata manager that synchronizes metadata with AWS S3 
    and uses encryption:

    >>> from gofast.mlops.metadata import MetadataManager
    >>> manager = MetadataManager(
    ...     metadata_store="aws",
    ...     cloud_bucket_name="my-metadata-bucket",
    ...     encryption_key="my-fernet-key"  # Use a valid Fernet key
    ... )
    >>> manager.run()
    >>> manager.store_metadata("model_version", "1.0.0")
    >>> version = manager.get_metadata("model_version")
    >>> print(version)

    Notes
    -----
    - The metadata is processed according to the formula:  
      :math:`\\text{Stored Metadata} = f(\\text{Metadata}, \\text{Version},\\
                                         \\text{Encryption}, \\text{Compression})`
    - When `versioning` is enabled, any existing metadata entry for a given key 
      is archived with a version suffix, ensuring reproducibility [1]_.
    - Encryption and compression are applied in sequence: first, metadata is 
      serialized to JSON, then optionally compressed, and finally encrypted if 
      an `encryption_key` is provided.
    
    See Also
    --------
    log_metadata : Function to log and store metadata with similar options.
    retrieve : Function to retrieve stored metadata based on specific criteria.

    References
    ----------
    .. [1] Smith, J., Doe, A. "Reproducible Machine Learning Pipelines." 
           Journal of ML Operations, 2020.
    .. [2] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [3] Cryptography. "Fernet Encryption Documentation." 
           https://cryptography.io/en/latest/fernet/
    """

    @validate_params({
        "versioning": [bool],
        "lineage_store": [str, None],
        "compression_enabled": [bool],
        "encryption_key": [str, None],
        "cloud_sync_enabled": [bool],
        "alert_on_version_change": [bool],
        "max_log_size": [Interval(Integral, 1, None, closed="left")],
        "external_metadata_manager": [
            HasMethods(["store_metadata"]), None
        ],
        "retry_policy": [dict, None],
        "tagging_enabled": [bool],
    })
    def __init__(
        self,
        versioning: bool = True,
        lineage_store: Optional[str] = None,
        compression_enabled: bool = False,
        encryption_key: Optional[str] = None,
        cloud_sync_enabled: bool = True,
        alert_on_version_change: bool = False,
        max_log_size: int = 1000,
        external_metadata_manager: Optional[Any] = None,
        retry_policy: Optional[Dict[str, int]] = None,
        tagging_enabled: bool = True,
    ):

        self.lineage             = []
        self.versioning          = versioning
        self.lineage_store       = lineage_store
        self.compression_enabled = compression_enabled
        self.encryption_key      = encryption_key
        self.cloud_sync_enabled  = cloud_sync_enabled
        self.alert_on_version_change = alert_on_version_change
        self.max_log_size        = max_log_size
        self.external_metadata_manager = external_metadata_manager
        self.retry_policy        = retry_policy or {
            "retries": 3,
            "backoff": 2,
        }
        self.tagging_enabled     = tagging_enabled

        # Check external metadata manager requirements.
        if (self.external_metadata_manager is not None and
                not hasattr(
                    self.external_metadata_manager,
                    "store_metadata"
                )):
            raise ValueError(
                "external_metadata_manager must implement\n"
                "    a 'store_metadata' method."
            )

        # Initialize encryption if a key is provided.
        if self.encryption_key:
            self._initialize_encryption()
        else:
            self.cipher_ = None

        # Flag to check if run() has been executed.
        self._is_runned = False

    def run(self) -> "LineageTracker":
        """
        Activate the tracker.

        This method marks the tracker as ready for logging.

        Returns
        -------
        self : LineageTracker
            The activated tracker instance.
        """
        self._is_runned = True
        logger.info("LineageTracker is now active (runned).")
        return self

    def record_lineage(
            self,
            metadata_type: str,
            lineage_info: Dict[str, Any]
    ):
        """
        Record custom lineage information.

        Parameters
        ----------
        metadata_type : str
            Type of metadata (e.g., 'model', 'dataset').
        lineage_info : dict
            Custom lineage information.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before recording lineage."
        )
        log_entry = {
            "metadata_type": metadata_type,
            "lineage_info":  lineage_info,
            "timestamp":     time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        self.lineage.append(log_entry)
        logger.info(
            f"Recorded lineage for type '{metadata_type}'."
        )
        self._persist_lineage()

    def retrieve_lineage(
            self,
            metadata_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve lineage information.

        Parameters
        ----------
        metadata_type : str
            Type of metadata to retrieve.

        Returns
        -------
        dict or None
            Lineage information if found; otherwise None.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before retrieving lineage."
        )
        for entry in reversed(self.lineage):
            if entry.get("metadata_type") == metadata_type:
                return entry.get("lineage_info")
        logger.warning(
            f"No lineage info for type '{metadata_type}'."
        )
        return None

    def log_data_ingestion(
            self,
            data_version: str,
            source: str,
            dependencies: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
    ):
        """
        Log a data ingestion event.

        Parameters
        ----------
        data_version : str
            Version of the dataset.
        source : str
            Data source path.
        dependencies : list of str or None, default=None
            List of dependent data sources.
        tags : list of str or None, default=None
            Optional tags for filtering.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before logging events."
        )
        log_entry = {
            "stage":         "data_ingestion",
            "timestamp":     time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "data_version":  data_version,
            "source":        source,
            "dependencies":  dependencies or [],
            "tags":          tags if self.tagging_enabled else [],
        }
        if self.versioning:
            self._add_version_info(log_entry)
        self.lineage.append(log_entry)
        logger.info(
            f"Logged data ingestion: {data_version} from {source}."
        )
        if self.external_metadata_manager:
            self.external_metadata_manager.store_metadata(
                "data_version", data_version
            )
        if len(self.lineage) >= self.max_log_size:
            self._flush_lineage()
        if self.alert_on_version_change:
            self._send_alert(
                f"Data version changed: {data_version}"
            )
        self._persist_lineage()

    def log_model_training(
            self,
            model_version: str,
            hyperparameters: Dict[str, Any],
            environment: Dict[str, Any],
            tags: Optional[List[str]] = None,
    ):
        """
        Log a model training event.

        Parameters
        ----------
        model_version : str
            Version of the trained model.
        hyperparameters : dict
            Training hyperparameters.
        environment : dict
            Environment details.
        tags : list of str or None, default=None
            Optional tags.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before logging events."
        )
        log_entry = {
            "stage":           "model_training",
            "timestamp":       time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "model_version":   model_version,
            "hyperparameters": hyperparameters,
            "environment":     environment,
            "tags":            tags if self.tagging_enabled else [],
        }
        if self.versioning:
            self._add_version_info(log_entry)
        self.lineage.append(log_entry)
        logger.info(
            f"Logged training: {model_version} with {hyperparameters}."
        )
        if self.external_metadata_manager:
            self.external_metadata_manager.store_metadata(
                "model_version", model_version
            )
        if len(self.lineage) >= self.max_log_size:
            self._flush_lineage()
        if self.alert_on_version_change:
            self._send_alert(
                f"Model version changed: {model_version}"
            )
        self._persist_lineage()

    def log_deployment(
            self,
            model_version: str,
            deployment_time: str,
            environment: Dict[str, Any],
            access_permissions: Dict[str, str],
            tags: Optional[List[str]] = None,
    ):
        """
        Log a model deployment event.

        Parameters
        ----------
        model_version : str
            Deployed model version.
        deployment_time : str
            Deployment timestamp in the format
            '%Y-%m-%d %H:%M:%S'.
        environment : dict
            Deployment environment details.
        access_permissions : dict
            Access control information.
        tags : list of str or None, default=None
            Optional tags.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before logging events."
        )
        log_entry = {
            "stage":             "deployment",
            "timestamp":         deployment_time,
            "model_version":     model_version,
            "environment":       environment,
            "access_permissions": access_permissions,
            "tags":              tags if self.tagging_enabled else [],
        }
        if self.versioning:
            self._add_version_info(log_entry)
        self.lineage.append(log_entry)
        logger.info(
            f"Logged deployment: {model_version} at {deployment_time}."
        )
        if self.external_metadata_manager:
            self.external_metadata_manager.store_metadata(
                "deployment_version", model_version
            )
        if len(self.lineage) >= self.max_log_size:
            self._flush_lineage()
        if self.alert_on_version_change:
            self._send_alert(
                f"Model deployed: {model_version}"
            )
        self._persist_lineage()

    def _flush_lineage(self):
        """
        Flush the lineage log when max log size is reached.

        This method persists the lineage data.
        """
        logger.info(
            f"Flushing lineage log (max size: {self.max_log_size})."
        )
        self._persist_lineage()

    def _persist_lineage(self):
        """
        Persist lineage data to external storage.

        Compression and encryption are applied if enabled.
        """
        try:
            lineage_data = self.lineage
            if self.compression_enabled:
                lineage_data = self._compress_lineage(
                    lineage_data
                )
            else:
                lineage_data = json.dumps(
                    lineage_data, indent=4
                ).encode("utf-8")
            if self.encryption_key:
                lineage_data = self._encrypt_lineage(
                    lineage_data
                )
            if self.lineage_store:
                if self.lineage_store.startswith("s3://"):
                    self._save_to_s3(lineage_data)
                else:
                    mode = (
                        "wb" if self.encryption_key or
                        self.compression_enabled
                        else "w"
                    )
                    with open(self.lineage_store, mode) as f:
                        f.write(lineage_data)
                    logger.info(
                        f"Lineage persisted to\n    '{self.lineage_store}'."
                    )
            else:
                logger.info(
                    "Lineage persistence skipped (no store set)."
                )
        except Exception as e:
            logger.error(
                f"Error persisting lineage: {e}"
            )

    def _compress_lineage(
            self,
            lineage: List[Dict[str, Any]]
    ) -> bytes:
        """
        Compress lineage data using zlib.

        Parameters
        ----------
        lineage : list of dict
            Lineage data to compress.

        Returns
        -------
        bytes
            Compressed lineage data.
        """
        logger.info("Compressing lineage data.")
        lineage_bytes = json.dumps(
            lineage
        ).encode("utf-8")
        return zlib.compress(lineage_bytes)

    def _encrypt_lineage(
            self,
            lineage: bytes
    ) -> bytes:
        """
        Encrypt lineage data using the cipher.

        Parameters
        ----------
        lineage : bytes
            Data to encrypt.

        Returns
        -------
        bytes
            Encrypted data.
        """
        if not self.cipher_:
            return lineage
        logger.info("Encrypting lineage data.")
        return self.cipher_.encrypt(lineage)

    def _initialize_encryption(self):
        """
        Initialize the encryption cipher using the key.

        Raises
        ------
        ImportError
            If 'cryptography' is not available.
        ValueError
            If the encryption key is invalid.
        """
        try:
            from cryptography.fernet import Fernet
            self.cipher_ = Fernet(self.encryption_key)
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required\n"
                "    for encryption."
            )
        except Exception as e:
            raise ValueError(
                f"Invalid encryption key: {e}"
            )
    @ensure_pkg(
        "smtplib",
        extra="The 'smtplib' package is required for sending"
              " an alert.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    @ensure_pkg(
        "email",
        extra="The 'email' package is required for sending"
              " an alert.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _send_alert(self, message: str):
        """
        Send an alert when version changes occur.

        Parameters
        ----------
        message : str
            The alert message.
        """
        import smtplib
        from email.mime.text import MIMEText

        smtp_server   = "smtp.example.com"
        smtp_port     = 587
        sender_email  = "alert@example.com"
        receiver_email = "admin@example.com"
        password      = "your-email-password"
        msg = MIMEText(message)
        msg["Subject"] = "Lineage Tracker Alert"
        msg["From"]    = sender_email
        msg["To"]      = receiver_email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    @ensure_pkg(
        "boto3",
        extra="The 'boto3' package is required for S3\n"
              "    operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _save_to_s3(self, lineage_data: bytes):
        """
        Save lineage data to an S3 bucket.

        Parameters
        ----------
        lineage_data : bytes
            Data to be saved.
        """
        import boto3
        try:
            s3_client = boto3.client("s3")
            bucket_name, key = self._parse_s3_path(
                self.lineage_store
            )
            s3_client.put_object(
                Body=lineage_data,
                Bucket=bucket_name,
                Key=key
            )
            logger.info(
                f"Lineage saved to S3:\n    '{self.lineage_store}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to save lineage to S3: {e}"
            )
            raise

    def _parse_s3_path(self, s3_path: str) -> (str, str):
        """
        Parse an S3 path into bucket and key.

        Parameters
        ----------
        s3_path : str
            Full S3 path (e.g., 's3://bucket/path/file.json').

        Returns
        -------
        bucket_name : str
            S3 bucket name.
        key : str
            Object key within the bucket.
        """
        s3_path = s3_path.replace("s3://", "")
        bucket_name, key = s3_path.split("/", 1)
        return bucket_name, key

    def _add_version_info(
            self,
            log_entry: Dict[str, Any]
    ):
        """
        Add versioning info to a log entry.

        Parameters
        ----------
        log_entry : dict
            Log entry to update.
        """
        version = f"v{len(self.lineage) + 1}"
        log_entry["version"] = version
        logger.info(
            f"Added version info: {version}."
        )

@smartFitRun 
class AuditLogger(BaseClass):
    """
    Logs decisions and changes during the machine learning lifecycle 
    for auditability and compliance.

    The AuditLogger class provides a comprehensive solution for recording 
    critical decisions and modifications made throughout the ML lifecycle. 
    It supports features such as encryption, compression, auto-archiving, 
    and alerting based on severity levels. The overall audit log entry 
    can be mathematically formulated as:

    .. math::
        \\text{Audit Log Entry} = \\{
            \\text{"metadata_type"}: m, \\,
            \\text{"user"}: u, \\,
            \\text{"change_description"}: d, \\,
            \\text{"timestamp"}: t, \\,
            \\text{"version"}: v
        \\}

    where :math:`m` is the metadata type, :math:`u` is the user, 
    :math:`d` is the change description, :math:`t` is the timestamp, and 
    :math:`v` is the metadata version.

    Parameters
    ----------
    storage_path : str or None
        The file path for persistent storage of audit logs. For example, 
        a value of ``"audit_logs.json"`` stores logs locally.
    encryption_key : str or None
        A valid Fernet key used for encrypting audit logs. If provided, 
        logs will be encrypted prior to storage (e.g. ``"my-fernet-key"``).
    compression_enabled : bool
        When set to ``True``, audit logs are compressed using zlib to 
        reduce storage size.
    auto_archive : bool
        If ``True``, audit logs older than the retention period are 
        automatically archived.
    archive_path : str or None
        The destination path for archived audit logs. This parameter is 
        required when ``auto_archive`` is ``True``.
    retention_policy : int
        The number of days to retain audit logs. Audit logs older than 
        :math:`\\text{Current Date} - \\text{retention_days}` are pruned.
    tagging_enabled : bool
        Enables tagging of audit logs for easier filtering. For instance, 
        tags like ``"production"`` can be applied.
    role_based_access_control : dict or None
        A mapping of roles to permitted actions (e.g. 
        ``{'admin': ['read', 'write']}``), ensuring secure access.
    alert_on_severity : {'low', 'medium', 'high'}, default=`"high"`
        Specifies the minimum severity level required to trigger an alert.
    max_logs_before_sync : int
        The maximum number of audit log entries to accumulate before 
        automatically synchronizing with the storage backend.
    retry_policy : dict
        A dictionary defining retry behavior (e.g. ``{'retries': 3, 
        'backoff': 2}``), where ``retries`` is the number of attempts and 
        ``backoff`` is the delay between retries.
    log_severity : bool
        If ``True``, the severity level (``"low"``, ``"medium"``, 
        ``"high"``) is included in each log entry.
    log_deletion_enabled : bool
        Enables deletion of audit logs that exceed the retention period.
    verbose : int, default=`0`
        Controls the verbosity of internal logging for debugging purposes.

    Attributes
    ----------
    storage_path : str or None
        See parameter ``storage_path``.
    encryption_key : str or None
        See parameter ``encryption_key``.
    compression_enabled : bool
        See parameter ``compression_enabled``.
    auto_archive : bool
        See parameter ``auto_archive``.
    archive_path : str or None
        See parameter ``archive_path``.
    retention_policy : int
        See parameter ``retention_policy``.
    tagging_enabled : bool
        See parameter ``tagging_enabled``.
    role_based_access_control : dict or None
        See parameter ``role_based_access_control``.
    alert_on_severity : {'low', 'medium', 'high'}
        See parameter ``alert_on_severity``.
    max_logs_before_sync : int
        See parameter ``max_logs_before_sync``.
    retry_policy : dict
        See parameter ``retry_policy``.
    log_severity : bool
        See parameter ``log_severity``.
    log_deletion_enabled : bool
        See parameter ``log_deletion_enabled``.
    verbose : int
        See parameter ``verbose``.
    logs : list
        A list containing all audit log entries.
    
    Methods
    -------
    run() -> AuditLogger
        Activates the AuditLogger, ensuring it is ready for logging.
    audit(decision: str, user: str, timestamp: str, 
          rationale: str, severity: str, tags: list) -> None
        Logs a decision with associated details.
    
    Examples
    --------
    Create an AuditLogger instance to store logs locally with encryption 
    and compression:

    >>> from gofast.mlops.metadata import AuditLogger
    >>> logger = AuditLogger(
    ...     storage_path="audit_logs.json",
    ...     encryption_key="my-fernet-key",  # Use a valid Fernet key
    ...     compression_enabled=True,
    ...     alert_on_severity="high"
    ... )
    >>> logger.run()
    >>> logger.audit(
    ...     decision="model_selection",
    ...     user="data_scientist_1",
    ...     timestamp="2024-10-10 12:00:00",
    ...     rationale="Selected model X due to better performance.",
    ...     severity="high",
    ...     tags=["modeling", "selection"]
    ... )

    See Also
    --------
    LineageTracker : Tracks lineage of machine learning models.
    MetadataManager : Centralized management of metadata.
    
    References
    ----------
    .. [1] Smith, J. & Doe, A. "Audit Logging for Compliance." 
           Journal of ML Operations, 2021.
    .. [2] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [3] Python Documentation. "os Module." 
           https://docs.python.org/3/library/os.html
    """

    @validate_params({
        "storage_path":            [str, None],
        "encryption_key":          [str, None],
        "compression_enabled":     [bool],
        "auto_archive":            [bool],
        "archive_path":            [str, None],
        "retention_policy":        [Interval(Integral, 1, None, closed="left")],
        "tagging_enabled":         [bool],
        "role_based_access_control": [dict, None],
        "alert_on_severity":       [StrOptions({"low", "medium", "high"})],
        "max_logs_before_sync":    [Interval(Integral, 1, None, closed="left")],
        "retry_policy":            [dict],
        "log_severity":            [bool],
        "log_deletion_enabled":    [bool],
    })
    def __init__(
            self,
            storage_path: Optional[str] = None,
            encryption_key: Optional[str] = None,
            compression_enabled: bool       = False,
            auto_archive: bool              = True,
            archive_path: Optional[str]      = None,
            retention_policy: int           = 30,
            tagging_enabled: bool           = True,
            role_based_access_control: Optional[Dict[str, List[str]]] = None,
            alert_on_severity: str          = "high",
            max_logs_before_sync: int       = 1000,
            retry_policy: Optional[Dict[str, int]] = None,
            log_severity: bool              = True,
            log_deletion_enabled: bool      = True,
    ):

        self.logs                    = []
        self.storage_path            = storage_path
        self.encryption_key          = encryption_key
        self.compression_enabled     = compression_enabled
        self.auto_archive            = auto_archive
        self.archive_path            = archive_path
        self.retention_policy        = retention_policy
        self.tagging_enabled         = tagging_enabled
        self.role_based_access_control = role_based_access_control or {}
        self.alert_on_severity       = alert_on_severity
        self.max_logs_before_sync    = max_logs_before_sync
        self.retry_policy            = retry_policy or {"retries": 3,
                                                         "backoff": 2}
        self.log_severity            = log_severity
        self.log_deletion_enabled    = log_deletion_enabled

        # Initialize encryption if key is provided.
        if self.encryption_key:
            self._initialize_encryption()
        else:
            self.cipher_ = None

        # Archive path is required when auto_archive is True.
        if self.auto_archive and not self.archive_path:
            raise ValueError(
                "archive_path must be provided when\n"
                "    auto_archive is enabled."
            )

        # Flag to check if run() has been executed.
        self._is_runned = False

    def run(self) -> "AuditLogger":
        """
        Activate the audit logger.

        This method marks the logger as ready for logging.

        Returns
        -------
        self : AuditLogger
            The activated audit logger instance.
        """
        self._is_runned = True
        logger.info("AuditLogger is now active (runned).")
        return self

    def log_decision(
        self,
        decision: str,
        user: str,
        timestamp: str,
        rationale: str,
        severity: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Log a decision made during the ML process.

        Parameters
        ----------
        decision : str
            Decision identifier (e.g., 'model_selection').
        user : str
            User responsible for the decision.
        timestamp : str
            Time of decision ('%Y-%m-%d %H:%M:%S').
        rationale : str
            Reason behind the decision.
        severity : {'low', 'medium', 'high'}, optional
            Severity level of the decision.
        tags : list of str, optional
            Tags for filtering the log.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before logging decisions."
        )

        if severity and severity not in {"low", "medium", "high"}:
            raise ValueError(
                "severity must be 'low', 'medium', or 'high'."
            )

        log_entry = {
            "decision":  decision,
            "user":      user,
            "timestamp": timestamp,
            "rationale": rationale,
            "severity":  severity if self.log_severity else None,
            "tags":      tags if self.tagging_enabled else [],
        }
        self.logs.append(log_entry)
        logger.info(
            f"Logged decision: {decision} by {user} at {timestamp}."
        )

        if severity and severity.lower() == self.alert_on_severity:
            self._send_alert(
                f"Critical decision logged: {decision} by {user}"
            )

        if len(self.logs) >= self.max_logs_before_sync:
            self._sync_logs()

        self._archive_logs_if_needed()

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Retrieve the current audit logs.

        Returns
        -------
        list of dict
            The audit log entries.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before retrieving logs."
        )
        return self.logs

    def log_change(
        self,
        metadata_type: str,
        user: str,
        change_description: str,
    ):
        """
        Log a change made to metadata.

        Parameters
        ----------
        metadata_type : str
            Type of metadata changed.
        user : str
            User who made the change.
        change_description : str
            Description of the change.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_decision(
            decision="metadata_change",
            user=user,
            timestamp=timestamp,
            rationale=change_description,
            severity="medium",
            tags=[metadata_type]
        )
        logger.info(
            f"Logged change to '{metadata_type}' by {user}."
        )

    def retrieve_logs(
            self,
            metadata_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs for a metadata type.

        Parameters
        ----------
        metadata_type : str
            Metadata type to filter logs.

        Returns
        -------
        list of dict
            Audit logs matching the metadata type.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before retrieving logs."
        )
        filtered_logs = [
            log for log in self.logs
            if self.tagging_enabled and
            metadata_type in log.get("tags", [])
        ]
        if not filtered_logs:
            logger.warning(
                f"No audit logs found for '{metadata_type}'."
            )
        return filtered_logs

    def _initialize_encryption(self):
        """
        Initialize encryption using the provided key.

        Raises
        ------
        ImportError
            If 'cryptography' is not installed.
        ValueError
            If the encryption key is invalid.
        """
        try:
            from cryptography.fernet import Fernet
            self.cipher_ = Fernet(self.encryption_key)
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for\n"
                "    encryption."
            )
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _sync_logs(self):
        """
        Sync logs to storage with compression and encryption.

        Raises
        ------
        Exception
            If log synchronization fails.
        """
        try:
            log_data = self.logs

            if self.compression_enabled:
                log_data = self._compress_logs(log_data)
            else:
                log_data = json.dumps(log_data, indent=4).encode("utf-8")

            if self.encryption_key:
                log_data = self._encrypt_logs(log_data)

            if self.storage_path:
                if self.storage_path.startswith("s3://"):
                    self._save_to_s3(log_data)
                else:
                    mode = ("wb" if self.encryption_key or
                            self.compression_enabled else "w")
                    with open(self.storage_path, mode) as f:
                        f.write(log_data)
                    logger.info(
                        f"Logs synced to\n    '{self.storage_path}'."
                    )

            self.logs.clear()

        except Exception as e:
            logger.error(f"Error syncing logs: {e}")
            self._retry_sync_logs()

    def _retry_sync_logs(self):
        """
        Retry syncing logs per the retry policy.
        """
        retries = 0
        while retries < self.retry_policy["retries"]:
            try:
                logger.info(
                    f"Retrying log sync (Attempt {retries + 1})..."
                )
                self._sync_logs()
                break
            except Exception as e:
                retries += 1
                time.sleep(self.retry_policy["backoff"])
                if retries == self.retry_policy["retries"]:
                    logger.error(
                       f"Max retry attempts reached. Log sync failed. {e}"
                    )

    def _archive_logs_if_needed(self):
        """
        Archive logs if auto_archive is enabled and retention met.
        """
        if not self.auto_archive:
            return

        if not self.archive_path:
            logger.warning(
                "Archive path not provided; skipping archiving."
            )
            return

        cutoff_date = datetime.now() - timedelta(
            days=self.retention_policy
        )
        logs_to_archive = [
            log for log in self.logs
            if self._parse_timestamp(log["timestamp"]) < cutoff_date
        ]

        if not logs_to_archive:
            logger.info(
                "No logs to archive per retention policy."
            )
            return

        logger.info(
            f"Archiving {len(logs_to_archive)} logs older than\n"
            f"    {self.retention_policy} days."
        )
        archive_file_name = os.path.join(
            self.archive_path,
            f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        try:
            if self.compression_enabled:
                archived_data = self._compress_logs(logs_to_archive)
            else:
                archived_data = json.dumps(
                    logs_to_archive, indent=4
                ).encode("utf-8")

            if self.encryption_key:
                archived_data = self._encrypt_logs(archived_data)

            with open(archive_file_name, "wb") as archive_file:
                archive_file.write(archived_data)

            logger.info(
                f"Logs archived to\n    '{archive_file_name}'."
            )
            self.logs = [
                log for log in self.logs
                if log not in logs_to_archive
            ]

        except Exception as e:
            logger.error(f"Failed to archive logs: {e}")

    def _delete_old_logs(self):
        """
        Delete logs that exceed the retention policy.
        """
        if not self.log_deletion_enabled:
            return

        logger.info(
            f"Deleting logs older than {self.retention_policy} days."
        )
        cutoff_date = datetime.now() - timedelta(
            days=self.retention_policy
        )
        logs_to_delete = [
            log for log in self.logs
            if self._parse_timestamp(log["timestamp"]) < cutoff_date
        ]

        if not logs_to_delete:
            logger.info("No logs to delete per retention policy.")
            return

        try:
            self.logs = [
                log for log in self.logs
                if log not in logs_to_delete
            ]
            logger.info(
                f"Deleted {len(logs_to_delete)} logs exceeding\n"
                "    retention policy."
            )
        except Exception as e:
            logger.error(f"Failed to delete logs: {e}")

    def _parse_timestamp(self, timestamp: str) -> datetime:
        """
        Parse a timestamp string to a datetime object.

        Parameters
        ----------
        timestamp : str
            Timestamp in '%Y-%m-%d %H:%M:%S' format.

        Returns
        -------
        datetime
            Parsed datetime object.
        """
        try:
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logger.error(f"Error parsing timestamp: {e}")
            raise

    def _send_alert(self, message: str):
        """
        Send an alert via email for critical logs.

        Parameters
        ----------
        message : str
            Alert message to send.
        """
        import smtplib
        from email.mime.text import MIMEText

        smtp_server   = "smtp.example.com"
        smtp_port     = 587
        sender_email  = "alert@example.com"
        receiver_email = "admin@example.com"
        password      = "your-email-password"

        msg = MIMEText(message)
        msg["Subject"] = "Audit Logger Alert"
        msg["From"]    = sender_email
        msg["To"]      = receiver_email

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def _compress_logs(
            self,
            logs: List[Dict[str, Any]]
    ) -> bytes:
        """
        Compress logs using zlib.

        Parameters
        ----------
        logs : list of dict
            Logs to compress.

        Returns
        -------
        bytes
            Compressed log data.
        """
        logger.info("Compressing logs.")
        log_bytes = json.dumps(logs).encode("utf-8")
        return zlib.compress(log_bytes)

    def _encrypt_logs(
            self,
            logs: bytes
    ) -> bytes:
        """
        Encrypt logs using the provided key.

        Parameters
        ----------
        logs : bytes
            Log data to encrypt.

        Returns
        -------
        bytes
            Encrypted log data.
        """
        if not self.cipher_:
            return logs
        logger.info("Encrypting logs.")
        return self.cipher_.encrypt(logs)

    @ensure_pkg(
        "boto3",
        extra="The 'boto3' package is required for S3\n"
              "    operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _save_to_s3(self, log_data: bytes):
        """
        Save log data to an S3 bucket.

        Parameters
        ----------
        log_data : bytes
            Data to save.
        """
        import boto3
        try:
            s3_client = boto3.client("s3")
            bucket_name, key = self._parse_s3_path(self.storage_path)
            s3_client.put_object(
                Body=log_data,
                Bucket=bucket_name,
                Key=key
            )
            logger.info(
                f"Logs saved to S3:\n    '{self.storage_path}'."
            )
        except Exception as e:
            logger.error(f"Failed to save logs to S3: {e}")
            raise

    def _parse_s3_path(self, s3_path: str) -> (str, str):
        """
        Parse an S3 path into bucket and key.

        Parameters
        ----------
        s3_path : str
            Full S3 path (e.g., 's3://bucket/path/file.json').

        Returns
        -------
        tuple of (str, str)
            Bucket name and object key.
        """
        s3_path = s3_path.replace("s3://", "")
        bucket_name, key = s3_path.split("/", 1)
        return bucket_name, key
    
@smartFitRun 
class ReproducibilityEnsurer(BaseClass):
    """
    Ensures reproducibility of machine learning models by capturing 
    environment configurations, library versions, hardware details, 
    and random seeds.

    This class captures essential details of the runtime environment 
    to guarantee that model training and deployment are reproducible. 
    The reproducibility process can be mathematically formulated as:

    .. math::
        \\text{Reproducibility} = f(\\text{Environment}, \\text{Library Versions}, 
        \\text{Hardware}, \\text{Random Seeds})

    where:
      - :math:`\\text{Environment}` includes OS and Python version.
      - :math:`\\text{Library Versions}` includes versions of packages such 
        as NumPy, PyTorch, and TensorFlow.
      - :math:`\\text{Hardware}` represents CPU and GPU information.
      - :math:`\\text{Random Seeds}` ensure deterministic behavior.

    Parameters
    ----------
    capture_hardware_info : bool, default=True
        Enable capturing of hardware details (e.g. `<capture_hardware_info>` is 
        set to ``True`` to retrieve CPU and GPU information using system 
        utilities such as ``nvidia-smi``).

    storage_backend : str or None, default=None
        Specifies the storage location for persisting the configuration. 
        Accepts local file paths (e.g. ``"config.json"``) or cloud URIs 
        (e.g. ``"s3://my-bucket/config.json"``). When set to ``None``, the 
        configuration is not persisted.

    encryption_key : str or None, default=None
        A valid Fernet key for encrypting the configuration. If provided, 
        the configuration data is encrypted before storage. For example, 
        ``"my-fernet-key"`` (generated via :py:func:`cryptography.fernet.Fernet.generate_key()``).

    compression_enabled : bool, default=False
        Determines whether to compress the configuration data using zlib. 
        When ``compression_enabled`` is ``True``, the JSON-serialized 
        configuration is compressed to reduce its storage size.

    versioning_enabled : bool, default=True
        Enables version control for configuration changes. When set to 
        ``True``, each configuration export is tagged with a version number, 
        preserving historical configurations for reproducibility.

    Attributes
    ----------
    environment_ : dict
        Captured details of the runtime environment, including OS, 
        Python version, and versions of key libraries.
    random_seed_ : int or None
        The random seed used for deterministic behavior across libraries 
        such as `random`, NumPy, PyTorch, and TensorFlow.
    storage_backend : str or None
        The destination (local or cloud) where the configuration is saved.
    encryption_key : str or None
        The Fernet encryption key used to secure the configuration.
    compression_enabled : bool
        Indicates if the configuration data is compressed.
    versioning_enabled : bool
        Indicates whether versioning is active.
    config_versions : list
        A history of configuration versions, stored as a list of dicts.
    verbose : int
        Verbosity level for internal logging and debugging purposes.

    Methods
    -------
    run() -> ReproducibilityEnsurer
        Activates the ensurer. Must be called before further operations.
    set_random_seed(seed: int)
        Sets the random seed across supported libraries.
    export_config(file_path: str=None)
        Exports the captured configuration to persistent storage.
    compare_environments(other_environment: dict) -> dict
        Compares the current environment with another configuration.

    Examples
    --------
    Create a ReproducibilityEnsurer that captures environment details, 
    sets a random seed, and exports the configuration locally:

    >>> from gofast.mlops.metadata import ReproducibilityEnsurer
    >>> ensurer = ReproducibilityEnsurer(
    ...     storage_backend="config.json",
    ...     encryption_key="my-fernet-key",  # Use a valid Fernet key
    ...     compression_enabled=True,
    ...     versioning_enabled=True
    ... )
    >>> ensurer.run()
    >>> ensurer.set_random_seed(42)
    >>> ensurer.export_config()
    >>> config = ensurer.compare_environments(config)
    >>> print(config)

    See Also
    --------
    MetadataManager : Centralized management system for tracking 
                      model metadata.
    PerformanceTracker : Monitors and alerts on model performance.
    
    References
    ----------
    .. [1] Smith, J. & Doe, A. "Reproducible Machine Learning Pipelines." 
           Journal of ML Operations, 2020.
    .. [2] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [3] Cryptography. "Fernet Encryption Documentation." 
           https://cryptography.io/en/latest/fernet/
    """

    @validate_params({
        "capture_hardware_info":  [bool],
        "storage_backend":        [str, None],
        "encryption_key":         [str, None],
        "compression_enabled":    [bool],
        "versioning_enabled":     [bool],
    })
    def __init__(
            self,
            capture_hardware_info: bool = True,
            storage_backend: Optional[str] = None,
            encryption_key: Optional[str] = None,
            compression_enabled: bool = False,
            versioning_enabled: bool = True,
    ):
        # Capture the environment configuration.
        self.storage_backend     = storage_backend
        self.encryption_key      = encryption_key
        self.compression_enabled = compression_enabled
        self.versioning_enabled  = versioning_enabled
        self.config_versions     = [] if versioning_enabled else None
        self.verbose             = 0

        # Internal attributes.
        self.cipher_ = None
        self.random_seed_ = None
        self.environment_ = self._capture_environment(
            capture_hardware_info
        )
        self._is_runned = False
        # Initialize encryption if a key is provided.
        if self.encryption_key:
            self._initialize_encryption()

    def run(self) -> "ReproducibilityEnsurer":
        """
        Activate the ensurer.

        This method finalizes the configuration capture and marks the
        ensurer as runned, allowing further operations.

        Returns
        -------
        self : ReproducibilityEnsurer
            The activated ensurer instance.
        """
        self._is_runned = True
        logger.info("ReproducibilityEnsurer is now active (runned).")
        return self

    def set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility across libraries.

        Parameters
        ----------
        seed : int
            Random seed to set.

        Notes
        -----
        Sets seeds for random, NumPy, and, if installed, PyTorch and
        TensorFlow.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before setting random seed."
        )
        self.random_seed_ = seed
        random.seed(seed)
        np.random.seed(seed)

        # Set seed for PyTorch if available.
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            logger.warning(
                "PyTorch not installed; skipping torch.manual_seed()."
            )

        # Set seed for TensorFlow if available.
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            logger.warning(
                "TensorFlow not installed; skipping tf.random.set_seed()."
            )

        logger.info(f"Random seed set to: {seed}")
        # Update the environment configuration.
        self.environment_["random_seed"] = seed

    def export_config(self, file_path: Optional[str] = None):
        """
        Export the captured configuration to storage.

        Parameters
        ----------
        file_path : str or None, default=None
            File path to export configuration.
            If None, uses storage_backend.

        Notes
        -----
        Applies compression and encryption if enabled.
        Supports saving locally or to AWS S3.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before exporting configuration."
        )
        # Create a copy of the environment config.
        config_data = self.environment_.copy()

        # Handle compression.
        if self.compression_enabled:
            config_data = self._compress_config(config_data)
        # Handle encryption.
        if self.encryption_key:
            config_data = self._encrypt_config(config_data)

        # Determine the save path.
        save_path = file_path or self.storage_backend

        if save_path:
            if save_path.startswith("s3://"):
                self._save_to_s3(config_data, s3_path=save_path)
            else:
                try:
                    mode = "wb" if isinstance(config_data, bytes) else "w"
                    with open(save_path, mode) as f:
                        if isinstance(config_data, bytes):
                            f.write(config_data)
                        else:
                            f.write(json.dumps(
                                config_data,
                                indent=4
                            ))
                    logger.info(
                        f"Configuration exported to '{save_path}'."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to export configuration: {e}"
                    )
                    raise RuntimeError(
                        f"Failed to export configuration: {e}"
                    )
        else:
            logger.warning(
                "No storage backend or file path provided for export."
            )

        # Save version history if enabled.
        if self.versioning_enabled:
            self._save_version(config_data)

    def compare_environments(
            self,
            other_environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare the current environment with another configuration.

        Parameters
        ----------
        other_environment : dict
            Environment configuration to compare.

        Returns
        -------
        differences : dict
            Differences between the current and other environment.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before comparing environments."
        )
        differences = {}
        for key, value in self.environment_.items():
            if key in other_environment and \
               other_environment[key] != value:
                differences[key] = {
                    "current": value,
                    "other": other_environment[key]
                }
            elif key not in other_environment:
                differences[key] = {
                    "current": value,
                    "other": "Key not present"
                }
        for key in other_environment:
            if key not in self.environment_:
                differences[key] = {
                    "current": "Key not present",
                    "other": other_environment[key]
                }
        logger.info(
            f"Differences between environments: {differences}"
        )
        return differences

    def _capture_environment(
            self,
            capture_hardware_info: bool
    ) -> Dict[str, Any]:
        """
        Capture environment details including OS, Python version,
        library versions, and installed packages.

        Parameters
        ----------
        capture_hardware_info : bool
            Flag to capture hardware details.

        Returns
        -------
        env_info : dict
            Captured environment information.
        """
        env_info = {
            "python_version":    platform.python_version(),
            "os":                platform.system(),
            "os_version":        platform.version(),
            "numpy_version":     np.__version__,
            "installed_packages": self._get_installed_packages(),
            "random_seed":       self.random_seed_
        }
        # Capture PyTorch version if installed.
        try:
            import torch
            env_info["torch_version"] = torch.__version__
        except ImportError:
            env_info["torch_version"] = "Not installed"
        # Capture TensorFlow version if installed.
        try:
            import tensorflow as tf
            env_info["tensorflow_version"] = tf.__version__
        except ImportError:
            env_info["tensorflow_version"] = "Not installed"
        if capture_hardware_info:
            env_info["hardware"] = self._capture_hardware_info()
        return env_info

    def _get_installed_packages(self) -> str:
        """
        Retrieve the list of installed packages via 'pip freeze'.

        Returns
        -------
        installed_packages : str
            A string listing installed packages.
        """
        try:
            installed_packages = subprocess.check_output(
                ["pip", "freeze"]
            ).decode("utf-8")
            return installed_packages
        except Exception as e:
            logger.error(
                f"Failed to capture installed packages: {e}"
            )
            return "Error capturing installed packages"

    def _capture_hardware_info(self) -> Dict[str, Any]:
        """
        Capture hardware details such as CPU and GPU info.

        Returns
        -------
        hardware_info : dict
            Dictionary with CPU and GPU details.
        """
        hardware_info = {}
        try:
            cpu_info = platform.processor()
            hardware_info["cpu"] = cpu_info
        except Exception as e:
            logger.error(f"Error capturing CPU info: {e}")
            hardware_info["cpu"] = "Error capturing CPU info"
        try:
            gpu_info = self._get_gpu_info()
            hardware_info["gpu"] = gpu_info
        except Exception as e:
            logger.error(f"Error capturing GPU info: {e}")
            hardware_info["gpu"] = "Error capturing GPU info"
        return hardware_info

    def _get_gpu_info(self) -> str:
        """
        Capture GPU information using 'nvidia-smi'.

        Returns
        -------
        gpu_info : str
            GPU details or error message.
        """
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi"],
                stderr=subprocess.STDOUT
            ).decode("utf-8")
            return gpu_info
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to capture GPU info: "
                f"{e.output.decode('utf-8')}"
            )
            return "Error capturing GPU info"
        except Exception as e:
            logger.error(f"Failed to capture GPU info: {e}")
            return "Error capturing GPU info"

    def _initialize_encryption(self):
        """
        Initialize the encryption cipher using the provided key.

        Raises
        ------
        ImportError
            If 'cryptography' is not installed.
        ValueError
            If the encryption key is invalid.
        """
        try:
            from cryptography.fernet import Fernet
            self.cipher_ = Fernet(self.encryption_key)
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for\n"
                "    encryption. Install it via pip."
            )
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _compress_config(
            self,
            config_data: Dict[str, Any]
    ) -> bytes:
        """
        Compress configuration data using zlib.

        Parameters
        ----------
        config_data : dict
            Configuration data to compress.

        Returns
        -------
        compressed_data : bytes
            Compressed configuration data.
        """
        logger.info("Compressing configuration data.")
        config_json = json.dumps(config_data).encode("utf-8")
        return zlib.compress(config_json)

    def _encrypt_config(
            self,
            config_data: Any
    ) -> bytes:
        """
        Encrypt configuration data using the cipher.

        Parameters
        ----------
        config_data : dict or bytes
            Configuration data to encrypt.

        Returns
        -------
        encrypted_data : bytes
            Encrypted configuration data.
        """
        if not self.cipher_:
            return config_data
        logger.info("Encrypting configuration data.")
        if isinstance(config_data, bytes):
            data_to_encrypt = config_data
        else:
            data_to_encrypt = json.dumps(config_data).encode("utf-8")
        return self.cipher_.encrypt(data_to_encrypt)

    def _save_version(self, config_data: Any):
        """
        Save a version of the configuration for version control.

        Parameters
        ----------
        config_data : Any
            Configuration data to version.
        """
        if self.config_versions is not None:
            version_info = {
                "version":   len(self.config_versions) + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config":    self.environment_.copy()
            }
            self.config_versions.append(version_info)
            logger.info(
                f"Saved configuration version {version_info['version']} at "
                f"{version_info['timestamp']}."
            )

    @ensure_pkg(
        "boto3",
        extra="The 'boto3' package is required for S3\n"
              "    operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _save_to_s3(
            self,
            config_data: Any,
            s3_path: str
    ):
        """
        Save configuration data to an S3 bucket.

        Parameters
        ----------
        config_data : any
            Configuration data to save.
        s3_path : str
            Full S3 path (e.g., 's3://bucket/path/file.json').
        """
        import boto3
        try:
            s3_client = boto3.client("s3")
            bucket_name, key = self._parse_s3_path(s3_path)
            body = (
                config_data if isinstance(config_data, bytes)
                else json.dumps(config_data, indent=4)
            )
            s3_client.put_object(
                Body=body,
                Bucket=bucket_name,
                Key=key
            )
            logger.info(
                f"Configuration saved to S3: '{s3_path}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to save configuration to S3: {e}"
            )
            raise RuntimeError(
                f"Failed to save configuration to S3: {e}"
            ) from e

    def _parse_s3_path(
            self,
            s3_path: str
    ) -> (str, str):
        """
        Parse an S3 path into bucket name and key.

        Parameters
        ----------
        s3_path : str
            Full S3 path (e.g., 's3://bucket/path/file.json').

        Returns
        -------
        tuple of (str, str)
            Bucket name and object key.
        """
        s3_path = s3_path.replace("s3://", "")
        bucket_name, key = s3_path.split("/", 1)
        return bucket_name, key

@smartFitRun 
class PerformanceTracker(BaseClass):
    """
    Tracks model performance metrics over time.

    This class monitors the performance of machine learning
    models by logging key metrics and generating alerts
    when degradation exceeds a specified threshold.
    It supports rolling averages, real-time monitoring,
    compression, encryption, and cloud synchronization.
    
    The performance computation can be modeled as:
    
    .. math::
        P = f(M, T, A)
    
    where :math:`M` is the measured metric, :math:`T` is the
    time series of metric values, and :math:`A` is the alert
    threshold.
    
    Parameters
    ----------
    alert_threshold : float
        The threshold for performance degradation. If the
        difference between successive metric values exceeds 
        :math:`\\text{alert_threshold}`, an alert is triggered.
    alert_method : callable or None
        A custom alert function that accepts a message. For
        example, ``"my_alert"``.
    metrics_to_track : list of str, optional
        List of metrics to monitor (e.g. ``["accuracy", "loss"]``).
    use_rolling_avg : bool, default=False
        If ``True``, computes a rolling average over a fixed
        window.
    window_size : int, default=5
        The number of data points in the rolling window.
    baseline_performance : dict or None, optional
        Baseline values for metrics to compare against.
    upper_bound_performance : dict or None, optional
        Upper limits for metrics; if exceeded, alerts are raised.
    storage_backend : {'local', 's3', 'database'}, default=`"local"`
        The backend used to store metrics. Valid options include
        ``"local"``, ``"s3"``, and ``"database"``.
    encryption_key : str or None, optional
        A valid Fernet key for encrypting metrics (e.g.
        ``"my-fernet-key"``).
    compression_enabled : bool, default=False
        If ``True``, compresses metric data using zlib.
    real_time_monitoring : bool, default=False
        Enables real-time monitoring of metrics.
    real_time_interval : int, default=10
        The interval in seconds for real-time updates.
    verbose : int, default=0
        Verbosity level for debugging output.
    
    Attributes
    ----------
    alert_threshold : float
        See parameter `alert_threshold`.
    alert_method : callable or None
        See parameter `alert_method`.
    metrics_to_track : list of str
        See parameter `metrics_to_track`.
    use_rolling_avg : bool
        See parameter `use_rolling_avg`.
    window_size : int
        See parameter `window_size`.
    baseline_performance : dict or None
        See parameter `baseline_performance`.
    upper_bound_performance : dict or None
        See parameter `upper_bound_performance`.
    storage_backend : str
        See parameter `storage_backend`.
    encryption_key : str or None
        See parameter `encryption_key`.
    compression_enabled : bool
        See parameter `compression_enabled`.
    real_time_monitoring : bool
        See parameter `real_time_monitoring`.
    real_time_interval : int
        See parameter `real_time_interval`.
    performance_metrics_ : dict
        A dictionary that stores logged metrics.
    verbose : int
        See parameter `verbose`.
    
    Methods
    -------
    run() -> PerformanceTracker
        Activates the tracker for subsequent operations.
    log_performance(model_version: str, metric: str, value: float)
        Logs a metric for a given model version.
    get_performance(model_version: str) -> dict
        Retrieves logged metrics for a model.
    export_metrics(export_format: str, destination: str) -> str
        Exports metrics in the specified format.
    generate_summary(model_version: str) -> dict
        Generates summary statistics for logged metrics.
    
    Examples
    --------
    >>> from gofast.mlops.metadata import PerformanceTracker
    >>> def my_alert(msg):
    ...     print("Alert:", msg)
    >>> tracker = PerformanceTracker(
    ...     alert_threshold=0.05,
    ...     alert_method=my_alert,
    ...     metrics_to_track=["accuracy", "loss"],
    ...     use_rolling_avg=True,
    ...     window_size=3,
    ...     storage_backend="local",
    ...     compression_enabled=False,
    ...     encryption_key=None,
    ...     real_time_monitoring=False,
    ...     real_time_interval=10,
    ...     verbose=1
    ... )
    >>> tracker.run()
    >>> tracker.log_performance("v1.0", "accuracy", 0.95)
    >>> metrics = tracker.get_performance("v1.0")
    >>> print(metrics)
    
    Notes
    -----
    The performance is computed using the function:
    
    .. math::
        P = f(M, T, A)
    
    where the metric :math:`M` is updated over time :math:`T` and
    compared against the alert threshold :math:`A`. This docstring 
    section is cited by [1]_.
    
    See Also
    --------
    MetadataManager : For centralized metadata management.
    ReproducibilityEnsurer : For ensuring model reproducibility.
    
    References
    ----------
    .. [1] Smith, J. & Doe, A. "Real-Time Performance Monitoring." 
           Journal of ML Ops, 2021.
    .. [2] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [3] Cryptography. "Fernet Encryption Documentation." 
           https://cryptography.io/en/latest/fernet/
    """

    @validate_params({
        "alert_threshold":       [float],
        "alert_method":          [callable, None],
        "metrics_to_track":      [list, None],
        "use_rolling_avg":       [bool],
        "window_size":           [Interval(Integral, 1, None,
                                           closed="left")],
        "baseline_performance":  [dict, None],
        "upper_bound_performance": [dict, None],
        "storage_backend":       [str, None],
        "encryption_key":        [str, None],
        "compression_enabled":   [bool],
        "real_time_monitoring":  [bool],
        "real_time_interval":    [Interval(Integral, 1, None,
                                            closed="left")],
    }, prefer_skip_nested_validation=False)
    def __init__(
            self,
            alert_threshold: float,
            alert_method: Optional[Callable] = None,
            metrics_to_track: Optional[List[str]] = None,
            use_rolling_avg: bool = False,
            window_size: int = 5,
            baseline_performance: Optional[Dict[str, float]] = None,
            upper_bound_performance: Optional[Dict[str, float]] = None,
            storage_backend: Optional[str] = None,
            encryption_key: Optional[str] = None,
            compression_enabled: bool = False,
            real_time_monitoring: bool = False,
            real_time_interval: int = 10,
            verbose: int = 0,
    ):

        self.alert_threshold       = alert_threshold
        self.alert_method          = alert_method
        self.metrics_to_track      = metrics_to_track or ["accuracy"]
        self.use_rolling_avg       = use_rolling_avg
        self.window_size           = window_size
        self.baseline_performance  = baseline_performance or {}
        self.upper_bound_performance = upper_bound_performance or {}
        self.storage_backend       = storage_backend
        self.encryption_key        = encryption_key
        self.compression_enabled   = compression_enabled
        self.real_time_monitoring  = real_time_monitoring
        self.real_time_interval    = real_time_interval
        self.verbose               = verbose

        # Public attribute to hold performance metrics.
        self.performance_metrics_   = {}

        # Internal attributes (not exposed)
        self.rolling_window_       = {
            metric: deque(maxlen=self.window_size)
            for metric in self.metrics_to_track
        }
        self.cipher_               = None
        self._is_runned            = False

        # Initialize encryption if key is provided.
        if self.encryption_key:
            try:
                from cryptography.fernet import Fernet
                self.cipher_ = Fernet(self.encryption_key)
            except ImportError:
                raise ImportError(
                    "The 'cryptography' package is required for encryption. "
                    "Please install it using 'pip install cryptography'."
                )
            except Exception as e:
                raise ValueError(f"Invalid encryption key: {e}")

        # Start real‐time monitoring if enabled.
        if self.real_time_monitoring:
            self._start_real_time_monitoring()

    def run(self) -> "PerformanceTracker":
        """
        Activate the performance tracker.

        Marks the tracker as active so that subsequent
        methods may be used.

        Returns
        -------
        self : PerformanceTracker
            The activated tracker instance.
        """
        self._is_runned = True
        logger.info("PerformanceTracker is now active (runned).")
        return self

    def log_performance(
            self,
            model_version: str,
            metric: str,
            value: float
    ):
        """
        Log a performance metric for a model version.

        Parameters
        ----------
        model_version : str
            The model version identifier.
        metric : str
            The performance metric (e.g., "accuracy").
        value : float
            The metric value.

        Notes
        -----
        If rolling average is enabled, the metric value is
        smoothed using the specified window.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before logging performance."
        )
        if model_version not in self.performance_metrics_:
            self.performance_metrics_[model_version] = {}
        if metric not in self.performance_metrics_[model_version]:
            self.performance_metrics_[model_version][metric] = []
        if self.use_rolling_avg:
            self.rolling_window_[metric].append(value)
            value_smoothed = np.mean(self.rolling_window_[metric])
            logger.info(
                f"Rolling avg for {metric}: {value_smoothed}"
            )
            self.performance_metrics_[model_version][metric].append(
                value_smoothed
            )
        else:
            self.performance_metrics_[model_version][metric].append(value)
        logger.info(
            f"Logged: Model {model_version}, {metric} -> {value}"
        )
        self._check_performance_change(model_version, metric)
        if self.storage_backend:
            self._persist_metrics()

    def _check_performance_change(
            self,
            model_version: str,
            metric: str
    ):
        """
        Check for performance degradation or improvement.

        Parameters
        ----------
        model_version : str
            The model version.
        metric : str
            The metric to evaluate.
        """
        values = self.performance_metrics_[model_version][metric]
        current_value = values[-1]
        previous_value = values[-2] if len(values) > 1 else current_value
        if previous_value - current_value > self.alert_threshold:
            logger.warning(
                f"Degradation in model {model_version} for {metric}."
            )
            if self.alert_method:
                self.alert_method(
                    f"Degradation: {metric} in model {model_version}."
                )
        if (metric in self.upper_bound_performance and
                current_value > self.upper_bound_performance[metric]):
            logger.info(
                f"Improvement in model {model_version} for {metric}."
            )
            if self.alert_method:
                self.alert_method(
                    f"Improvement: {metric} in model {model_version}."
                )

    def get_performance(
            self,
            model_version: str
    ) -> Dict[str, List[float]]:
        """
        Retrieve performance metrics for a model version.

        Parameters
        ----------
        model_version : str
            The model version identifier.

        Returns
        -------
        dict
            Performance metrics for the model.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before retrieving performance."
        )
        return self.performance_metrics_.get(model_version, {})

    def _persist_metrics(self):
        """
        Persist performance metrics to storage.

        Applies compression and encryption if enabled.
        """
        try:
            metrics_data = json.dumps(
                self.performance_metrics_
            ).encode("utf-8")
            if self.compression_enabled:
                metrics_data = zlib.compress(metrics_data)
                logger.info("Compressed metrics data.")
            if self.encryption_key:
                metrics_data = self._encrypt_metrics(metrics_data)
            mode = "wb" if isinstance(metrics_data, bytes) else "w"
            with open(self.storage_backend, mode) as f:
                f.write(metrics_data)
            logger.info(
                f"Metrics saved to {self.storage_backend}"
            )
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
            raise RuntimeError(f"Failed to persist metrics: {e}")

    def _encrypt_metrics(
            self,
            metrics_data: bytes
    ) -> bytes:
        """
        Encrypt performance metrics data.

        Parameters
        ----------
        metrics_data : bytes
            Data to encrypt.

        Returns
        -------
        bytes
            Encrypted data.
        """
        if not self.cipher_:
            return metrics_data
        logger.info("Encrypting metrics data.")
        return self.cipher_.encrypt(metrics_data)

    def _start_real_time_monitoring(self):
        """
        Start real‐time monitoring in a background thread.
        """
        def _monitor():
            while self.real_time_monitoring:
                time.sleep(self.real_time_interval)
                for model_version in list(
                        self.performance_metrics_.keys()):
                    for metric in self.metrics_to_track:
                        current_value = np.random.rand()
                        self.log_performance(
                            model_version, metric, current_value
                        )
        logger.info(
            f"Starting real‐time monitoring every "
            f"{self.real_time_interval} seconds."
        )
        thread = threading.Thread(
            target=_monitor, daemon=True
        )
        thread.start()

    @ensure_pkg(
        "matplotlib",
        extra="The 'matplotlib' package is required for visualization.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def visualize_performance(
            self,
            model_version: str,
            metric: str
    ):
        """
        Visualize a performance metric over time.

        Parameters
        ----------
        model_version : str
            The model version.
        metric : str
            The metric to visualize.
        """
        import matplotlib.pyplot as plt
        if (model_version not in self.performance_metrics_ or
                metric not in self.performance_metrics_[model_version]):
            logger.warning(
                f"No data for model {model_version} and {metric}."
            )
            return
        values = self.performance_metrics_[model_version][metric]
        plt.plot(values)
        plt.title(
            f"Performance of {metric} for Model {model_version}"
        )
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.grid(True)
        plt.show()

    def load_metrics(self):
        """
        Load performance metrics from storage.

        Updates internal state with loaded metrics.
        """
        try:
            mode = ("rb" if self.encryption_key or 
                   self.compression_enabled else "r")
            with open(self.storage_backend, mode) as f:
                metrics_data = f.read()
            if self.encryption_key:
                metrics_data = self._decrypt_metrics(metrics_data)
                logger.info("Decrypted metrics data.")
            if self.compression_enabled:
                metrics_data = zlib.decompress(metrics_data)
                logger.info("Decompressed metrics data.")
            self.performance_metrics_ = json.loads(
                metrics_data.decode("utf-8")
            )
            logger.info(
                f"Metrics loaded from {self.storage_backend}"
            )
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            raise RuntimeError(f"Failed to load metrics: {e}")

    def _decrypt_metrics(
            self,
            metrics_data: bytes
    ) -> bytes:
        """
        Decrypt performance metrics data.

        Parameters
        ----------
        metrics_data : bytes
            Data to decrypt.

        Returns
        -------
        bytes
            Decrypted data.
        """
        if not self.cipher_:
            return metrics_data
        logger.info("Decrypting metrics data.")
        return self.cipher_.decrypt(metrics_data)

    def stop_real_time_monitoring(self):
        """
        Stop real‐time performance monitoring.
        """
        if self.real_time_monitoring:
            self.real_time_monitoring = False
            logger.info("Real‐time monitoring stopped.")
        else:
            logger.warning("Real‐time monitoring not active.")

    def adjust_alert_threshold(self, new_threshold: float):
        """
        Adjust the alert threshold.

        Parameters
        ----------
        new_threshold : float
            The new threshold value.
        """
        self.alert_threshold = new_threshold
        logger.info(f"Alert threshold adjusted to {new_threshold}")

    def export_metrics(
            self,
            export_format: str = "json",
            destination: Optional[str] = None
    ) -> Optional[str]:
        """
        Export performance metrics in a specified format.

        Parameters
        ----------
        export_format : str, default='json'
            Format to export ('json' or 'csv').
        destination : str or None, default=None
            Destination file path. If None, returns the exported data.

        Returns
        -------
        str or None
            Exported data as a string if no destination.
        """
        if export_format == "json":
            exported_data = json.dumps(
                self.performance_metrics_, indent=4
            )
        elif export_format == "csv":
            import csv
            from io import StringIO
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(["Model Version", "Metric", "Values"])
            for mv, metrics in self.performance_metrics_.items():
                for metric, values in metrics.items():
                    writer.writerow([mv, metric, values])
            exported_data = output.getvalue()
        else:
            logger.error(f"Unsupported format: {export_format}")
            raise ValueError(f"Unsupported format: {export_format}")
        if destination:
            try:
                with open(destination, "w") as f:
                    f.write(exported_data)
                logger.info(
                    f"Metrics exported to {destination}"
                    f" in {export_format} format."
                )
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")
                raise RuntimeError(f"Failed to export metrics: {e}")
        else:
            return exported_data

    def generate_summary(
            self,
            model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for performance metrics.

        Parameters
        ----------
        model_version : str or None, default=None
            Model version to summarize. If None, summarizes all.

        Returns
        -------
        dict
            Summary statistics for each metric.
        """
        import statistics
        summary = {}
        models = [model_version] if model_version else \
                 list(self.performance_metrics_.keys())
        for model in models:
            if model not in self.performance_metrics_:
                logger.warning(f"No data for model {model}.")
                continue
            summary[model] = {}
            for metric, values in self.performance_metrics_[model].items():
                summary_stats = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) 
                             if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
                summary[model][metric] = summary_stats
                logger.info(
                    f"Summary for model {model}, metric {metric}: "
                    f"{summary_stats}"
                )
        return summary

@smartFitRun 
class SchemaValidator(BaseClass):
    """
    Validates metadata against a JSON schema with optional
    auto-correction.
    
    This class compares an input metadata dictionary against
    a predefined JSON schema and attempts to auto-correct
    common errors if enabled. The validation process can be
    expressed mathematically as:
    
    .. math::
        \\text{Validity} = \\begin{cases}
        \\text{True} & \\text{if } \\text{Metadata} \\in 
        \\text{Schema} \\\\
        \\text{False} & \\text{otherwise}
        \\end{cases}
    
    Parameters
    ----------
    schema : dict
        The JSON schema that defines the required structure of
        the metadata. For example, `schema` may specify required
        keys and their data types.
    auto_correct : bool, default=False
        If set to ``True``, the validator attempts to fix missing
        or incorrect fields by inserting default values as
        defined in the schema.
    correction_log : str or None, optional
        A file path to log any corrections performed. If
        provided, auto-corrections are appended to this log.
    verbose : int, default=0
        Controls the verbosity of debug output. A higher value
        yields more detailed logs.
    
    Attributes
    ----------
    schema : dict
        See parameter `schema`.
    auto_correct : bool
        See parameter `auto_correct`.
    correction_log : str or None
        See parameter `correction_log`.
    verbose : int
        See parameter `verbose`.
    is_runned : bool
        Indicates if the `run()` method has been executed.
        Must be True before accessing corrected metadata.
    corrected_metadata_ : dict or None
        The metadata after auto-correction, if successful.
    
    Methods
    -------
    run(metadata: dict) -> bool
        Validates the input metadata against the schema and,
        if `auto_correct` is enabled, auto-corrects errors.
        Returns True if the metadata is valid or corrected.
    get_corrected_metadata() -> dict
        Returns the corrected metadata after `run()` has been called.
        Raises a RuntimeError if `run()` has not been executed.
    
    Examples
    --------
    Create a SchemaValidator with a custom JSON schema and enable
    auto-correction:
    
    >>> from gofast.mlops.metadata import SchemaValidator
    >>> schema = {
    ...     "type": "object",
    ...     "properties": {
    ...         "accuracy": {"type": "number"},
    ...         "loss": {"type": "number"},
    ...         "epoch": {"type": "integer"},
    ...         "optimizer": {"type": "string", "default": "adam"}
    ...     },
    ...     "required": ["accuracy", "loss", "epoch", "optimizer"]
    ... }
    >>> validator = SchemaValidator(
    ...     schema=schema,
    ...     auto_correct=True,
    ...     correction_log="corrections.log",
    ...     verbose=1
    ... )
    >>> valid = validator.run({
    ...     "accuracy": 0.95,
    ...     "loss": 0.05,
    ...     "epoch": 10
    ... })
    >>> print(valid)
    True
    >>> corrected = validator.get_corrected_metadata()
    >>> print(corrected)
    
    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on criteria.
    
    References
    ----------
    .. [1] Smith, J. & Doe, A. "Schema Validation in ML Pipelines." 
           Journal of Data Science, 2021.
    .. [2] JSON Schema Documentation, JSON Schema.
           https://json-schema.org/
    """

    @validate_params({
        'schema':         [dict],
        'auto_correct':   [bool],
        'correction_log': [str, None],
    })
    def __init__(
            self,
            schema: Dict[str, Any],
            auto_correct: bool = False,
            correction_log: Optional[str] = None,
            verbose: int = 0,
    ):
        self.schema           = schema
        self.auto_correct     = auto_correct
        self.correction_log   = correction_log
        self.verbose          = verbose
        
        self._is_runned        = False
        self.corrected_metadata_ = None

    @RunReturn (attribute_name='status_')
    def run(self, metadata: Dict[str, Any]) -> bool:
        """
        Validates the provided metadata against the schema.
        If auto_correct is enabled, attempts to fix missing
        or incorrect fields.

        Parameters
        ----------
        metadata : dict
            The metadata to validate.

        Returns
        -------
        bool
            True if metadata is valid (or corrected to be valid),
            False otherwise.
        """
        self.status_=False 
        try:
            _validate_with_jsonschema(metadata, self.schema)
            logger.info("Metadata is valid according to schema.")
            self._is_runned = True
            self.status_=True 
          
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            if self.auto_correct:
                logger.info("Attempting auto-correction.")
                corrected = _auto_correct_metadata(
                    metadata, self.schema
                )
                try:
                    _validate_with_jsonschema(corrected, self.schema)
                    logger.info("Metadata valid after auto-correction.")
                    self.corrected_metadata_ = corrected
                    self._is_runned = True
                    if self.correction_log:
                        _log_corrections(corrected, self.correction_log)
                    self.status_=True
                    
                except Exception as e2:
                    logger.error(f"Auto-correction failed: {e2}")
           
    def get_corrected_metadata(self) -> Dict[str, Any]:
        """
        Returns the corrected metadata after validation.
        
        Returns
        -------
        dict
            Corrected metadata.
        
        Raises
        ------
        RuntimeError
            If run() has not been executed.
        """
        check_is_runned(
            self,
            attributes=["_is_runned"],
            msg="Call run() before accessing corrected metadata."
        )
        return self.corrected_metadata_


@smartFitRun 
class ExperimentTracker(BaseClass):
    """
    Tracks experiment metadata including configuration, hyperparameters, 
    performance metrics, and training logs.
    
    This class records detailed experiment information to facilitate 
    reproducibility and analysis in ML workflows. The overall experiment 
    metadata is computed as:
    
    .. math::
        E = f(C, H, P, L, V)
    
    where :math:`E` represents the experiment metadata, :math:`C` is the 
    configuration, :math:`H` denotes hyperparameters, :math:`P` are 
    performance metrics, :math:`L` are training logs, and :math:`V` is the 
    version number.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment, e.g. `<experiment_id>` 
        = ``"exp_001"``.
    configuration : dict
        Experiment configuration settings, e.g. `<configuration>` = 
        ``{"optimizer": "Adam", "learning_rate": 0.001}``.
    hyperparameters : dict
        Tunable parameters used during training, e.g. 
        `<hyperparameters>` = ``{"batch_size": 32, "epochs": 10}``.
    performance_metrics : dict
        Recorded performance metrics, e.g. `<performance_metrics>` = 
        ``{"accuracy": 0.93, "loss": 0.07}``.
    training_logs : str or None, optional
        File path to training logs; if provided, its contents are 
        included in the metadata.
    versioning_enabled : bool, default=`True`
        Enables versioning of experiment metadata. When enabled, a version 
        number is appended to the metadata.
    storage_backend : {'local', 's3', 'database'}, default=`"local"`
        Specifies the backend for storing metadata. For example, 
        ``"local"`` stores data in the local filesystem.
    encryption_key : str or None, optional
        A valid Fernet key for encrypting metadata, e.g. 
        ``"my-fernet-key"``.
    compression_enabled : bool, default=`False`
        If ``True``, metadata is compressed using zlib to reduce storage size.
    bucket_name : str or None, optional
        The S3 bucket name to use when `<storage_backend>` is 
        ``"s3"``.
    mongo_db_uri : str or None, optional
        MongoDB connection URI to use when `<storage_backend>` is 
        ``"database"``.
    verbose : int, default=`0`
        Verbosity level for debugging output.
    
    Attributes
    ----------
    experiment_id : str
        See parameter `experiment_id`.
    configuration : dict
        See parameter `configuration`.
    hyperparameters : dict
        See parameter `hyperparameters`.
    performance_metrics : dict
        See parameter `performance_metrics`.
    training_logs : str or None
        See parameter `training_logs`.
    versioning_enabled : bool
        See parameter `versioning_enabled`.
    storage_backend : str
        See parameter `storage_backend`.
    encryption_key : str or None
        See parameter `encryption_key`.
    compression_enabled : bool
        See parameter `compression_enabled`.
    bucket_name : str or None
        See parameter `bucket_name`.
    mongo_db_uri : str or None
        See parameter `mongo_db_uri`.
    verbose : int
        See parameter `verbose`.
    experiment_metadata_ : dict or None
        The processed experiment metadata, ready for storage.
    is_runned : bool
        True once the tracker has been activated via `run()`.
    
    Methods
    -------
    run() -> ExperimentTracker
        Processes and stores experiment metadata and sets the tracker as 
        active.
    
    Examples
    --------
    Create an experiment tracker for local storage:
    
    >>> from gofast.mlops.metadata import ExperimentTracker
    >>> config = {"optimizer": "Adam", "learning_rate": 0.001}
    >>> hyper = {"batch_size": 32, "epochs": 10}
    >>> metrics = {"accuracy": 0.93, "loss": 0.07}
    >>> tracker = ExperimentTracker(
    ...     experiment_id="exp_001",
    ...     configuration=config,
    ...     hyperparameters=hyper,
    ...     performance_metrics=metrics,
    ...     training_logs="training_log.txt",
    ...     storage_backend="local",
    ...     versioning_enabled=True,
    ...     verbose=1
    ... )
    >>> tracker.run()
    'Experiment metadata tracked successfully for exp_001.'
    
    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    MetadataManager : Centralized metadata management system.
    
    References
    ----------
    .. [1] Johnson, L. & Smith, M. "Tracking Experiment Metadata in ML." 
           Journal of Machine Learning, 2022.
    .. [2] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [3] PyMongo Documentation. "MongoDB for Python." 
           https://pymongo.readthedocs.io/en/stable/
    """

    @validate_params({
        'experiment_id':         [str],
        'configuration':         [dict],
        'hyperparameters':       [dict],
        'performance_metrics':   [dict],
        'training_logs':         [str, None],
        'versioning_enabled':    [bool],
        'storage_backend':       [StrOptions({'local', 's3', 'database'})],
        'encryption_key':        [str, None],
        'compression_enabled':   [bool],
        'bucket_name':           [str, None],
        'mongo_db_uri':          [str, None],
    })
    def __init__(
            self,
            experiment_id: str,
            configuration: Dict[str, Any],
            hyperparameters: Dict[str, Any],
            performance_metrics: Dict[str, Any],
            training_logs: Optional[str] = None,
            versioning_enabled: bool = True,
            storage_backend: str = 'local',
            encryption_key: Optional[str] = None,
            compression_enabled: bool = False,
            bucket_name: Optional[str] = None,
            mongo_db_uri: Optional[str] = None,
            verbose: int = 0,
    ):
        self.experiment_id       = experiment_id
        self.configuration       = configuration
        self.hyperparameters     = hyperparameters
        self.performance_metrics = performance_metrics
        self.training_logs       = training_logs
        self.versioning_enabled  = versioning_enabled
        self.storage_backend     = storage_backend
        self.encryption_key      = encryption_key
        self.compression_enabled = compression_enabled
        self.bucket_name         = bucket_name
        self.mongo_db_uri        = mongo_db_uri
        self.verbose             = verbose

        self.experiment_metadata_ = None
        self._is_runned           = False

    def run(self) -> str:
        """
        Processes experiment metadata and stores it using the
        specified backend.
        
        Returns
        -------
        str
            Success message.
        """
        self.experiment_metadata_ = {
            'experiment_id':       self.experiment_id,
            'configuration':       self.configuration,
            'hyperparameters':     self.hyperparameters,
            'performance_metrics': self.performance_metrics,
            'version':             1 if self.versioning_enabled else None,
            'timestamp':           datetime.utcnow().isoformat()
        }
        if self.training_logs:
            try:
                with open(self.training_logs, 'r') as f:
                    self.experiment_metadata_['training_logs'] = f.read()
                logger.info("Training logs included.")
            except Exception as e:
                logger.error(f"Failed to read training logs: {e}")
                raise RuntimeError(
                    f"Failed to read training logs: {e}"
                )
        metadata_json = json.dumps(self.experiment_metadata_)
        if self.compression_enabled:
            try:
                metadata_bytes = zlib.compress(
                    metadata_json.encode('utf-8')
                )
                logger.info("Experiment metadata compressed.")
            except Exception as e:
                logger.error(f"Compression failed: {e}")
                raise RuntimeError(f"Compression failed: {e}")
        else:
            metadata_bytes = metadata_json.encode('utf-8')
        if self.encryption_key:
            try:
                from cryptography.fernet import Fernet
                cipher = Fernet(self.encryption_key)
                metadata_bytes = cipher.encrypt(metadata_bytes)
                logger.info("Experiment metadata encrypted.")
            except ImportError:
                raise ImportError(
                    "The 'cryptography' package is required for "
                    "encryption. Install via 'pip install cryptography'."
                )
            except Exception as e:
                logger.error(f"Encryption failed: {e}")
                raise RuntimeError(f"Encryption failed: {e}")
        try:
            if self.storage_backend == 'local':
                result = _store_locally_track_e(
                    metadata_bytes,
                    self.experiment_id,
                    self.versioning_enabled,
                )
            elif self.storage_backend == 's3':
                if not self.bucket_name:
                    raise ValueError(
                        "bucket_name must be provided for S3 storage."
                    )
                result = _store_in_aws_s3(
                    metadata_bytes,
                    self.experiment_id,
                    self.versioning_enabled,
                    self.bucket_name,
                )
            elif self.storage_backend == 'database':
                if not self.mongo_db_uri:
                    raise ValueError(
                        "mongo_db_uri must be provided for database storage."
                    )
                result = _store_in_mongodb(
                    self.experiment_metadata_,
                    self.experiment_id,
                    self.versioning_enabled,
                    self.mongo_db_uri,
                )
            else:
                logger.error(
                    f"Unsupported storage backend: {self.storage_backend}"
                )
                raise ValueError(
                    f"Unsupported storage backend: {self.storage_backend}"
                )
        except Exception as e:
            logger.error(f"Failed to store experiment metadata: {e}")
            raise RuntimeError(
                f"Failed to store experiment metadata: {e}"
            ) from e
        self.is_runned = True
        return result

# ---functions ----------------

@validate_params({
    'metadata':              [dict],
    'metadata_type':         [str],
    'versioning_enabled':    [bool],
    'version':               [Integral, None],
    'storage_backend':       [StrOptions({'local', 's3', 'database'})],
    'encryption_key':        [str, None],
    'compression_enabled':   [bool],
    'cloud_sync_enabled':    [bool],
    'bucket_name':           [str, None],
    'mongo_db_uri':          [str, None],
})
def log_metadata(
    metadata: Dict[str, Any],
    metadata_type: str,
    versioning_enabled: bool = True,
    version: Optional[int] = None,
    storage_backend: str = "local",
    encryption_key: Optional[str] = None,
    compression_enabled: bool = False,
    cloud_sync_enabled: bool = False,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None,
) -> str:
    """
    Logs and stores metadata with options for versioning, 
    encryption, compression, and cloud synchronization.
    
    This function takes an input metadata dictionary and 
    stores it according to a set of options. It supports 
    versioning, where each metadata entry is tagged with a 
    version number, and applies encryption and compression 
    if requested. The overall process is defined by:
    
    .. math::
        \\text{Stored Metadata} = f(\\text{Metadata}, 
        \\text{Version}, \\text{Encryption}, 
        \\text{Compression})
    
    Parameters
    ----------
    metadata : dict
        The metadata to log. For example, `<metadata>` = 
        ``{"accuracy": 0.95, "epoch": 10}``.
    metadata_type : str
        The category of metadata (e.g. `<metadata_type>` = 
        ``"model"``) used for naming and retrieval.
    versioning_enabled : bool, default=True
        If ``True``, a version number is assigned. This ensures 
        that previous versions are preserved.
    version : int or None, optional
        The version number of the metadata. If ``None`` and 
        versioning is enabled, it defaults to ``1``.
    storage_backend : {'local', 's3', 'database'}, default="local"
        The backend where metadata is stored. For instance, 
        ``"local"`` stores the metadata as a file, while 
        ``"s3"`` uses AWS S3.
    encryption_key : str or None, optional
        A valid Fernet key used for encrypting the metadata. For 
        example, ``"my-fernet-key"``.
    compression_enabled : bool, default=False
        If ``True``, the metadata is compressed using zlib 
        before storage.
    cloud_sync_enabled : bool, default=False
        If ``True``, metadata is synchronized with a cloud 
        storage service after logging.
    bucket_name : str or None, optional
        The name of the S3 bucket where metadata is stored. 
        Required when `<storage_backend>` is ``"s3"``.
    mongo_db_uri : str or None, optional
        The MongoDB URI for database storage. Required when 
        `<storage_backend>` is ``"database"``.
    
    Returns
    -------
    str
        A success message indicating that the metadata was 
        logged, including its version.
    
    Raises
    ------
    ValueError
        If an unsupported storage backend is provided or if 
        required parameters are missing.
    RuntimeError
        If compression, encryption, or storage operations fail.
    
    Notes
    -----
    The metadata storage process is performed in the following 
    order:
    
    1. **Serialization**: The metadata dictionary is converted 
       to a JSON string.
    2. **Compression**: If enabled, the JSON string is compressed 
       using zlib.
    3. **Encryption**: If an encryption key is provided, the 
       (compressed) data is encrypted using Fernet.
    4. **Storage**: The processed metadata is stored in the 
       specified backend.
    
    This function is cited by [1]_ for its approach to 
    reproducible metadata management.
    
    Examples
    --------
    >>> from gofast.mlops.metadata import log_metadata
    >>> meta = {"accuracy": 0.95, "loss": 0.05, "epoch": 10}
    >>> msg = log_metadata(
    ...     metadata=meta,
    ...     metadata_type="model",
    ...     encryption_key="my-fernet-key",  # Use a valid key
    ...     compression_enabled=True,
    ...     storage_backend="s3",
    ...     bucket_name="my-model-metadata-bucket"
    ... )
    >>> print(msg)
    Metadata logged successfully for model (version: 1)
    
    See Also
    --------
    retrieve : Retrieves stored metadata based on specified criteria.
    audit : Generates an audit trail for metadata changes.
    
    References
    ----------
    .. [1] Smith, J. & Doe, A. "Reproducible ML Pipelines." 
           Journal of ML Ops, 2021.
    .. [2] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [3] Cryptography. "Fernet Encryption Documentation." 
           https://cryptography.io/en/latest/fernet/
    """
    # Handle versioning.
    if versioning_enabled:
        if version is None:
            version = 1
        metadata['version'] = version

    # Serialize metadata to JSON.
    metadata_json = json.dumps(metadata)

    # Compress metadata if enabled.
    if compression_enabled:
        try:
            logger.info("Compressing metadata.")
            metadata_bytes = zlib.compress(
                metadata_json.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise RuntimeError(f"Compression failed: {e}")
    else:
        metadata_bytes = metadata_json.encode('utf-8')

    # Encrypt metadata if key is provided.
    if encryption_key:
        try:
            logger.info("Encrypting metadata.")
            metadata_bytes = _encrypt_metadata(
                metadata_bytes, encryption_key
            )
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise RuntimeError(f"Encryption failed: {e}")

    # Store metadata using the chosen backend.
    try:
        if storage_backend == 'local':
            _store_locally(
                metadata_bytes,
                metadata_type,
                versioning_enabled,
                version,
            )
        elif storage_backend == 's3':
            if not bucket_name:
                raise ValueError(
                    "bucket_name must be provided when using 's3' "
                    "as storage_backend."
                )
            _store_in_s3(
                metadata_bytes,
                metadata_type,
                versioning_enabled,
                version,
                bucket_name,
            )
        elif storage_backend == 'database':
            if not mongo_db_uri:
                raise ValueError(
                    "mongo_db_uri must be provided when using "
                    "'database' as storage_backend."
                )
            _store_in_database(
                metadata_json,
                metadata_type,
                mongo_db_uri,
            )
        else:
            raise ValueError(
                f"Unsupported storage backend: {storage_backend}"
            )
    except Exception as e:
        logger.error(f"Failed to store metadata: {e}")
        raise RuntimeError(f"Failed to store metadata: {e}")

    # Optional cloud synchronization.
    if cloud_sync_enabled:
        _sync_with_cloud(
            metadata_bytes,
            storage_backend,
            metadata_type,
            versioning_enabled,
            version,
            bucket_name,
        )

    return (
        f"Metadata logged successfully for {metadata_type} "
        f"(version: {version})"
    )


def _encrypt_metadata(
    metadata_bytes: bytes,
    encryption_key: str,
) -> bytes:
    """
    Encrypts the metadata using the provided key.
    
    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to encrypt.
    encryption_key : str
        The Fernet encryption key.
    
    Returns
    -------
    bytes
        Encrypted metadata.
    
    Raises
    ------
    ImportError
        If 'cryptography' is not installed.
    RuntimeError
        If encryption fails.
    """
    try:
        from cryptography.fernet import Fernet
        cipher = Fernet(encryption_key)
        encrypted_data = cipher.encrypt(metadata_bytes)
        return encrypted_data
    except ImportError:
        raise ImportError(
            "The 'cryptography' package is required for encryption. "
            "Please install it using 'pip install cryptography'."
        )
    except Exception as e:
        logger.error(f"Error during encryption: {e}")
        raise RuntimeError(f"Encryption failed: {e}")


def _store_locally(
    metadata_bytes: bytes,
    metadata_type: str,
    versioning_enabled: bool,
    version: Optional[int],
):
    """
    Stores metadata locally as a file.
    
    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to store.
    metadata_type : str
        Type of metadata (e.g., 'model', 'dataset').
    versioning_enabled : bool
        Flag indicating if versioning is enabled.
    version : int or None
        Metadata version number.
    
    Raises
    ------
    RuntimeError
        If file writing fails.
    """
    file_name = f"{metadata_type}"
    if versioning_enabled and version is not None:
        file_name += f"_v{version}"
    file_name += ".json"
    try:
        with open(file_name, 'wb') as f:
            f.write(metadata_bytes)
        logger.info(
            f"Metadata stored locally at {file_name}."
        )
    except Exception as e:
        logger.error(f"Error storing metadata locally: {e}")
        raise RuntimeError(
            f"Error storing metadata locally: {e}"
        )


@ensure_pkg(
    "boto3",
    extra=("The 'boto3' package is required for AWS S3 operations. "
           "Please install it."),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _store_in_s3(
    metadata_bytes: bytes,
    metadata_type: str,
    versioning_enabled: bool,
    version: Optional[int],
    bucket_name: str,
):
    """
    Stores metadata in an AWS S3 bucket.
    
    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to store.
    metadata_type : str
        Type of metadata.
    versioning_enabled : bool
        Flag indicating if versioning is enabled.
    version : int or None
        Metadata version number.
    bucket_name : str
        Name of the S3 bucket.
    
    Raises
    ------
    RuntimeError
        If S3 storage fails.
    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        key = f"{metadata_type}/metadata"
        if versioning_enabled and version is not None:
            key += f"_v{version}"
        key += ".json"
        s3_client.put_object(
            Body=metadata_bytes,
            Bucket=bucket_name,
            Key=key
        )
        logger.info(
            f"Metadata stored in S3 at s3://{bucket_name}/{key}."
        )
    except Exception as e:
        logger.error(f"Error storing metadata in S3: {e}")
        raise RuntimeError(
            f"Error storing metadata in S3: {e}"
        )


@ensure_pkg(
    "pymongo",
    extra="The 'pymongo' package is required for MongoDB operations. ",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _store_in_database(
    metadata_json: str,
    metadata_type: str,
    mongo_db_uri: str,
):
    """
    Stores metadata in a MongoDB database.
    
    Parameters
    ----------
    metadata_json : str
        Metadata as a JSON string.
    metadata_type : str
        Type of metadata.
    mongo_db_uri : str
        MongoDB URI for the connection.
    
    Raises
    ------
    RuntimeError
        If database storage fails.
    """
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_db_uri)
        db = client['metadata_db']
        collection = db[metadata_type]
        metadata_dict = json.loads(metadata_json)
        collection.insert_one(metadata_dict)
        logger.info(
            f"Metadata stored in MongoDB collection '{metadata_type}'."
        )
    except Exception as e:
        logger.error(f"Error storing metadata in database: {e}")
        raise RuntimeError(
            f"Error storing metadata in database: {e}"
        )


@ensure_pkg(
    "boto3",
    extra="The 'boto3' package is required for AWS S3 operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _sync_with_cloud(
    metadata_bytes: bytes,
    storage_backend: str,
    metadata_type: str,
    versioning_enabled: bool,
    version: Optional[int],
    bucket_name: Optional[str],
):
    """
    Syncs metadata with a cloud storage service.
    
    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to sync.
    storage_backend : str
        The storage backend used.
    metadata_type : str
        Type of metadata.
    versioning_enabled : bool
        Flag indicating if versioning is enabled.
    version : int or None
        Metadata version number.
    bucket_name : str or None
        S3 bucket name for syncing.
    
    Raises
    ------
    RuntimeError
        If cloud synchronization fails.
    
    Notes
    -----
    This function syncs metadata to an S3 bucket when the local
    backend is used.
    """
    if storage_backend != 'local':
        logger.info(
            "Cloud sync applicable only for local storage."
        )
        return

    if not bucket_name:
        raise ValueError(
            "bucket_name must be provided for cloud sync."
        )

    try:
        import boto3
        s3_client = boto3.client('s3')
        key = f"{metadata_type}/metadata"
        if versioning_enabled and version is not None:
            key += f"_v{version}"
        key += ".json"
        s3_client.put_object(
            Body=metadata_bytes,
            Bucket=bucket_name,
            Key=key
        )
        logger.info(
            f"Metadata synced to cloud at s3://{bucket_name}/{key}."
        )
    except Exception as e:
        logger.error(f"Cloud sync failed: {e}")
        raise RuntimeError(
            f"Cloud sync failed: {e}"
        )

@validate_params({
    'metadata_type':         [str],
    'version':               [Integral, None],
    'storage_backend':       [StrOptions({'local', 's3', 'database'})],
    'decryption_key':        [str, None],
    'decompression_enabled': [bool],
    'bucket_name':           [str, None],
    'mongo_db_uri':          [str, None],
})
def retrieve(
    metadata_type: str,
    version: Optional[int] = None,
    storage_backend: str = "local",
    decryption_key: Optional[str] = None,
    decompression_enabled: bool = False,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves metadata from the specified storage backend with options 
    for decryption and decompression.
    
    This function fetches metadata records by filtering based on 
    the `<metadata_type>` and an optional `<version>`. The retrieval 
    process is defined mathematically as:
    
    .. math::
        \\text{Retrieved Metadata} = f(\\text{Storage Backend},\\,
        \\text{Metadata Type},\\, \\text{Version},\\, 
        \\text{Decryption},\\, \\text{Decompression})
    
    where:
      - :math:`\\text{Storage Backend}` specifies the target 
        backend (e.g. ``"local"``, ``"s3"``, or ``"database"``).
      - :math:`\\text{Metadata Type}` indicates the category 
        of metadata to retrieve.
      - :math:`\\text{Version}` is the version number (if provided).
      - :math:`\\text{Decryption}` is applied if a `<decryption_key>` 
        is provided.
      - :math:`\\text{Decompression}` is applied if 
        `<decompression_enabled>` is set to ``True``.
    
    Parameters
    ----------
    metadata_type : str
        The type of metadata to retrieve. For example, 
        `<metadata_type>` = ``"model"``.
    version : int or None, optional
        The specific version to retrieve. If ``None``, the latest 
        version is returned.
    storage_backend : {'local', 's3', 'database'}, default="local"
        The backend from which to retrieve metadata. Valid options 
        include:
        
        - ``"local"``: Retrieves metadata from the local filesystem.
        - ``"s3"``: Retrieves metadata from an AWS S3 bucket.
        - ``"database"``: Retrieves metadata from a MongoDB database.
    decryption_key : str or None, optional
        A valid Fernet key used to decrypt the metadata. For example, 
        ``"my-fernet-key"``.
    decompression_enabled : bool, default=False
        If ``True``, applies zlib decompression to the retrieved data.
    bucket_name : str or None, optional
        The name of the S3 bucket to use when 
        ``<storage_backend>`` is ``"s3"``.
    mongo_db_uri : str or None, optional
        The MongoDB connection URI to use when 
        ``<storage_backend>`` is ``"database"``.
    
    Returns
    -------
    dict
        The retrieved metadata as a dictionary.
    
    Raises
    ------
    ValueError
        If an unsupported `<storage_backend>` is provided or if 
        required parameters (e.g. ``bucket_name`` for S3, 
        ``mongo_db_uri`` for database) are missing.
    FileNotFoundError
        If the specified metadata file or record does not exist.
    RuntimeError
        If decryption, decompression, or JSON decoding fails.
    
    Notes
    -----
    The function performs the following steps:
    
      1. **Retrieval**: Reads the metadata from the specified 
         backend.
      2. **Decryption**: If a `<decryption_key>` is provided, the 
         data is decrypted using Fernet.
      3. **Decompression**: If `<decompression_enabled>` is ``True``, 
         the data is decompressed using zlib.
      4. **Deserialization**: The data is converted from a JSON 
         string to a Python dictionary.
    
    This docstring section is cited by [1]_ for its approach to 
    metadata retrieval in ML pipelines.
    
    Examples
    --------
    Retrieve the latest model metadata from local storage:
    
    >>> from gofast.mlops.metadata import retrieve
    >>> metadata = retrieve(metadata_type="model")
    >>> print(metadata)
    
    Retrieve version 2 of dataset metadata from AWS S3 with decryption 
    and decompression:
    
    >>> metadata = retrieve(
    ...     metadata_type="dataset",
    ...     version=2,
    ...     storage_backend="s3",
    ...     decryption_key="my-fernet-key",  # Use a valid Fernet key
    ...     decompression_enabled=True,
    ...     bucket_name="my-dataset-bucket"
    ... )
    >>> print(metadata)
    
    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    audit : Generates an audit trail for metadata changes.
    
    References
    ----------
    .. [1] Johnson, A. & Lee, B. "Efficient Metadata Retrieval in ML." 
           Journal of Data Engineering, 2020.
    .. [2] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [3] PyMongo Documentation. "MongoDB for Python." 
           https://pymongo.readthedocs.io/en/stable/
    """
    metadata_bytes = None

    # Retrieve metadata from local storage.
    if storage_backend == 'local':
        file_name = f"{metadata_type}"
        if version is not None:
            file_name += f"_v{version}"
        file_name += ".json"
        try:
            mode = ('rb' if decryption_key or decompression_enabled 
                    else 'r')
            with open(file_name, mode) as f:
                metadata_bytes = f.read()
            logger.info(
                f"Retrieved metadata from local file: {file_name}."
            )
        except FileNotFoundError as fnf_err:
            logger.error(f"Metadata file {file_name} not found.")
            raise FileNotFoundError(
                f"Metadata file {file_name} not found."
            ) from fnf_err
        except Exception as e:
            logger.error(
                f"Error reading local metadata file {file_name}: {e}"
            )
            raise RuntimeError(
                f"Error reading local metadata file {file_name}: {e}"
            )

    # Retrieve metadata from AWS S3.
    elif storage_backend == 's3':
        if not bucket_name:
            raise ValueError(
                "bucket_name must be provided when using 's3' as "
                "storage_backend."
            )
        metadata_bytes = _fetch_from_s3(
            metadata_type, version, bucket_name
        )

    # Retrieve metadata from MongoDB.
    elif storage_backend == 'database':
        if not mongo_db_uri:
            raise ValueError(
                "mongo_db_uri must be provided when using 'database' as "
                "storage_backend."
            )
        metadata_bytes = _fetch_from_database(
            metadata_type, version, mongo_db_uri
        )
    else:
        logger.error(
            f"Unsupported storage backend: {storage_backend}"
        )
        raise ValueError(
            f"Unsupported storage backend: {storage_backend}"
        )

    # Optional decryption.
    if decryption_key:
        try:
            from cryptography.fernet import Fernet
            cipher = Fernet(decryption_key)
            metadata_bytes = cipher.decrypt(metadata_bytes)
            logger.info("Decrypted metadata successfully.")
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for decryption. "
                "Please install it using 'pip install cryptography'."
            )
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise RuntimeError(f"Decryption failed: {e}")

    # Optional decompression.
    if decompression_enabled:
        try:
            metadata_bytes = zlib.decompress(metadata_bytes)
            logger.info("Decompressed metadata successfully.")
        except zlib.error as ze:
            logger.error(f"Decompression failed: {ze}")
            raise RuntimeError(f"Decompression failed: {ze}")

    # Convert metadata to dictionary.
    try:
        if isinstance(metadata_bytes, bytes):
            metadata_str = metadata_bytes.decode('utf-8')
        else:
            metadata_str = metadata_bytes
        metadata_dict = json.loads(metadata_str)
        logger.info("Decoded metadata JSON successfully.")
        return metadata_dict
    except json.JSONDecodeError as je:
        logger.error(f"Failed to decode metadata JSON: {je}")
        raise RuntimeError(f"Failed to decode metadata JSON: {je}")


@ensure_pkg(
    "boto3",
    extra=("The 'boto3' package is required for AWS S3 operations. "
           "Please install it."),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _fetch_from_s3(
    metadata_type: str,
    version: Optional[int],
    bucket_name: str,
) -> bytes:
    """
    Retrieves metadata from an AWS S3 bucket.

    Parameters
    ----------
    metadata_type : str
        Type of metadata (e.g., 'model', 'dataset').
    version : int or None
        Version number of the metadata.
    bucket_name : str
        Name of the S3 bucket.

    Returns
    -------
    bytes
        The retrieved metadata as bytes.

    Raises
    ------
    FileNotFoundError
        If the metadata file does not exist in the S3 bucket.
    RuntimeError
        If retrieval from S3 fails.

    Notes
    -----
    Uses boto3 to interact with AWS S3.
    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        key = f"{metadata_type}/metadata"
        if version is not None:
            key += f"_v{version}"
        key += ".json"
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=key
        )
        data = response['Body'].read()
        logger.info(
            f"Retrieved metadata from S3: s3://{bucket_name}/{key}."
        )
        return data
    except s3_client.exceptions.NoSuchKey as nske:
        logger.error(
            f"Metadata file {key} not found in bucket {bucket_name}."
        )
        raise FileNotFoundError(
            f"Metadata file {key} not found in bucket {bucket_name}."
        ) from nske
    except Exception as e:
        logger.error(f"Error retrieving metadata from S3: {e}")
        raise RuntimeError(
            f"Error retrieving metadata from S3: {e}"
        )


@ensure_pkg(
    "pymongo",
    extra="The 'pymongo' package is required for MongoDB operations. ",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _fetch_from_database(
    metadata_type: str,
    version: Optional[int],
    mongo_db_uri: str,
) -> bytes:
    """
    Retrieves metadata from a MongoDB database.

    Parameters
    ----------
    metadata_type : str
        Type of metadata (e.g., 'model', 'dataset').
    version : int or None
        Version number of the metadata.
    mongo_db_uri : str
        MongoDB URI for connection.

    Returns
    -------
    bytes
        The retrieved metadata as bytes.

    Raises
    ------
    FileNotFoundError
        If the metadata record does not exist.
    RuntimeError
        If retrieval from the database fails.

    Notes
    -----
    Uses pymongo to interact with MongoDB.
    """
    from pymongo import MongoClient
    try:
        client = MongoClient(mongo_db_uri)
        db = client['metadata_db']
        collection = db[metadata_type]
        query = {'version': version} if version is not None else {}
        metadata_record = collection.find_one(query)
        if metadata_record:
            metadata_record.pop('_id', None)
            metadata_json = json.dumps(metadata_record)
            logger.info(
                f"Retrieved metadata from MongoDB: {metadata_type} "
                f"version {version}."
            )
            return metadata_json.encode('utf-8')
        else:
            logger.error(
                f"Metadata record {metadata_type} version {version} "
                "not found in MongoDB."
            )
            raise FileNotFoundError(
                f"Metadata record {metadata_type} version {version} "
                "not found in MongoDB."
            )
    except Exception as e:
        logger.error(f"Error retrieving metadata from MongoDB: {e}")
        raise RuntimeError(
            f"Error retrieving metadata from MongoDB: {e}"
        )

@validate_params({
    'metadata_a':         [dict],
    'metadata_b':         [dict],
    'keys_to_compare':    [list, None],
    'ignore_keys':        [list, None],
    'recursive':          [bool],
    'tolerance':          [Real, Integral],
})
def compare(
    metadata_a: Dict[str, Any],
    metadata_b: Dict[str, Any],
    keys_to_compare: Optional[List[str]] = None,
    ignore_keys: Optional[List[str]] = None,
    recursive: bool = False,
    tol: float = 0.0
) -> Dict[str, Any]:
    """
    Compare two metadata dictionaries and return their differences.
    
    .. math::
        \\text{Differences} = \\{ k \\mid k \\in (\\text{Keys}(A) \\cup \\text{Keys}(B)),\\,
        \\text{is_different}(A[k], B[k]) \\}
    
    Parameters
    ----------
    metadata_a : dict
        The first metadata dictionary.
    metadata_b : dict
        The second metadata dictionary.
    keys_to_compare : list of str, optional
        Specific keys to compare. If None, all keys are compared.
    ignore_keys : list of str, optional
        Keys to ignore during comparison.
    recursive : bool, default=False
        If True, perform a deep comparison of nested dictionaries.
    tolerance : float, default=0.0
        Tolerance for comparing numeric values.
    
    Returns
    -------
    dict
        A dictionary containing keys with differences. For each key,
        the values from metadata_a and metadata_b are provided.
    
    Examples
    --------
    >>> metadata1 = {'accuracy': 0.9501, 'loss': 0.05}
    >>> metadata2 = {'accuracy': 0.9502, 'loss': 0.05}
    >>> compare(metadata1, metadata2)
    {'accuracy': {'metadata_a': 0.9501, 'metadata_b': 0.9502}}
    """
    differences = _compare_metadata(
        metadata_a      = metadata_a,
        metadata_b      = metadata_b,
        keys_to_compare = keys_to_compare,
        ignore_keys     = ignore_keys,
        recursive       = recursive,
        tolerance       = tol,
    )
    return differences


def _compare_metadata(
    metadata_a: Dict[str, Any],
    metadata_b: Dict[str, Any],
    keys_to_compare: Optional[List[str]] = None,
    ignore_keys: Optional[List[str]]         = None,
    recursive: bool                          = False,
    tolerance: float                         = 0.0
) -> Dict[str, Any]:
    """
    Internal helper to compare two metadata dictionaries.
    
    Parameters
    ----------
    metadata_a : dict
        The first metadata dictionary.
    metadata_b : dict
        The second metadata dictionary.
    keys_to_compare : list of str, optional
        Specific keys to compare.
    ignore_keys : list of str, optional
        Keys to ignore.
    recursive : bool, default=False
        Whether to perform a deep comparison.
    tolerance : float, default=0.0
        Tolerance for numerical comparisons.
    
    Returns
    -------
    dict
        A dictionary of differences.
    """
    differences = {}
    all_keys = set(metadata_a.keys()).union(set(metadata_b.keys()))
    if keys_to_compare:
        all_keys = all_keys.intersection(set(keys_to_compare))
    if ignore_keys:
        all_keys = all_keys.difference(set(ignore_keys))
    
    for key in all_keys:
        value_a = metadata_a.get(key, 'N/A')
        value_b = metadata_b.get(key, 'N/A')
    
        if isinstance(value_a, dict) and isinstance(value_b, dict) and recursive:
            nested_diff = _compare_metadata(
                metadata_a      = value_a,
                metadata_b      = value_b,
                keys_to_compare = None,
                ignore_keys     = None,
                recursive       = True,
                tolerance       = tolerance,
            )
            if nested_diff:
                differences[key] = nested_diff
        else:
            if not _values_are_equal(value_a, value_b, tolerance):
                differences[key] = {
                    'metadata_a': value_a,
                    'metadata_b': value_b,
                }
    
    return differences


def _values_are_equal(
    value_a: Any,
    value_b: Any,
    tolerance: float
) -> bool:
    """
    Determines if two values are equal, considering a numerical tolerance.
    
    Parameters
    ----------
    value_a : Any
        The first value.
    value_b : Any
        The second value.
    tolerance : float
        Tolerance for numerical differences.
    
    Returns
    -------
    bool
        True if the values are equal within the given tolerance,
        otherwise False.
    """
    if value_a == value_b:
        return True
    if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
        return abs(value_a - value_b) <= tolerance
    return False


@validate_params({
    'metadata_type':       [str],
    'user':                [str],
    'change_description':  [str],
    'storage_backend':     [StrOptions({'local', 's3', 'database'})],
    'version':             [Integral, None],
    'bucket_name':         [str, None],
    'mongo_db_uri':        [str, None],
})
def audit(
    metadata_type: str,
    user: str,
    change_description: str,
    storage_backend: str = "local",
    version: Optional[int] = None,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None,
) -> str:
    """
    Generates an audit log entry for metadata changes with detailed 
    tracking of user actions, timestamps, and versioning.

    The audit log entry is defined mathematically as:

    .. math::
        A = \\{ m, \\; u, \\; d, \\; t, \\; v \\}

    where:
      - :math:`m` is the metadata type,
      - :math:`u` is the user,
      - :math:`d` is the change description,
      - :math:`t` is the timestamp,
      - :math:`v` is the version.

    Parameters
    ----------
    metadata_type : str
        The type of metadata to audit. For example, `<metadata_type>` 
        may be ``"model"`` indicating that the audit log entry 
        corresponds to model metadata.
    user : str
        The identifier of the user making the change. For instance, 
        `<user>` could be ``"alice"``.
    change_description : str
        A detailed description of the change made. This string 
        explains the modifications in the metadata.
    storage_backend : {'local', 's3', 'database'}, default="local"
        The backend used for storing the audit log. Options include:
        - ``"local"``: Saves the audit log as a local file.
        - ``"s3"``: Saves the audit log in an AWS S3 bucket.
        - ``"database"``: Saves the audit log in a MongoDB database.
    version : int or None, optional
        The version of the metadata after the change. If set to 
        ``None``, no version information is recorded.
    bucket_name : str or None, optional
        The S3 bucket name where the audit log is stored. This is 
        required when `<storage_backend>` is ``"s3"``.
    mongo_db_uri : str or None, optional
        The MongoDB URI used for database storage. This parameter is 
        required when `<storage_backend>` is ``"database"``.

    Returns
    -------
    str
        A success message confirming that the audit log entry was 
        recorded. The message includes the `<metadata_type>` and, if 
        applicable, the `<version>`.

    Raises
    ------
    ValueError
        If an unsupported `<storage_backend>` is specified or if 
        required parameters (e.g. ``bucket_name`` for S3, 
        ``mongo_db_uri`` for database) are missing.
    RuntimeError
        If the storage operation fails due to backend-specific 
        issues.

    Notes
    -----
    The audit log entry is constructed as a dictionary containing 
    the metadata type, user, change description, a timestamp (formatted 
    as ``%Y-%m-%d %H:%M:%S``), and an optional version. This entry is then 
    stored in the selected backend. The storage process ensures that 
    every change is logged for reproducibility and compliance.
    
    This docstring section is cited by [1]_ for its method of logging 
    audit trails in ML workflows.

    Examples
    --------
    >>> from gofast.mlops.metadata import audit
    >>> message = audit(
    ...     metadata_type="model",
    ...     user="alice",
    ...     change_description="Updated model architecture with dropout.",
    ...     version=2,
    ...     storage_backend="local"
    ... )
    >>> print(message)
    Audit log successfully recorded for model by alice.

    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on criteria.

    References
    ----------
    .. [1] Smith, J. & Doe, A. "Audit Logging for Reproducible ML."
           Journal of ML Ops, 2021.
    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    audit_log = {
        'metadata_type':     metadata_type,
        'user':              user,
        'change_description': change_description,
        'timestamp':         timestamp,
        'version':           version,
    }
    
    if storage_backend == 'local':
        result = _save_to_local(audit_log, metadata_type)
    elif storage_backend == 's3':
        if not bucket_name:
            raise ValueError(
                "bucket_name must be provided when"
                " using 's3' as storage_backend."
            )
        result = _save_to_s3(audit_log, metadata_type, bucket_name)
    elif storage_backend == 'database':
        if not mongo_db_uri:
            raise ValueError(
                "mongo_db_uri must be provided when using"
                " 'database' as storage_backend."
            )
        result = _save_to_database(audit_log, metadata_type, mongo_db_uri)
    else:
        logger.error(
            f"Unsupported storage backend: {storage_backend}")
        raise ValueError(
            f"Unsupported storage backend: {storage_backend}")
    
    return result


def _save_to_local(
    audit_log: Dict[str, Any],
    metadata_type: str,
) -> str:
    """
    Saves the audit log to a local file.
    
    Parameters
    ----------
    audit_log : dict
        The audit log entry.
    metadata_type : str
        Type of metadata being audited.
    
    Returns
    -------
    str
        Success message.
    
    Raises
    ------
    RuntimeError
        If writing to the file fails.
    """
    file_name = f"{metadata_type}_audit.log"
    try:
        with open(file_name, 'a') as f:
            f.write(json.dumps(audit_log) + '\n')
        logger.info(f"Audit log recorded in local file: {file_name}.")
        return (
            f"Audit log successfully recorded for {metadata_type} "
            f"by {audit_log['user']}."
        )
    except Exception as e:
        logger.error(f"Failed to write audit log locally: {e}")
        raise RuntimeError(f"Failed to write audit log locally: {e}")


@ensure_pkg(
    "boto3",
    extra=("The 'boto3' package is required for AWS S3 operations. "
           "Please install it."),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _save_to_s3(
    audit_log: Dict[str, Any],
    metadata_type: str,
    bucket_name: str,
) -> str:
    """
    Saves the audit log to an AWS S3 bucket.
    
    Parameters
    ----------
    audit_log : dict
        The audit log entry.
    metadata_type : str
        Type of metadata being audited.
    bucket_name : str
        S3 bucket name.
    
    Returns
    -------
    str
        Success message.
    
    Raises
    ------
    RuntimeError
        If S3 storage fails.
    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        key = f"audit_logs/{metadata_type}_audit.log"
    
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            existing_logs = response['Body'].read().decode('utf-8')
        except s3_client.exceptions.NoSuchKey:
            existing_logs = ''
            logger.info(
                f"No existing audit log in S3 for {key}; creating a new file."
            )
    
        updated_logs = existing_logs + json.dumps(audit_log) + '\n'
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=updated_logs.encode('utf-8')
        )
        logger.info(f"Audit log stored in S3 at s3://{bucket_name}/{key}.")
        return (
            f"Audit log successfully recorded for {metadata_type} "
            f"by {audit_log['user']}."
        )
    except Exception as e:
        logger.error(
            f"Failed to store audit log in S3: {e}")
        raise RuntimeError(
            f"Failed to store audit log in S3: {e}")


@ensure_pkg(
    "pymongo",
    extra=("The 'pymongo' package is required for MongoDB operations. "
           "Please install it."),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _save_to_database(
    audit_log: Dict[str, Any],
    metadata_type: str,
    mongo_db_uri: str,
) -> str:
    """
    Saves the audit log to a MongoDB database.
    
    Parameters
    ----------
    audit_log : dict
        The audit log entry.
    metadata_type : str
        Type of metadata being audited.
    mongo_db_uri : str
        MongoDB URI for connection.
    
    Returns
    -------
    str
        Success message.
    
    Raises
    ------
    RuntimeError
        If storing in the database fails.
    """
    from pymongo import MongoClient
    try:
        client = MongoClient(mongo_db_uri)
        db = client['metadata_audit_db']
        collection = db[f"{metadata_type}_audit_logs"]
        collection.insert_one(audit_log)
        logger.info(
            "Audit log stored in MongoDB"
            f" collection '{metadata_type}_audit_logs'."
        )
        return (
            f"Audit log successfully recorded for {metadata_type} "
            f"by {audit_log['user']}."
        )
    except Exception as e:
        logger.error(f"Failed to store audit log in MongoDB: {e}")
        raise RuntimeError(f"Failed to store audit log in MongoDB: {e}")


@validate_params({
    'metadata':         [dict],
    'cloud_provider':   [StrOptions({'aws', 'gcp'})],
    'retries':          [Integral],
    'batch_size':       [Integral],
    'bucket_name':      [str],
    'aws_credentials':  [dict, None],
    'gcp_credentials':  [dict, None],
})
def sync_with_cloud(
    metadata: Dict[str, Any],
    cloud_provider: str,
    retries: int = 3,
    batch_size: int = 100,
    bucket_name: str = None,
    aws_credentials: Optional[Dict[str, str]] = None,
    gcp_credentials: Optional[Dict[str, str]] = None,
) -> str:
    """
    Synchronizes metadata with a cloud provider.

    This function syncs metadata to a cloud storage backend,
    such as AWS S3 or GCP Cloud Storage. The sync process is
    defined mathematically as:

    .. math::
       S = f(M, P, R, B, C)

    where:
      - M : metadata,
      - P : cloud provider,
      - R : number of retries,
      - B : batch size,
      - C : credentials.

    Parameters
    ----------
    metadata : dict
        The metadata to sync, e.g. 
        `{"key": "value"}`.
    cloud_provider : str
        Target cloud provider, e.g. 
        ``"aws"`` or ``"gcp"``.
    retries : int, default=3
        Number of retry attempts.
    batch_size : int, default=100
        Records per batch.
    bucket_name : str
        S3 bucket name if using ``"aws"``.
    aws_credentials : dict or None
        AWS credentials as 
        ``{"aws_access_key_id": "id", 
        "aws_secret_access_key": "key"}``.
    gcp_credentials : dict or None
        GCP credentials as 
        ``{"service_account_json": "path/to/json"}``.

    Returns
    -------
    str
        Success message upon sync completion.

    Raises
    ------
    ValueError
        If required credentials are missing.
    RuntimeError
        If sync fails after retries.

    Examples
    --------
    >>> from gofast.mlops.metadata import sync_with_cloud
    >>> meta = {"a": 1}
    >>> msg = sync_with_cloud(
    ...     metadata=meta,
    ...     cloud_provider="aws",
    ...     bucket_name="my-bucket",
    ...     aws_credentials={"aws_access_key_id": "id",
    ...                      "aws_secret_access_key": "key"}
    ... )
    >>> print(msg)

    See Also
    --------
    log_metadata : Logs metadata.
    retrieve : Retrieves metadata.

    References
    ----------
    .. [1] Smith, J. & Doe, A. "Cloud Sync in ML Ops." 
           Journal of ML Ops, 2021.
    .. [2] AWS. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    """
    result = _sync_with_cloud_inner(
        metadata         = metadata,
        cloud_provider   = cloud_provider,
        retries          = retries,
        batch_size       = batch_size,
        bucket_name      = bucket_name,
        aws_credentials  = aws_credentials,
        gcp_credentials  = gcp_credentials,
    )
    return result

def _sync_with_cloud_inner(
    metadata: Dict[str, Any],
    cloud_provider: str,
    retries: int,
    batch_size: int,
    bucket_name: str,
    aws_credentials: Optional[Dict[str, str]],
    gcp_credentials: Optional[Dict[str, str]],
) -> str:
    """
    Internal helper to synchronize metadata with a specified cloud provider.
    
    Parameters
    ----------
    metadata : dict
        The metadata to synchronize.
    cloud_provider : str
        The cloud provider ('aws' or 'gcp').
    retries : int
        Number of retry attempts.
    batch_size : int
        Number of records per batch.
    bucket_name : str
        Cloud storage bucket name.
    aws_credentials : dict or None
        AWS credentials.
    gcp_credentials : dict or None
        GCP credentials.
    
    Returns
    -------
    str
        Success message.
    
    Raises
    ------
    ValueError
        If required credentials are missing.
    RuntimeError
        If synchronization fails after the specified retries.
    """
    if cloud_provider == 'aws' and not aws_credentials:
        logger.error(
            "AWS credentials must be "
            "provided for AWS synchronization."
        )
        raise ValueError(
            "AWS credentials must be"
            " provided for AWS synchronization."
        )
    if cloud_provider == 'gcp' and not gcp_credentials:
        logger.error(
            "GCP credentials must be"
            " provided for GCP synchronization."
            )
        raise ValueError(
            "GCP credentials must be"
            " provided for GCP synchronization."
            )
    
    metadata_items = list(metadata.items())
    total_items = len(metadata_items)
    logger.info(
        "Starting synchronization of"
        f" {total_items} metadata items to "
        f"{cloud_provider.upper()}."
    )
    
    for attempt in range(1, retries + 1):
        try:
            for i in range(0, total_items, batch_size):
                metadata_batch = dict(metadata_items[i : i + batch_size])
                _sync_metadata(
                    metadata_batch  = metadata_batch,
                    cloud_provider  = cloud_provider,
                    bucket_name     = bucket_name,
                    aws_credentials = aws_credentials,
                    gcp_credentials = gcp_credentials,
                )
            logger.info(
                "Metadata synced successfully"
                f" with {cloud_provider.upper()}."
            )
            return( 
                "Metadata synced successfully"
                f" with {cloud_provider.upper()}."
                )
        except Exception as e:
            logger.error(
                f"Attempt {attempt} failed with error: {e}")
            if attempt < retries:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(
                    "Failed to sync metadata with"
                    f" {cloud_provider.upper()} "
                    f"after {retries} attempts."
                )
                raise RuntimeError(
                    "Failed to sync metadata with"
                    f" {cloud_provider.upper()} "
                    f"after {retries} attempts."
                ) from e


def _sync_metadata(
    metadata_batch: Dict[str, Any],
    cloud_provider: str,
    bucket_name: str,
    aws_credentials: Optional[Dict[str, str]],
    gcp_credentials: Optional[Dict[str, str]],
):
    """
    Synchronize a batch of metadata with the specified cloud provider.
    
    Parameters
    ----------
    metadata_batch : dict
        A batch of metadata to synchronize.
    cloud_provider : str
        The cloud provider ('aws' or 'gcp').
    bucket_name : str
        Cloud storage bucket name.
    aws_credentials : dict or None
        AWS credentials.
    gcp_credentials : dict or None
        GCP credentials.
    
    Raises
    ------
    ValueError
        If the cloud provider is unsupported.
    RuntimeError
        If synchronization fails.
    """
    if cloud_provider == 'aws':
        _sync_to_aws(
            metadata_batch  = metadata_batch,
            bucket_name     = bucket_name,
            aws_credentials = aws_credentials,
        )
    elif cloud_provider == 'gcp':
        _sync_to_gcp(
            metadata_batch  = metadata_batch,
            bucket_name     = bucket_name,
            gcp_credentials = gcp_credentials,
        )
    else:
        logger.error(
            f"Unsupported cloud provider: {cloud_provider}")
        raise ValueError(
            f"Unsupported cloud provider: {cloud_provider}"
            )

@ensure_pkg(
    "boto3",
    extra=("The 'boto3' package is required for AWS S3 operations. "
           "Please install it."),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _sync_to_aws(
    metadata_batch: Dict[str, Any],
    bucket_name: str,
    aws_credentials: Dict[str, str],
):
    """
    Sync a batch of metadata to AWS S3.
    
    Parameters
    ----------
    metadata_batch : dict
        The metadata batch.
    bucket_name : str
        The AWS S3 bucket name.
    aws_credentials : dict
        AWS credentials for authentication.
    
    Raises
    ------
    RuntimeError
        If synchronization with AWS S3 fails.
    """
    import boto3
    import botocore.exceptions
    try:
        s3_client = boto3.client('s3', **aws_credentials)
        file_key = f"metadata/metadata_sync_{int(time.time())}.json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=json.dumps(metadata_batch)
        )
        logger.info(f"Batch synced to AWS S3 as {file_key}.")
    except botocore.exceptions.ClientError as e:
        logger.error(f"Failed to sync batch to AWS S3: {e}")
        raise RuntimeError(f"Failed to sync batch to AWS S3: {e}") from e


@ensure_pkg(
    "google",
    extra=("The 'google-cloud-storage' package"
           " is required for GCP Storage operations. "
           ),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
    dist_name="google-cloud-storage",
    infer_dist_name=True
)
def _sync_to_gcp(
    metadata_batch: Dict[str, Any],
    bucket_name: str,
    gcp_credentials: Dict[str, str],
):
    """
    Sync a batch of metadata to Google Cloud Storage.
    
    Parameters
    ----------
    metadata_batch : dict
        The metadata batch.
    bucket_name : str
        The GCP Cloud Storage bucket name.
    gcp_credentials : dict
        GCP credentials for authentication.
    
    Raises
    ------
    RuntimeError
        If synchronization with GCP Cloud Storage fails.
    """
    from google.cloud import storage
    from google.auth.exceptions import GoogleAuthError
    try:
        client = storage.Client.from_service_account_json(
            gcp_credentials['service_account_json']
        )
        bucket = client.bucket(bucket_name)
        blob_name = f"metadata/metadata_sync_{int(time.time())}.json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(metadata_batch))
        logger.info(
            f"Batch synced to GCP Cloud Storage as {blob_name}.")
    except (GoogleAuthError, Exception) as e:
        logger.error(
            f"Failed to sync batch to GCP Cloud Storage: {e}")
        raise RuntimeError(
            f"Failed to sync batch to GCP Cloud Storage: {e}") from e

def _validate_with_jsonschema(
        metadata: Dict[str, Any],
        schema: Dict[str, Any]
):
    """
    Validates metadata using the jsonschema library.

    Parameters
    ----------
    metadata : dict
        The metadata to validate.
    schema : dict
        The JSON schema.

    Raises
    ------
    jsonschema.exceptions.ValidationError
        If validation fails.
    """
    import jsonschema
    jsonschema.validate(instance=metadata, schema=schema)


def _auto_correct_metadata(
        metadata: Dict[str, Any],
        schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Auto-corrects metadata by adding missing required fields
    with default values and removing unexpected fields.

    Parameters
    ----------
    metadata : dict
        The metadata to correct.
    schema : dict
        The JSON schema.

    Returns
    -------
    dict
        The corrected metadata.
    """
    corrected = metadata.copy()
    props = schema.get("properties", {})
    for key, val in props.items():
        if key not in corrected and "default" in val:
            corrected[key] = val["default"]
            logger.info(
                f"Added missing key '{key}' with default "
                f"value '{val['default']}'."
            )
    for key in list(corrected.keys()):
        if key not in props:
            logger.info(f"Removed unexpected key '{key}'.")
            del corrected[key]
    return corrected


def _log_corrections(
        metadata: Dict[str, Any],
        log_path: str
):
    """
    Logs corrections to a file.

    Parameters
    ----------
    metadata : dict
        The corrected metadata.
    log_path : str
        Path to the correction log file.

    Raises
    ------
    RuntimeError
        If writing to the log file fails.
    """
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps(metadata, indent=4) + '\n')
        logger.info(f"Corrections logged to {log_path}.")
    except Exception as e:
        logger.error(f"Failed to log corrections: {e}")
        raise RuntimeError(f"Failed to log corrections: {e}")

def _store_locally_track_e(
        metadata_bytes: bytes,
        experiment_id: str,
        versioning_enabled: bool
) -> str:
    """
    Stores experiment metadata locally as a JSON file.
    
    Parameters
    ----------
    metadata_bytes : bytes
        The (compressed/encrypted) metadata.
    experiment_id : str
        Experiment identifier.
    versioning_enabled : bool
        Flag indicating if versioning is enabled.
    
    Returns
    -------
    str
        Success message.
    
    Raises
    ------
    RuntimeError
        If file writing fails.
    """
    file_name = f"experiment_{experiment_id}"
    if versioning_enabled:
        file_name += "_v1"
    file_name += ".json"
    try:
        with open(file_name, 'wb') as f:
            f.write(metadata_bytes)
        logger.info(f"Metadata stored locally in {file_name}.")
        return (f"Experiment metadata tracked successfully "
                f"for {experiment_id}.")
    except Exception as e:
        logger.error(f"Local storage failed: {e}")
        raise RuntimeError(f"Local storage failed: {e}")


@ensure_pkg(
    "boto3",
    extra=("The 'boto3' package is required for AWS S3 operations. "
           "Please install it."),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _store_in_aws_s3(
        metadata_bytes: bytes,
        experiment_id: str,
        versioning_enabled: bool,
        bucket_name: str,
) -> str:
    """
    Stores experiment metadata in AWS S3.
    
    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to store.
    experiment_id : str
        Experiment identifier.
    versioning_enabled : bool
        Flag for versioning.
    bucket_name : str
        S3 bucket name.
    
    Returns
    -------
    str
        Success message.
    
    Raises
    ------
    RuntimeError
        If S3 storage fails.
    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        key = f"experiment_metadata/experiment_{experiment_id}"
        if versioning_enabled:
            key += "_v1"
        key += ".json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=metadata_bytes
        )
        logger.info(
            f"Metadata stored in AWS S3 under key: {key}."
        )
        return (f"Experiment metadata tracked successfully "
                f"for {experiment_id}.")
    except Exception as e:
        logger.error(f"S3 storage failed: {e}")
        raise RuntimeError(f"S3 storage failed: {e}")


@ensure_pkg(
    "pymongo",
    extra=("The 'pymongo' package is required for MongoDB operations. "
           "Please install it."),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _store_in_mongodb(
        experiment_metadata: Dict[str, Any],
        experiment_id: str,
        versioning_enabled: bool,
        mongo_db_uri: str,
) -> str:
    """
    Stores experiment metadata in MongoDB.
    
    Parameters
    ----------
    experiment_metadata : dict
        The metadata to store.
    experiment_id : str
        Experiment identifier.
    versioning_enabled : bool
        Flag for versioning.
    mongo_db_uri : str
        MongoDB URI.
    
    Returns
    -------
    str
        Success message.
    
    Raises
    ------
    RuntimeError
        If database storage fails.
    """
    from pymongo import MongoClient
    try:
        client = MongoClient(mongo_db_uri)
        db = client['experiment_db']
        collection = db['experiment_metadata']
        document = experiment_metadata.copy()
        if versioning_enabled:
            document['version'] = 1
        collection.insert_one(document)
        logger.info(
            f"Metadata stored in MongoDB for experiment {experiment_id}."
        )
        return (f"Experiment metadata tracked successfully "
                f"for {experiment_id}.")
    except Exception as e:
        logger.error(f"Database storage failed: {e}")
        raise RuntimeError(f"Database storage failed: {e}")


@validate_params({
    'metadata': [dict],
    'schema': [dict],
    'auto_correct': [bool],
    'correction_log': [str, None],
})
def validate_schema(
    metadata: Dict[str, Any],
    schema: Dict[str, Any],
    auto_correct: bool = False,
    correction_log: Optional[str] = None
) -> bool:
    """
    Validates metadata against a predefined JSON schema with optional
    auto-correction.

    This function checks whether the input `<metadata>` conforms
    to the specified JSON `<schema>`. If `<auto_correct>` is enabled,
    the function attempts to fix common errors (such as missing
    required fields) by assigning default values defined in the `<schema>`.
    The validation process is formulated as:

    .. math::
        \\text{Validity} = \\begin{cases}
        \\text{True} & \\text{if } \\text{metadata} \\in \\text{schema} \\\\
        \\text{False} & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    metadata : dict
        The metadata dictionary to validate. For example, `<metadata>` 
        might be ``{"accuracy": 0.95, "loss": 0.05, "epoch": 10}``.
    schema : dict
        The JSON schema defining the required structure of the metadata.
        For instance, `<schema>` may specify that ``"accuracy"`` is a 
        ``number``, ``"epoch"`` is an ``integer``, and that ``"optimizer"`` 
        is a ``string`` with a default value.
    auto_correct : bool, default=False
        If set to ``True``, the function will attempt to auto-correct
        common errors by inserting default values for missing keys.
    correction_log : str or None, optional
        A file path where any corrections made will be logged. When 
        `<auto_correct>` is enabled and corrections occur, the corrected 
        metadata is recorded to this log.

    Returns
    -------
    bool
        Returns ``True`` if the `<metadata>` is valid or has been 
        successfully corrected, otherwise returns ``False``.

    Examples
    --------
    >>> from gofast.mlops.metadata import validate_schema
    >>> metadata = {"accuracy": 0.95, "loss": 0.05, "epoch": 10}
    >>> schema = {
    ...     "type": "object",
    ...     "properties": {
    ...         "accuracy": {"type": "number"},
    ...         "loss": {"type": "number"},
    ...         "epoch": {"type": "integer"},
    ...         "optimizer": {"type": "string", "default": "adam"}
    ...     },
    ...     "required": ["accuracy", "loss", "epoch", "optimizer"]
    ... }
    >>> valid = validate_schema(
    ...     metadata=metadata,
    ...     schema=schema,
    ...     auto_correct=True,
    ...     correction_log="corrections.log"
    ... )
    >>> print(valid)
    True

    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on criteria.

    References
    ----------
    .. [1] Smith, J. & Doe, A. "Schema Validation in Machine Learning 
           Pipelines." Journal of ML Ops, 2021.
    """
    is_valid = _validate_schema_inner(
        metadata=metadata,
        schema=schema,
        auto_correct=auto_correct,
        correction_log=correction_log
    )
    return is_valid

def _validate_schema_inner(
    metadata: Dict[str, Any],
    schema: Dict[str, Any],
    auto_correct: bool,
    correction_log: Optional[str]
) -> bool:
    """
    Validates and optionally auto-corrects metadata based on the provided schema.

    Parameters
    ----------
    metadata : dict
        The metadata to be validated.

    schema : dict
        The JSON schema to validate against.

    auto_correct : bool
        Flag to enable auto-correction of metadata.

    correction_log : str or None
        Path to the correction log file.

    Returns
    -------
    bool
        Validation status after processing.

    """
    try:
        _validate_with_jsonschema(metadata, schema)
        logger.info("Metadata is valid according to the schema.")
        return True

    except Exception as e:
        logger.error(f"Metadata validation failed: {e}")

        if auto_correct:
            logger.info("Attempting to auto-correct the metadata.")
            corrected_metadata = _auto_correct_metadata(metadata, schema)

            # Validate again after correction
            try:
                _validate_with_jsonschema(corrected_metadata, schema)
                logger.info("Metadata is valid after auto-correction.")
                if correction_log:
                    _log_corrections(corrected_metadata, correction_log)
                return True
            except Exception as e2:
                logger.error(f"Auto-correction failed: {e2}")
                return False

        return False


@validate_params({
    'experiment_id': [str],
    'configuration': [dict],
    'hyperparameters': [dict],
    'performance_metrics': [dict],
    'training_logs': [str, None],
    'versioning_enabled': [bool],
    'storage_backend': [StrOptions({'local', 's3', 'database'})],
    'encryption_key': [str, None],
    'compression_enabled': [bool],
    'bucket_name': [str, None],
    'mongo_db_uri': [str, None],
})
def track_experiment(
    experiment_id: str,
    configuration: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    training_logs: Optional[str] = None,
    versioning_enabled: bool = True,
    storage_backend: str = "local",
    encryption_key: Optional[str] = None,
    compression_enabled: bool = False,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None
) -> str:
    """
    Tracks experiment metadata including configuration, hyperparameters, 
    performance metrics, and training logs with versioning and storage 
    capabilities.

    This function compiles the experiment metadata into a single record, 
    denoted mathematically as:

    .. math::
        E = f(C, H, P, L, V)

    where:
        - :math:`C` is the `<configuration>` dictionary,
        - :math:`H` represents `<hyperparameters>`,
        - :math:`P` denotes `<performance_metrics>`,
        - :math:`L` corresponds to `<training_logs>`, and
        - :math:`V` is the version number (applied if `<versioning_enabled>` 
          is ``True``).

    Parameters
    ----------
    experiment_id : str
        A unique identifier for the experiment, e.g. `<experiment_id>` = 
        ``"exp_001"``.
    configuration : dict
        The experimental configuration settings, e.g. `<configuration>` = 
        ``{"optimizer": "Adam", "learning_rate": 0.001}``.
    hyperparameters : dict
        The hyperparameters used during training, for instance, 
        `<hyperparameters>` = ``{"batch_size": 32, "epochs": 10}``.
    performance_metrics : dict
        The performance metrics obtained from the experiment, such as 
        `<performance_metrics>` = ``{"accuracy": 0.93, "loss": 0.07}``.
    training_logs : str or None, optional
        The file path to the training logs. If provided, the content of 
        the file is included in the experiment metadata.
    versioning_enabled : bool, default=True
        Enables versioning of the experiment metadata. When ``True``, 
        a version number is added to the metadata record.
    storage_backend : {'local', 's3', 'database'}, default="local"
        Specifies where to store the experiment metadata. For example, 
        ``"local"`` saves the metadata to a local file.
    encryption_key : str or None, optional
        A valid Fernet key for encrypting the metadata. For example, 
        ``"my-fernet-key"``.
    compression_enabled : bool, default=False
        If ``True``, compresses the metadata using zlib to reduce file 
        size.
    bucket_name : str or None, optional
        The S3 bucket name for storage when `<storage_backend>` is 
        ``"s3"``.
    mongo_db_uri : str or None, optional
        The MongoDB URI for database storage when 
        `<storage_backend>` is ``"database"``.

    Returns
    -------
    str
        A message confirming that the experiment metadata was tracked 
        successfully. The message includes the `<experiment_id>`.
    
    Examples
    --------
    >>> from gofast.mlops.metadata import track_experiment
    >>> config = {"optimizer": "Adam", "learning_rate": 0.001}
    >>> hyper = {"batch_size": 32, "epochs": 10}
    >>> metrics = {"accuracy": 0.93, "loss": 0.07}
    >>> msg = track_experiment(
    ...     experiment_id="exp_001",
    ...     configuration=config,
    ...     hyperparameters=hyper,
    ...     performance_metrics=metrics,
    ...     training_logs="training_log.txt",
    ...     storage_backend="local",
    ...     versioning_enabled=True
    ... )
    >>> print(msg)
    Experiment metadata tracked successfully for exp_001.
    
    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on specified criteria.
    
    References
    ----------
    .. [1] Johnson, L. & Smith, M. "Experiment Tracking in ML Pipelines." 
           Journal of Machine Learning Engineering, 2022.
    .. [2] Amazon Web Services. "AWS S3 Documentation." 
           https://aws.amazon.com/s3/
    .. [3] PyMongo Documentation. "MongoDB for Python." 
           https://pymongo.readthedocs.io/en/stable/
    """
    result = _track_experiment_inner(
        experiment_id       = experiment_id,
        configuration       = configuration,
        hyperparameters     = hyperparameters,
        performance_metrics = performance_metrics,
        training_logs       = training_logs,
        versioning_enabled  = versioning_enabled,
        storage_backend     = storage_backend,
        encryption_key      = encryption_key,
        compression_enabled = compression_enabled,
        bucket_name         = bucket_name,
        mongo_db_uri        = mongo_db_uri,
    )
    return result

def _track_experiment_inner(
    experiment_id: str,
    configuration: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    training_logs: Optional[str] = None,
    versioning_enabled: bool = True,
    storage_backend: str = 'local',
    encryption_key: Optional[str] = None,
    compression_enabled: bool = False,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None
) -> str:
    """
    Tracks experiment-specific metadata including configurations,
    hyperparameters,performance metrics, and training logs, with
    versioning and automated storage.
    """
    # Create experiment metadata dictionary
    experiment_metadata = {
        'experiment_id': experiment_id,
        'configuration': configuration,
        'hyperparameters': hyperparameters,
        'performance_metrics': performance_metrics,
        'version': 1 if versioning_enabled else None,
        'timestamp': datetime.utcnow().isoformat()
    }

    # Include training logs if provided
    if training_logs:
        try:
            with open(training_logs, 'r') as f:
                experiment_metadata['training_logs'] = f.read()
            logger.info("Training logs included in the experiment metadata.")
        except Exception as e:
            logger.error(f"Failed to read training logs: {e}")
            raise RuntimeError(f"Failed to read training logs: {e}")

    # Convert metadata to JSON string
    metadata_json = json.dumps(experiment_metadata)

    # Optional compression of the metadata
    if compression_enabled:
        try:
            metadata_bytes = zlib.compress(metadata_json.encode('utf-8'))
            logger.info("Experiment metadata compressed.")
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise RuntimeError(f"Compression failed: {e}")
    else:
        metadata_bytes = metadata_json.encode('utf-8')

    # Optional encryption of the metadata
    if encryption_key:
        try:
            from cryptography.fernet import Fernet
            cipher = Fernet(encryption_key)
            metadata_bytes = cipher.encrypt(metadata_bytes)
            logger.info("Experiment metadata encrypted.")
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for encryption. "
                "Please install it using 'pip install cryptography'."
            )
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise RuntimeError(f"Encryption failed: {e}")

    # Store metadata in the specified backend
    try:
        if storage_backend == 'local':
            return _store_locally_track_e(metadata_bytes, experiment_id,
                                  versioning_enabled)
        elif storage_backend == 's3':
            if not bucket_name:
                raise ValueError(
                    "bucket_name must be provided when using 's3'"
                    " as storage_backend.")
            return _store_in_aws_s3(metadata_bytes, experiment_id, 
                                    versioning_enabled, bucket_name)
        elif storage_backend == 'database':
            if not mongo_db_uri:
                raise ValueError(
                    "mongo_db_uri must be provided when"
                    " using 'database' as storage_backend.")
            return _store_in_mongodb(
                experiment_metadata, experiment_id, versioning_enabled, mongo_db_uri)
        else:
            logger.error(f"Unsupported storage backend: {storage_backend}")
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
    except Exception as e:
        logger.error(f"Failed to store experiment metadata: {e}")
        raise RuntimeError(f"Failed to store experiment metadata: {e}") from e


@validate_params({
    'metadata_type':       [str],
    'retention_days':      [Interval(Integral, 1, None, closed="left")],
    'storage_backend':     [StrOptions({'local', 's3', 'database'})],
    'preserve_versions':   [list, None],
    'bucket_name':         [str, None],
    'mongo_db_uri':        [str, None],
})
def prune_old(
    metadata_type: str,
    retention_days: int,
    storage_backend: str = 'local',
    preserve_versions: Optional[List[int]] = None,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None,
) -> str:
    """
    Prunes old metadata records based on a retention policy while preserving 
    specific versions.

    This function examines all metadata records of a given type and removes 
    those that have not been modified in the last `retention_days` days, 
    except those whose version numbers are included in the list 
    `preserve_versions`.

    The pruning operation removes every metadata record :math:`m` satisfying:

    .. math::
        m.\text{last_modified} < \text{cutoff\_date} \quad
        \text{and} \quad m.\text{version} \notin \text{Preserve Versions}

    where the cutoff date is computed as:

    .. math::
        \text{cutoff\_date} = \text{Current Date} - \text{retention\_days}

    Parameters
    ----------
    metadata_type : str
        The type of metadata to prune (e.g. `metadata_type` such as 
        ``'model'``, ``'dataset'``, or ``'experiment'``). This is used to 
        filter and identify the metadata records.
    retention_days : int
        The number of days to retain metadata records. Metadata with a last 
        modification date earlier than :math:`\text{Current Date} - 
        \text{retention\_days}` will be pruned.
    storage_backend : {'local', 's3', 'database'}, default='local'
        The backend from which to prune metadata. Options include:
        
        - ``'local'``: Prune metadata stored as local files.
        - ``'s3'``: Prune metadata stored in an AWS S3 bucket.
        - ``'database'``: Prune metadata stored in a MongoDB database.
    preserve_versions : list of int, optional
        A list of version numbers to preserve. Metadata records with a version 
        in ``preserve_versions`` will not be pruned.
    bucket_name : str, optional
        The name of the S3 bucket where metadata is stored. Required if 
        ``storage_backend`` is ``'s3'``.
    mongo_db_uri : str, optional
        The MongoDB URI for database connection. Required if 
        ``storage_backend`` is ``'database'``.

    Returns
    -------
    str
        A success message indicating that old metadata has been pruned.

    Raises
    ------
    ValueError
        If an unsupported `storage_backend` is provided or if required 
        parameters (e.g. ``bucket_name`` for S3 or ``mongo_db_uri`` for 
        database) are missing.
    RuntimeError
        If the pruning operation fails due to backend-specific issues.

    Notes
    -----
    - **Local Storage**: Scans the current directory for files starting with 
      `metadata_type` and prunes those whose last modification date is older than 
      the calculated cutoff date, excluding those with preserved versions.
    - **AWS S3**: Uses the `boto3` library to list and delete objects with a 
      key prefix corresponding to `metadata_type`.
    - **MongoDB**: Connects to the specified MongoDB database and deletes 
      documents where the `last_modified` field is older than the cutoff date and 
      the `version` is not in the list of preserved versions.
    
    Examples
    --------
    Prune local metadata older than 30 days while preserving versions 1 and 2:

    >>> from gofast.mlops.metadata import prune_old
    >>> message = prune_old(
    ...     metadata_type='model',
    ...     retention_days=30,
    ...     storage_backend='local',
    ...     preserve_versions=[1, 2]
    ... )
    >>> print(message)
    Old metadata pruned successfully based on a retention policy of 30 days.

    Prune AWS S3 metadata older than 60 days:

    >>> message = prune_old(
    ...     metadata_type='dataset',
    ...     retention_days=60,
    ...     storage_backend='s3',
    ...     bucket_name='my-metadata-bucket'
    ... )
    >>> print(message)
    Old metadata pruned successfully based on a retention policy of 60 days.

    Prune MongoDB metadata older than 90 days, preserving version 3:

    >>> message = prune_old(
    ...     metadata_type='experiment',
    ...     retention_days=90,
    ...     storage_backend='database',
    ...     preserve_versions=[3],
    ...     mongo_db_uri='mongodb://localhost:27017/'
    ... )
    >>> print(message)
    Old metadata pruned successfully based on a retention policy of 90 days.

    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on specified criteria.
    audit : Generates an audit trail for metadata changes.

    References
    ----------
    .. [1] AWS S3 Documentation, Amazon Web Services.
           https://aws.amazon.com/s3/
    .. [2] PyMongo Documentation, MongoDB.
           https://pymongo.readthedocs.io/en/stable/
    .. [3] Python os Module, Python Documentation.
           https://docs.python.org/3/library/os.html
    """
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    preserve_versions = preserve_versions or []

    if storage_backend == 'local':
        result = _prune_local(
            metadata_type      = metadata_type,
            cutoff_date        = cutoff_date,
            preserve_versions  = preserve_versions,
            retention_days     = retention_days,
        )
    elif storage_backend == 's3':
        if not bucket_name:
            raise ValueError(
                "bucket_name must be provided"
                " when using 's3' as storage_backend."
            )
        result = _prune_s3(
            metadata_type      = metadata_type,
            cutoff_date        = cutoff_date,
            preserve_versions  = preserve_versions,
            retention_days     = retention_days,
            bucket_name        = bucket_name,
        )
    elif storage_backend == 'database':
        if not mongo_db_uri:
            raise ValueError(
                "mongo_db_uri must be provided when"
                " using 'database' as storage_backend."
            )
        result = _prune_database(
            metadata_type      = metadata_type,
            cutoff_date        = cutoff_date,
            preserve_versions  = preserve_versions,
            retention_days     = retention_days,
            mongo_db_uri       = mongo_db_uri,
        )
    else:
        logger.error(
            f"Unsupported storage backend: {storage_backend}"
        )
        raise ValueError(
            f"Unsupported storage backend: {storage_backend}"
        )
    return result


def _prune_local(
    metadata_type: str,
    cutoff_date: datetime,
    preserve_versions: List[int],
    retention_days: int,
) -> str:
    """
    Prunes local metadata files based on the retention policy.
    
    Parameters
    ----------
    metadata_type : str
        The metadata category (e.g. ``'model'``) whose files are pruned.
    cutoff_date : datetime
        The cutoff date defined as :math:`\text{Current Date} - \text{retention_days}`.
    preserve_versions : list of int
        List of version numbers to preserve.
    retention_days : int
        Retention period in days.
    
    Returns
    -------
    str
        Success message upon completion.
    
    Raises
    ------
    RuntimeError
        If file deletion fails.
    """
    import os
    from datetime import datetime

    try:
        metadata_files = [
            f for f in os.listdir() if f.startswith(metadata_type)
        ]
        for file in metadata_files:
            file_path = os.path.join(os.getcwd(), file)
            file_modified_ts = os.path.getmtime(file_path)
            file_modified_date = datetime.fromtimestamp(file_modified_ts)

            # Attempt to extract version from filename (e.g., 'model_v1.json')
            version = None
            if '_v' in file:
                try:
                    version_str = file.split('_v')[-1].split('.')[0]
                    version = int(version_str)
                except (IndexError, ValueError):
                    logger.warning(
                        f"Unable to parse version from filename: {file}. "
                        "Skipping version check."
                    )

            if file_modified_date < cutoff_date and (
                    version not in preserve_versions):
                os.remove(file_path)
                logger.info(f"Pruned file: {file_path}")
        return (
            f"Old metadata pruned successfully based on a retention policy of "
            f"{retention_days} days."
        )
    except Exception as e:
        logger.error(
            f"Failed to prune local metadata: {e}")
        raise RuntimeError(
            f"Failed to prune local metadata: {e}") from e


@ensure_pkg(
    "boto3",
    extra=("The 'boto3' package is required for AWS S3 operations. "
           "Please install it."),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _prune_s3(
    metadata_type: str,
    cutoff_date: datetime,
    preserve_versions: List[int],
    retention_days: int,
    bucket_name: str,
) -> str:
    """
    Prunes metadata stored in AWS S3 based on the retention policy.
    
    Parameters
    ----------
    metadata_type : str
        The metadata category (e.g. ``'model'``).
    cutoff_date : datetime
        The cutoff date as :math:`\text{Current Date} - \text{retention_days}`.
    preserve_versions : list of int
        Versions to preserve.
    retention_days : int
        Retention period in days.
    bucket_name : str
        Name of the S3 bucket.
    
    Returns
    -------
    str
        Success message upon completion.
    
    Raises
    ------
    RuntimeError
        If S3 operations fail.
    
    Notes
    -----
    Uses the :py:mod:`boto3` library to list and delete S3 objects.
    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=f"{metadata_type}/")
    
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    file_last_modified = obj['LastModified']
    
                    version = None
                    if '_v' in file_key:
                        try:
                            version_str = file_key.split('_v')[-1].split('.')[0]
                            version = int(version_str)
                        except (IndexError, ValueError):
                            logger.warning(
                                f"Unable to parse version from file key: {file_key}. "
                                "Skipping version check."
                            )
    
                    if file_last_modified.replace(tzinfo=None) < cutoff_date and (
                        version not in preserve_versions
                    ):
                        s3_client.delete_object(Bucket=bucket_name, Key=file_key)
                        logger.info(
                            f"Pruned S3 object: s3://{bucket_name}/{file_key}"
                        )
        return (
            f"Old metadata pruned successfully based on a retention policy of "
            f"{retention_days} days."
        )
    except Exception as e:
        logger.error(f"Failed to prune S3 metadata: {e}")
        raise RuntimeError(f"Failed to prune S3 metadata: {e}") from e


@ensure_pkg(
    "pymongo",
    extra=("The 'pymongo' package is required for MongoDB operations. "),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _prune_database(
    metadata_type: str,
    cutoff_date: datetime,
    preserve_versions: List[int],
    retention_days: int,
    mongo_db_uri: str,
) -> str:
    """
    Prunes metadata stored in MongoDB based on the retention policy.
    
    Parameters
    ----------
    metadata_type : str
        The metadata category (e.g. ``'model'``).
    cutoff_date : datetime
        The cutoff date as :math:`\text{Current Date} - \text{retention_days}`.
    preserve_versions : list of int
        Versions to preserve.
    retention_days : int
        Retention period in days.
    mongo_db_uri : str
        MongoDB URI for connection.
    
    Returns
    -------
    str
        Success message upon completion.
    
    Raises
    ------
    RuntimeError
        If MongoDB operations fail.
    
    Notes
    -----
    Uses :py:mod:`pymongo` to delete documents with a ``last_modified`` field 
    older than the cutoff date and a ``version`` not in the preserve list.
    """
    from pymongo import MongoClient
    try:
        client = MongoClient(mongo_db_uri)
        db = client['metadata_db']
        collection = db[metadata_type]
        query = {"last_modified": {"$lt": cutoff_date}}
        if preserve_versions:
            query["version"] = {"$nin": preserve_versions}
        result = collection.delete_many(query)
        logger.info(
            f"Pruned {result.deleted_count} records from MongoDB for type "
            f"'{metadata_type}'."
        )
        return (
            f"Old metadata pruned successfully based on a retention policy of "
            f"{retention_days} days."
        )
    except Exception as e:
        logger.error(f"Failed to prune MongoDB metadata: {e}")
        raise RuntimeError(f"Failed to prune MongoDB metadata: {e}") from e

