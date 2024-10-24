# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Incorporate security best practices into machine learning workflows, 
ensuring models and data are protected throughout their lifecycle.
"""
#XXX TO OPTIMIZE 
import os
import json
import threading
from numbers import Integral 
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sklearn.utils._param_validation import StrOptions

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 
from ..api.property import BaseClass
from ..compat.sklearn import validate_params, Interval 
from ..tools.funcutils import ensure_pkg 
from .._gofastlog import gofastlog 

# Initialize logger
logger=gofastlog.get_gofast_logger(__name__)

__all__ = [
    'BaseSecurity',
    'DataEncryption',
    'ModelProtection',
    'SecureDeployment',
    'AuditTrail',
    'AccessControl'
]


class BaseSecurity(BaseClass):
    """
    Provides security functionalities such as data encryption,
    decryption, and secure logging. Supports key management,
    encryption algorithms, and secure log handling.

    Parameters
    ----------
    encryption_key : str or None, default=None
        The encryption key to use. If `None`, a new key is generated
        using ``Fernet.generate_key()``.

    log_file : str, default='security_log.json'
        The path to the log file where security events are recorded.

    encryption_algorithm : {'fernet', 'sha256'}, default='fernet'
        The encryption algorithm to use. Options are `'fernet'` for
        symmetric encryption or `'sha256'` for hashing.

    log_encryption_enabled : bool, default=False
        If `True`, the log entries are encrypted using the specified
        encryption algorithm.

    log_retention_days : int or None, default=None
        The number of days to retain logs. If `None`, logs are not
        pruned based on age.

    Attributes
    ----------
    encryption_key_ : bytes
        The encryption key used for encryption and decryption.

    encryption_algorithm_ : str
        The encryption algorithm used.

    log_file_ : str
        The path to the log file.

    log_encryption_enabled_ : bool
        Indicates if log encryption is enabled.

    log_retention_days_ : int or None
        The number of days to retain logs.

    Methods
    -------
    encrypt(data)
        Encrypts data using the selected encryption algorithm.

    decrypt(encrypted_data)
        Decrypts data using the selected encryption algorithm.

    export_key(file_path, encrypted=False)
        Exports the encryption key to a file, with optional encryption.

    import_key(file_path, encrypted=False)
        Imports the encryption key from a file, with optional decryption.

    rotate_key(new_key=None)
        Rotates the encryption key and re-encrypts the log file if necessary.

    log_event(event_type, details)
        Logs an event for security auditing, with optional encryption.

    retrieve_logs(decrypt=True)
        Retrieves the logs stored in the log file, with optional decryption.

    prune_old_logs()
        Prunes logs that exceed the log retention policy.

    backup_logs(backup_path)
        Backs up the log file to a specified path.

    Notes
    -----
    The class uses the `cryptography` library for encryption and
    decryption operations. When using the `'fernet'` algorithm, data
    is encrypted symmetrically. The `'sha256'` algorithm is used for
    hashing purposes and does not support decryption.

    Examples
    --------
    >>> from gofast.mlops.security import BaseSecurity
    >>> security = BaseSecurity()
    >>> encrypted_data = security.encrypt(b'secret data')
    >>> decrypted_data = security.decrypt(encrypted_data)
    >>> print(decrypted_data)
    b'secret data'

    See Also
    --------
    cryptography.fernet.Fernet : Symmetric encryption using Fernet.
    hashlib.sha256 : Secure hashing using SHA-256.

    References
    ----------
    .. [1] D. Eastlake and P. Jones, "US Secure Hash Algorithm 1 (SHA1)",
       RFC 3174, 2001.

    """

    @validate_params({
        'encryption_key': [str, None],
        'log_file': [str],
        'encryption_algorithm': [StrOptions({'fernet', 'sha256'})],
        'log_encryption_enabled': [bool],
        'log_retention_days': [Interval(Integral, 1, None, closed='left'), None],
    })
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        log_file: str = 'security_log.json',
        encryption_algorithm: str = 'fernet',
        log_encryption_enabled: bool = False,
        log_retention_days: Optional[int] = None
    ):
        self.encryption_key_ = encryption_key or self._generate_key()
        self.encryption_algorithm_ = encryption_algorithm
        self.log_file_ = log_file
        self.log_encryption_enabled_ = log_encryption_enabled
        self.log_retention_days_ = log_retention_days
        self.cipher_ = self._initialize_cipher()

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for encryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _generate_key(self) -> bytes:
        """Generates a new encryption key using Fernet."""
        from cryptography.fernet import Fernet
        return Fernet.generate_key()

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for encryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _initialize_cipher(self):
        """Initialize cipher based on the selected encryption algorithm."""
        if self.encryption_algorithm_ == 'fernet':
            from cryptography.fernet import Fernet
            return Fernet(self.encryption_key_)
        elif self.encryption_algorithm_ == 'sha256':
            import hashlib
            return hashlib.sha256()
        else:
            raise ValueError("Unsupported encryption algorithm")

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for encryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypts data using the selected encryption algorithm.

        Parameters
        ----------
        data : bytes
            The data to encrypt.

        Returns
        -------
        encrypted_data : bytes
            The encrypted data.

        Raises
        ------
        ValueError
            If the encryption algorithm is unsupported.

        Notes
        -----
        When using the `'fernet'` algorithm, the data is encrypted
        symmetrically using the Fernet cipher. For the `'sha256'`
        algorithm, the data is hashed using SHA-256 and cannot be
        decrypted.

        Examples
        --------
        >>> encrypted_data = security.encrypt(b'secret data')

        """
        if self.encryption_algorithm_ == 'fernet':
            return self.cipher_.encrypt(data)
        elif self.encryption_algorithm_ == 'sha256':
            import hashlib
            return hashlib.sha256(data).hexdigest().encode()
        else:
            raise ValueError("Unsupported encryption algorithm for encryption")

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for decryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts data using the selected encryption algorithm.

        Parameters
        ----------
        encrypted_data : bytes
            The data to decrypt.

        Returns
        -------
        data : bytes
            The decrypted data.

        Raises
        ------
        ValueError
            If the encryption algorithm does not support decryption.

        Notes
        -----
        Decryption is only supported when using the `'fernet'`
        encryption algorithm.

        Examples
        --------
        >>> decrypted_data = security.decrypt(encrypted_data)

        """
        if self.encryption_algorithm_ == 'fernet':
            return self.cipher_.decrypt(encrypted_data)
        else:
            raise ValueError("Decryption is only supported for Fernet algorithm")

    def export_key(self, file_path: str, encrypted: bool = False):
        """
        Exports the encryption key to a file, with optional encryption.

        Parameters
        ----------
        file_path : str
            The path to the file where the key will be saved.

        encrypted : bool, default=False
            If `True`, the key is encrypted before saving.

        Raises
        ------
        ValueError
            If encryption is requested but the algorithm does not support it.

        Notes
        -----
        When `encrypted` is `True`, the key is encrypted using the
        current cipher before being saved to the file.

        Examples
        --------
        >>> security.export_key('keyfile.key', encrypted=True)

        """
        key_data = self.encryption_key_
        if encrypted:
            if self.encryption_algorithm_ == 'fernet':
                key_data = self.cipher_.encrypt(self.encryption_key_)
            else:
                raise ValueError("Key encryption is only supported for Fernet algorithm")
        with open(file_path, 'wb') as key_file:
            key_file.write(key_data)

    def import_key(self, file_path: str, encrypted: bool = False):
        """
        Imports the encryption key from a file, with optional decryption.

        Parameters
        ----------
        file_path : str
            The path to the file from which the key will be loaded.

        encrypted : bool, default=False
            If `True`, the key is decrypted after loading.

        Raises
        ------
        ValueError
            If decryption is requested but the algorithm does not support it.

        Notes
        -----
        When `encrypted` is `True`, the key is decrypted using the
        current cipher after being loaded from the file.

        Examples
        --------
        >>> security.import_key('keyfile.key', encrypted=True)

        """
        with open(file_path, 'rb') as key_file:
            key_data = key_file.read()
            if encrypted:
                if self.encryption_algorithm_ == 'fernet':
                    key_data = self.cipher_.decrypt(key_data)
                else:
                    raise ValueError("Key decryption is only supported for Fernet algorithm")
            self.encryption_key_ = key_data
            self.cipher_ = self._initialize_cipher()

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for key rotation.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def rotate_key(self, new_key: Optional[bytes] = None):
        """
        Rotates the encryption key and re-encrypts the log file if necessary.

        Parameters
        ----------
        new_key : bytes or None, default=None
            The new encryption key. If `None`, a new key is generated
            using ``Fernet.generate_key()``.

        Notes
        -----
        When the key is rotated and log encryption is enabled, the
        existing logs are re-encrypted using the new key.

        Examples
        --------
        >>> security.rotate_key()

        """
        old_key = self.encryption_key_
        self.encryption_key_ = new_key or self._generate_key()
        self.cipher_ = self._initialize_cipher()
        logger.info("Encryption key rotated.")

        # Re-encrypt logs if enabled
        if self.log_encryption_enabled_:
            self._reencrypt_logs(old_key)

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for re-encrypting logs.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _reencrypt_logs(self, old_key: bytes):
        """Re-encrypts the log file with the new key."""
        from cryptography.fernet import Fernet
        old_cipher = Fernet(old_key) # XXX It seems Fix here  
        logs = self.retrieve_logs(decrypt=True)
        with open(self.log_file_, 'wb') as f:
            for log in logs:
                log_data = json.dumps(log).encode()
                encrypted_log = self.encrypt(log_data)
                f.write(encrypted_log + b'\n')
        logger.info("Re-encrypted logs with new encryption key.")

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """
        Logs an event for security auditing, with optional encryption.

        Parameters
        ----------
        event_type : str
            The type of the event to log.

        details : dict of str to Any
            Additional details about the event.

        Notes
        -----
        The log entry includes a timestamp in ISO format, the event type,
        and the event details. If log encryption is enabled, the log
        entry is encrypted before being written to the log file.

        Examples
        --------
        >>> security.log_event('login_attempt', {'user': 'admin'})

        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details
        }
        log_data = json.dumps(log_entry).encode()
        if self.log_encryption_enabled_:
            log_data = self.encrypt(log_data)

        with open(self.log_file_, 'ab') as f:
            f.write(log_data + b'\n')

        logger.info(f"Logged event: {event_type} - {details}")

    def retrieve_logs(self, decrypt: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves the logs stored in the log file, with optional decryption.

        Parameters
        ----------
        decrypt : bool, default=True
            If `True`, decrypts the log entries if they are encrypted.

        Returns
        -------
        logs : list of dict
            A list of log entries.

        Raises
        ------
        ValueError
            If decryption is requested but the algorithm does not support it.

        Notes
        -----
        If log encryption is enabled and `decrypt` is `True`, the log
        entries are decrypted before being parsed.

        Examples
        --------
        >>> logs = security.retrieve_logs()

        """
        logs = []
        with open(self.log_file_, 'rb') as f:
            for line in f:
                line = line.strip()
                if self.log_encryption_enabled_ and decrypt:
                    line = self.decrypt(line)
                logs.append(json.loads(line))
        return logs

    def prune_old_logs(self):
        """
        Prunes logs that exceed the log retention policy.

        Notes
        -----
        Removes log entries that are older than the retention period
        specified in `log_retention_days_`. If `log_retention_days_`
        is `None`, no pruning is performed.

        Examples
        --------
        >>> security.prune_old_logs()

        """
        if not self.log_retention_days_:
            return
        cutoff_date = datetime.utcnow() - timedelta(days=self.log_retention_days_)
        pruned_logs = []
        with open(self.log_file_, 'rb') as f:
            for line in f:
                line_data = line.strip()
                if self.log_encryption_enabled_:
                    line_data = self.decrypt(line_data)
                log = json.loads(line_data)
                log_date = datetime.fromisoformat(log['timestamp'])
                if log_date >= cutoff_date:
                    pruned_logs.append(line)
        with open(self.log_file_, 'wb') as f:
            f.writelines(pruned_logs)
        logger.info(f"Pruned logs older than {self.log_retention_days_} days.")

    def backup_logs(self, backup_path: str):
        """
        Backs up the log file to a specified path.

        Parameters
        ----------
        backup_path : str
            The destination path for the backup file.

        Notes
        -----
        The method creates the backup directory if it does not exist.

        Examples
        --------
        >>> security.backup_logs('backups/security_log_backup.json')

        """
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(self.log_file_, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        logger.info(f"Backed up logs to {backup_path}")



from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import zlib


class DataEncryption(BaseSecurity):
    """
    A class for handling data encryption and decryption with optional 
    compression, supporting various encryption algorithms.

    Parameters
    ----------
    encryption_algorithm : str, default="fernet"
        The encryption algorithm to use. Supported algorithms include 
        'fernet', 'aes', and 'chacha20'.
    compression : bool, default=False
        If True, the data is compressed before encryption and decompressed 
        after decryption.
    encryption_key : str, optional
        The key used for encryption. Required for certain encryption 
        algorithms like 'aes' and 'chacha20'.

    Attributes
    ----------
    compression_ : bool
        Whether data compression is enabled.
    encryption_algorithm_ : str
        The algorithm used for encryption.
    encryption_key_ : str
        The key used for encryption.

    Examples
    --------
    >>> from gofast.mlops.security import DataEncryption
    >>> encryptor = DataEncryption(encryption_algorithm='aes', 
                                   compression=True, encryption_key="my_secret_key")
    >>> encrypted_data = encryptor.encrypt_data(b"Secret data")
    >>> decrypted_data = encryptor.decrypt_data(encrypted_data)

    See Also
    --------
    BaseSecurity : The base class that provides core security functionality.

    Notes
    -----
    This class supports various encryption methods, including symmetric 
    encryption (e.g., AES and ChaCha20) and asymmetric encryption using 
    the 'fernet' algorithm.
    """

    @validate_params({
        'encryption_algorithm': [StrOptions({"fernet", "aes", "chacha20"})],
        'compression': [bool],
        'encryption_key': [Optional[str]]
    })
    def __init__(
        self, 
        encryption_algorithm: str = "fernet", 
        compression: bool = False, 
        encryption_key: Optional[str] = None
    ):
        """
        Initialize the DataEncryption class with the specified encryption 
        algorithm and optional compression.

        Raises
        ------
        ValueError
            If an unsupported encryption algorithm is provided or if the 
            encryption key is missing for certain algorithms.
        """
        super().__init__(encryption_key)
        self.compression_ = compression
        self.encryption_algorithm_ = encryption_algorithm
        self.additional_ciphers_ = {
            'aes': algorithms.AES(self.encryption_key),
            'chacha20': algorithms.ChaCha20(self.encryption_key[:32], b"0" * 16)
        }

    def _apply_compression(self, data: bytes) -> bytes:
        """Apply compression to the data if compression is enabled."""
        if self.compression_:
            return zlib.compress(data)
        return data

    def _apply_decompression(self, data: bytes) -> bytes:
        """Decompress the data if compression is enabled."""
        if self.compression_:
            return zlib.decompress(data)
        return data

    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data using the specified encryption algorithm.

        Parameters
        ----------
        data : bytes
            The data to be encrypted.

        Returns
        -------
        bytes
            The encrypted data.

        Raises
        ------
        ValueError
            If an unsupported encryption algorithm is specified.
        """
        data = self._apply_compression(data)
        
        if self.encryption_algorithm_ == "fernet":
            encrypted_data = self.encrypt(data)
        else:
            encrypted_data = self._encrypt_with_cipher(data)
        
        self.log_event('data_encryption', {'data_length': len(data)})
        return encrypted_data

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using the specified encryption algorithm.

        Parameters
        ----------
        encrypted_data : bytes
            The encrypted data to be decrypted.

        Returns
        -------
        bytes
            The decrypted data.

        Raises
        ------
        ValueError
            If an unsupported encryption algorithm is specified.
        """
        if self.encryption_algorithm_ == "fernet":
            decrypted_data = self.decrypt(encrypted_data)
        else:
            decrypted_data = self._decrypt_with_cipher(encrypted_data)
        
        decrypted_data = self._apply_decompression(decrypted_data)
        self.log_event('data_decryption', {'data_length': len(decrypted_data)})
        return decrypted_data

    def encrypt_file(self, file_path: str, output_path: str):
        """
        Encrypt a file and save the encrypted content to a new file.

        Parameters
        ----------
        file_path : str
            The path of the file to be encrypted.
        output_path : str
            The path where the encrypted file will be saved.
        """
        with open(file_path, 'rb') as f:
            file_data = f.read()
        encrypted_data = self.encrypt_data(file_data)
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        self.log_event('file_encryption', {'file': file_path, 'output': output_path})

    def decrypt_file(self, encrypted_file_path: str, output_path: str):
        """
        Decrypt an encrypted file and save the decrypted content to a new file.

        Parameters
        ----------
        encrypted_file_path : str
            The path of the encrypted file.
        output_path : str
            The path where the decrypted file will be saved.
        """
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = self.decrypt_data(encrypted_data)
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        self.log_event('file_decryption', {'file': encrypted_file_path, 'output': output_path})

    def _encrypt_with_cipher(self, data: bytes) -> bytes:
        """
        Encrypt data using the specified cipher algorithm.

        Parameters
        ----------
        data : bytes
            The data to be encrypted.

        Returns
        -------
        bytes
            The encrypted data.

        Raises
        ------
        ValueError
            If an unsupported encryption algorithm is specified.
        """
        if self.encryption_algorithm_ not in self.additional_ciphers_:
            raise ValueError(f"Unsupported encryption algorithm: {self.encryption_algorithm_}")

        cipher = Cipher(self.additional_ciphers_[self.encryption_algorithm_], 
                        modes.CFB(b"0" * 16), 
                        backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(data) + encryptor.finalize()

    def _decrypt_with_cipher(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using the specified cipher algorithm.

        Parameters
        ----------
        encrypted_data : bytes
            The data to be decrypted.

        Returns
        -------
        bytes
            The decrypted data.

        Raises
        ------
        ValueError
            If an unsupported encryption algorithm is specified.
        """
        if self.encryption_algorithm_ not in self.additional_ciphers_:
            raise ValueError(f"Unsupported encryption algorithm: {self.encryption_algorithm_}")

        cipher = Cipher(self.additional_ciphers_[self.encryption_algorithm_], 
                        modes.CFB(b"0" * 16), 
                        backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_data) + decryptor.finalize()

    def change_encryption_algorithm(self, algorithm: str):
        """
        Change the encryption algorithm used by the class.

        Parameters
        ----------
        algorithm : str
            The new encryption algorithm to use. Supported algorithms are
            'fernet', 'aes', and 'chacha20'.

        Raises
        ------
        ValueError
            If an unsupported algorithm is specified.
        """
        if algorithm not in ['fernet', 'aes', 'chacha20']:
            raise ValueError("Supported algorithms: 'fernet', 'aes', 'chacha20'")
        self.encryption_algorithm_ = algorithm
        self.log_event('encryption_algorithm_change', {'algorithm': algorithm})


import hashlib
import hmac
from typing import Optional, Dict


class ModelProtection(BaseSecurity):
    """
    A class for securing machine learning models by offering functionality 
    for encryption, signing, verification, and backup. This class allows 
    users to protect their models using HMAC-based signing and secure 
    encryption algorithms.

    Parameters
    ----------
    secret_key : bytes, optional
        A secret key used for signing the model. If not provided, a default 
        key is generated.
    encryption_key : str, optional
        A key used for encrypting the model data.
    hash_algorithm : str, default='sha256'
        The hash algorithm used for signing the model. Supported algorithms 
        include 'sha256', 'sha512', and 'md5'.
    compression : bool, default=False
        If True, the model data is compressed before encryption and decompressed 
        after decryption.

    Attributes
    ----------
    secret_key_ : bytes
        The secret key used for signing and verifying the model.
    encryption_key_ : str
        The key used for encrypting and decrypting the model data.
    hash_algorithm_ : str
        The algorithm used for hashing and signing the model.
    compression_ : bool
        Indicates whether data compression is enabled.

    Examples
    --------
    >>> from gofast.mlops.security import ModelProtection
    >>> protector = ModelProtection(secret_key=b'super_secret_key', compression=True)
    >>> model_data = b"model binary data"
    >>> signature = protector.sign_model(model_data)
    >>> verified = protector.verify_model(model_data, signature)

    See Also
    --------
    DataEncryption : Class for handling general data encryption and decryption.
    """

    @validate_params({
        'secret_key': [Optional[bytes]],
        'encryption_key': [Optional[str]],
        'hash_algorithm': [StrOptions({"sha256", "sha512", "md5"})],
        'compression': [bool]
    })
    def __init__(
        self, 
        secret_key: Optional[bytes] = None, 
        encryption_key: Optional[str] = None, 
        hash_algorithm: str = 'sha256', 
        compression: bool = False
    ):
        """
        Initializes the ModelProtection class with the given encryption key,
        hash algorithm, and optional compression.
        
        Raises
        ------
        ValueError
            If an unsupported hash algorithm is provided.
        """
        super().__init__(encryption_key)
        self.secret_key_ = secret_key or hashlib.sha256(b'default_key').digest()
        self.hash_algorithm_ = hash_algorithm.lower()
        self.compression_ = compression
        self.allowed_hash_algorithms = {'sha256', 'sha512', 'md5'}
        if self.hash_algorithm_ not in self.allowed_hash_algorithms:
            raise ValueError(f"Invalid hash algorithm: {self.hash_algorithm_}")

    def _apply_compression(self, data: bytes) -> bytes:
        """Compress the data if compression is enabled."""
        if self.compression_:
            return zlib.compress(data)
        return data

    def _apply_decompression(self, data: bytes) -> bytes:
        """Decompress the data if compression is enabled."""
        if self.compression_:
            return zlib.decompress(data)
        return data

    def sign_model(self, model_data: bytes) -> str:
        """
        Sign the model data using HMAC with the selected hash algorithm.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be signed.

        Returns
        -------
        str
            The hexadecimal HMAC signature of the model data.
        """
        if self.hash_algorithm_ == 'sha256':
            signature = hmac.new(self.secret_key_, model_data, hashlib.sha256).hexdigest()
        elif self.hash_algorithm_ == 'sha512':
            signature = hmac.new(self.secret_key_, model_data, hashlib.sha512).hexdigest()
        elif self.hash_algorithm_ == 'md5':
            signature = hmac.new(self.secret_key_, model_data, hashlib.md5).hexdigest()

        self.log_event('model_sign', {'signature': signature})
        return signature

    def verify_model(self, model_data: bytes, signature: str) -> bool:
        """
        Verify the integrity of the model by checking its signature.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be verified.
        signature : str
            The expected signature of the model.

        Returns
        -------
        bool
            True if the model's signature matches, False otherwise.
        """
        if self.hash_algorithm_ == 'sha256':
            calculated_signature = hmac.new(
                self.secret_key_, model_data, hashlib.sha256).hexdigest()
        elif self.hash_algorithm_ == 'sha512':
            calculated_signature = hmac.new(
                self.secret_key_, model_data, hashlib.sha512).hexdigest()
        elif self.hash_algorithm_ == 'md5':
            calculated_signature = hmac.new(
                self.secret_key_, model_data, hashlib.md5).hexdigest()

        verified = hmac.compare_digest(calculated_signature, signature)
        self.log_event('model_verification', {'verified': verified})
        return verified

    def encrypt_model(self, model_data: bytes) -> bytes:
        """
        Encrypt the model data after optional compression.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be encrypted.

        Returns
        -------
        bytes
            The encrypted model data.
        """
        model_data = self._apply_compression(model_data)
        encrypted_model = self.encrypt(model_data)
        self.log_event('model_encryption', {'model_size': len(model_data)})
        return encrypted_model

    def decrypt_model(self, encrypted_model_data: bytes) -> bytes:
        """
        Decrypt the model data and decompress if necessary.

        Parameters
        ----------
        encrypted_model_data : bytes
            The encrypted model data to be decrypted.

        Returns
        -------
        bytes
            The decrypted model data.
        """
        decrypted_model = self.decrypt(encrypted_model_data)
        decrypted_model = self._apply_decompression(decrypted_model)
        self.log_event('model_decryption', {'model_size': len(decrypted_model)})
        return decrypted_model

    def track_model_version(self, model_data: bytes, version: str) -> str:
        """
        Track and log the version of the model, returning a version ID.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model.
        version : str
            The version of the model.

        Returns
        -------
        str
            A unique version ID for the model based on its content and version.
        """
        version_id = hashlib.sha256(model_data + version.encode()).hexdigest()
        self.log_event('model_version_track', {'version': version, 'version_id': version_id})
        return version_id

    def rotate_secret_key(self, new_secret_key: Optional[bytes] = None):
        """
        Rotate the secret key used for signing and verifying the model.

        Parameters
        ----------
        new_secret_key : bytes, optional
            A new secret key. If not provided, a new default key will be generated.
        """
        self.secret_key_ = new_secret_key or hashlib.sha256(b'new_default_key').digest()
        self.log_event('secret_key_rotation', {'new_key': True})

    def integrity_check(self, model_data: bytes) -> bool:
        """
        Perform a basic integrity check on the model, such as verifying the model size.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be checked.

        Returns
        -------
        bool
            True if the integrity check passes, False otherwise.
        """
        expected_size = 5000000  # Expected model size in bytes (e.g., 5MB)
        if len(model_data) > expected_size * 1.2 or len(model_data) < expected_size * 0.8:
            self.log_event('integrity_check', {'status': 'failed', 'size': len(model_data)})
            return False

        self.log_event('integrity_check', {'status': 'passed', 'size': len(model_data)})
        return True

    def backup_model(self, model_data: bytes, backup_path: str):
        """
        Backup the model by encrypting and saving it to a file.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be backed up.
        backup_path : str
            The file path where the encrypted model backup will be saved.
        """
        encrypted_model = self.encrypt_model(model_data)
        with open(backup_path, 'wb') as backup_file:
            backup_file.write(encrypted_model)
        self.log_event('model_backup', {'backup_path': backup_path})

    def restore_model(self, backup_path: str) -> bytes:
        """
        Restore the model from an encrypted backup file.

        Parameters
        ----------
        backup_path : str
            The path to the encrypted backup file.

        Returns
        -------
        bytes
            The decrypted model data.
        """
        with open(backup_path, 'rb') as backup_file:
            encrypted_model = backup_file.read()
        decrypted_model = self.decrypt_model(encrypted_model)
        self.log_event('model_restore', {'backup_path': backup_path})
        return decrypted_model


import jwt
from typing import Optional, List, Dict, Callable
from sklearn.utils._param_validation import validate_params, StrOptions
from gofast.tools.funcutils import ensure_pkg

class SecureDeployment(BaseSecurity):
    """
    A class for managing secure deployment workflows, including token-based 
    authentication, role-based access control (RBAC), IP whitelisting, and 
    auditing. Provides functionality for generating, verifying, and revoking 
    authentication tokens and refresh tokens.

    Parameters
    ----------
    secret_key : str
        A secret key used for signing tokens (JWT).
    auth_method : str, default='token'
        The authentication method, default is token-based authentication.
    encryption_key : str, optional
        The encryption key used for secure data transmission (if needed).
    allow_refresh_tokens : bool, default=True
        If True, allows the generation and verification of refresh tokens.
    token_blacklist : list of str, optional
        A list of tokens that are revoked or blacklisted.

    Attributes
    ----------
    secret_key_ : str
        The secret key used to sign and verify tokens.
    auth_method_ : str
        The authentication method in use.
    allow_refresh_tokens_ : bool
        Indicates if refresh tokens are enabled.
    token_blacklist_ : list of str
        A list containing blacklisted tokens.
    
    Examples
    --------
    >>> from gofast.mlops.security import SecureDeployment
    >>> deploy = SecureDeployment(secret_key="super_secret_key")
    >>> token = deploy.generate_token(user_id="user123", roles=["admin"])
    >>> valid = deploy.verify_token(token)

    See Also
    --------
    ModelProtection : Class for securing models with encryption and signing.
    """

    @validate_params({
        'secret_key': [str],
        'auth_method': [StrOptions({"token", "password"})],
        'encryption_key': [Optional[str]],
        'allow_refresh_tokens': [bool],
        'token_blacklist': [Optional[List[str]]] 
    })
    def __init__(
        self, 
        secret_key: str, 
        auth_method: str = 'token', 
        encryption_key: Optional[str] = None, 
        allow_refresh_tokens: bool = True,
        token_blacklist: Optional[List[str]] = None
    ):
        """
        Initialize the SecureDeployment class with the given secret key, 
        authentication method, and token handling options.
        """
        super().__init__(encryption_key)
        self.secret_key_ = secret_key
        self.auth_method_ = auth_method
        self.allow_refresh_tokens_ = allow_refresh_tokens
        self.token_blacklist_ = token_blacklist or []

    def generate_token(
        self, 
        user_id: str, 
        expires_in: int = 3600, 
        roles: Optional[List[str]] = None, 
        custom_claims: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a JWT token for a user with optional expiration and roles.

        Parameters
        ----------
        user_id : str
            The user ID for whom the token is being generated.
        expires_in : int, default=3600
            The time (in seconds) until the token expires.
        roles : list of str, optional
            A list of roles to include in the token.
        custom_claims : dict of str, optional
            Custom claims to add to the token payload.

        Returns
        -------
        str
            The encoded JWT token.
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in),
            'roles': roles or []
        }
        
        if custom_claims:
            payload.update(custom_claims)
        
        token = jwt.encode(payload, self.secret_key_, algorithm='HS256')
        self.log_event('token_generation', {'user_id': user_id, 'roles': roles})
        return token

    def generate_refresh_token(self, user_id: str, expires_in: int = 86400) -> str:
        """
        Generate a refresh token for a user.

        Parameters
        ----------
        user_id : str
            The user ID for whom the refresh token is being generated.
        expires_in : int, default=86400
            The time (in seconds) until the refresh token expires.

        Returns
        -------
        str
            The encoded refresh token.

        Raises
        ------
        ValueError
            If refresh tokens are not enabled in the class configuration.
        """
        if not self.allow_refresh_tokens_:
            raise ValueError("Refresh tokens are not enabled.")
        
        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in),
            'type': 'refresh_token'
        }
        refresh_token = jwt.encode(payload, self.secret_key_, algorithm='HS256')
        self.log_event('refresh_token_generation', {'user_id': user_id})
        return refresh_token

    def verify_token(self, token: str) -> bool:
        """
        Verify the validity of a token.

        Parameters
        ----------
        token : str
            The JWT token to be verified.

        Returns
        -------
        bool
            True if the token is valid and not expired, False otherwise.
        """
        if token in self.token_blacklist_:
            self.log_event('token_verification', {'valid': False, 'error': 'Revoked'})
            return False

        try:
            jwt.decode(token, self.secret_key_, algorithms=['HS256'])
            self.log_event('token_verification', {'valid': True})
            return True
        except jwt.ExpiredSignatureError:
            self.log_event('token_verification', {'valid': False, 'error': 'Expired'})
            return False
        except jwt.InvalidTokenError:
            self.log_event('token_verification', {'valid': False, 'error': 'Invalid'})
            return False

    def revoke_token(self, token: str):
        """
        Revoke a token by adding it to the blacklist.

        Parameters
        ----------
        token : str
            The token to be revoked.
        """
        self.token_blacklist_.append(token)
        self.log_event('token_revocation', {'token': token})

    def enforce_rbac(self, required_roles: List[str], token: str) -> bool:
        """
        Enforce role-based access control (RBAC) for a user.

        Parameters
        ----------
        required_roles : list of str
            The roles required for accessing a resource.
        token : str
            The JWT token to verify the user's roles.

        Returns
        -------
        bool
            True if the user has at least one of the required roles, False otherwise.
        """
        try:
            decoded = jwt.decode(token, self.secret_key_, algorithms=['HS256'])
            user_roles = decoded.get('roles', [])
            if any(role in required_roles for role in user_roles):
                self.log_event('rbac_check', {'status': 'success', 
                                              'user_roles': user_roles})
                return True
            self.log_event('rbac_check', {'status': 'failure', 
                                          'required_roles': required_roles,
                                          'user_roles': user_roles})
            return False
        except jwt.InvalidTokenError:
            self.log_event('rbac_check', {'status': 'failure', 'error': 'Invalid Token'})
            return False

    def ip_whitelisting(self, allowed_ips: List[str], current_ip: str) -> bool:
        """
        Check if the current IP is in the whitelist.

        Parameters
        ----------
        allowed_ips : list of str
            A list of IPs that are allowed to access the resource.
        current_ip : str
            The IP address of the current request.

        Returns
        -------
        bool
            True if the current IP is whitelisted, False otherwise.
        """
        if current_ip in allowed_ips:
            self.log_event('ip_whitelist_check', {'status': 'allowed', 'ip': current_ip})
            return True
        else:
            self.log_event('ip_whitelist_check', {'status': 'denied', 'ip': current_ip})
            return False

    def audit_event(self, event: str, metadata: Dict[str, str],
                    ip_address: Optional[str] = None, 
                    user_agent: Optional[str] = None):
        """
        Log an audit event with optional IP address and user agent.

        Parameters
        ----------
        event : str
            The event to audit.
        metadata : dict of str
            Additional metadata related to the event.
        ip_address : str, optional
            The IP address of the user.
        user_agent : str, optional
            The user agent of the browser or client.

        Returns
        -------
        None
        """
        log_data = {
            'event': event,
            'metadata': metadata,
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        self.log_event('audit', log_data)


class AuditTrail(BaseSecurity):
    """
    A class to handle audit trail logging, allowing flexible configurations 
    for logging levels, batch logging, and external logger integrations. 
    Provides functionality to log events with metadata, flush logs at 
    specified intervals, and send logs to external services if configured.

    Parameters
    ----------
    logging_level : str, default='INFO'
        The logging level to use ('INFO', 'WARNING', 'ERROR').
    external_logger : callable, optional
        A custom external logger function to log events. It should accept two 
        parameters: `event_type` (str) and `event_details` (dict).
    batch_logging : bool, default=False
        If True, enables batch logging where events are logged in batches.
    batch_size : int, default=10
        The number of events to accumulate before flushing the log when 
        `batch_logging` is True.
    flush_interval : int, default=60
        The time interval in seconds to flush the batch log when 
        `batch_logging` is enabled.
    include_metadata : bool, default=True
        If True, includes metadata such as `user_id`, `ip_address`, and 
        `user_agent` in the event logs.

    Attributes
    ----------
    event_log_ : list of dict
        A list to store event logs when `batch_logging` is enabled.
    last_flush_time_ : datetime
        Tracks the last time the event log was flushed.
    
    Examples
    --------
    >>> from gofast.mlops.security import AuditTrail
    >>> audit_trail = AuditTrail(logging_level="INFO", batch_logging=True)
    >>> audit_trail.log_event("login_attempt", {"success": True}, user_id="user123")
    >>> audit_trail.change_logging_level("ERROR")

    See Also
    --------
    SecureDeployment : Class for managing secure deployment, token generation, 
                       and verification.
    """

    @validate_params({
        'logging_level': [StrOptions({"INFO", "WARNING", "ERROR"})],
        'external_logger': [Optional[Callable[[str, Dict[str, Any]], None]]],
        'batch_logging': [bool],
        'batch_size': [int],
        'flush_interval': [int],
        'include_metadata': [bool]
    })
    def __init__(
        self,
        logging_level: str = "INFO",
        external_logger: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        batch_logging: bool = False,
        batch_size: int = 10,
        flush_interval: int = 60,
        include_metadata: bool = True,
    ):
        """
        Initialize the `AuditTrail` class with the given configuration options 
        for logging levels, external logging services, and batch logging.

        """
        super().__init__()
        self.logging_level = logging_level.upper()
        self.external_logger = external_logger
        self.batch_logging = batch_logging
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.include_metadata = include_metadata
        self.event_log_ = []
        self.last_flush_time_ = datetime.datetime.now()

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Logs an individual event with optional metadata.

        Parameters
        ----------
        event_type : str
            The type of event being logged (e.g., "login_attempt").
        details : dict of str
            Specific details of the event.
        user_id : str, optional
            The ID of the user associated with the event.
        ip_address : str, optional
            The IP address of the user associated with the event.
        user_agent : str, optional
            The user agent of the browser or client used by the user.
        metadata : dict of str, optional
            Additional metadata related to the event.

        Notes
        -----
        This method supports batch logging, console logging, and can send 
        logs to an external logger if provided.
        """
        event_details = {
            "event_type": event_type,
            "details": details,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

        if self.include_metadata:
            event_details.update({
                "user_id": user_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "metadata": metadata or {}
            })

        if self.batch_logging:
            self._batch_log_event(event_details)
        else:
            self._log_event_to_console(event_details)

        if self.external_logger:
            self.external_logger(event_type, event_details)

    def _batch_log_event(self, event_details: Dict[str, Any]):
        """
        Logs events in batches and flushes the log if necessary.

        Parameters
        ----------
        event_details : dict of str
            The event details to be logged in batches.
        """
        self.event_log_.append(event_details)
        if len(self.event_log_) >= self.batch_size or self._should_flush():
            self._flush_log()

    def _flush_log(self):
        """
        Flushes the current batch of event logs to the console or external logger.
        """
        logger.info(f"Flushing {len(self.event_log_)} audit events.")
        for event in self.event_log_:
            self._log_event_to_console(event)
        self.event_log_.clear()
        self.last_flush_time_ = datetime.datetime.now()

    def _log_event_to_console(self, event_details: Dict[str, Any]):
        """
        Logs events to the console based on the logging level.

        Parameters
        ----------
        event_details : dict of str
            The event details to log.
        """
        if self.logging_level == "INFO":
            logger.info(event_details)
        elif self.logging_level == "WARNING":
            logger.warning(event_details)
        elif self.logging_level == "ERROR":
            logger.error(event_details)

    def _should_flush(self) -> bool:
        """
        Determines if the event log should be flushed based on the 
        `flush_interval`.

        Returns
        -------
        bool
            True if the event log should be flushed, False otherwise.
        """
        time_since_last_flush = (datetime.datetime.now() - self.last_flush_time_).total_seconds()
        return time_since_last_flush >= self.flush_interval

    def log_batch_events(self, events: List[Dict[str, Any]]):
        """
        Logs multiple events at once, with the same parameters as `log_event`.

        Parameters
        ----------
        events : list of dict
            A list of event dictionaries to log.
        """
        for event in events:
            self.log_event(
                event_type=event.get("event_type"),
                details=event.get("details"),
                user_id=event.get("user_id"),
                ip_address=event.get("ip_address"),
                user_agent=event.get("user_agent"),
                metadata=event.get("metadata"),
            )

    def integrate_with_cloud_logging(
            self, cloud_logging_service: Callable[[Dict[str, Any]], None]):
        """
        Integrates the audit trail with an external cloud logging service.

        Parameters
        ----------
        cloud_logging_service : callable
            A function to handle cloud-based logging of events.
        """
        logger.info("Integrating with external cloud logging service.")
        self.external_logger = cloud_logging_service

    def change_logging_level(self, new_level: str):
        """
        Changes the logging level for future events.

        Parameters
        ----------
        new_level : str
            The new logging level to set ('INFO', 'WARNING', 'ERROR').
        """
        self.logging_level = new_level.upper()
        logger.info(f"Logging level changed to: {self.logging_level}")


class AccessControl(BaseSecurity):
    """
    A class for managing role-based access control (RBAC), allowing for role 
    assignment, permission management, and user-specific permissions. It 
    supports dynamic role creation, permission checks, and temporary 
    permissions.

    Parameters
    ----------
    encryption_key : str, optional
        The encryption key for security purposes.
    default_roles : dict of {str: list of str}, optional
        The default roles and associated users.
        Example: {'admin': ['admin_user'], 'user': ['user1', 'user2']}
    default_permissions : dict of {str: list of str}, optional
        The default permissions and roles that can access them.
        Example: {'view': ['admin', 'user'], 'edit': ['admin']}
    allow_custom_roles : bool, default=True
        Whether to allow the creation of custom roles.
    allow_role_inheritance : bool, default=True
        Whether to allow role inheritance when adding custom roles.

    Attributes
    ----------
    roles_ : dict of {str: list of str}
        Stores the roles and the list of users associated with each role.
    permissions_ : dict of {str: list of str}
        Stores the permissions and the roles that are allowed to access them.
    custom_roles_ : dict of {str: list of str}
        Stores any custom roles dynamically added during runtime.
    allow_custom_roles_ : bool
        Indicates if custom roles are allowed.
    allow_role_inheritance_ : bool
        Indicates if role inheritance is allowed.
    """

    def __init__(
        self, 
        encryption_key: Optional[str] = None, 
        default_roles: Optional[Dict[str, List[str]]] = None, 
        default_permissions: Optional[Dict[str, List[str]]] = None, 
        allow_custom_roles: bool = True, 
        allow_role_inheritance: bool = True
    ):
        """
        Initializes the `AccessControl` class with default roles and permissions, 
        and enables custom roles and role inheritance by default.
        """
        super().__init__(encryption_key)
        self.roles_ = default_roles or {'admin': [], 'user': [], 'guest': []}
        self.permissions_ = default_permissions or {
            'deploy': ['admin'], 
            'modify': ['admin', 'user'], 
            'view': ['admin', 'user', 'guest']
        }
        self.custom_roles_ = {}  # Store any custom roles added dynamically
        self.allow_custom_roles_ = allow_custom_roles
        self.allow_role_inheritance_ = allow_role_inheritance

    def add_user(self, username: str, role: str):
        """
        Adds a user to a role. If custom roles are enabled, the role can be 
        a custom role.

        Parameters
        ----------
        username : str
            The name of the user to add to the role.
        role : str
            The role to assign to the user.
        """
        if role in self.roles_ or (self.allow_custom_roles_ and role in self.custom_roles_):
            self.roles_.setdefault(role, []).append(username)
            self.log_event('add_user', {'username': username, 'role': role})

    def remove_user(self, username: str, role: str):
        """
        Removes a user from a role.

        Parameters
        ----------
        username : str
            The name of the user to remove from the role.
        role : str
            The role from which to remove the user.
        """
        if role in self.roles_ and username in self.roles_[role]:
            self.roles_[role].remove(username)
            self.log_event('remove_user', {'username': username, 'role': role})

    def add_custom_role(self, role_name: str, inherits_from: Optional[str] = None):
        """
        Adds a custom role, optionally inheriting users from another role.

        Parameters
        ----------
        role_name : str
            The name of the custom role to create.
        inherits_from : str, optional
            The role to inherit users from, if role inheritance is enabled.
        """
        if not self.allow_custom_roles_:
            raise ValueError("Custom roles are disabled.")
        
        if inherits_from and self.allow_role_inheritance_:
            self.custom_roles_[role_name] = self.roles_.get(inherits_from, [])
        else:
            self.custom_roles_[role_name] = []
        
        self.log_event('add_custom_role', {'role_name': role_name, 'inherits_from': inherits_from})

    def add_permission(self, permission: str, roles: List[str]):
        """
        Adds a permission to specified roles.

        Parameters
        ----------
        permission : str
            The name of the permission to add.
        roles : list of str
            The roles that will be granted this permission.
        """
        for role in roles:
            if role in self.roles_ or (self.allow_custom_roles_ and role in self.custom_roles_):
                self.permissions_.setdefault(permission, []).append(role)
                self.log_event('add_permission', {'permission': permission, 'role': role})

    def remove_permission(self, permission: str, role: str):
        """
        Removes a permission from a role.

        Parameters
        ----------
        permission : str
            The name of the permission to remove.
        role : str
            The role to remove the permission from.
        """
        if permission in self.permissions_ and role in self.permissions_[permission]:
            self.permissions_[permission].remove(role)
            self.log_event('remove_permission', {'permission': permission, 'role': role})

    def check_permission(self, username: str, permission: str) -> bool:
        """
        Checks if a user has a specific permission.

        Parameters
        ----------
        username : str
            The name of the user whose permissions are being checked.
        permission : str
            The permission to check.

        Returns
        -------
        bool
            True if the user has the permission, otherwise False.
        """
        for role, users in {**self.roles_, **self.custom_roles_}.items():
            if username in users and role in self.permissions_.get(permission, []):
                self.log_event('check_permission', {'username': username, 
                                                    'permission': permission, 
                                                    'granted': True})
                return True
        self.log_event('check_permission', {
            'username': username, 
            'permission': permission,
            'granted': False})
        return False

    def assign_temporary_permission(self, username: str, permission: str, duration: int):
        """
        Assigns a temporary permission to a user for a specified duration.

        Parameters
        ----------
        username : str
            The user to assign the temporary permission to.
        permission : str
            The permission to assign.
        duration : int
            The duration (in seconds) for the temporary permission.
        """
        self.permissions_.setdefault(permission, []).append(username)
        self.log_event('assign_temporary_permission', {
            'username': username, 
            'permission': permission,
            'duration': duration})
        
        # Schedule a task to revoke the temporary permission after the duration expires
        threading.Timer(duration, self._revoke_temp_permission, args=[username, permission]).start()

    def _revoke_temp_permission(self, username: str, permission: str):
        """
        Revokes a temporary permission from a user.
        """
        if permission in self.permissions_ and username in self.permissions_[permission]:
            self.permissions_[permission].remove(username)
            self.log_event('revoke_temp_permission', {'username': username, 'permission': permission})

    def get_role_users(self, role: str) -> List[str]:
        """
        Retrieves all users assigned to a specific role.

        Parameters
        ----------
        role : str
            The role whose users are to be retrieved.

        Returns
        -------
        list of str
            A list of usernames assigned to the role.
        """
        return self.roles_.get(role, []) + self.custom_roles_.get(role, [])

    def get_user_permissions(self, username: str) -> List[str]:
        """
        Retrieves all permissions assigned to a user.

        Parameters
        ----------
        username : str
            The username whose permissions are to be retrieved.

        Returns
        -------
        list of str
            A list of permissions assigned to the user.
        """
        user_permissions = []
        for permission, roles in self.permissions_.items():
            for role in roles:
                if username in self.roles_.get(role, []) or username in self.custom_roles_.get(role, []):
                    user_permissions.append(permission)
        self.log_event('get_user_permissions', {'username': username, 'permissions': user_permissions})
        return user_permissions

    def log_role_change(self, username: str, old_role: str, new_role: str):
        """
        Logs a role change for a user.

        Parameters
        ----------
        username : str
            The username whose role is being changed.
        old_role : str
            The old role the user had.
        new_role : str
            The new role assigned to the user.
        """
        self.log_event('role_change', {'username': username, 'old_role': old_role, 'new_role': new_role})
