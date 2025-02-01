# -*- coding: utf-8 -*-
# test_metadata.py

import os
import json
import pytest
import time
# Assume these are imported from the gofast.mlops.metadata module
from gofast.mlops.metadata import (
    MetadataManager,
    LineageTracker,
    #AuditLogger,
    ReproducibilityEnsurer,
    PerformanceTracker,
    log_metadata,
    retrieve,
    compare,
    audit,
    sync_with_cloud,
    validate_schema,
    track_experiment,
    prune_old,
)

# Mock implementations or fixtures can be added here if necessary
# Constants for testing
TEST_METADATA_TYPE = 'test_model'
TEST_METADATA = {'accuracy': 0.95, 'loss': 0.05}
TEST_METADATA_UPDATED = {'accuracy': 0.96, 'loss': 0.04}
TEST_USER = 'test_user'
TEST_CHANGE_DESCRIPTION = 'Updated accuracy and loss metrics.'
TEST_BUCKET_NAME = 'test-bucket'
TEST_MONGO_URI = 'mongodb://localhost:27017/test_db'
TEST_EXPERIMENT_ID = 'exp_test_001'
TEST_SCHEMA = {
    'type': 'object',
    'properties': {
        'accuracy': {'type': 'number'},
        'loss': {'type': 'number'},
    },
    'required': ['accuracy', 'loss']
}


# Test MetadataManager
def test_metadata_manager_local(tmp_path, monkeypatch):
    """
    Test MetadataManager by storing and retrieving metadata locally.
    
    Parameters
    ----------
    tmp_path : pathlib.Path
         Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
         Used to change the current directory.
         
    Example
    -------
    >>> from gofast.mlops.metadata import MetadataManager
    >>> manager = MetadataManager(metadata_store='local')
    >>> manager.fit({"model_version": "1.0"})
    >>> manager.store_metadata("model_version", "1.1")
    >>> value = manager.get_metadata("model_version")
    >>> assert value == "1.1"
    """
    monkeypatch.chdir(tmp_path)
    # For local storage, we do not need a cloud bucket
    manager = MetadataManager(
        metadata_store="local",
        cloud_bucket_name=None
    )
    # Simulate "fitting" the manager with initial metadata
    manager.fit({"model_version": "1.0", "data_version": "2022-01-01"})
    manager.store_metadata("model_version", "1.1")
    value = manager.get_metadata("model_version")
    assert value == "1.1"
    # Verify that the local backup file was created
    backup_file = tmp_path / "metadata_backup.json"
    assert backup_file.exists()


# Test LineageTracker
def test_lineage_tracker(tmp_path, monkeypatch):
    """
    Test LineageTracker functionality using local storage.
    
    Parameters
    ----------
    tmp_path : pathlib.Path
         Temporary directory.
    monkeypatch : pytest.MonkeyPatch
         Used to set the working directory.
    """
    monkeypatch.chdir(tmp_path)
    tracker = LineageTracker(
        lineage_store="local",
        encryption_key=None,
        compression_enabled=False,
        alert_on_version_change=False
    )
    # Activate the tracker
    tracker.run()
    # Log data ingestion and retrieve it
    tracker.log_data_ingestion(
        data_version="v1.0.0",
        source="local_source",
        dependencies=[],
        tags=["test"]
    )
    retrieved = tracker.retrieve_lineage("data_ingestion")
    assert retrieved is not None


# Test AuditLogger
def test_audit_logger_local(tmp_path, monkeypatch):
    """
    Test AuditLogger functionality using local file storage.
    
    Parameters
    ----------
    tmp_path : pathlib.Path
         Temporary directory.
    monkeypatch : pytest.MonkeyPatch
         Used to change directory.
    """
    monkeypatch.chdir(tmp_path)
    msg = audit(
        metadata_type="model",
        user="tester",
        change_description="Test change",
        storage_backend="local",
        version=1
    )
    assert "Audit log successfully recorded" in msg
    # Verify that the audit log file exists.
    audit_file = tmp_path / "model_audit.log"
    assert audit_file.exists()

# Test ReproducibilityEnsurer
def test_reproducibility_ensurer(tmp_path, monkeypatch):
    """
    Test ReproducibilityEnsurer: setting random seed, exporting config,
    and comparing environments.
    
    Parameters
    ----------
    tmp_path : pathlib.Path
         Temporary directory.
    monkeypatch : pytest.MonkeyPatch
         Used to change directory.
    """
    monkeypatch.chdir(tmp_path)
    ensurer = ReproducibilityEnsurer(
        storage_backend="local",
        encryption_key=None,
        compression_enabled=False,
        versioning_enabled=True
    )
    ensurer.run()
    ensurer.set_random_seed(42)
    ensurer.export_config(file_path="config.json")
    with open("config.json", "r") as f:
        config = json.load(f)
    assert "python_version" in config
    differences = ensurer.compare_environments(config)
    # Expect no differences between the exported config and itself.
    assert isinstance(differences, dict)


# Test PerformanceTracker
def test_performance_tracker(tmp_path, monkeypatch):
    """
    Test PerformanceTracker by logging performance and exporting metrics.
    
    Parameters
    ----------
    tmp_path : pathlib.Path
         Temporary directory.
    monkeypatch : pytest.MonkeyPatch
         Used to set current directory.
    """
    monkeypatch.chdir(tmp_path)
    def dummy_alert(msg):
        print("ALERT:", msg)
    tracker = PerformanceTracker(
        alert_threshold=0.01,
        alert_method=dummy_alert,
        metrics_to_track=["accuracy"],
        use_rolling_avg=True,
        window_size=3,
        storage_backend="local",
        compression_enabled=False,
        encryption_key=None,
        real_time_monitoring=False,
        real_time_interval=10
    )
    tracker.run()
    tracker.log_performance("v1.0", "accuracy", 0.95)
    tracker.log_performance("v1.0", "accuracy", 0.94)
    perf = tracker.get_performance("v1.0")
    assert "accuracy" in perf
    exported = tracker.export_metrics(export_format="json")
    exported_dict = json.loads(exported)
    assert "v1.0" in exported_dict


# Test log_metadata and retrieve functions
@pytest.mark.skip ("test is made locally due to local files.")
def test_log_and_retrieve_metadata_local(tmp_path, monkeypatch):
    """
    Test log_metadata and retrieve functions using local storage.
    
    Parameters
    ----------
    tmp_path : pathlib.Path
         Temporary directory.
    monkeypatch : pytest.MonkeyPatch
         Used to change the working directory.
    """
    monkeypatch.chdir(tmp_path)
    meta = {"accuracy": 0.95, "loss": 0.05, "epoch": 10}
    msg = log_metadata(
        metadata=meta,
        metadata_type="model",
        encryption_key=None,
        compression_enabled=False,
        storage_backend="local"
    )
    assert "Metadata logged successfully" in msg
    # The file should be named 'model_v1.json'
    assert os.path.exists("model_v1.json")
    
    # Now test retrieve by simulating a stored file.
    with open("model_v1.json", "wb") as f:
        f.write(json.dumps({"accuracy": 0.95, "loss": 0.05,
                            "epoch": 10, "version": 1}).encode("utf-8"))
    retrieved = retrieve(metadata_type="model")
    assert retrieved["accuracy"] == 0.95

# Test compare function
def test_compare_function():
    """
    Test the compare function with shallow and deep comparisons.
    """
    meta1 = {
        "accuracy": 0.9501,
        "loss": 0.05,
        "epoch": 10,
        "optimizer": "adam",
        "params": {"learning_rate": 0.001}
    }
    meta2 = {
        "accuracy": 0.9502,
        "loss": 0.05,
        "epoch": 10,
        "optimizer": "adam",
        "params": {"learning_rate": 0.0015}
    }
    diff = compare(meta1, meta2)
    assert "accuracy" in diff
    diff_deep = compare(meta1, meta2, recursive=True, tol=0.0005)
    assert "params" in diff_deep


# Test audit function
def test_audit_function_local(tmp_path, monkeypatch):
    """
    Test the audit function using local storage.
    """
    monkeypatch.chdir(tmp_path)
    msg = audit(
        metadata_type="model",
        user="tester",
        change_description="Test audit change",
        storage_backend="local",
        version=1
    )
    assert "Audit log successfully recorded" in msg
    assert os.path.exists("model_audit.log")


# Test sync_with_cloud function (using monkeypatch to simulate cloud sync)
def test_sync_with_cloud(monkeypatch):
    """
    Test sync_with_cloud by monkey-patching the inner sync function.
    """
    # Define a dummy sync inner function
    def dummy_sync(**kwargs):
        return "Dummy cloud sync success."
    monkeypatch.setattr(
        "gofast.mlops.metadata._sync_with_cloud_inner", dummy_sync
    )
    meta = {"dummy": "data"}
    result = sync_with_cloud(
        metadata=meta,
        cloud_provider="aws",
        bucket_name="dummy_bucket",
        aws_credentials={"aws_access_key_id": "key", "aws_secret_access_key": "secret"}
    )
    assert "Dummy cloud sync success." in result


# Test validate_schema function
def test_validate_schema_function(tmp_path, monkeypatch):
    """
    Test validate_schema with auto-correction enabled.
    """
    monkeypatch.chdir(tmp_path)
    meta = {"accuracy": 0.95, "loss": 0.05, "epoch": 10}
    schema = {
        "type": "object",
        "properties": {
            "accuracy": {"type": "number"},
            "loss": {"type": "number"},
            "epoch": {"type": "integer"},
            "optimizer": {"type": "string", "default": "adam"}
        },
        "required": ["accuracy", "loss", "epoch", "optimizer"]
    }
    is_valid = validate_schema(
        metadata=meta,
        schema=schema,
        auto_correct=True,
        correction_log="corrections.log"
    )
    assert is_valid is True
    # Check that the correction log file was created
    assert os.path.exists("corrections.log")


# Test track_experiment function
def test_track_experiment_function(tmp_path, monkeypatch):
    """
    Test track_experiment function using local storage.
    """
    monkeypatch.chdir(tmp_path)
    configuration = {"optimizer": "Adam", "learning_rate": 0.001}
    hyperparameters = {"batch_size": 32, "epochs": 10}
    performance_metrics = {"accuracy": 0.93, "loss": 0.07}
    # Create a dummy training log file.
    training_log = tmp_path / "training_log.txt"
    training_log.write_text("Training log content.")
    
    msg = track_experiment(
        experiment_id="exp_001",
        configuration=configuration,
        hyperparameters=hyperparameters,
        performance_metrics=performance_metrics,
        training_logs=str(training_log),
        storage_backend="local",
        compression_enabled=False,
        versioning_enabled=True
    )
    assert "Experiment metadata tracked successfully" in msg
    # Expected file: experiment_exp_001_v1.json
    expected_file = tmp_path / "experiment_exp_001_v1.json"
    assert expected_file.exists()

# Test prune_old function
def test_prune_old_function(tmp_path, monkeypatch):
    """
    Test prune_old function for local metadata pruning.
    """
    monkeypatch.chdir(tmp_path)
    # Create dummy metadata files.
    file_names = ["model_v1.json", "model_v2.json", "model_v3.json"]
    current_time = time.time()
    for i, file_name in enumerate(file_names, start=1):
        with open(file_name, "w") as f:
            json.dump({"dummy": "data", "version": i}, f)
        # Set modification time: v1 and v2 -> 100 days old; v3 -> 10 days old.
        mod_time = current_time - (100 * 86400) if i < 3 else current_time - (10 * 86400)
        os.utime(file_name, (mod_time, mod_time))
    
    msg = prune_old(
        metadata_type="model",
        retention_days=30,
        storage_backend="local",
        preserve_versions=[3]
    )
    assert "Old metadata pruned successfully" in msg
    remaining_files = os.listdir(tmp_path)
    assert "model_v3.json" in remaining_files
    assert "model_v1.json" not in remaining_files
    assert "model_v2.json" not in remaining_files


if __name__=='__main__': 
    pytest.main([__file__])