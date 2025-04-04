# -*- coding: utf-8 -*-
## tests/test_automation.py

import pytest
import logging
from unittest.mock import MagicMock, patch #, create_autospec
from datetime import datetime# , timedelta
import time
import pickle

# from unittest.mock import patch, MagicMock
from gofast.mlops.automation import (
    SimpleAutomation,
    AutomationManager,
    RetrainingScheduler,
    AirflowAutomation,
    KubeflowAutomation,
    KafkaAutomation,
    RabbitMQAutomation,
)
from gofast.mlops._config import INSTALL_DEPENDENCIES
from gofast.utils.deps_utils import is_module_installed 

# Enable dependency installation during tests
INSTALL_DEPENDENCIES = True  # noqa

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_model():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression()

@pytest.fixture
def tmp_state_file(tmp_path):
    return tmp_path / "test_state.pkl"

# --- AutomationManager Tests ---
@pytest.mark.skipif (
    not is_module_installed("apscheduler"), 
    reason="APScheduller needs to be required to the test to proceed."
    )
class TestAutomationManager:
    @pytest.fixture
    def automator(self, tmp_state_file):
        return AutomationManager(state_file=str(tmp_state_file))

    def test_add_operation(self, automator):
        mock_func = MagicMock()
        automator.add_operation("test_op", mock_func, 5)
        assert "test_op" in automator.operations
        assert automator.operations["test_op"]["interval"] == 5

    def test_run_without_operations(self, automator):
        with pytest.raises(RuntimeError):
            automator.run()

    def test_state_persistence(self, automator, tmp_state_file):
        mock_func = MagicMock()
        automator.add_operation("persist_test", mock_func, 10)
        automator.run()
        automator.shutdown()
        
        assert tmp_state_file.exists()
        with open(tmp_state_file, "rb") as f:
            state = pickle.load(f)
        assert "persist_test" in state

# --- AirflowAutomation Tests ---
@pytest.mark.skipif(
    not is_module_installed("airflow"), 
    reason="Airflow needs to be required to the test to proceed."
    )
class TestAirflowAutomation:
    @pytest.fixture
    def airflow(self):
        return AirflowAutomation(
            dag_id="test_dag",
            start_date=datetime.now(),
            schedule_interval="@daily"
        )

    @patch('airflow.models.DAG')
    def test_dag_creation(self, mock_dag, airflow):
        assert airflow.dag is not None
        mock_dag.assert_called_once()

    @patch('airflow.operators.python.PythonOperator')
    def test_add_task(self, mock_operator, airflow):
        mock_func = MagicMock()
        airflow.add_task_to_airflow("test_task", mock_func)
        mock_operator.assert_called_once_with(
            task_id="test_task",
            python_callable=mock_func,
            dag=airflow.dag
        )

# --- KubeflowAutomation Tests ---
@pytest.mark.skipif(
    not is_module_installed("kubeflow"), 
    reason="Kubeflow needs to be required to the test to proceed."
    )
class TestKubeflowAutomation:
    @pytest.fixture
    def kubeflow(self):
        return KubeflowAutomation(host="http://localhost:8080")

    @patch('kfp.Client')
    def test_pipeline_creation(self, mock_client, kubeflow):
        mock_func = MagicMock()
        run_id = kubeflow.create_kubeflow_pipeline(
            "test_pipe", "test_task", mock_func
        )
        mock_client.return_value.create_run_from_pipeline_func.assert_called()
        assert isinstance(run_id, str)

# --- KafkaAutomation Tests ---
@pytest.mark.skipif(
    not is_module_installed("kafka"), 
    reason="Kafka needs to be required to the test to proceed."
    )
class TestKafkaAutomation:
    @pytest.fixture
    def kafka_auto(self):
        return KafkaAutomation(
            kafka_servers=["localhost:9092"],
            topic="test_topic"
        )

    @patch('kafka.KafkaConsumer')
    def test_message_processing(self, mock_consumer, kafka_auto):
        mock_msg = MagicMock()
        mock_msg.value = b"test message"
        mock_consumer.return_value = [mock_msg]

        processor = MagicMock()
        kafka_auto.run()
        kafka_auto.process_kafka_message(processor)
        
        processor.assert_called_once_with(mock_msg)

# --- RabbitMQAutomation Tests ---
@pytest.mark.skipif(
    not is_module_installed("rabbit"), 
    reason="Rabbit needs to be required to the test to proceed."
    )
class TestRabbitMQAutomation:
    @pytest.fixture
    def rabbit_auto(self):
        return RabbitMQAutomation(
            host="localhost",
            queue="test_queue"
        )

    @patch('pika.BlockingConnection')
    def test_message_handling(self, mock_conn, rabbit_auto):
        mock_channel = MagicMock()
        mock_conn.return_value.channel.return_value = mock_channel

        processor = MagicMock()
        rabbit_auto.run()
        rabbit_auto.process_rabbitmq_message(processor)
        
        mock_channel.basic_consume.assert_called_once()

# --- RetrainingScheduler Tests ---
@pytest.mark.skipif (
    not is_module_installed("apscheduler"), 
    reason="APScheduller needs to be required to the test to proceed."
    )
class TestRetrainingScheduler:
    @pytest.fixture
    def scheduler(self, sample_model):
        s = RetrainingScheduler(max_workers=2)
        s.run(model=sample_model)
        return s

    def test_performance_evaluation(self, scheduler, sample_model):
        metric = MagicMock(return_value=0.95)
        score = scheduler.evaluate_performance(metric)
        metric.assert_called_once_with(sample_model)
        assert score == 0.95

    @patch('gofast.mlops.automation.validate_params')
    def test_retraining_trigger(self, mock_validator, scheduler):
        mock_metric = MagicMock(return_value=0.75)
        scheduler.monitor_model(mock_metric, 0.8, 10)
        assert len(scheduler.operations) == 1

# --- SimpleAutomation Tests ---
@pytest.mark.skipif (
    not is_module_installed("apscheduler"), 
    reason="APScheduller needs to be required to the test to proceed."
    )
class TestSimpleAutomation:
    @pytest.fixture
    def simple_auto(self):
        return SimpleAutomation()

    def test_task_lifecycle(self, simple_auto):
        mock_func = MagicMock()
        simple_auto.add_task("test_task", mock_func, 0.1)
        simple_auto.run()
        time.sleep(0.2)  # Allow time for execution
        simple_auto.shutdown()
        
        assert mock_func.call_count >= 1

# --- Common Utility Tests ---
def test_check_is_runned():
    from gofast.utils.validator import check_is_runned
    from gofast.exceptions import NotRunnedError
    
    class TestClass:
        _is_runned = False
    
    with pytest.raises(NotRunnedError):
        check_is_runned(TestClass(), ['_is_runned'], msg="Error")

def test_ensure_pkg():
    from gofast.utils.deps_utils import ensure_pkg
    
    @ensure_pkg("nonexistent_package", extra="Test message")
    def test_func():
        pass
    
    with pytest.raises(ImportError):
        test_func()
        
# Run the test suite
if __name__ == "__main__":
    pytest.main([__file__])
