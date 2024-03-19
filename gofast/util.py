# util.py

import os
import logging
import logging.config  
import yaml

def ensure_logging_directory(log_path):
    """Ensure that the logging directory exists."""
    os.makedirs(log_path, exist_ok=True)

def load_logging_configuration(config_file_path, default_level=logging.INFO):
    """Load logging configuration from a YAML file."""
    with open(config_file_path, 'rt') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config) 

def setup_logging(default_path='_gflog.yml', default_level=logging.INFO):
    """Setup logging configuration with fallback."""
    package_dir = os.path.dirname(__file__)
    log_path = os.environ.get('LOG_PATH', os.path.join(package_dir, 'gflogs'))
    ensure_logging_directory(log_path)

    config_file_path = os.path.join(package_dir, default_path)
    try:
        load_logging_configuration(config_file_path, default_level)
    except Exception as e:
        logging.basicConfig(level=default_level)
        logging.warning(f"Failed to load logging configuration from {config_file_path}."
                        f" Error: {e}. Using basicConfig with level={default_level}.")

# Example usage of the setup_logging function within util.py
if __name__ == "__main__":
    setup_logging()


