# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Track bugs and issues, and also deal with all GOFast exceptions.
 
"""
 
import os
import yaml
import logging
import logging.config
from string import Template

__all__= ["gofastlog"]

class gofastlog:
    """
    A class to configure logging for the `gofast` module, facilitating 
    tracking of all exceptions.
    """
    @staticmethod
    def load_configuration(
            config_path=None, use_default_logger=True, verbose=False):
        """
        Configures logging based on a specified configuration file.

        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file. Supports .yaml, .yml, .ini, and .json formats.
            If None or not provided, uses basic logging configuration or a default logger
            setup, depending on `use_default_logger`.
        use_default_logger : bool, optional
            Whether to use the default logger configuration if no config_path is provided.
            Defaults to True.
        verbose : bool, optional
            If True, prints additional information during configuration. Defaults to False.

        """
        if not config_path:
            if use_default_logger:
                gofastlog.set_default_logger()
            else:
                logging.basicConfig()
            return

        if verbose:
            print(f"Configuring logging with: {config_path}")

        if config_path.endswith((".yaml", ".yml")):
            gofastlog._configure_from_yaml(config_path, verbose)
        elif config_path.endswith(".ini"):
            logging.config.fileConfig(config_path, disable_existing_loggers=False)
        else:
            logging.warning(f"Unsupported logging configuration format: {config_path}")

    @staticmethod
    def _configure_from_yaml(yaml_path, verbose=False):
        """
        Configures logging from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file.
        verbose : bool, optional
            If True, prints additional information during configuration.
        """
        full_path = os.path.abspath(yaml_path)
        if not os.path.exists(full_path):
            logging.error(f"The YAML config file {full_path} does not exist.")
            return

        if verbose:
            print(f"Loading YAML config from {full_path}")

        with open(full_path, "rt") as f:
            config = yaml.safe_load(f.read())
            
        logging.config.dictConfig(config)

    @staticmethod
    def set_default_logger():
        """
        Sets up a default logger configuration. This can be customized to suit default
        logging preferences.
        """
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    @staticmethod
    def get_gofast_logger(logger_name=''):
        """
        Creates and retrieves a named logger.
    
        Parameters
        ----------
        logger_name : str, optional
            The name of the logger. If empty, returns the root logger.
    
        Returns
        -------
        logging.Logger
            The logger instance with the specified name.
        """
        return logging.getLogger(logger_name)
    
    @staticmethod
    def load_configure_set_logfile(config_file='_gflog.yml', app_name='gofast'):
        """
        Configures logging from a YAML file located in a specific directory 
        defined by an environment variable or a direct path.
    
        Parameters
        ----------
        config_file : str, optional
            The name of the configuration file. Defaults to '_gflog.yml'.
        app_name : str, optional
            The application name used to derive the path from environment variables.
    
        Raises
        ------
        FileNotFoundError
            If the specified configuration file does not exist.
        ValueError
            If the configuration file format is not supported.
        """
        config_path = os.getenv(app_name.upper() + '_LOG_CONFIG_PATH', '')
        if config_path:
            full_path = os.path.join(config_path, config_file)
        else:
            raise EnvironmentError(
                f"{app_name.upper()}_LOG_CONFIG_PATH environment variable is not set.")
    
        if not full_path.endswith(('.yaml', '.yml')):
            raise ValueError("Only .yaml or .yml config files are supported.")
    
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The config file {full_path} does not exist.")
    
        with open(full_path, 'rt') as f:
            config = yaml.safe_load(f.read())
            
        logging.config.dictConfig(config)
    
    @staticmethod
    def set_logger_output(
            log_filename="gofast.log", date_format='%Y-%m-%d %H:%M:%S', file_mode="w",
            format_="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.DEBUG):
        """
        Configures the logging output, including the file name, date format, 
        logging level, and message format.
    
        Parameters
        ----------
        log_filename : str, optional
            The name of the log file.
        date_format : str, optional
            The date format used in log messages.
        file_mode : str, optional
            The mode for opening the log file ('a' for append, 'w' for overwrite).
        format_ : str, optional
            The format of the log messages.
        level : logging.Level, optional
            The logging level (e.g., logging.DEBUG, logging.INFO).
    
        """
        handler = logging.FileHandler(log_filename, mode=file_mode)
        handler.setLevel(level)
        formatter = logging.Formatter(format_, datefmt=date_format)
        handler.setFormatter(formatter)
    
        logger = gofastlog.get_gofast_logger()
        logger.setLevel(level)
        logger.addHandler(handler)
        # Remove duplicate handlers if any exist to prevent repeated log messages.
        logger.handlers = list(set(logger.handlers))

def setup_logging(config_path='path/to/_gflog.yml'):
    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f.read())

    # Use the default path from the YAML file or override it with an environment variable
    log_path = os.getenv('LOG_PATH', config.get('default_log_path', '/fallback/path'))

    # Now replace the placeholder or directly use the `log_path` 
    # in your logging configuration
    if 'handlers' in config:
        for handler in config['handlers'].values():
            if 'filename' in handler:
                handler['filename'] = handler['filename'].replace('${LOG_PATH}', log_path)

    logging.config.dictConfig(config)

def setup_logging_with_template(config_path='path/to/_gflog.yml'):
    # Load the YAML configuration file
    with open(config_path, 'rt') as f:
        config_text = f.read()
    
    # Substitute environment variable placeholders
    config_text = Template(config_text).substitute(LOG_PATH=os.getenv(
        'LOG_PATH', 'default/log/path'))
    
    # Load the substituted configuration as a dictionary
    config = yaml.safe_load(config_text)
    
    # Configure logging
    logging.config.dictConfig(config) 
    
def setup_logging_with_expandvars(
        config_path='path/to/_gflog.yml'):
    # Load the YAML configuration file
    with open(config_path, 'rt') as f:
        config_text = f.read()
    
    # Substitute environment variable placeholders
    config_text = os.path.expandvars(config_text)
    
    # Load the substituted configuration as a dictionary
    config = yaml.safe_load(config_text)
    
    # Configure logging
    logging.config.dictConfig(config)

if __name__=='__main__':
    print(os.path.abspath(gofastlog.__name__))
    

    
    
    
    
    