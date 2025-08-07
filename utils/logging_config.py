"""
Logging configuration for the institutional trading platform
"""
import logging
import sys
import os
from typing import Optional, Union

def setup_logging(log_level: Union[int, str] = logging.INFO, 
                 log_file: str = "logs/trading_platform.log") -> logging.Logger:
    """
    Setup logging configuration with UTF-8 support and proper error handling.
    
    Args:
        log_level: Logging level (e.g., logging.INFO, 'DEBUG', 'INFO')
        log_file: Path to the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("institutional_platform")
    
    try:
        # Convert string log level to int if needed
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)
            
        logger.setLevel(log_level)
        
        # Clear any existing handlers to avoid duplicate logs
        if logger.hasHandlers():
            logger.debug("Clearing existing log handlers")
            logger.handlers.clear()
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir and not os.path.exists(log_dir):
            logger.debug(f"Creating log directory: {log_dir}")
            os.makedirs(log_dir, exist_ok=True)
        
        # Define log format with timestamp, logger name, level, and source location
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Set UTF-8 encoding for console output if supported
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except Exception as e:
                logger.warning(f"Could not set console encoding to UTF-8: {e}")
        
        logger.addHandler(console_handler)
        logger.debug("Console logging handler configured")
        
        # File handler with error handling
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"File logging configured: {os.path.abspath(log_file)}")
        except Exception as e:
            logger.error(f"Failed to configure file logging to {log_file}: {e}")
            # Continue with just console logging
        
        logger.info(f"Logging initialized at level: {logging.getLevelName(log_level)}")
        
    except Exception as e:
        # Fallback to basic logging if our setup fails
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("institutional_platform")
        logger.error(f"Error setting up logging: {e}")
        logger.info("Falling back to basic logging configuration")
    
    return logger

