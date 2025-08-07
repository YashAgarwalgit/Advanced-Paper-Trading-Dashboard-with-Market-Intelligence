"""
General utility functions for the trading platform.

This module provides various utility functions for common operations:
- Safe type conversion
- String formatting
- File I/O operations
- Data validation
- Context managers for resource handling
"""
import json
import os
import tempfile
import shutil
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator, TypeVar, Type, Tuple, Union, Callable
from contextlib import contextmanager

# Configure module logger
logger = logging.getLogger(__name__)

# Type variable for generic type hints
T = TypeVar('T')

def safe_float(value: Any, default: float = 0.0, context: str = None) -> float:
    """
    Safely convert a value to float with error handling and logging.
    
    Args:
        value: The value to convert to float
        default: Default value to return if conversion fails
        context: Optional context string for error messages
        
    Returns:
        float: The converted float value or default if conversion fails
    """
    try:
        if value is None:
            return default
            
        result = float(value)
        logger.debug(f"Converted value to float: {value} -> {result} (context: {context or 'N/A'})")
        return result
        
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Failed to convert value to float: {value}. "
            f"Using default: {default}. Error: {str(e)} (context: {context or 'N/A'})"
        )
        return default

def safe_int(value: Any, default: int = 0, context: str = None) -> int:
    """
    Safely convert a value to int with error handling and logging.
    
    Args:
        value: The value to convert to int
        default: Default value to return if conversion fails
        context: Optional context string for error messages
        
    Returns:
        int: The converted int value or default if conversion fails
    """
    try:
        if value is None:
            return default
            
        # First convert to float to handle string floats, then to int
        float_val = float(value)
        result = int(float_val)
        
        # Log if there was a decimal part that was truncated
        if float_val != result:
            logger.warning(
                f"Truncated decimal part when converting to int: {float_val} -> {result} "
                f"(context: {context or 'N/A'})"
            )
        else:
            logger.debug(f"Converted value to int: {value} -> {result} (context: {context or 'N/A'})")
            
        return result
        
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Failed to convert value to int: {value}. "
            f"Using default: {default}. Error: {str(e)} (context: {context or 'N/A'})"
        )
        return default

def format_currency(amount: float, currency: str = "₹", decimals: int = 2) -> str:
    """
    Format a numeric amount as a currency string with proper formatting.
    
    Args:
        amount: The numeric amount to format
        currency: Currency symbol to use (default: ₹ for Indian Rupee)
        decimals: Number of decimal places to show (default: 2)
        
    Returns:
        str: Formatted currency string
    """
    try:
        # Ensure amount is a valid number
        amount_float = float(amount)
        
        # Format with thousands separators and specified decimal places
        format_str = f"{{:,.{decimals}f}}"
        formatted = format_str.format(amount_float)
        
        # Add currency symbol
        result = f"{currency}{formatted}"
        
        logger.debug(f"Formatted currency: {amount_float} -> {result}")
        return result
        
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to format currency: {amount} ({currency}). Error: {e}")
        return f"{currency}0.00"

def format_percentage(value: float, decimals: int = 2, include_sign: bool = True) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: The decimal value to format (0.05 for 5%)
        decimals: Number of decimal places to show (default: 2)
        include_sign: Whether to include the % sign (default: True)
        
    Returns:
        str: Formatted percentage string
    """
    try:
        # Ensure value is a valid number
        value_float = float(value) * 100  # Convert to percentage
        
        # Format with specified decimal places
        format_str = f"{{:,.{decimals}f}}"
        formatted = format_str.format(value_float)
        
        # Add percentage sign if requested
        result = f"{formatted}%" if include_sign else formatted
        
        logger.debug(f"Formatted percentage: {value} -> {result}")
        return result
        
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to format percentage: {value}. Error: {e}")
        return "0.00%" if include_sign else "0.00"

def calculate_percentage_change(current: float, previous: float, default: float = 0.0) -> float:
    """
    Calculate the percentage change between two values.
    
    Args:
        current: Current value
        previous: Previous value to compare against
        default: Default value to return if calculation is not possible (e.g., division by zero)
        
    Returns:
        float: Percentage change between current and previous values
    """
    try:
        if previous == 0:
            logger.warning(
                f"Cannot calculate percentage change: previous value is zero. "
                f"Using default: {default}"
            )
            return default
            
        change = ((current - previous) / previous) * 100
        logger.debug(
            f"Calculated percentage change: current={current}, previous={previous}, "
            f"change={change:.2f}%"
        )
        return change
        
    except (TypeError, ValueError) as e:
        logger.error(
            f"Error calculating percentage change: current={current}, "
            f"previous={previous}. Error: {e}"
        )
        return default

def truncate_string(text: str, max_length: int = 50, ellipsis: str = "...") -> str:
    """
    Truncate a string to a maximum length, optionally adding an ellipsis.
    
    Args:
        text: The input string to truncate
        max_length: Maximum allowed length of the string (including ellipsis if added)
        ellipsis: String to append at the end when truncating (default: "...")
        
    Returns:
        str: Truncated string with ellipsis if needed
    """
    if not isinstance(text, str):
        logger.warning(f"Expected string for truncation, got {type(text).__name__}")
        return str(text)[:max_length]
        
    if len(text) <= max_length:
        return text
        
    ellipsis_len = len(ellipsis)
    if max_length <= ellipsis_len:
        return ellipsis[:max_length]
        
    truncated = text[:max_length - ellipsis_len] + ellipsis
    logger.debug(f"Truncated string (length {len(text)} -> {len(truncated)}): {truncated}")
    return truncated

@contextmanager
def atomic_write(filepath: str, mode: str = 'w', **kwargs):
    """
    Context manager for atomic file writing to prevent partial writes.
    
    Args:
        filepath: Path to the target file
        mode: File open mode (default: 'w' for text write)
        **kwargs: Additional arguments to pass to tempfile.NamedTemporaryFile
        
    Yields:
        file: A file-like object to write to
        
    Raises:
        OSError: If file operations fail
    """
    logger.debug(f"Starting atomic write to {filepath}")
    tmp = None
    
    try:
        # Ensure directory exists
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            logger.debug(f"Creating directory: {dirname}")
            os.makedirs(dirname, exist_ok=True)
        
        # Create temporary file in the same directory as the target
        with tempfile.NamedTemporaryFile(
            mode=mode, 
            delete=False, 
            dir=os.path.dirname(filepath) or None,
            prefix=f".tmp_{os.path.basename(filepath)}.",
            **kwargs
        ) as f:
            tmp = f.name
            logger.debug(f"Temporary file created: {tmp}")
            yield f
            
            # Ensure all data is written to disk
            f.flush()
            os.fsync(f.fileno())
            
        # Atomic rename (works on POSIX systems, falls back to copy on Windows)
        if os.name == 'nt':  # Windows
            if os.path.exists(filepath):
                os.unlink(filepath)
            os.rename(tmp, filepath)
        else:  # POSIX
            os.replace(tmp, filepath)
            
        logger.debug(f"Atomic write completed successfully: {filepath}")
        
    except Exception as e:
        logger.error(f"Atomic write failed for {filepath}: {e}", exc_info=True)
        
        # Clean up temporary file if it exists
        if tmp and os.path.exists(tmp):
            try:
                logger.debug(f"Cleaning up temporary file: {tmp}")
                os.unlink(tmp)
            except Exception as cleanup_error:
                logger.error(
                    f"Failed to clean up temporary file {tmp}: {cleanup_error}",
                    exc_info=True
                )
        
        raise  # Re-raise the original exception

def save_json_atomic(data: Dict[str, Any], filepath: str) -> bool:
    """Atomically save JSON data to file"""
    try:
        with atomic_write(filepath) as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        return False

def load_json_safe(filepath: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON from file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
    return None

def validate_ticker(ticker: str) -> bool:
    """Validate ticker format"""
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic validation - alphanumeric with dots and hyphens
    return ticker.replace('.', '').replace('-', '').replace('=', '').replace('^', '').isalnum()

def get_timestamp_iso() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()

def chunks(lst: List[Any], n: int) -> Generator[List[Any], None, None]:
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]