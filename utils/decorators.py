"""
Utility decorators for caching, retry logic, and performance monitoring.

This module provides decorators for common cross-cutting concerns:
- Retry logic for failed operations
- Asynchronous retry logic
- Performance measurement
- Error handling and logging
"""
import functools
import time
import asyncio
import logging
import sys
from typing import Callable, Any, Optional, TypeVar, Type, Tuple, Dict, List, Union
from functools import wraps
import inspect

# Type variable for generic function typing
T = TypeVar('T')

# Configure module logger
logger = logging.getLogger(__name__)

# Set default retry configuration
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0

def retry_on_failure(
    max_retries: int = DEFAULT_RETRY_ATTEMPTS,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff: float = DEFAULT_RETRY_BACKOFF,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    log_retries: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator for functions that may fail with configurable backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay on each retry
        exceptions: Exception type(s) to catch and retry on
        log_retries: Whether to log retry attempts
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal delay
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0 and log_retries:
                        logger.info(f"{func.__qualname__} succeeded after {attempt} retries")
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__qualname__} failed after {max_retries} retries: {str(e)}",
                            exc_info=True
                        )
                        break
                        
                    if log_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__qualname__}: "
                            f"{str(e)}. Retrying in {current_delay:.1f}s..."
                        )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # If we get here, all retries have been exhausted
            logger.critical(
                f"{func.__qualname__} failed after {max_retries} retries. Last error: {str(last_exception)}",
                exc_info=True
            )
            raise last_exception if last_exception else RuntimeError("Unknown error in retry decorator")
            
        # Add retry info to the function for introspection
        wrapper.retry_info = {
            'max_retries': max_retries,
            'delay': delay,
            'backoff': backoff,
            'exceptions': exceptions
        }
        
        return wrapper
    return decorator

def async_retry_on_failure(
    max_retries: int = DEFAULT_RETRY_ATTEMPTS,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff: float = DEFAULT_RETRY_BACKOFF,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    log_retries: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async retry decorator for coroutines that may fail with configurable backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay on each retry
        exceptions: Exception type(s) to catch and retry on
        log_retries: Whether to log retry attempts
        
    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal delay
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0 and log_retries:
                        logger.info(f"Async {func.__qualname__} succeeded after {attempt} retries")
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Async {func.__qualname__} failed after {max_retries} retries: {str(e)}",
                            exc_info=True
                        )
                        break
                        
                    if log_retries:
                        logger.warning(
                            f"Async attempt {attempt + 1}/{max_retries} failed for {func.__qualname__}: "
                            f"{str(e)}. Retrying in {current_delay:.1f}s..."
                        )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # If we get here, all retries have been exhausted
            logger.critical(
                f"Async {func.__qualname__} failed after {max_retries} retries. Last error: {str(last_exception)}",
                exc_info=True
            )
            raise last_exception if last_exception else RuntimeError("Unknown error in async retry decorator")
        
        # Add retry info to the function for introspection
        async_wrapper.retry_info = {
            'max_retries': max_retries,
            'delay': delay,
            'backoff': backoff,
            'exceptions': exceptions,
            'async': True
        }
        
        return async_wrapper
    return decorator

def measure_performance(level: int = logging.DEBUG) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to measure and log function execution time.
    
    Args:
        level: Logging level to use for the performance message
        
    Returns:
        Decorated function with performance measurement
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                logger.log(
                    level,
                    f"{func.__qualname__} executed in {duration:.6f} seconds"
                )
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                logger.log(
                    level,
                    f"Async {func.__qualname__} executed in {duration:.6f} seconds"
                )
        
        # Return the appropriate wrapper based on whether the function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    # Handle the case where the decorator is used without parentheses
    if callable(level):
        func = level
        level = logging.DEBUG
        return decorator(func)
    
    return decorator
