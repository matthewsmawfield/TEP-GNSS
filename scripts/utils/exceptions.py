#!/usr/bin/env python3
"""
TEP GNSS Analysis - Exception Handling Utilities
================================================

Provides specific exception types and safe error handling patterns
to replace the excessive bare 'except Exception:' usage throughout
the codebase.

Author: Matthew Lukin Smawfield  
Theory: Temporal Equivalence Principle (TEP)
"""

import functools
from typing import Optional, Callable, Any, Union, List
from pathlib import Path
import logging
from scripts.utils.logger import TEPLogger

# Instantiate the logger
logger = TEPLogger().logger

# Custom exception hierarchy for TEP-specific errors
class TEPError(Exception):
    """Base exception for all TEP-related errors"""
    pass


class TEPDataError(TEPError):
    """Errors related to data loading, validation, or processing"""
    pass


class TEPNetworkError(TEPError):
    """Network-related errors during data acquisition"""
    pass


class TEPFileError(TEPError):
    """File I/O related errors"""
    pass


class TEPConfigurationError(TEPError):
    """Configuration and environment variable errors"""
    pass


class TEPAnalysisError(TEPError):
    """Errors during statistical analysis or fitting"""
    pass


class SafeErrorHandler:
    """
    Provides safe error handling patterns to replace bare exception catching.
    Each method handles specific categories of errors appropriately.
    """
    
    @staticmethod
    def safe_file_operation(
        operation: Callable,
        error_message: str = "File operation failed",
        logger_func: Optional[Callable] = None,
        return_on_error: Any = None
    ):
        """
        Safely execute file operations with specific exception handling.
        
        Args:
            operation: Function to execute
            error_message: Message to log on error
            logger_func: Optional logging function
            return_on_error: Value to return on error
            
        Returns:
            Result of operation or return_on_error value
        """
        if logger_func is None:
            logger_func = logger.warning # Use the centralized logger by default
            
        try:
            return operation()
        except FileNotFoundError as e:
            logger_func(f"{error_message}: File not found - {e}")
            return return_on_error
        except PermissionError as e:
            logger_func(f"{error_message}: Permission denied - {e}")
            return return_on_error
        except IsADirectoryError as e:
            logger_func(f"{error_message}: Expected a file but found a directory - {e}")
            return return_on_on_error
        except OSError as e:
            logger_func(f"{error_message}: An OS error occurred - {e}")
            return return_on_error
        except (UnicodeDecodeError, UnicodeError) as e:
            logger_func(f"{error_message} - encoding issue: {e}")
            return return_on_error
    
    @staticmethod
    def safe_network_operation(
        operation: Callable,
        error_message: str = "Network operation failed",
        logger_func: Optional[Callable] = None,
        return_on_error: Any = None,
        max_retries: int = 0
    ):
        """
        Safely execute network operations with specific exception handling.
        
        Args:
            operation: Function to execute
            error_message: Message to log on error
            logger_func: Optional logging function
            return_on_error: Value to return on error
            max_retries: Number of retry attempts
            
        Returns:
            Result of operation or return_on_error value
        """
        import urllib.error
        import socket
        import time
        
        if logger_func is None:
            logger_func = logger.warning # Use the centralized logger by default for warnings
        error_logger_func = logger.error # Use error level for network failures

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return operation()
            except (urllib.error.URLError, urllib.error.HTTPError, socket.error, 
                   ConnectionError, TimeoutError) as e:
                last_error = e
                if attempt == 0:
                    logger_func(f"{error_message}: {e}")
                
                if attempt < max_retries:
                    logger.info(f"Retrying network operation (attempt {attempt + 2}/{max_retries + 1})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                break
        
        error_logger_func(f"{error_message} - all retries failed: {last_error}")
        return return_on_error
    
    @staticmethod
    def safe_data_operation(
        operation: Callable,
        error_message: str = "Data operation failed",
        logger_func: Optional[Callable] = None,
        return_on_error: Any = None
    ):
        """
        Safely execute data processing operations.
        
        Args:
            operation: Function to execute
            error_message: Message to log on error
            logger_func: Optional logging function
            return_on_error: Value to return on error
            
        Returns:
            Result of operation or return_on_error value
        """
        if logger_func is None:
            logger_func = logger.warning # Use the centralized logger by default
            
        try:
            return operation()
        except (ValueError, TypeError, KeyError, IndexError) as e:
            logger_func(f"{error_message} - data error: {e}")
            return return_on_error
        except (MemoryError, OverflowError) as e:
            logger_func(f"{error_message} - resource error: {e}")
            return return_on_error
        except ImportError as e:
            logger_func(f"{error_message} - missing dependency: {e}")
            return return_on_error
    
    @staticmethod
    def safe_analysis_operation(
        operation: Callable,
        error_message: str = "Analysis operation failed",
        logger_func: Optional[Callable] = None,
        return_on_error: Any = None
    ):
        """
        Safely execute statistical analysis operations.
        
        Args:
            operation: Function to execute
            error_message: Message to log on error
            logger_func: Optional logging function
            return_on_error: Value to return on error
            
        Returns:
            Result of operation or return_on_error value
        """
        if logger_func is None:
            logger_func = logger.warning # Use the centralized logger by default
            
        try:
            return operation()
        except (RuntimeError, ArithmeticError, ZeroDivisionError) as e:
            logger_func(f"{error_message} - computation error: {e}")
            return return_on_error
        except (ValueError, TypeError) as e:
            logger_func(f"{error_message} - parameter error: {e}")
            return return_on_error
        # Let scipy-specific errors be handled by caller
        except ImportError as e:
            logger_func(f"{error_message} - missing scipy/numpy: {e}")
            return return_on_error


def safe_operation(
    error_types: Union[type, tuple] = Exception,
    error_message: str = "Operation failed",
    logger_func: Optional[Callable] = None,
    return_on_error: Any = None,
    reraise: bool = False
):
    """
    Decorator for safe operation execution with specific error handling.
    
    Args:
        error_types: Exception type(s) to catch
        error_message: Message to log on error
        logger_func: Optional logging function
        return_on_error: Value to return on error
        reraise: Whether to re-raise the exception after logging
        
    Returns:
        Decorated function
    """
    if logger_func is None:
        logger_func = logger.error # Use the centralized logger for decorator by default
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                logger_func(f"{error_message}: {e}")
                if reraise:
                    raise
                return return_on_error
        return wrapper
    return decorator


def validate_file_exists(file_path: Union[str, Path], description: str = "File") -> Path:
    """
    Validate that a file exists with proper error handling.
    
    Args:
        file_path: Path to validate
        description: Description for error messages
        
    Returns:
        Path: Validated path object
        
    Raises:
        TEPFileError: If file doesn't exist or is not accessible
    """
    path = Path(file_path)
    
    if not path.exists():
        raise TEPFileError(f"{description} not found: {path}")
    
    if not path.is_file():
        raise TEPFileError(f"{description} is not a file: {path}")
    
    if not os.access(path, os.R_OK):
        raise TEPFileError(f"{description} is not readable: {path}")
    
    return path


def validate_directory_exists(dir_path: Union[str, Path], description: str = "Directory") -> Path:
    """
    Validate that a directory exists with proper error handling.
    
    Args:
        dir_path: Path to validate
        description: Description for error messages
        
    Returns:
        Path: Validated path object
        
    Raises:
        TEPFileError: If directory doesn't exist or is not accessible
    """
    path = Path(dir_path)
    
    if not path.exists():
        raise TEPFileError(f"{description} not found: {path}")
    
    if not path.is_dir():
        raise TEPFileError(f"{description} is not a directory: {path}")
    
    if not os.access(path, os.R_OK):
        raise TEPFileError(f"{description} is not readable: {path}")
    
    return path


def safe_csv_read(file_path: Union[str, Path], **kwargs):
    """
    Safely read CSV files with proper error handling.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        pd.DataFrame or None if failed
        
    Raises:
        TEPDataError: For data-related errors
        TEPFileError: For file-related errors
    """
    import pandas as pd
    
    path = validate_file_exists(file_path, "CSV file")
    
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        raise TEPDataError(f"CSV file is empty: {path}")
    except pd.errors.ParserError as e:
        raise TEPDataError(f"Failed to parse CSV file {path}: {e}")
    except (UnicodeDecodeError, UnicodeError) as e:
        raise TEPDataError(f"Encoding error reading CSV file {path}: {e}")


def safe_json_read(file_path: Union[str, Path]):
    """
    Safely read JSON files with proper error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        dict: Parsed JSON data
        
    Raises:
        TEPDataError: For data-related errors
        TEPFileError: For file-related errors
    """
    import json
    
    path = validate_file_exists(file_path, "JSON file")
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise TEPDataError(f"Invalid JSON in file {path}: {e}")
    except (UnicodeDecodeError, UnicodeError) as e:
        raise TEPDataError(f"Encoding error reading JSON file {path}: {e}")


def safe_json_write(data: dict, file_path: Union[str, Path], indent: int = 2):
    """
    Safely write JSON files with proper error handling.
    
    Args:
        data: Data to write
        file_path: Target file path
        indent: JSON indentation
        
    Raises:
        TEPFileError: For file-related errors
    """
    import json
    
    path = Path(file_path)
    
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=indent)
        temp_path.rename(path)
        
    except (OSError, PermissionError) as e:
        raise TEPFileError(f"Failed to write JSON file {path}: {e}")
    except (TypeError, ValueError) as e:
        raise TEPDataError(f"Cannot serialize data to JSON: {e}")


# Import os for file access checks
import os
