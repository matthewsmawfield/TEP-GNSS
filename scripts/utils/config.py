#!/usr/bin/env python3
"""
TEP GNSS Analysis - Centralized Configuration Management
========================================================

Provides consistent environment variable handling and configuration management
across all TEP analysis steps. Eliminates the inconsistent parsing patterns
that were scattered throughout the codebase.

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import os
from typing import Optional, Union, List
from pathlib import Path
import multiprocessing as mp


class TEPConfig:
    """Centralized configuration management for TEP GNSS Analysis"""
    
    # Default values for common configuration parameters
    DEFAULTS = {
        # Analysis parameters
        'TEP_BINS': 40,
        'TEP_MAX_DISTANCE_KM': 13000.0,
        'TEP_MIN_BIN_COUNT': 200,
        'TEP_BOOTSTRAP_ITER': 1000,
        'TEP_NULL_ITERATIONS': 100,
        
        # Processing parameters
        'TEP_WORKERS': None,  # Will default to CPU count
        'TEP_MEMORY_LIMIT_GB': 8.0,
        
        # Data parameters
        'TEP_MIN_STATIONS': 0,
        'TEP_DATE_START': '2023-01-01',
        'TEP_DATE_END': '2025-06-30',
        
        # Network timeouts
        'TEP_NETWORK_TIMEOUT': 30,
        'TEP_DOWNLOAD_TIMEOUT': 60,
        
        # File limits
        'TEP_FILES_PER_CENTER': None,  # None means unlimited
        
        # Feature flags
        'TEP_PROCESS_ALL_CENTERS': True,
        'TEP_WRITE_PAIR_LEVEL': True,
        'TEP_ENABLE_JACKKNIFE': True,
        'TEP_ENABLE_ANISOTROPY': True,
        'TEP_ENABLE_TEMPORAL': True,
        'TEP_ENABLE_LOSO': True,
        'TEP_ENABLE_LODO': True,
        'TEP_ENABLE_ENHANCED_ANISOTROPY': True,
        
        # Rebuild flags
        'TEP_REBUILD_COORDS': False,
        'TEP_REBUILD_CLK': False,
        'TEP_REBUILD_METADATA': False,
        'TEP_SKIP_COORDS': False,
        
        # Advanced options
        'TEP_USE_REAL_COHERENCY': False,
        'TEP_COHERENCY_F1': 0.001,
        'TEP_COHERENCY_F2': 0.01,
        
        # Statistical validation limits
        'TEP_LOSO_MAX_STATIONS': 50,
        'TEP_LODO_MAX_DAYS': 100,
        
        # Optional metadata flags
        'TEP_FETCH_CLOCK_METADATA': False,
        'TEP_REQUIRE_CLOCK_METADATA': False,
    }
    
    @staticmethod
    def get_int(key: str, default: Optional[int] = None) -> int:
        """
        Get integer configuration value with proper error handling.
        
        Args:
            key: Environment variable name
            default: Default value if key not found or invalid
            
        Returns:
            int: Configuration value
            
        Raises:
            ValueError: If no default provided and key is invalid
        """
        if default is None:
            default = TEPConfig.DEFAULTS.get(key)
            
        value = os.getenv(key)
        if value is None:
            if default is None:
                raise ValueError(f"Required configuration {key} not found and no default provided")
            return default
            
        try:
            return int(value)
        except ValueError as e:
            if default is None:
                raise ValueError(f"Invalid integer value for {key}: '{value}'") from e
            return default
    
    @staticmethod
    def get_float(key: str, default: Optional[float] = None) -> float:
        """
        Get float configuration value with proper error handling.
        
        Args:
            key: Environment variable name
            default: Default value if key not found or invalid
            
        Returns:
            float: Configuration value
            
        Raises:
            ValueError: If no default provided and key is invalid
        """
        if default is None:
            default = TEPConfig.DEFAULTS.get(key)
            
        value = os.getenv(key)
        if value is None:
            if default is None:
                raise ValueError(f"Required configuration {key} not found and no default provided")
            return default
            
        try:
            return float(value)
        except ValueError as e:
            if default is None:
                raise ValueError(f"Invalid float value for {key}: '{value}'") from e
            return default
    
    @staticmethod
    def get_bool(key: str, default: Optional[bool] = None) -> bool:
        """
        Get boolean configuration value with proper error handling.
        
        Args:
            key: Environment variable name  
            default: Default value if key not found
            
        Returns:
            bool: Configuration value
        """
        if default is None:
            default = TEPConfig.DEFAULTS.get(key, False)
            
        value = os.getenv(key)
        if value is None:
            return default
            
        return str(value).lower() in ('1', 'true', 'yes', 'on')
    
    @staticmethod
    def get_str(key: str, default: Optional[str] = None) -> str:
        """
        Get string configuration value.
        
        Args:
            key: Environment variable name
            default: Default value if key not found
            
        Returns:
            str: Configuration value
            
        Raises:
            ValueError: If no default provided and key not found
        """
        if default is None:
            default = TEPConfig.DEFAULTS.get(key)
            
        value = os.getenv(key)
        if value is None:
            if default is None:
                raise ValueError(f"Required configuration {key} not found and no default provided")
            return default
        return value
    
    @staticmethod
    def get_optional_int(key: str) -> Optional[int]:
        """
        Get optional integer that can be None.
        Handles special values like 'all', 'unlimited', 'max' as None.
        
        Args:
            key: Environment variable name
            
        Returns:
            Optional[int]: Integer value or None for unlimited
        """
        value = os.getenv(key)
        if value is None:
            return None
            
        value_lower = value.strip().lower()
        if value_lower in ('all', 'max', 'unlimited', 'inf', 'infinite', ''):
            return None
            
        try:
            return int(value)
        except ValueError:
            return None
    
    @staticmethod
    def get_path(key: str, default: Optional[Union[str, Path]] = None) -> Path:
        """
        Get path configuration value.
        
        Args:
            key: Environment variable name
            default: Default path if key not found
            
        Returns:
            Path: Configuration path
        """
        if default is None:
            default = TEPConfig.DEFAULTS.get(key)
            
        value = os.getenv(key)
        if value is None:
            if default is None:
                raise ValueError(f"Required path configuration {key} not found")
            return Path(default)
        return Path(value)
    
    @staticmethod
    def get_file_limits() -> dict:
        """
        Get file limit configuration with proper inheritance.
        Handles the complex per-center file limit logic from step_1.
        
        Returns:
            dict: File limits per center
        """
        # Global limit with special value handling
        global_limit = TEPConfig.get_optional_int('TEP_FILES_PER_CENTER')
        
        # Per-center limits with inheritance
        limits = {
            'igs_combined': TEPConfig.get_optional_int('TEP_FILES_PER_CENTER_IGS') or global_limit,
            'code': TEPConfig.get_optional_int('TEP_FILES_PER_CENTER_CODE') or global_limit,
            'esa_final': TEPConfig.get_optional_int('TEP_FILES_PER_CENTER_ESA') or global_limit,
        }
        
        return limits
    
    @staticmethod
    def get_worker_count(env_var: str = 'TEP_WORKERS') -> int:
        """
        Get a valid worker count from an environment variable.
        Caps the value at the number of available CPU cores to prevent over-subscription.
        """
        default_workers = mp.cpu_count()
        try:
            user_workers = int(os.getenv(env_var, default_workers))
            # Cap at the number of physical cores
            return max(1, min(user_workers, default_workers))
        except (ValueError, TypeError):
            return default_workers
    
    @staticmethod
    def get_date_range() -> tuple:
        """
        Get date range configuration with validation.
        
        Returns:
            tuple: (start_date_str, end_date_str)
            
        Raises:
            ValueError: If date format is invalid
        """
        start_date = TEPConfig.get_str('TEP_DATE_START')
        end_date = TEPConfig.get_str('TEP_DATE_END')
        
        # Basic validation of date format
        try:
            from datetime import datetime
            datetime.fromisoformat(start_date)
            datetime.fromisoformat(end_date)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD format: {e}") from e
        
        return start_date, end_date
    
    @classmethod
    def validate_configuration(cls) -> List[str]:
        """
        Validate current configuration and return list of issues.
        
        Returns:
            List[str]: List of configuration issues (empty if valid)
        """
        issues = []
        
        try:
            # Test critical numeric values
            bins = cls.get_int('TEP_BINS')
            if bins < 10:
                issues.append(f"TEP_BINS ({bins}) should be at least 10")
            
            max_dist = cls.get_float('TEP_MAX_DISTANCE_KM')
            if max_dist < 1000:
                issues.append(f"TEP_MAX_DISTANCE_KM ({max_dist}) should be at least 1000")
            
            min_bin = cls.get_int('TEP_MIN_BIN_COUNT')
            if min_bin < 50:
                issues.append(f"TEP_MIN_BIN_COUNT ({min_bin}) should be at least 50")
            
            # Test date range
            start_date, end_date = cls.get_date_range()
            from datetime import datetime
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            if end < start:
                issues.append(f"End date ({end_date}) must be after start date ({start_date})")
            
            # Test worker count
            workers = cls.get_worker_count()
            if workers < 1:
                issues.append(f"Worker count ({workers}) must be at least 1")
            
        except (ValueError, TypeError) as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues

    @classmethod
    def print_configuration(cls, logger_func=print):
        """
        Print current configuration for debugging.
        
        Args:
            logger_func: Function to use for logging (default: print)
        """
        logger_func("=== TEP Configuration ===")
        
        # Analysis parameters
        logger_func("Analysis Parameters:")
        logger_func(f"  TEP_BINS: {cls.get_int('TEP_BINS')}")
        logger_func(f"  TEP_MAX_DISTANCE_KM: {cls.get_float('TEP_MAX_DISTANCE_KM')}")
        logger_func(f"  TEP_MIN_BIN_COUNT: {cls.get_int('TEP_MIN_BIN_COUNT')}")
        logger_func(f"  TEP_BOOTSTRAP_ITER: {cls.get_int('TEP_BOOTSTRAP_ITER')}")
        
        # Processing parameters
        logger_func("Processing Parameters:")
        logger_func(f"  TEP_WORKERS: {cls.get_worker_count('TEP_WORKERS')}")
        logger_func(f"  TEP_MEMORY_LIMIT_GB: {cls.get_float('TEP_MEMORY_LIMIT_GB')}")
        
        # File limits
        logger_func("File Limits:")
        limits = cls.get_file_limits()
        for center, limit in limits.items():
            logger_func(f"  {center.upper()}: {'unlimited' if limit is None else limit}")
        
        # Date range
        start_date, end_date = cls.get_date_range()
        logger_func(f"Date Range: {start_date} to {end_date}")
        
        # Feature flags
        logger_func("Feature Flags:")
        flags = [
            'TEP_PROCESS_ALL_CENTERS', 'TEP_WRITE_PAIR_LEVEL', 'TEP_ENABLE_JACKKNIFE',
            'TEP_ENABLE_ANISOTROPY', 'TEP_ENABLE_TEMPORAL', 'TEP_ENABLE_LOSO',
            'TEP_ENABLE_LODO', 'TEP_ENABLE_ENHANCED_ANISOTROPY'
        ]
        for flag in flags:
            logger_func(f"  {flag}: {cls.get_bool(flag)}")
        
        logger_func("========================")


# Convenience instances for common use patterns
config = TEPConfig()
