#!/usr/bin/env python3
"""
TEP-GNSS Analysis Configuration

Theoretical Framework:
    The analysis implements phase-coherent cross-spectral methods to detect
    signatures of the Temporal Equivalence Principle in Global Navigation
    Satellite System timing data. The methodology follows Smawfield (2025)
    and employs multi-center validation across CODE, IGS, and ESA products.

Parameter Specifications:
    TEP_LAMBDA_RANGE: Predicted correlation lengths for screened scalar
                      fields (Khoury & Weltman 2004; Damour & Polyakov 1994)
    TEP_COUPLING_BOUND: Maximum coupling strength β/MPl consistent with
                        PPN constraints (Cassini: |γ-1| < 2.3×10⁻⁵)
    TEP_SCREENING_SCALE: Environmental screening scale / Temporal Topology characteristic length near Earth's surface

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


class TEPConfig:
    """Centralized configuration management for TEP GNSS Analysis"""

    # Default values for common configuration parameters
    DEFAULTS = {
        # Analysis parameters
        "TEP_BINS": 40,  # Total number of distance bins for correlation analysis
        "TEP_MAX_DISTANCE_KM": 13000.0,  # Maximum distance in km for binning and correlation analysis
        "TEP_MIN_BIN_COUNT": 50,  # Minimum number of pairs required per distance bin for correlation fitting
        "TEP_BOOTSTRAP_ITER": 5000,  # Number of bootstrap iterations for uncertainty estimation (enhanced)
        "TEP_NULL_ITERATIONS": 20,  # Number of null hypothesis iterations per analysis center (20 × 3 centers = 60 per scrambling type, 180 total tests)
        "TEP_MIN_BINS_FOR_FIT": 5,  # Minimum number of valid bins required for fitting a correlation model
        "TEP_CORRELATION_LENGTH_INITIAL_GUESS": 3000,  # Initial guess for the correlation length (λ) in km during model fitting
        "TEP_MIN_DISTANCE_FOR_FIT": 100.0,  # Minimum distance in km to include pairs in model fitting
        "TEP_MAX_DISTANCE_FOR_FIT": 13000.0,  # Maximum distance in km to include pairs in model fitting (aligned with TEP_MAX_DISTANCE_KM)
        "TEP_NUM_ELEV_BINS_FOR_JACKKNIFE": 5,  # Number of elevation bins used in jackknife resampling for anisotropy analysis
        "TEP_MIN_POINTS_FOR_TREND": 3,  # Minimum data points required to compute a trend
        "TEP_MIN_DISTANCE_FOR_CIRCULAR_STATS": 100.0,  # Minimum distance in km for calculating circular statistics
        "TEP_MAX_DISTANCE_FOR_CIRCULAR_STATS": 13000.0,  # Maximum distance in km for calculating circular statistics (aligned with TEP_MAX_DISTANCE_KM)
        "TEP_NUM_BINS_FOR_CIRCULAR_STATS": 10,  # Number of bins for circular statistics (e.g., azimuth, local time difference)
        "TEP_MIN_PAIRS_PER_BIN": 50,  # Minimum number of pairs required per bin for general statistical analysis
        # Outlier detection thresholds for distance in km
        "TEP_MIN_DISTANCE_OUTLIER_KM": 1.0,  # Minimum distance in km below which pairs are considered outliers
        "TEP_MAX_DISTANCE_OUTLIER_KM": 20000.0,  # Maximum distance in km above which pairs are considered outliers
        # Analysis-relevant thresholds for validation
        "TEP_ANALYSIS_MAX_DISTANCE_KM": 13000.0,  # Maximum distance used in main analysis (for validation alignment)
        "TEP_ANALYSIS_MIN_DISTANCE_KM": 100.0,  # Minimum distance used in main analysis (for validation alignment)
        # Processing parameters
        "TEP_WORKERS": 14,  # Number of worker processes to use for parallel processing (optimized for comprehensive analysis)
        "TEP_MEMORY_LIMIT_GB": 12.0,  # Memory limit in GB for data processing to prevent excessive RAM usage
        # Data parameters
        "TEP_MIN_STATIONS": 0,  # Minimum number of stations required for analysis (0 means no minimum)
        "TEP_DATE_START": "2023-01-01",  # Start date for data acquisition and analysis (YYYY-MM-DD)
        "TEP_DATE_END": "2025-06-30",  # End date for data acquisition and analysis (YYYY-MM-DD)
        # Network timeouts
        "TEP_NETWORK_TIMEOUT": 30,  # General network timeout in seconds for requests
        "TEP_DOWNLOAD_TIMEOUT": 60,  # Timeout in seconds for individual file downloads
        "TEP_MAX_PARALLEL_DOWNLOADS": 10,  # Maximum number of concurrent file downloads
        "TEP_MIN_FILE_SIZE_MB": 1.0,  # Minimum expected file size in MB for a successful download (used for validation)
        # Data Source URLs
        "TEP_IGS_COORDS_URL": "https://files.igs.org/pub/station/general/IGSNetworkWithFormer.json",  # URL for IGS station coordinates in JSON format
        "TEP_IGS_CLK_URL_TEMPLATE": "https://igs.bkg.bund.de/root_ftp/IGS/products/{week:04d}/IGS0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz",  # Template for IGS clock product download URLs
        "TEP_CODE_CLK_URL_TEMPLATE": "http://ftp.aiub.unibe.ch/CODE/{year}/COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz",  # Template for CODE clock product download URLs
        "TEP_ESA_CLK_URL_TEMPLATE": "http://navigation-office.esa.int/products/gnss-products/{week}/ESA0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz",  # Template for ESA clock product download URLs
        # File limits
        "TEP_MAX_PAIR_FILES": None,  # Maximum number of pair files to process (None means unlimited)
        # Feature flags
        "TEP_PROCESS_ALL_CENTERS": True,  # Enable processing for all analysis centers (CODE, IGS, ESA)
        "TEP_WRITE_PAIR_LEVEL": True,  # Enable writing of pair-level data to CSV files
        "TEP_ENABLE_JACKKNIFE": True,  # Enable Jackknife resampling for uncertainty estimation
        "TEP_ENABLE_ANISOTROPY": True,  # Enable anisotropy analysis (directional dependence)
        "TEP_ENABLE_TEMPORAL": True,  # Enable temporal analysis (time-dependent signals)
        "TEP_ENABLE_LOSO": True,  # Enable Leave-One-Station-Out cross-validation
        "TEP_ENABLE_LODO": True,  # Enable Leave-One-Day-Out cross-validation
        "TEP_ENABLE_ENHANCED_ANISOTROPY": True,  # Enable enhanced anisotropy analysis with directional sectors
        # Cross-validation flags
        "TEP_ENABLE_MONTHLY_CV": True,  # Enable monthly cross-validation
        "TEP_ENABLE_STATION_BLOCKS_CV": True,  # Enable cross-validation using station blocks
        "TEP_ENABLE_BOOTSTRAP_CV": True,  # Enable bootstrap cross-validation
        # Rebuild flags
        "TEP_REBUILD_COORDS": True,  # If True, force recalculation and download of station coordinates
        "TEP_REBUILD_CLK": False,  # If True, force re-download and reprocessing of clock products
        "TEP_REBUILD_METADATA": False,  # If True, force re-extraction and reprocessing of metadata
        "TEP_SKIP_COORDS": False,  # If True, skip coordinate download and use existing local files
        # Advanced options
        "TEP_USE_REAL_COHERENCY": False,  # If True, use real part of coherence; otherwise, use magnitude
        "TEP_COHERENCY_F1": 1e-5,  # Lower frequency bound (in Hz) for coherence calculation
        "TEP_COHERENCY_F2": 5e-4,  # Upper frequency bound (in Hz) for coherence calculation
        # Statistical validation limits
        "TEP_LOSO_MAX_STATIONS": 50,  # Maximum number of stations to sample for Leave-One-Station-Out (LOSO) cross-validation
        "TEP_LODO_MAX_DAYS": 100,  # Maximum number of days to sample for Leave-One-Day-Out (LODO) cross-validation
        # Cross-validation parameters
        "TEP_MONTHLY_CV_FOLDS": 10,  # Number of folds for monthly cross-validation
        "TEP_STATION_BLOCK_SIZE": 10,  # Size of station blocks for block cross-validation
        "TEP_LOSO_SAMPLE_SIZE": 50,  # Sample size for Leave-One-Station-Out (LOSO) cross-validation
        "TEP_LODO_SAMPLE_SIZE": 100,  # Sample size for Leave-One-Day-Out (LODO) cross-validation
        "TEP_BOOTSTRAP_SAMPLES": 100,  # Number of bootstrap samples for general cross-validation (reduced for system stability)
        # Robust Block Bootstrap Configuration (Step 3.1) - Memory Optimized
        "TEP_STATION_BOOTSTRAP_SAMPLES": 20,  # Number of bootstrap samples for station-level analysis (reduced for memory)
        "TEP_DAY_BOOTSTRAP_SAMPLES": 30,  # Number of bootstrap samples for day-level analysis (reduced for memory)
        "TEP_HYBRID_BOOTSTRAP_SAMPLES": 20,  # Number of bootstrap samples for hybrid (station-day) analysis (reduced for memory)
        "TEP_BOOTSTRAP_MIN_STATIONS": 50,  # Minimum number of stations required for bootstrap sampling (reduced for memory)
        "TEP_BOOTSTRAP_MIN_DAYS": 50,  # Minimum number of days required for bootstrap sampling (reduced for memory)
        "TEP_BOOTSTRAP_CONFIDENCE_LEVEL": 0.95,  # Confidence level for bootstrap confidence intervals
        # Optional metadata flags
        "TEP_FETCH_CLOCK_METADATA": False,  # If True, attempt to fetch additional clock metadata
        "TEP_REQUIRE_CLOCK_METADATA": False,  # If True, pipeline will fail if clock metadata cannot be fetched
        # Scientific constants for TEP analysis
        "TEP_ROTATION_SIGNATURE_GRADIENT_STRENGTH": 0.05,  # Expected gradient strength of rotational signature
        "TEP_ROTATION_SIGNATURE_LONGITUDE_CORR": 0.3,  # Expected longitude correlation for rotational signature
        "TEP_ANISOTROPY_CV_MODERATE_LOWER": 0.2,  # Lower threshold for moderate anisotropy coefficient of variation
        "TEP_ANISOTROPY_CV_MODERATE_UPPER": 0.5,  # Upper threshold for moderate anisotropy coefficient of variation
        "TEP_ANISOTROPY_CV_ISOTROPIC_THRESHOLD": 0.1,  # Threshold for considering anisotropy as isotropic
        "TEP_ANISOTROPY_CV_CHAOTIC_THRESHOLD": 0.8,  # Threshold for considering anisotropy as chaotic
        "TEP_DIPOLE_STRENGTH_THRESHOLD": 0.3,  # Threshold for detecting a significant dipole strength
        "TEP_MIN_EPOCHS": 20,  # Minimum number of epochs required for time-series analysis
        "TEP_INITIAL_LAMBDA_GUESS": 3000,  # Initial guess for the correlation length parameter in TEP models
        # NEW: Helical Motion Analysis Configuration (ADDITIONS ONLY)
        "TEP_ENABLE_CHANDLER_WOBBLE": True,
        "TEP_ENABLE_3D_HARMONICS": True,
        "TEP_ENABLE_BEAT_FREQUENCIES": True,
        "TEP_ENABLE_RELATIVE_MOTION_BEATS": True,  # NEW: Enhanced relative motion analysis
        "TEP_ENABLE_MESH_DANCE_ANALYSIS": True,  # NEW: The ultimate "dance" analysis
        "TEP_ENABLE_NUTATION_ANALYSIS": False,  # Requires multi-year data
        "TEP_CHANDLER_PERIOD_DAYS": 425.0,  # Consistent with code default
        "TEP_CHANDLER_WINDOW_DAYS": 600,  # Window length for high-S/N Chandler fit
        "TEP_CHANDLER_WINDOW_STEP_DAYS": 60,  # Step between window centers
        "TEP_SPHERICAL_THETA_BINS": 6,
        "TEP_SPHERICAL_PHI_BINS": 12,
        "TEP_BEAT_MIN_PERIOD_DAYS": 7.0,
        "TEP_BEAT_MIN_CYCLES": 3,
        "TEP_BEAT_SIGNIFICANCE_THRESHOLD": 0.3,  # Much more sensitive threshold
        "TEP_MESH_COHERENCE_THRESHOLD": 0.05,  # More sensitive mesh analysis
        "TEP_MIN_CORRELATION_THRESHOLD": 0.2,  # Minimum correlation to consider significant
        "TEP_SIGNIFICANCE_THRESHOLD": 2.0,  # 2.0σ threshold (95% confidence) for planetary opposition analysis
        "TEP_NUTATION_PERIOD_YEARS": 18.6,
        # NEW: Chunked Data Loading Optimization
        "TEP_MIN_CHUNK_SIZE": 25000,  # Minimum rows for a data chunk
        "TEP_MAX_CHUNK_SIZE": 100000,  # Maximum rows for a data chunk
        "TEP_CHUNK_CONSOLIDATION_THRESHOLD": 100,  # Number of chunks before consolidation
        # NEW: Astronomical Event Analysis Configuration
        "TEP_ENABLE_JUPITER_OPPOSITION": True,  # Jupiter opposition pulse analysis
        "TEP_ENABLE_SATURN_OPPOSITION": True,  # Saturn opposition analysis (smaller signal)
        "TEP_ENABLE_MARS_OPPOSITION": True,  # Mars opposition analysis (weakest signal)
        "TEP_ENABLE_LUNAR_STANDSTILL": True,
        # NEW: Sampling and Random Seed Configuration
        "TEP_ENABLE_MONTE_CARLO_ORBITAL_TEST": True,  # Enable Monte Carlo orbital surrogate test by default
        "TEP_MONTE_CARLO_N_SURROGATES": 10000,  # Default number of surrogate iterations
        "TEP_MONTE_CARLO_SEED": 42,  # Monte Carlo test random seed
        "TEP_ANISOTROPY_SAMPLING_FRAC": 1.0,  # Fraction of data to sample for anisotropy analysis
        "TEP_RANDOM_SEED": 42,  # Random seed for reproducible sampling
        # NEW: Visualization Configuration
        "TEP_HEXBIN_GRID_SIZE": 20,  # Grid size for hexbin plots
        "TEP_ANISOTROPY_DIST_BINS": 20,  # Number of distance bins for anisotropy analysis
        "TEP_MIN_PAIRS_PER_BIN": 50,  # Minimum number of pairs required per bin for general statistical analysis
        "TEP_ANISOTROPY_LON_BINS": 20,  # Number of longitude bins for anisotropy analysis
        "TEP_MIN_BINS_FOR_FIT": 5,  # Minimum number of valid bins required for fitting a correlation model
        "TEP_ENABLE_SOLAR_ECLIPSE": True,  # Solar eclipse ionospheric effects
        "TEP_ENABLE_PERIHELION_APHELION": True,  # Earth perihelion/aphelion analysis
        "TEP_EVENT_WINDOW_DAYS": 60,  # ±60 days around each event
        "TEP_JUPITER_AMPLITUDE_FRACTION": 0.0022,  # 0.22% of solar annual amplitude
        "TEP_SATURN_AMPLITUDE_FRACTION": 0.00019,  # 0.019% of solar annual amplitude
        "TEP_MARS_AMPLITUDE_FRACTION": 0.00005,  # 0.005% of solar annual amplitude (estimated)
        "TEP_EVENT_MIN_PAIRS": 1000,  # Minimum pairs needed per event window
        "TEP_STORM_SCRUBBING": False,  # Enable geomagnetic storm removal
        # NEW: Data Loading & Logging Configuration
        "TEP_LOAD_BATCH_SIZE": 10,  # Number of files to load in a batch
        "TEP_FILE_LOGGING_INTERVAL": 50,  # Log progress every N files
        "TEP_ENABLE_TEMPORAL_ORBITAL_TRACKING": True,  # Enable temporal orbital tracking analysis
        "TEP_LOGGING_INTERVAL_FILES": 50,  # Log progress every N files processed
        # Major Lunar Standstill Configuration
        "TEP_LUNAR_STANDSTILL_WINDOW_MONTHS": 6,  # ±6 months around peak
        "TEP_SIDEREAL_DAY_HOURS": 23.934469591,  # Precise sidereal day length
        "TEP_LUNAR_PRE_STANDSTILL_MONTHS": 12,  # 12 months before standstill peak
        "TEP_LUNAR_DURING_STANDSTILL_MONTHS": 24,  # 24 months during standstill cycle
        "TEP_LUNAR_STANDSTILL_PEAK_DATE": "2024-12-15",  # Approximate peak date
        # Solar Eclipse Configuration
        "TEP_ECLIPSE_WINDOW_HOURS": 12,  # ±12 hours around eclipse
        "TEP_SOLAR_ECLIPSE_DATE": "2024-04-08",  # Total solar eclipse date
        "TEP_ECLIPSE_TOTALITY_DURATION": 4.5,  # Duration of totality in minutes
    }

    # Site-themed colors for visualizations
    theme_colors = {
        "primary": "#2D0140",  # Dark purple
        "secondary": "#495773",  # Warm beige
        "accent": "#4A90C2",  # Medium blue
        "highlight": "#F39C12",  # Orange
        "text": "#220126",  # Dark text
        "background": "#F8F8FF",  # Off-white background
        "grid": "#CCCCCC",  # Light grey for grids
        "border": "#999999",  # Medium grey for borders
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
                raise ValueError(
                    f"Required configuration {key} not found and no default provided"
                )
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
                raise ValueError(
                    f"Required configuration {key} not found and no default provided"
                )
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

        return str(value).lower() in ("1", "true", "yes", "on")

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
                raise ValueError(
                    f"Required configuration {key} not found and no default provided"
                )
            return default
        return value

    @staticmethod
    def get_list(key: str, default: Optional[List[str]] = None) -> List[str]:
        """
        Get list configuration value (e.g., comma-separated strings).

        Args:
            key: Environment variable name
            default: Default value if key not found

        Returns:
            List[str]: Configuration value as a list of strings

        Raises:
            ValueError: If no default provided and key not found
        """
        value = os.getenv(key)
        if value is None:
            if default is None:
                default_from_defaults = TEPConfig.DEFAULTS.get(key)
                if default_from_defaults is None:
                    raise ValueError(
                        f"Required configuration {key} not found and no default provided"
                    )
                if isinstance(default_from_defaults, str):
                    return [
                        s.strip() for s in default_from_defaults.split(",") if s.strip()
                    ]
                return default_from_defaults
            return default

        return [s.strip() for s in value.split(",") if s.strip()]

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
        if value_lower in ("all", "max", "unlimited", "inf", "infinite", ""):
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
        global_limit = TEPConfig.get_optional_int("TEP_MAX_PAIR_FILES")

        # Per-center limits with inheritance
        limits = {
            "igs_combined": TEPConfig.get_optional_int("TEP_FILES_PER_CENTER_IGS")
            or global_limit,
            "code": TEPConfig.get_optional_int("TEP_FILES_PER_CENTER_CODE")
            or global_limit,
            "esa_final": TEPConfig.get_optional_int("TEP_FILES_PER_CENTER_ESA")
            or global_limit,
        }

        return limits

    @staticmethod
    def get_worker_count(env_var: str = "TEP_WORKERS") -> int:
        """
        Get a valid worker count from an environment variable.
        Allows oversubscription for I/O-bound tasks (up to 2x CPU cores).
        """
        default_workers = mp.cpu_count()
        try:
            user_workers = int(os.getenv(env_var, default_workers))
            # Allow up to 2x CPU cores for I/O-bound tasks, but cap at 32
            max_workers = min(32, default_workers * 2)
            return max(1, min(user_workers, max_workers))
        except (ValueError, TypeError):
            return default_workers

    @staticmethod
    def get_adaptive_lambda_bounds(distances: np.ndarray) -> tuple:
        """
        Get adaptive lambda bounds based on data characteristics.

        Args:
            distances: Array of distance values

        Returns:
            tuple: (lower_bounds, upper_bounds) for curve_fit
        """
        if len(distances) == 0:
            return ([1e-10, 100, -1], [5, 15000, 1])

        # Base bounds on 80% of maximum distance, but cap at 15,000 km
        max_reasonable_lambda = min(15000, np.max(distances) * 0.8)
        return ([1e-10, 100, -1], [5, max_reasonable_lambda, 1])

    @staticmethod
    def get_date_range() -> tuple:
        """
        Get date range configuration with validation.

        Returns:
            tuple: (start_date_str, end_date_str)

        Raises:
            ValueError: If date format is invalid
        """
        start_date = TEPConfig.get_str("TEP_DATE_START")
        end_date = TEPConfig.get_str("TEP_DATE_END")

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
            bins = cls.get_int("TEP_BINS")
            if bins < 10:
                issues.append(f"TEP_BINS ({bins}) should be at least 10")

            max_dist = cls.get_float("TEP_MAX_DISTANCE_KM")
            if max_dist < 1000:
                issues.append(
                    f"TEP_MAX_DISTANCE_KM ({max_dist}) should be at least 1000"
                )

            min_bin = cls.get_int("TEP_MIN_BIN_COUNT")
            if min_bin < 1:
                issues.append(f"TEP_MIN_BIN_COUNT ({min_bin}) should be at least 1")

            # Test distance limit consistency
            max_dist_general = cls.get_float("TEP_MAX_DISTANCE_KM")
            max_dist_fit = cls.get_float("TEP_MAX_DISTANCE_FOR_FIT")
            if max_dist_fit > max_dist_general:
                issues.append(
                    f"TEP_MAX_DISTANCE_FOR_FIT ({max_dist_fit} km) cannot exceed TEP_MAX_DISTANCE_KM ({max_dist_general} km)"
                )

            # Test date range
            start_date, end_date = cls.get_date_range()
            from datetime import datetime

            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            if end < start:
                issues.append(
                    f"End date ({end_date}) must be after start date ({start_date})"
                )

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
            logger_func(
                f"  {center.upper()}: {'unlimited' if limit is None else limit}"
            )

        # Date range
        start_date, end_date = cls.get_date_range()
        logger_func(f"Date Range: {start_date} to {end_date}")

        # Feature flags
        logger_func("Feature Flags:")
        flags = [
            "TEP_PROCESS_ALL_CENTERS",
            "TEP_WRITE_PAIR_LEVEL",
            "TEP_ENABLE_JACKKNIFE",
            "TEP_ENABLE_ANISOTROPY",
            "TEP_ENABLE_TEMPORAL",
            "TEP_ENABLE_LOSO",
            "TEP_ENABLE_LODO",
            "TEP_ENABLE_ENHANCED_ANISOTROPY",
        ]
        for flag in flags:
            logger_func(f"  {flag}: {cls.get_bool(flag)}")

        logger_func("========================")

    @staticmethod
    def get(key: str, default=None):
        """
        Generic get method for backward compatibility.
        Automatically determines the type and calls the appropriate method.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value with appropriate type
        """
        # Try to get from defaults first to determine type
        default_value = TEPConfig.DEFAULTS.get(key, default)

        if default_value is None:
            # No default, try to infer type from environment variable
            env_value = os.getenv(key)
            if env_value is None:
                raise ValueError(
                    f"Required configuration {key} not found and no default provided"
                )

            # Try to convert to appropriate type
            try:
                if env_value.lower() in ("true", "false"):
                    return TEPConfig.get_bool(key, default)
                elif (
                    "." in env_value
                    and env_value.replace(".", "").replace("-", "").isdigit()
                ):
                    return TEPConfig.get_float(key, default)
                elif env_value.isdigit() or (
                    env_value.startswith("-") and env_value[1:].isdigit()
                ):
                    return TEPConfig.get_int(key, default)
                else:
                    return TEPConfig.get_str(key, default)
            except:
                return TEPConfig.get_str(key, default)

        # Use default to determine type
        if isinstance(default_value, bool):
            return TEPConfig.get_bool(key, default)
        elif isinstance(default_value, (int, float)):
            if isinstance(default_value, int):
                return TEPConfig.get_int(key, default)
            else:
                return TEPConfig.get_float(key, default)
        elif isinstance(default_value, list):
            return TEPConfig.get_list(key, default)
        else:
            return TEPConfig.get_str(key, default)


# Convenience instances for common use patterns
config = TEPConfig()
