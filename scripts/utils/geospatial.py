#!/usr/bin/env python3
"""
TEP GNSS Geospatial Utilities
============================

Shared geospatial analysis functions to avoid code duplication across analysis steps.
"""

import numpy as np
from typing import Union


def compute_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute azimuth from station 1 to station 2 in degrees.
    
    Args:
        lat1, lon1: Latitude and longitude of first station in degrees
        lat2, lon2: Latitude and longitude of second station in degrees
        
    Returns:
        float: Azimuth in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    azimuth = np.arctan2(y, x)
    return (np.degrees(azimuth) + 360) % 360


def classify_ew_ns(azimuth: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """
    Classify direction as East-West or North-South based on azimuth.
    
    Args:
        azimuth: Azimuth in degrees (0-360) - can be scalar or array
        
    Returns:
        str or array: 'EW' for East-West, 'NS' for North-South
    """
    # East-West: 45-135° and 225-315° (±45° around E and W)
    if isinstance(azimuth, (int, float)):
        return 'EW' if (45 <= azimuth <= 135) or (225 <= azimuth <= 315) else 'NS'
    else:
        # Vectorized version for arrays
        return np.where(
            ((azimuth >= 45) & (azimuth <= 135)) | ((azimuth >= 225) & (azimuth <= 315)),
            'EW', 'NS'
        )


def compute_great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
        
    Returns:
        float: Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    R = 6371.0
    return R * c


def classify_directional_sector(azimuth: Union[float, np.ndarray], n_sectors: int = 8) -> Union[str, np.ndarray]:
    """
    Classify azimuth into directional sectors.
    
    Args:
        azimuth: Azimuth in degrees (0-360)
        n_sectors: Number of sectors (default: 8 for N, NE, E, SE, S, SW, W, NW)
        
    Returns:
        str or array: Sector name(s)
    """
    if n_sectors == 8:
        sector_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        sector_size = 45.0
        # Adjust so North is centered at 0°
        adjusted_azimuth = (azimuth + sector_size/2) % 360
        sector_idx = (adjusted_azimuth // sector_size).astype(int) if hasattr(azimuth, 'astype') else int(adjusted_azimuth // sector_size)
        
        if isinstance(azimuth, (int, float)):
            return sector_names[sector_idx]
        else:
            return np.array([sector_names[i] for i in sector_idx])
    else:
        # Generic sector classification
        sector_size = 360.0 / n_sectors
        adjusted_azimuth = (azimuth + sector_size/2) % 360
        sector_idx = (adjusted_azimuth // sector_size).astype(int) if hasattr(azimuth, 'astype') else int(adjusted_azimuth // sector_size)
        return sector_idx
