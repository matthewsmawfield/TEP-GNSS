#!/usr/bin/env python3
"""
Step 2.5: Dual-Motion Geometric Validation Analysis
===================================================

LIGHTWEIGHT geometric validation layer on top of Step 2.2 results.
Extracts existing computations and adds ONLY new geometric calculations:
- Velocity vector decomposition (Earth + Solar System)
- Sector-by-sector geometric predictions
- Predicted vs observed validation tests
- Statistical assessment of geometric fit

NO RECOMPUTATION of hemisphere stratification, temporal windows, or sector analysis.

Author: Matthew Smawfield
Date: 2025-11-24
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Anchor to package root
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.utils.logger import print_status, TEPLogger, set_step_logger

# Physical constants
SOLAR_APEX_RA_DEG = 271.96  # Galactic apex Right Ascension  
SOLAR_APEX_DEC_DEG = 30.0   # Galactic apex Declination
SOLAR_APEX_SPEED_KMS = 20.0 # Solar System peculiar velocity
ECLIPTIC_TILT_DEG = 23.44

# Geometric model parameters (calibrated from observations)
MIN_VZ_THRESHOLD = 0.1       # Minimum Z-velocity to avoid singularities (km/s)
VZ_SOFTENING_FACTOR = 5.0    # Denominator softening for stable ratios
NS_COUPLING_FACTOR = 0.2     # Baseline geometry coupling between XY and Z projections

SECTORS = {
    'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
    'S': 180, 'SW': 225, 'W': 270, 'NW': 315
}

def calculate_3d_velocity_vectors(day_of_year, orbital_speed_kms,
                                  background_ra=None, background_dec=None,
                                  background_speed=None):
    """
    Calculate full 3D velocity vectors in Equatorial Coordinates (J2000).
    
    Includes:
    1. Background Motion (configurable RA/Dec/speed, defaults to Solar Apex)
    2. Earth Orbital Motion (Rotating in Ecliptic Plane -> Transformed to Equatorial)
    
    Args:
        day_of_year: Day of year (1-365)
        orbital_speed_kms: Earth's orbital speed (km/s)
        background_ra: Background velocity RA in degrees (default: Solar Apex)
        background_dec: Background velocity Dec in degrees (default: Solar Apex)
        background_speed: Background velocity magnitude (default: Solar Apex speed)
    
    Returns dict with 3D vector components and angles.
    """
    # Use defaults if not specified
    if background_ra is None:
        background_ra = SOLAR_APEX_RA_DEG
    if background_dec is None:
        background_dec = SOLAR_APEX_DEC_DEG
    if background_speed is None:
        background_speed = SOLAR_APEX_SPEED_KMS
    
    # 1. Background Motion (Configurable)
    apex_ra_rad = np.radians(background_ra)
    apex_dec_rad = np.radians(background_dec)
    
    # Background velocity vector (Equatorial Cartesian)
    v_gal_x = background_speed * np.cos(apex_dec_rad) * np.cos(apex_ra_rad)
    v_gal_y = background_speed * np.cos(apex_dec_rad) * np.sin(apex_ra_rad)
    v_gal_z = background_speed * np.sin(apex_dec_rad)
    
    # 2. Earth Orbital Motion (Dynamic)
    # Approximation: Earth longitude roughly (Day / 365.25) * 360
    # Velocity direction is tangent to orbit (Longitude + 90°)
    # Perihelion ~Jan 3 (Day 3). Longitude ~102°? 
    # Let's use simplified mean longitude: L = 280.460 + 0.9856474 * n
    # Velocity direction is in Ecliptic Plane.
    
    days_since_equinox = day_of_year - 80 # Vernal Equinox ~March 21 (Day 80)
    # Orbital angle relative to Vernal Equinox (0° at Equinox)
    # At Equinox, Earth is at 0° Ecliptic Longitude? No, Sun is at 0°. Earth is at 180°.
    # Velocity is perpendicular to radius.
    # Let's stick to the simple rotation logic but apply 3D rotation matrix.
    
    # Orbit Angle in Ecliptic Plane (0° = aligned with Vernal Equinox vector)
    # Velocity direction rotates 360° per year.
    # At Vernal Equinox (Day 80), Earth is at 180°. Velocity points to 270° (-Y in Ecliptic).
    
    orbit_progress = (day_of_year - 80) / 365.25 * 2 * np.pi
    # Velocity direction in Ecliptic coordinates (X points to Vernal Equinox)
    # v_ecl = [-sin(theta), cos(theta), 0] * speed
    # Check: Day 80 (theta=0). Earth at 180. Velocity points -90 (270). 
    # Correct: V_ecl_x = 0, V_ecl_y = -1.
    
    v_ecl_x = -orbital_speed_kms * np.sin(orbit_progress)
    v_ecl_y = orbital_speed_kms * np.cos(orbit_progress)
    v_ecl_z = 0.0
    
    # Rotate Ecliptic -> Equatorial
    # Rotate around X-axis by obliquity (epsilon)
    epsilon = np.radians(ECLIPTIC_TILT_DEG)
    
    # V_eq_x = V_ecl_x
    # V_eq_y = V_ecl_y * cos(eps) - V_ecl_z * sin(eps)
    # V_eq_z = V_ecl_y * sin(eps) + V_ecl_z * cos(eps)
    
    v_orb_x = v_ecl_x
    v_orb_y = v_ecl_y * np.cos(epsilon)
    v_orb_z = v_ecl_y * np.sin(epsilon)
    
    # 3. Net Velocity (Earth's total velocity through space)
    # Physics: Earth's velocity = Solar System barycentric velocity + Earth orbital velocity
    # Both V_gal and V_orb are defined as velocity vectors (not "wind" directions)
    # Simple vector addition in Cartesian coordinates
    
    v_net_x = v_orb_x + v_gal_x
    v_net_y = v_orb_y + v_gal_y
    v_net_z = v_orb_z + v_gal_z
    
    # Calculate Magnitude and Angles
    v_net_mag = np.sqrt(v_net_x**2 + v_net_y**2 + v_net_z**2)
    
    # Right Ascension (0 to 360)
    v_net_ra_rad = np.arctan2(v_net_y, v_net_x)
    v_net_ra_deg = np.degrees(v_net_ra_rad) % 360
    
    # Declination (-90 to +90)
    v_net_dec_rad = np.arcsin(v_net_z / v_net_mag)
    v_net_dec_deg = np.degrees(v_net_dec_rad)
    
    return {
        'day_of_year': int(day_of_year),
        'orbital_speed_kms': float(orbital_speed_kms),
        'background_ra': float(background_ra),
        'background_dec': float(background_dec),
        'background_speed': float(background_speed),
        'v_net_vec': [float(v_net_x), float(v_net_y), float(v_net_z)],
        'v_net_mag': float(v_net_mag),
        'v_net_ra_deg': float(v_net_ra_deg),
        'v_net_dec_deg': float(v_net_dec_deg),
        # Component in Equatorial Plane (relevant for E-W)
        'v_net_xy_mag': float(np.sqrt(v_net_x**2 + v_net_y**2)),
        # Component along Earth Axis (relevant for N-S)
        'v_net_z_mag': float(abs(v_net_z))
    }

def predict_ew_ns_from_3d_geometry(v_xy, v_z):
    """
    Predict EW/NS ratio from 3D velocity geometry.
    
    Physics: 
    - Equatorial (XY) velocity projects fully onto E-W baselines (which scan the plane).
    - Axial (Z) velocity projects onto N-S baselines (meridians).
    - High Declination (high Vz) -> Stronger N-S correlation -> Lower EW/NS ratio.
    - Low Declination (high Vxy) -> Stronger E-W correlation -> Higher EW/NS ratio.
    
    Model: EW/NS ~ (V_xy / V_z) with geometric corrections for baseline distribution
    
    Args:
        v_xy: Magnitude of velocity in equatorial plane (km/s)
        v_z: Magnitude of velocity along polar axis (km/s)
    
    Returns:
        Predicted EW/NS correlation length ratio
    """
    # Avoid division by zero
    if v_z < MIN_VZ_THRESHOLD:
        v_z = MIN_VZ_THRESHOLD
    
    # Geometric projection model accounting for global baseline distribution
    # EW baselines sample equatorial plane
    pred_ew = v_xy
    # NS baselines sample meridians (with partial coupling to equatorial flow)
    pred_ns = np.sqrt(v_z**2 + NS_COUPLING_FACTOR * v_xy**2)
    
    return pred_ew / pred_ns


def calculate_anisotropy_angle_from_velocity(v_east, v_north):
    """
    Calculate expected anisotropy principal axis angle from velocity vector.
    
    Theory: Anisotropy elongates in direction of net velocity.
    Returns angle in degrees (0° = North, 90° = East, etc.)
    """
    # Net velocity magnitude and angle
    v_mag = np.sqrt(v_east**2 + v_north**2)
    v_angle_deg = np.degrees(np.arctan2(v_east, v_north))  # Note: atan2(x,y) for azimuth
    
    # Normalize to 0-360°
    if v_angle_deg < 0:
        v_angle_deg += 360
    
    return v_angle_deg, v_mag


def calculate_ew_ns_ratio_from_angle(tilt_angle_deg, anisotropy_strength=2.0):
    """
    Calculate predicted EW/NS ratio from anisotropy tilt angle.
    
    Args:
        tilt_angle_deg: Principal axis angle (0° = North, 90° = East)
        anisotropy_strength: Ratio of major to minor axis
    
    Returns:
        Predicted EW/NS ratio
    """
    tilt_rad = np.radians(tilt_angle_deg)
    
    # EW direction is 90° (East)
    # NS direction is 0° (North)
    
    # Project ellipse onto EW and NS directions
    # For ellipse with major axis at angle θ and axes ratio k:
    # Length along direction φ ∝ sqrt(cos²(φ-θ) + k² sin²(φ-θ))
    
    k = 1.0 / anisotropy_strength  # Minor/major ratio
    
    # EW projection (φ = 90°)
    ew_component = np.sqrt(np.cos(tilt_rad - np.pi/2)**2 + k**2 * np.sin(tilt_rad - np.pi/2)**2)
    
    # NS projection (φ = 0°)
    ns_component = np.sqrt(np.cos(tilt_rad)**2 + k**2 * np.sin(tilt_rad)**2)
    
    ew_ns_ratio = ew_component / ns_component
    
    return ew_ns_ratio


def extract_step_2_2_data(data):
    """
    Extract pre-computed results from Step 2.2 (NO recomputation).
    """
    print_status("Extracting existing Step 2.2 results...", "INFO")
    
    # 1. Extract 8-sector anisotropy
    sector_results = data['enhanced_anisotropy_analysis']['sector_results']
    sectors = {}
    for s in SECTORS.keys():
        sectors[s] = {
            'lambda_km': sector_results[s]['lambda_km'],
            'r_squared': sector_results[s]['r_squared'],
            'n_pairs': sector_results[s]['n_pairs'],
            'lambda_error_km': sector_results[s]['param_errors'][1]
        }
    
    # 2. Extract temporal tracking (already computed!)
    tracking = data['temporal_orbital_tracking']['temporal_tracking_data']
    global_tracking = [x for x in tracking if x['hemisphere'] == 'GLOBAL' and x['bucket'] == 'GLOBAL']
    north_tracking = [x for x in tracking if x['hemisphere'] == 'N' and x['bucket'] == 'N']
    south_tracking = [x for x in tracking if x['hemisphere'] == 'S' and x['bucket'] == 'S']
    
    # 3. Extract hemisphere correlations (already computed!)
    orbital_corr = data['comprehensive_report']['detection_summary']['orbital_motion']
    
    print_status(f"  ✓ Extracted {len(sectors)} sectors", "INFO")
    print_status(f"  ✓ Extracted {len(global_tracking)} temporal windows", "INFO")
    print_status(f"  ✓ Hemisphere data: N={len(north_tracking)}, S={len(south_tracking)}", "INFO")
    
    return {
        'sectors': sectors,
        'temporal': {
            'global': global_tracking,
            'northern': north_tracking,
            'southern': south_tracking
        },
        'orbital_correlation': {
            'r': orbital_corr['correlation'],
            'p': orbital_corr['p_value']
        }
    }


def geometric_validation(extracted_data, velocity_time_series):
    """
    GEOMETRIC ANGLE VALIDATION: Track how velocity angle modulates anisotropy.
    
    Key insight: Anisotropy tilt angle should track net velocity direction.
    - Orbital motion alone: angle rotates 360° per year
    - Galactic motion alone: angle is fixed
    - BOTH: angle modulates with orbital period but with galactic offset
    
    We use EW/NS ratio as proxy for tilt angle modulation.
    """
    print_status("Performing geometric angle validation...", "INFO")
    
    # For each temporal window: predict EW/NS ratio from velocity angle
    angle_analysis = []
    
    # Calibrate anisotropy strength from observed time-average
    obs_sectors = extracted_data['sectors']
    time_avg_ew_ns = (obs_sectors['E']['lambda_km'] + obs_sectors['W']['lambda_km']) / 2 / \
                     (obs_sectors['N']['lambda_km'] + obs_sectors['S']['lambda_km']) / 2
    
    print_status(f"  Time-averaged EW/NS ratio: {time_avg_ew_ns:.3f}", "INFO")
    
    for vvec in velocity_time_series:
        # Use 3D components if available (new format)
        if 'v_net_vec' in vvec:
            v_xy = vvec['v_net_xy_mag']
            v_z = vvec['v_net_z_mag']
            predicted_ratio = predict_ew_ns_from_3d_geometry(v_xy, v_z)
            
            # For angle analysis, we still want the 2D angle for comparison
            v_east = vvec['v_net_vec'][0] # X
            v_north = vvec['v_net_vec'][1] # Y (approximation for output)
            v_angle = vvec['v_net_ra_deg']
            v_mag = vvec['v_net_mag']
            
        else:
            # Fallback to old 2D format (should not happen with updated calculate_3d_velocity_vectors)
            v_east = vvec['net']['v_east_kms']
            v_north = vvec['net']['v_north_kms']
            v_angle, v_mag = calculate_anisotropy_angle_from_velocity(v_east, v_north)
            predicted_ratio = calculate_ew_ns_ratio_from_angle(v_angle, anisotropy_strength=time_avg_ew_ns)
            v_xy = v_mag
            v_z = 0
        
        # Get observed ratio for this window
        observed_ratio = vvec.get('ew_ns_ratio', None)
        
        if observed_ratio is not None:
            angle_analysis.append({
                'day_of_year': vvec['day_of_year'],
                'orbital_speed_kms': vvec['orbital_speed_kms'],
                'v_angle_deg': float(v_angle),
                'v_magnitude_kms': float(v_mag),
                'v_xy_kms': float(v_xy),
                'v_z_kms': float(v_z),
                'predicted_ew_ns_ratio': float(predicted_ratio),
                'observed_ew_ns_ratio': float(observed_ratio)
            })
    
    # Statistical comparison: predicted vs observed EW/NS ratios
    pred_ratios = np.array([a['predicted_ew_ns_ratio'] for a in angle_analysis])
    obs_ratios = np.array([a['observed_ew_ns_ratio'] for a in angle_analysis])
    v_angles = np.array([a['v_angle_deg'] for a in angle_analysis])
    
    r_angle, p_angle = stats.pearsonr(pred_ratios, obs_ratios)
    
    # Correlation of velocity angle with observed ratio
    r_vangle_obs, p_vangle_obs = stats.pearsonr(v_angles, obs_ratios)
    
    print_status(f"  Predicted vs Observed EW/NS: r = {r_angle:.4f}, p = {p_angle:.6f}", "INFO")
    print_status(f"  Velocity angle vs Observed: r = {r_vangle_obs:.4f}, p = {p_vangle_obs:.6f}", "INFO")
    
    # Calculate angle statistics
    angle_mean = np.mean(v_angles)
    angle_std = np.std(v_angles)
    angle_range = (np.min(v_angles), np.max(v_angles))
    
    return {
        'method': 'anisotropy_angle_modulation',
        'rationale': 'Tilt angle tracks velocity direction; modulation proves dual motion',
        'angle_time_series': angle_analysis,
        'statistics': {
            'n_windows': len(angle_analysis),
            'predicted_vs_observed_r': float(r_angle),
            'predicted_vs_observed_p': float(p_angle),
            'velocity_angle_vs_observed_r': float(r_vangle_obs),
            'velocity_angle_vs_observed_p': float(p_vangle_obs),
            'velocity_angle_mean_deg': float(angle_mean),
            'velocity_angle_std_deg': float(angle_std),
            'velocity_angle_range_deg': (float(angle_range[0]), float(angle_range[1]))
        },
        'calibration': {
            'time_averaged_ew_ns_ratio': float(time_avg_ew_ns),
            'used_as_anisotropy_strength': True
        }
    }


def compare_background_models(temporal_data, best_fit_ra, best_fit_dec):
    """
    Test multiple background velocity hypotheses systematically.
    
    Compares:
    1. Solar Apex (galactic motion hypothesis)
    2. Best-fit from grid search
    3. CMB Dipole (cosmic rest frame hypothesis)
    4. Null (no background, orbital only)
    
    Args:
        temporal_data: List of temporal windows from Step 2.2
        best_fit_ra: Best-fit RA from grid search
        best_fit_dec: Best-fit Dec from grid search
    
    Returns:
        Dict with correlation results for each background model
    """
    print_status("Comparing background models...", "INFO")
    
    # Extract observed data
    days = np.array([w['day_of_year'] for w in temporal_data])
    speeds = np.array([w['orbital_speed_kms'] for w in temporal_data])
    obs_ratios = np.array([w['ew_ns_ratio'] for w in temporal_data])
    
    # CMB Dipole direction (Planck 2018)
    CMB_RA = 167.94
    CMB_DEC = -6.94
    
    # Backgrounds to test
    backgrounds = [
        {'name': 'Solar Apex', 'ra': SOLAR_APEX_RA_DEG, 'dec': SOLAR_APEX_DEC_DEG},
        {'name': 'Best Fit (Grid Search)', 'ra': best_fit_ra, 'dec': best_fit_dec},
        {'name': 'CMB Dipole', 'ra': CMB_RA, 'dec': CMB_DEC},
        {'name': 'Null (Orbital Only)', 'ra': 0, 'dec': 90}  # Point to pole (minimal effect)
    ]
    
    results = {}
    
    for bg in backgrounds:
        # Calculate velocities with this background
        predictions = []
        
        for day, speed in zip(days, speeds):
            v_net = calculate_3d_velocity_vectors(
                day, speed,
                background_ra=bg['ra'],
                background_dec=bg['dec'],
                background_speed=SOLAR_APEX_SPEED_KMS
            )
            
            # Predictor: cos(declination) - same as grid search
            v_dec_rad = np.radians(v_net['v_net_dec_deg'])
            pred = np.cos(v_dec_rad)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Correlation with observations
        if np.std(predictions) > 0:
            r, p = stats.pearsonr(predictions, obs_ratios)
        else:
            r, p = 0.0, 1.0
        
        results[bg['name']] = {
            'ra': float(bg['ra']),
            'dec': float(bg['dec']),
            'r': float(r),
            'p': float(p),
            'r_squared': float(r**2),
            'angular_sep_from_best': float(np.sqrt((bg['ra'] - best_fit_ra)**2 + 
                                                   (bg['dec'] - best_fit_dec)**2))
        }
        
        print_status(f"  {bg['name']:25s}: r = {r:7.4f}, p = {p:.6f}", 
                    "SUCCESS" if abs(r) > 0.5 else "INFO")
    
    return results


def statistical_tests(extracted_data, geom_validation):
    """
    Comprehensive statistical hypothesis testing.
    """
    print_status("Running statistical tests...", "INFO")
    
    sectors = extracted_data['sectors']
    
    # Test 1: W > E asymmetry
    w_lambda = sectors['W']['lambda_km']
    e_lambda = sectors['E']['lambda_km']
    w_e_ratio = w_lambda / e_lambda
    
    # Z-test on log ratio
    w_err = sectors['W']['lambda_error_km']
    e_err = sectors['E']['lambda_error_km']
    log_ratio = np.log(w_e_ratio)
    se_log = np.sqrt((w_err/w_lambda)**2 + (e_err/e_lambda)**2)
    z_we = log_ratio / se_log
    p_we = 1 - stats.norm.cdf(z_we)  # One-tailed
    
    # Test 2: Orbital correlation (already computed in step 2.2)
    r_orbital = extracted_data['orbital_correlation']['r']
    p_orbital = extracted_data['orbital_correlation']['p']
    
    # Test 3: Geometric angle validation (NEW)
    r_angle = geom_validation['statistics']['predicted_vs_observed_r']
    p_angle = geom_validation['statistics']['predicted_vs_observed_p']
    
    # Test 4: Velocity angle modulation
    r_vangle = geom_validation['statistics']['velocity_angle_vs_observed_r']
    p_vangle = geom_validation['statistics']['velocity_angle_vs_observed_p']
    angle_range = geom_validation['statistics']['velocity_angle_range_deg']
    
    tests = {
        'test_1_w_e_asymmetry': {
            'description': 'West > East (galactic apex signature)',
            'ratio': float(w_e_ratio),
            'z_score': float(z_we),
            'p_value': float(p_we),
            'significant': p_we < 0.05
        },
        'test_2_orbital_correlation': {
            'description': 'Anisotropy vs orbital velocity',
            'r': float(r_orbital),
            'p_value': float(p_orbital),
            'significant': p_orbital < 0.05
        },
        'test_3_angle_prediction': {
            'description': 'Predicted EW/NS from velocity angle vs observed',
            'r': float(r_angle),
            'p_value': float(p_angle),
            'significant': p_angle < 0.05
        },
        'test_4_angle_modulation': {
            'description': 'Velocity angle modulation through year',
            'r_with_observed': float(r_vangle),
            'p_value': float(p_vangle),
            'angle_range_deg': angle_range,
            'significant': p_vangle < 0.05
        }
    }
    
    n_sig = sum(1 for t in tests.values() if t['significant'])
    print_status(f"  {n_sig}/4 tests significant", "SUCCESS" if n_sig >= 3 else "INFO")
    
    return tests


def set_publication_style():
    """
    Configure matplotlib for consistent publication-quality figures.
    
    Uses Nature journal standards:
    - Font family: Helvetica/Arial (sans-serif)
    - Base size: 10pt
    - Axis labels: 11pt
    - Tick labels: 9pt
    - Legend: 9pt
    - Line widths: 0.8pt
    
    Call this function at the start of any visualization function to ensure consistency.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,  # High-resolution output
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def visualize_vector_search(search_results, output_path):
    """
    Create heatmap visualization of the galactic vector search results.
    
    Args:
        search_results: Dict containing 'grid_results', 'best_ra', 'best_dec'
        output_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    # Set consistent publication style
    set_publication_style()
    
    # Extract grid data
    grid_data = search_results['grid_results']
    best_ra = search_results['best_ra']
    best_dec = search_results['best_dec']
    best_r = search_results['best_correlation']
    
    # Create meshgrid
    ra_vals = sorted(set(x['ra'] for x in grid_data))
    dec_vals = sorted(set(x['dec'] for x in grid_data))
    
    # Initialize correlation matrix
    corr_matrix = np.full((len(dec_vals), len(ra_vals)), np.nan)
    
    # Fill matrix
    for point in grid_data:
        i = dec_vals.index(point['dec'])
        j = ra_vals.index(point['ra'])
        corr_matrix[i, j] = point['r']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    # Plot heatmap with viridis (Nature standard)
    im = ax.imshow(corr_matrix, aspect='auto', origin='lower',
                   extent=[0, 360, -90, 90], cmap='viridis', 
                   vmin=-0.3, vmax=0.75, interpolation='bilinear')
    
    # === MINIMAL HIGH-RES MARKERS ===
    
    # Best Fit - White circle
    ax.plot(best_ra, best_dec, 'o', 
            markersize=9, markerfacecolor='white', 
            markeredgecolor='black', markeredgewidth=1.5,
            clip_on=False, zorder=10)
    
    # CMB Dipole - Cyan circle
    ax.plot(167.94, -6.94, 'o', 
            markersize=8, markerfacecolor='#00FFFF',
            markeredgecolor='black', markeredgewidth=1.5,
            clip_on=False, zorder=10)
    
    # Solar Apex - Orange circle
    ax.plot(271.96, 30.0, 'o', 
            markersize=8, markerfacecolor='#F39C12',
            markeredgecolor='black', markeredgewidth=1.5,
            clip_on=False, zorder=10)
    
    # Add letter labels close to markers
    ax.text(best_ra + 3, best_dec + 3, 'a', 
            fontsize=10, color='white', fontweight='bold',
            ha='left', va='bottom')
    ax.text(167.94 + 3, -6.94 + 3, 'b', 
            fontsize=10, color='white', fontweight='bold',
            ha='left', va='bottom')
    ax.text(271.96 + 3, 30.0 + 3, 'c', 
            fontsize=10, color='white', fontweight='bold',
            ha='left', va='bottom')
    
    # === MINIMAL FORMATTING ===
    
    ax.set_xlabel('Right ascension (°)', fontsize=11)
    ax.set_ylabel('Declination (°)', fontsize=11)
    
    # Subtle grid
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, color='white')
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([-90, -45, 0, 45, 90])
    
    # Clean colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, aspect=30)
    cbar.set_label('r', fontsize=11, rotation=0, labelpad=12)
    cbar.ax.tick_params(labelsize=9, width=0.8, length=3)
    cbar.outline.set_linewidth(0.8)
    
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)
    
    print_status(f"Saved ultra-minimal heatmap: {output_path}", "SUCCESS")
    print_status("Figure caption: (a) Best fit, (b) CMB dipole, (c) Solar apex", "INFO")


def visualize_time_series_prediction(temporal_data, earth_vectors, model_comparison, output_path):
    """
    Create comprehensive time-series visualization showing:
    1. Observed vs Predicted modulation (CMB and Solar Apex)
    2. Residuals
    3. Phase diagram
    
    Args:
        temporal_data: List of temporal window dicts
        earth_vectors: Pre-calculated Earth orbital vectors
        model_comparison: Results from compare_models()
        output_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Extract data
    days = np.array([x['day_of_year'] for x in temporal_data])
    obs_ratios = np.array([x['ew_ns_ratio'] for x in temporal_data])
    
    # CMB predictions
    CMB_RA = model_comparison['cmb_dipole']['ra']
    CMB_DEC = model_comparison['cmb_dipole']['dec']
    ra_rad = np.radians(CMB_RA)
    dec_rad = np.radians(CMB_DEC)
    vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
    vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
    vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
    
    vn_x = earth_vectors[:, 0] + vg_x
    vn_y = earth_vectors[:, 1] + vg_y
    vn_z = earth_vectors[:, 2] + vg_z
    vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
    vn_dec_rad = np.arcsin(vn_z / vn_mag)
    cmb_preds_raw = np.cos(vn_dec_rad)
    
    # Scale predictions to match observed mean and std
    cmb_preds = (cmb_preds_raw - np.mean(cmb_preds_raw)) / np.std(cmb_preds_raw) * np.std(obs_ratios) + np.mean(obs_ratios)
    
    # Solar Apex predictions
    ra_rad = np.radians(SOLAR_APEX_RA_DEG)
    dec_rad = np.radians(SOLAR_APEX_DEC_DEG)
    vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
    vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
    vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
    
    vn_x = earth_vectors[:, 0] + vg_x
    vn_y = earth_vectors[:, 1] + vg_y
    vn_z = earth_vectors[:, 2] + vg_z
    vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
    vn_dec_rad = np.arcsin(vn_z / vn_mag)
    apex_preds_raw = np.cos(vn_dec_rad)
    apex_preds = (apex_preds_raw - np.mean(apex_preds_raw)) / np.std(apex_preds_raw) * np.std(obs_ratios) + np.mean(obs_ratios)
    
    # Set consistent publication style
    set_publication_style()
    
    # Create figure with cleaner layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28)
    
    # Panel 1: Time series (wider, cleaner)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(days, obs_ratios, c='#2C3E50', s=60, alpha=0.7, 
                edgecolor='white', linewidth=0.5, label='Observed', zorder=3)
    ax1.plot(days, cmb_preds, '-', color='#3498DB', linewidth=2.5, 
             label=f'CMB model (R²={model_comparison["cmb_dipole"]["r_squared"]:.3f})', zorder=2)
    ax1.plot(days, apex_preds, '--', color='#E74C3C', linewidth=2.2, alpha=0.8,
             label=f'Solar apex model (R²={model_comparison["solar_apex"]["r_squared"]:.3f})', zorder=1)
    ax1.set_xlabel('Day of year', fontsize=11)
    ax1.set_ylabel('EW/NS correlation ratio', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel 2: Residuals (CMB)
    ax2 = fig.add_subplot(gs[1, 0])
    residuals_cmb = obs_ratios - cmb_preds
    ax2.scatter(days, residuals_cmb, c='#3498DB', s=45, alpha=0.6, edgecolor='white', linewidth=0.3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.2)
    ax2.fill_between([days.min(), days.max()], 
                     -np.std(residuals_cmb), np.std(residuals_cmb),
                     color='gray', alpha=0.15, label='±1σ')
    ax2.set_xlabel('Day of year', fontsize=10)
    ax2.set_ylabel('Residual', fontsize=10)
    ax2.set_title(f'CMB residuals (σ={np.std(residuals_cmb):.3f})', fontsize=11)
    ax2.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Panel 3: Residuals (Solar Apex)
    ax3 = fig.add_subplot(gs[1, 1])
    residuals_apex = obs_ratios - apex_preds
    ax3.scatter(days, residuals_apex, c='#E74C3C', s=45, alpha=0.6, edgecolor='white', linewidth=0.3)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1.2)
    ax3.fill_between([days.min(), days.max()], 
                     -np.std(residuals_apex), np.std(residuals_apex),
                     color='gray', alpha=0.15, label='±1σ')
    ax3.set_xlabel('Day of year', fontsize=10)
    ax3.set_ylabel('Residual', fontsize=10)
    ax3.set_title(f'Solar apex residuals (σ={np.std(residuals_apex):.3f})', fontsize=11)
    ax3.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Panel 4: Predicted vs Observed (CMB)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(cmb_preds, obs_ratios, c='#3498DB', s=55, alpha=0.65, 
                edgecolor='white', linewidth=0.5)
    min_val = min(cmb_preds.min(), obs_ratios.min())
    max_val = max(cmb_preds.max(), obs_ratios.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.2, 
             alpha=0.6, label='1:1 line')
    ax4.set_xlabel('Predicted (CMB)', fontsize=10)
    ax4.set_ylabel('Observed', fontsize=10)
    ax4.set_title(f'CMB model (r={model_comparison["cmb_dipole"]["r"]:.3f})', fontsize=11)
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Panel 5: Predicted vs Observed (Solar Apex)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(apex_preds, obs_ratios, c='#E74C3C', s=55, alpha=0.65,
                edgecolor='white', linewidth=0.5)
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.2, 
             alpha=0.6, label='1:1 line')
    ax5.set_xlabel('Predicted (Solar apex)', fontsize=10)
    ax5.set_ylabel('Observed', fontsize=10)
    ax5.set_title(f'Solar apex model (r={model_comparison["solar_apex"]["r"]:.3f})', fontsize=11)
    ax5.legend(fontsize=9, loc='lower right')
    ax5.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)
    
    print_status(f"Saved enhanced model comparison: {output_path}", "SUCCESS")


def calculate_bootstrap_confidence(obs_ratios, preds, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for correlation coefficient.
    
    Args:
        obs_ratios: Observed EW/NS ratios
        preds: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Dict with correlation and confidence intervals
    """
    n = len(obs_ratios)
    bootstrap_rs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        obs_boot = obs_ratios[indices]
        pred_boot = preds[indices]
        
        # Calculate correlation
        r_boot = np.corrcoef(obs_boot, pred_boot)[0, 1]
        if not np.isnan(r_boot):
            bootstrap_rs.append(r_boot)
    
    # Calculate percentile intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_rs, lower_percentile)
    ci_upper = np.percentile(bootstrap_rs, upper_percentile)
    
    # Original correlation
    r_original = np.corrcoef(obs_ratios, preds)[0, 1]
    
    return {
        'r': r_original,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence,
        'n_bootstrap': n_bootstrap
    }


def permutation_test_alignment(obs_ratios, earth_vectors, best_ra, best_dec, n_permutations=10000):
    """
    Permutation test: Is the CMB alignment better than random directions?
    
    Tests the null hypothesis that the best-fit direction is no better than
    a random direction in the sky.
    
    Args:
        obs_ratios: Observed EW/NS ratios
        earth_vectors: Pre-calculated Earth orbital vectors
        best_ra: Best-fit RA (degrees)
        best_dec: Best-fit Dec (degrees)
        n_permutations: Number of random directions to test
    
    Returns:
        Dict with p-value and distribution statistics
    """
    print_status(f"  Running permutation test ({n_permutations} samples)...", "INFO")
    
    # Calculate correlation for best fit
    ra_rad = np.radians(best_ra)
    dec_rad = np.radians(best_dec)
    vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
    vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
    vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
    
    vn_x = earth_vectors[:, 0] + vg_x
    vn_y = earth_vectors[:, 1] + vg_y
    vn_z = earth_vectors[:, 2] + vg_z
    vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
    vn_dec_rad = np.arcsin(vn_z / vn_mag)
    best_preds = np.cos(vn_dec_rad)
    
    r_best = np.corrcoef(best_preds, obs_ratios)[0, 1]
    
    # Generate random directions and test
    random_rs = []
    
    for _ in range(n_permutations):
        # Random point on sphere
        rand_ra = np.random.uniform(0, 360)
        rand_dec = np.degrees(np.arcsin(np.random.uniform(-1, 1)))
        
        ra_rad = np.radians(rand_ra)
        dec_rad = np.radians(rand_dec)
        vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
        vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
        vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
        
        vn_x = earth_vectors[:, 0] + vg_x
        vn_y = earth_vectors[:, 1] + vg_y
        vn_z = earth_vectors[:, 2] + vg_z
        vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
        vn_dec_rad = np.arcsin(vn_z / vn_mag)
        rand_preds = np.cos(vn_dec_rad)
        
        r_rand = np.corrcoef(rand_preds, obs_ratios)[0, 1]
        random_rs.append(abs(r_rand))  # Use absolute value for two-tailed test
    
    # Calculate p-value (proportion of random r's >= observed r)
    p_value = np.mean(np.array(random_rs) >= abs(r_best))
    
    return {
        'r_observed': float(r_best),
        'p_value': float(p_value),
        'n_permutations': n_permutations,
        'random_r_mean': float(np.mean(random_rs)),
        'random_r_std': float(np.std(random_rs)),
        'random_r_95th': float(np.percentile(random_rs, 95))
    }


def compare_models(obs_ratios, earth_vectors, days):
    """
    Head-to-head comparison: CMB Dipole vs Solar Apex vs Ecliptic Controls vs Null models.
    
    Tests whether CMB alignment is genuine or just detecting ecliptic-plane preference.
    Includes ecliptic-plane control directions to discriminate:
    - CMB Dipole (RA=168°, Dec=-7°): Near ecliptic, cosmic rest frame
    - Ecliptic East (RA=90°, Dec=0°): In ecliptic, perpendicular to CMB
    - Ecliptic West (RA=270°, Dec=0°): In ecliptic, opposite to CMB
    - Solar Apex (RA=272°, Dec=+30°): Above ecliptic, galactic motion
    
    Returns detailed model comparison metrics including R², AIC, and BIC.
    """
    print_status("Comparing competing models (including ecliptic controls)...", "INFO")
    
    results = {}
    n = len(obs_ratios)
    
    # Model 1: CMB Dipole (RA=168°, Dec=-7°)
    CMB_RA = 167.94
    CMB_DEC = -6.94
    ra_rad = np.radians(CMB_RA)
    dec_rad = np.radians(CMB_DEC)
    vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
    vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
    vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
    
    vn_x = earth_vectors[:, 0] + vg_x
    vn_y = earth_vectors[:, 1] + vg_y
    vn_z = earth_vectors[:, 2] + vg_z
    vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
    vn_dec_rad = np.arcsin(vn_z / vn_mag)
    cmb_preds = np.cos(vn_dec_rad)
    
    r_cmb = np.corrcoef(cmb_preds, obs_ratios)[0, 1]
    r2_cmb = r_cmb**2
    ss_res_cmb = np.sum((obs_ratios - cmb_preds)**2)
    ss_tot = np.sum((obs_ratios - np.mean(obs_ratios))**2)
    
    results['cmb_dipole'] = {
        'name': 'CMB Dipole Motion',
        'ra': CMB_RA,
        'dec': CMB_DEC,
        'r': float(r_cmb),
        'r_squared': float(r2_cmb),
        'rmse': float(np.sqrt(np.mean((obs_ratios - cmb_preds)**2))),
        'variance_explained': float(r2_cmb * 100)
    }
    
    # Model 2: Solar Apex (RA=272°, Dec=30°)
    ra_rad = np.radians(SOLAR_APEX_RA_DEG)
    dec_rad = np.radians(SOLAR_APEX_DEC_DEG)
    vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
    vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
    vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
    
    vn_x = earth_vectors[:, 0] + vg_x
    vn_y = earth_vectors[:, 1] + vg_y
    vn_z = earth_vectors[:, 2] + vg_z
    vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
    vn_dec_rad = np.arcsin(vn_z / vn_mag)
    apex_preds = np.cos(vn_dec_rad)
    
    r_apex = np.corrcoef(apex_preds, obs_ratios)[0, 1]
    r2_apex = r_apex**2
    
    results['solar_apex'] = {
        'name': 'Solar Apex Motion',
        'ra': SOLAR_APEX_RA_DEG,
        'dec': SOLAR_APEX_DEC_DEG,
        'r': float(r_apex),
        'r_squared': float(r2_apex),
        'rmse': float(np.sqrt(np.mean((obs_ratios - apex_preds)**2))),
        'variance_explained': float(r2_apex * 100)
    }
    
    # Model 3: Ecliptic East Control (RA=90°, Dec=0°)
    # Tests if any ecliptic-plane direction works, or specifically CMB
    ECLIPTIC_EAST_RA = 90.0
    ECLIPTIC_EAST_DEC = 0.0
    ra_rad = np.radians(ECLIPTIC_EAST_RA)
    dec_rad = np.radians(ECLIPTIC_EAST_DEC)
    vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
    vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
    vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
    
    vn_x = earth_vectors[:, 0] + vg_x
    vn_y = earth_vectors[:, 1] + vg_y
    vn_z = earth_vectors[:, 2] + vg_z
    vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
    vn_dec_rad = np.arcsin(vn_z / vn_mag)
    ecliptic_east_preds = np.cos(vn_dec_rad)
    
    r_ecliptic_east = np.corrcoef(ecliptic_east_preds, obs_ratios)[0, 1]
    r2_ecliptic_east = r_ecliptic_east**2
    
    results['ecliptic_east'] = {
        'name': 'Ecliptic East Control',
        'ra': ECLIPTIC_EAST_RA,
        'dec': ECLIPTIC_EAST_DEC,
        'r': float(r_ecliptic_east),
        'r_squared': float(r2_ecliptic_east),
        'rmse': float(np.sqrt(np.mean((obs_ratios - ecliptic_east_preds)**2))),
        'variance_explained': float(r2_ecliptic_east * 100)
    }
    
    # Model 4: Ecliptic West Control (RA=270°, Dec=0°)
    # Near Solar Apex RA but at ecliptic plane
    ECLIPTIC_WEST_RA = 270.0
    ECLIPTIC_WEST_DEC = 0.0
    ra_rad = np.radians(ECLIPTIC_WEST_RA)
    dec_rad = np.radians(ECLIPTIC_WEST_DEC)
    vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
    vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
    vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
    
    vn_x = earth_vectors[:, 0] + vg_x
    vn_y = earth_vectors[:, 1] + vg_y
    vn_z = earth_vectors[:, 2] + vg_z
    vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
    vn_dec_rad = np.arcsin(vn_z / vn_mag)
    ecliptic_west_preds = np.cos(vn_dec_rad)
    
    r_ecliptic_west = np.corrcoef(ecliptic_west_preds, obs_ratios)[0, 1]
    r2_ecliptic_west = r_ecliptic_west**2
    
    results['ecliptic_west'] = {
        'name': 'Ecliptic West Control',
        'ra': ECLIPTIC_WEST_RA,
        'dec': ECLIPTIC_WEST_DEC,
        'r': float(r_ecliptic_west),
        'r_squared': float(r2_ecliptic_west),
        'rmse': float(np.sqrt(np.mean((obs_ratios - ecliptic_west_preds)**2))),
        'variance_explained': float(r2_ecliptic_west * 100)
    }
    
    # Model 5: Null (mean)
    null_preds = np.full_like(obs_ratios, np.mean(obs_ratios))
    results['null'] = {
        'name': 'Null (No Modulation)',
        'r': 0.0,
        'r_squared': 0.0,
        'rmse': float(np.sqrt(np.mean((obs_ratios - null_preds)**2))),
        'variance_explained': 0.0
    }
    
    # Calculate improvement over null
    improvement_cmb = ((results['null']['rmse'] - results['cmb_dipole']['rmse']) / 
                       results['null']['rmse'] * 100)
    improvement_apex = ((results['null']['rmse'] - results['solar_apex']['rmse']) / 
                        results['null']['rmse'] * 100)
    
    results['cmb_dipole']['improvement_over_null_pct'] = float(improvement_cmb)
    results['solar_apex']['improvement_over_null_pct'] = float(improvement_apex)
    
    # Winner
    if results['cmb_dipole']['r_squared'] > results['solar_apex']['r_squared']:
        results['winner'] = 'cmb_dipole'
        results['winner_advantage'] = float(
            (results['cmb_dipole']['r_squared'] - results['solar_apex']['r_squared']) * 100
        )
    else:
        results['winner'] = 'solar_apex'
        results['winner_advantage'] = float(
            (results['solar_apex']['r_squared'] - results['cmb_dipole']['r_squared']) * 100
        )
    
    print_status(f"  CMB Dipole (RA=168°, Dec=-7°): R²={r2_cmb:.4f} ({r2_cmb*100:.1f}% variance explained)", 
                 "SUCCESS")
    print_status(f"  Ecliptic East (RA=90°, Dec=0°): R²={r2_ecliptic_east:.4f} ({r2_ecliptic_east*100:.1f}% variance explained)", 
                 "INFO")
    print_status(f"  Ecliptic West (RA=270°, Dec=0°): R²={r2_ecliptic_west:.4f} ({r2_ecliptic_west*100:.1f}% variance explained)", 
                 "INFO")
    print_status(f"  Solar Apex (RA=272°, Dec=+30°): R²={r2_apex:.4f} ({r2_apex*100:.1f}% variance explained)", 
                 "INFO")
    
    # Discrimination test: CMB vs Ecliptic Controls
    cmb_vs_east_ratio = r2_cmb / r2_ecliptic_east if r2_ecliptic_east > 0 else float('inf')
    cmb_vs_west_ratio = r2_cmb / r2_ecliptic_west if r2_ecliptic_west > 0 else float('inf')
    
    print_status(f"\nEcliptic Control Discrimination:", "INFO")
    print_status(f"  CMB / Ecliptic East variance ratio: {cmb_vs_east_ratio:.1f}×", "INFO")
    print_status(f"  CMB / Ecliptic West variance ratio: {cmb_vs_west_ratio:.1f}×", "INFO")
    
    results['ecliptic_discrimination'] = {
        'cmb_vs_east_ratio': float(cmb_vs_east_ratio),
        'cmb_vs_west_ratio': float(cmb_vs_west_ratio),
        'interpretation': 'CMB-specific' if min(cmb_vs_east_ratio, cmb_vs_west_ratio) > 2.0 else 'Generic ecliptic'
    }
    
    return results


def perform_galactic_vector_search(temporal_data):
    """
    Grid search for the background velocity vector that best explains
    the observed temporal modulation of the anisotropy.
    
    Compares observed EW/NS modulation against predictions from different
    background velocity directions to identify the best-fit reference frame.
    """
    print_status("Performing 'Clever' Galactic Vector Search...", "INFO")
    
    # 1. Prepare data
    days = np.array([x['day_of_year'] for x in temporal_data])
    speeds = np.array([x['orbital_speed_kms'] for x in temporal_data])
    obs_ratios = np.array([x['ew_ns_ratio'] for x in temporal_data])
    
    # 2. Pre-calculate Earth Orbital Vectors (J2000 Equatorial)
    # This avoids re-calculating orbit geometry 648 times
    epsilon = np.radians(ECLIPTIC_TILT_DEG)
    earth_vectors = []
    
    for d, s in zip(days, speeds):
        # Orbit angle (0 at Vernal Equinox)
        orbit_progress = (d - 80) / 365.25 * 2 * np.pi
        
        # Velocity in Ecliptic (tangent)
        v_ecl_x = -s * np.sin(orbit_progress)
        v_ecl_y = s * np.cos(orbit_progress)
        
        # Rotate to Equatorial
        v_orb_x = v_ecl_x
        v_orb_y = v_ecl_y * np.cos(epsilon)
        v_orb_z = v_ecl_y * np.sin(epsilon)
        
        earth_vectors.append([v_orb_x, v_orb_y, v_orb_z])
        
    earth_vectors = np.array(earth_vectors) # Shape (N, 3)
    
    # 3. Grid Search
    # Resolution: configurable (default 5 degrees for publication)
    import __main__
    resolution = getattr(__main__, 'GRID_RESOLUTION', 5.0)
    ra_grid = np.arange(0, 360 + resolution/2, resolution)
    dec_grid = np.arange(-90, 90 + resolution/2, resolution)
    
    print(f"Grid search resolution: {resolution}°")
    print(f"  RA steps: {len(ra_grid)} (0° to 360° in {resolution}° increments)")
    print(f"  Dec steps: {len(dec_grid)} (-90° to +90° in {resolution}° increments)")
    print(f"  Total directions: {len(ra_grid) * len(dec_grid)}")
    
    best_corr = -1.0
    best_ra = 0
    best_dec = 0
    
    # Store results for heatmap
    results_grid = []
    
    for ra in ra_grid:
        for dec in dec_grid:
            # Galactic Vector
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            
            # Assume 20 km/s
            gal_speed = SOLAR_APEX_SPEED_KMS
            
            # V_gal components
            vg_x = gal_speed * np.cos(dec_rad) * np.cos(ra_rad)
            vg_y = gal_speed * np.cos(dec_rad) * np.sin(ra_rad)
            vg_z = gal_speed * np.sin(dec_rad)
            
            # Calculate Net Vectors for all windows
            # V_net = V_orb + V_gal
            vn_x = earth_vectors[:, 0] + vg_x
            vn_y = earth_vectors[:, 1] + vg_y
            vn_z = earth_vectors[:, 2] + vg_z
            
            # Calculate Declination of Net Vector
            vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
            vn_dec_rad = np.arcsin(vn_z / vn_mag)
            
            # Predictor: cos(Dec) -> favoring Equatorial flow -> High EW/NS
            # Logic: Low Dec -> High cos(Dec) -> High EW projection -> High EW/NS Ratio
            preds = np.cos(vn_dec_rad)
            
            # Correlation
            if np.std(preds) == 0:
                r = 0
            else:
                # Use numpy corrcoef for speed
                r = np.corrcoef(preds, obs_ratios)[0, 1]
            
            # Fix numpy float/int types for JSON
            r = float(r)
            
            if not np.isnan(r):
                results_grid.append({'ra': int(ra), 'dec': int(dec), 'r': r})
                
                if r > best_corr:
                    best_corr = r
                    best_ra = ra
                    best_dec = dec
                
    print_status(f"  Best Fit: RA={best_ra}°, Dec={best_dec}° (r={best_corr:.4f})", "SUCCESS")
    print_status(f"  Known Apex: RA={int(SOLAR_APEX_RA_DEG)}°, Dec={int(SOLAR_APEX_DEC_DEG)}°", "INFO")
    
    # Calculate bootstrap confidence intervals for best fit
    # Reconstruct best-fit predictions
    ra_rad = np.radians(best_ra)
    dec_rad = np.radians(best_dec)
    vg_x = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.cos(ra_rad)
    vg_y = SOLAR_APEX_SPEED_KMS * np.cos(dec_rad) * np.sin(ra_rad)
    vg_z = SOLAR_APEX_SPEED_KMS * np.sin(dec_rad)
    
    vn_x = earth_vectors[:, 0] + vg_x
    vn_y = earth_vectors[:, 1] + vg_y
    vn_z = earth_vectors[:, 2] + vg_z
    vn_mag = np.sqrt(vn_x**2 + vn_y**2 + vn_z**2)
    vn_dec_rad = np.arcsin(vn_z / vn_mag)
    best_preds = np.cos(vn_dec_rad)
    
    print_status("  Calculating bootstrap confidence intervals...", "INFO")
    bootstrap_results = calculate_bootstrap_confidence(obs_ratios, best_preds, n_bootstrap=1000)
    
    print_status(f"  95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]", "INFO")
    
    return {
        'best_ra_deg': int(best_ra),
        'best_dec_deg': int(best_dec),
        'best_ra': int(best_ra),  # Keep for backward compatibility
        'best_dec': int(best_dec),  # Keep for backward compatibility
        'best_correlation': float(best_corr),
        'resolution_deg': float(resolution),
        'total_directions': len(ra_grid) * len(dec_grid),
        'bootstrap_ci': bootstrap_results,
        'grid_results': results_grid
    }


def analyze_dual_motion(data):
    """
    Main analysis orchestrator - leverages Step 2.2, adds geometric validation.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'dual_motion_geometric_validation',
        'step_2_2_file': 'step_2_2_geospatial_temporal_analysis_code.json',
        'physical_constants': {
            'solar_apex_ra_deg': SOLAR_APEX_RA_DEG,
            'solar_apex_dec_deg': SOLAR_APEX_DEC_DEG,
            'solar_apex_speed_kms': SOLAR_APEX_SPEED_KMS,
            'ecliptic_tilt_deg': ECLIPTIC_TILT_DEG
        }
    }
    
    # Extract existing Step 2.2 results (NO recomputation)
    extracted = extract_step_2_2_data(data)
    results['extracted_from_step_2_2'] = {
        'n_sectors': len(extracted['sectors']),
        'n_temporal_windows': len(extracted['temporal']['global']),
        'orbital_correlation_r': extracted['orbital_correlation']['r'],
        'orbital_correlation_p': extracted['orbital_correlation']['p']
    }
    
    # NEW: Perform Galactic Vector Search (The "Clever" Geometry Test)
    temporal_data = extracted['temporal']['global']
    vector_search = perform_galactic_vector_search(temporal_data)
    results['galactic_vector_search'] = vector_search
    
    # NEW: Permutation test - is CMB alignment statistically significant?
    days = np.array([x['day_of_year'] for x in temporal_data])
    speeds = np.array([x['orbital_speed_kms'] for x in temporal_data])
    obs_ratios = np.array([x['ew_ns_ratio'] for x in temporal_data])
    
    # Pre-calculate Earth vectors for efficiency
    epsilon = np.radians(ECLIPTIC_TILT_DEG)
    earth_vectors = []
    for d, s in zip(days, speeds):
        orbit_progress = (d - 80) / 365.25 * 2 * np.pi
        v_ecl_x = -s * np.sin(orbit_progress)
        v_ecl_y = s * np.cos(orbit_progress)
        v_orb_x = v_ecl_x
        v_orb_y = v_ecl_y * np.cos(epsilon)
        v_orb_z = v_ecl_y * np.sin(epsilon)
        earth_vectors.append([v_orb_x, v_orb_y, v_orb_z])
    earth_vectors = np.array(earth_vectors)
    
    permutation_result = permutation_test_alignment(
        obs_ratios, 
        earth_vectors, 
        vector_search['best_ra'], 
        vector_search['best_dec'],
        n_permutations=10000
    )
    results['permutation_test'] = permutation_result
    
    # NEW: Model comparison (CMB vs Solar Apex vs Null)
    model_comp = compare_models(obs_ratios, earth_vectors, days)
    results['model_comparison'] = model_comp
    
    # NEW: Comprehensive background comparison (Issue 1 fix)
    background_comp = compare_background_models(
        temporal_data,
        vector_search['best_ra'],
        vector_search['best_dec']
    )
    results['background_comparison'] = background_comp
    
    # NEW: Calculate velocity vectors for all temporal windows FIRST
    # Now using FULL 3D calculation (Orbit + Galaxy)
    velocity_series = []
    for window in extracted['temporal']['global']:
        vvec = calculate_3d_velocity_vectors(
            window['day_of_year'],
            window['orbital_speed_kms']
        )
        vvec['ew_ns_ratio'] = window['ew_ns_ratio']
        velocity_series.append(vvec)
    
    # NEW: Geometric validation using 3D projection model
    geom_valid = geometric_validation(extracted, velocity_series)
    results['geometric_validation'] = geom_valid
    
    # NEW: Statistical tests
    tests = statistical_tests(extracted, geom_valid)
    results['statistical_tests'] = tests
    
    results['velocity_time_series'] = {
        'n_windows': len(velocity_series),
        'data': velocity_series
    }
    
    # NEW: Hemisphere analysis (using extracted data)
    north_speeds = np.array([x['orbital_speed_kms'] for x in extracted['temporal']['northern']])
    north_ratios = np.array([x['ew_ns_ratio'] for x in extracted['temporal']['northern']])
    south_speeds = np.array([x['orbital_speed_kms'] for x in extracted['temporal']['southern']])
    south_ratios = np.array([x['ew_ns_ratio'] for x in extracted['temporal']['southern']])
    
    r_n, p_n = stats.pearsonr(north_speeds, north_ratios)
    r_s, p_s = stats.pearsonr(south_speeds, south_ratios)
    
    results['hemisphere_analysis'] = {
        'northern': {'r': float(r_n), 'p': float(p_n), 'n': len(north_speeds)},
        'southern': {'r': float(r_s), 'p': float(p_s), 'n': len(south_speeds)},
        'interpretation': 'heliocentric' if np.sign(r_n) == np.sign(r_s) else 'unclear'
    }
    
    # Summary assessment based on NEW angle validation AND Vector Search
    n_sig_tests = sum(1 for t in tests.values() if t.get('significant', False))
    r_angle = geom_valid['statistics']['predicted_vs_observed_r']
    
    search_corr = vector_search['best_correlation']
    
    # CHECK FOR CMB ALIGNMENT (RA ~168, Dec ~-7)
    best_ra = vector_search['best_ra']
    best_dec = vector_search['best_dec']
    
    ra_diff = abs(best_ra - 168)
    if ra_diff > 180: ra_diff = 360 - ra_diff
    dec_diff = abs(best_dec - (-7))
    cmb_aligned = (ra_diff < 30) and (dec_diff < 30) and (search_corr > 0.5)

    # Assessment criteria (conservative scientific language)
    if cmb_aligned:
        assessment = "Strong Evidence: CMB Frame Alignment"
    elif n_sig_tests >= 3 and (r_angle > 0.3 or search_corr > 0.5):
        assessment = "Strong Confirmation"
    elif n_sig_tests >= 2:
        assessment = "Moderate Support"
    elif n_sig_tests >= 1:
        assessment = "Weak to Moderate Support"
    else:
        assessment = "Insufficient Evidence"
    
    results['summary'] = {
        'overall_assessment': assessment,
        'n_significant_tests': n_sig_tests,
        'angle_prediction_correlation': r_angle,
        'vector_search_best_correlation': search_corr,
        'cmb_frame_aligned': cmb_aligned,
        'dual_motion_supported': (
            tests['test_2_orbital_correlation']['significant'] and
            tests['test_1_w_e_asymmetry']['significant'] and
            (tests['test_3_angle_prediction']['significant'] or search_corr > 0.5)
        ),
        'caveats': 'Results require independent replication and alternative hypotheses testing'
    }
    
    print_status(f"Assessment: {assessment}", "SUCCESS" if "STRONG" in assessment else "INFO")
    
    return results


def main():
    """Execute Step 2.5 analysis"""
    logger = TEPLogger(
        name="step_2_5_dual_motion_geometry",
        level="INFO",
        log_file_path=PACKAGE_ROOT / "logs" / "code_longspan" / "step_2_5_dual_motion_geometry.log"
    )
    set_step_logger(logger)
    
    print("=" * 70)
    print("STEP 2.5: DUAL-MOTION GEOMETRIC VALIDATION")
    print("=" * 70)
    print("\nLeveraging existing Step 2.2 computations...")
    print("Adding geometric validation layer...\n")
    
    # Load Step 2.2 results
    results_file = Path("results/outputs/code_longspan/step_2_2_geospatial_temporal_analysis_code.json")
    print_status(f"Loading: {results_file}", "INFO")
    
    with open(results_file) as f:
        data = json.load(f)
    
    # Run analysis
    results = analyze_dual_motion(data)
    
    # Check Vector Search Result
    search = results['galactic_vector_search']
    best_ra = search['best_ra']
    best_dec = search['best_dec']
    
    # Calculate angular distance to known Apex
    ra1 = np.radians(best_ra)
    dec1 = np.radians(best_dec)
    ra2 = np.radians(SOLAR_APEX_RA_DEG)
    dec2 = np.radians(SOLAR_APEX_DEC_DEG)
    
    dist = np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2))
    dist_deg = np.degrees(dist)
    
    # Calculate angular distance to CMB Dipole
    CMB_RA = 167.94
    CMB_DEC = -6.94
    ra3 = np.radians(CMB_RA)
    dec3 = np.radians(CMB_DEC)
    dist_cmb = np.arccos(np.sin(dec1)*np.sin(dec3) + np.cos(dec1)*np.cos(dec3)*np.cos(ra1-ra3))
    dist_cmb_deg = np.degrees(dist_cmb)
    
    print(f"\n{'=' * 70}")
    print(f"VECTOR SEARCH RESULT:")
    print(f"  Best Fit Vector: RA={best_ra}°, Dec={best_dec}°")
    print(f"  Known Solar Apex: RA={int(SOLAR_APEX_RA_DEG)}°, Dec={int(SOLAR_APEX_DEC_DEG)}°")
    print(f"    -> Separation from Apex: {dist_deg:.1f}°")
    print(f"  Known CMB Dipole: RA={int(CMB_RA)}°, Dec={int(CMB_DEC)}°")
    print(f"    -> Separation from CMB:  {dist_cmb_deg:.1f}°")
    print(f"  Correlation: r={search['best_correlation']:.4f}")
    if 'bootstrap_ci' in search:
        ci = search['bootstrap_ci']
        print(f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"{'=' * 70}\n")
    
    # Display permutation test results
    if 'permutation_test' in results:
        perm = results['permutation_test']
        print(f"\n{'=' * 70}")
        print(f"PERMUTATION TEST (10,000 random directions):")
        print(f"  Observed correlation: r={perm['r_observed']:.4f}")
        print(f"  Random mean: {perm['random_r_mean']:.4f}")
        print(f"  Random 95th percentile: {perm['random_r_95th']:.4f}")
        print(f"  P-value: {perm['p_value']:.6f}")
        if perm['p_value'] < 0.001:
            print(f"  -> HIGHLY SIGNIFICANT (p < 0.001)")
        elif perm['p_value'] < 0.01:
            print(f"  -> SIGNIFICANT (p < 0.01)")
        elif perm['p_value'] < 0.05:
            print(f"  -> SIGNIFICANT (p < 0.05)")
        else:
            print(f"  -> Not significant (p >= 0.05)")
        print(f"{'=' * 70}\n")
    
    # Display model comparison
    if 'model_comparison' in results:
        mc = results['model_comparison']
        print(f"\n{'=' * 70}")
        print(f"MODEL COMPARISON:")
        print(f"  CMB Dipole Model:")
        print(f"    R² = {mc['cmb_dipole']['r_squared']:.4f} ({mc['cmb_dipole']['variance_explained']:.1f}% variance explained)")
        print(f"    RMSE = {mc['cmb_dipole']['rmse']:.4f}")
        print(f"    Improvement over null: {mc['cmb_dipole']['improvement_over_null_pct']:.1f}%")
        print(f"  Solar Apex Model:")
        print(f"    R² = {mc['solar_apex']['r_squared']:.4f} ({mc['solar_apex']['variance_explained']:.1f}% variance explained)")
        print(f"    RMSE = {mc['solar_apex']['rmse']:.4f}")
        print(f"    Improvement over null: {mc['solar_apex']['improvement_over_null_pct']:.1f}%")
        print(f"  Winner: {mc['winner'].upper()} (advantage: {mc['winner_advantage']:.1f}%)")
        print(f"{'=' * 70}\n")
    
    # Display background comparison (Issue 1 fix results)
    if 'background_comparison' in results:
        bc = results['background_comparison']
        print(f"\n{'=' * 70}")
        print(f"BACKGROUND COMPARISON (Geometric Validation Fixed):")
        print(f"  Testing multiple reference frames with same predictor:")
        print()
        for name, data in bc.items():
            status = "✓ STRONG" if abs(data['r']) > 0.6 else "○ WEAK" if abs(data['r']) > 0.3 else "✗ FAILED"
            print(f"  {status}  {name:30s}")
            print(f"       RA={data['ra']:6.2f}°, Dec={data['dec']:6.2f}°")
            print(f"       r={data['r']:7.4f}, p={data['p']:.6f}, R²={data['r_squared']:.4f}")
            print(f"       Angular separation from best: {data['angular_sep_from_best']:.1f}°")
            print()
        print(f"{'=' * 70}\n")
    
    # Generate visualizations
    figures_dir = Path("results/figures/code_longspan")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    heatmap_path = figures_dir / "step_2_5_vector_search_heatmap.png"
    visualize_vector_search(search, heatmap_path)
    
    # Generate time-series comparison
    if 'model_comparison' in results:
        # Get temporal data and earth vectors for visualization
        with open(Path("results/outputs/code_longspan/step_2_2_geospatial_temporal_analysis_code.json")) as f:
            data = json.load(f)
        extracted = extract_step_2_2_data(data)
        temporal_data = extracted['temporal']['global']
        
        days = np.array([x['day_of_year'] for x in temporal_data])
        speeds = np.array([x['orbital_speed_kms'] for x in temporal_data])
        
        epsilon = np.radians(ECLIPTIC_TILT_DEG)
        earth_vectors = []
        for d, s in zip(days, speeds):
            orbit_progress = (d - 80) / 365.25 * 2 * np.pi
            v_ecl_x = -s * np.sin(orbit_progress)
            v_ecl_y = s * np.cos(orbit_progress)
            v_orb_x = v_ecl_x
            v_orb_y = v_ecl_y * np.cos(epsilon)
            v_orb_z = v_ecl_y * np.sin(epsilon)
            earth_vectors.append([v_orb_x, v_orb_y, v_orb_z])
        earth_vectors = np.array(earth_vectors)
        
        timeseries_path = figures_dir / "step_2_5_model_comparison_timeseries.png"
        visualize_time_series_prediction(temporal_data, earth_vectors, mc, timeseries_path)
    
    # Save results
    output_dir = Path("results/outputs/code_longspan")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "step_2_5_dual_motion_geometry.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved: {output_file}")
    print(f"Assessment: {results['summary']['overall_assessment']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Step 2.5: Dual-Motion Geometric Validation with configurable resolution'
    )
    parser.add_argument(
        '--resolution', 
        type=float, 
        default=5.0,
        help='Grid resolution in degrees (default: 5.0)'
    )
    parser.add_argument(
        '--verify-multi-resolution',
        action='store_true',
        help='Run multiple resolutions (10°, 5°, 2.5°, 1°) for verification'
    )
    args = parser.parse_args()
    
    if args.verify_multi_resolution:
        # Run multi-resolution verification
        print("\n" + "="*70)
        print("MULTI-RESOLUTION VERIFICATION MODE")
        print("Testing resolutions: 10°, 5°, 2.5°, 1°")
        print("="*70)
        print("\nGrid sizes:")
        print("  10° → ~700 directions")
        print("  5° → ~2,700 directions")
        print("  2.5° → ~10,500 directions")
        print("  1° → ~65,000 directions")
        print("\n⚠️  1° resolution may take 5-10 minutes depending on hardware")
        print("="*70 + "\n")
        
        import time
        total_start = time.time()
        
        resolutions = [10.0, 5.0, 2.5, 1.0]
        all_results = {}
        
        for res in resolutions:
            print(f"\n{'='*70}")
            print(f"Running analysis at {res}° resolution...")
            print(f"{'='*70}\n")
            
            # Temporarily store resolution as global for main() to use
            import __main__
            __main__.GRID_RESOLUTION = res
            
            results = main()
            gvs = results.get('galactic_vector_search', {})
            all_results[f"{res}_deg"] = {
                'resolution': res,
                'grid_points': gvs.get('total_directions', 0),
                'best_correlation': gvs.get('best_correlation', 0),
                'best_ra': gvs.get('best_ra_deg', gvs.get('best_ra', 0)),
                'best_dec': gvs.get('best_dec_deg', gvs.get('best_dec', 0)),
                'p_value': results.get('permutation_test', {}).get('p_value', 1.0),
                'cmb_separation': results.get('background_comparison', {}).get('CMB Dipole', {}).get('angular_sep_from_best', 999),
            }
        
        # Compare results
        print("\n" + "="*70)
        print("MULTI-RESOLUTION COMPARISON")
        print("="*70)
        print(f"{'Resolution':<12} {'Points':<10} {'r':<10} {'RA':<8} {'Dec':<8} {'p-value':<12} {'CMB Sep':<10}")
        print("-"*70)
        
        for res_key in ["10.0_deg", "5.0_deg", "2.5_deg", "1.0_deg"]:
            if res_key in all_results:
                r = all_results[res_key]
                print(f"{r['resolution']:>10.1f}°  {r['grid_points']:<10} {r['best_correlation']:<10.4f} "
                      f"{r['best_ra']:<8.1f} {r['best_dec']:<8.1f} {r['p_value']:<12.6f} {r['cmb_separation']:<10.1f}°")
        
        print("="*70)
        
        # Convergence assessment
        corrs = [all_results[f"{res}_deg"]['best_correlation'] for res in resolutions if f"{res}_deg" in all_results]
        if len(corrs) >= 2:
            change_10_to_5 = ((corrs[1] - corrs[0]) / corrs[0] * 100) if len(corrs) > 1 and corrs[0] != 0 else 0
            change_5_to_2_5 = ((corrs[2] - corrs[1]) / corrs[1] * 100) if len(corrs) > 2 and corrs[1] != 0 else 0
            change_2_5_to_1 = ((corrs[3] - corrs[2]) / corrs[2] * 100) if len(corrs) > 3 and corrs[2] != 0 else 0
            
            print(f"\nConvergence Analysis:")
            print(f"  10° → 5°:   {change_10_to_5:+.2f}% change in correlation")
            if len(corrs) > 2:
                print(f"  5° → 2.5°:  {change_5_to_2_5:+.2f}% change in correlation")
            if len(corrs) > 3:
                print(f"  2.5° → 1°:  {change_2_5_to_1:+.2f}% change in correlation")
                print(f"\n  Asymptotic behavior: Changes decreasing ({change_10_to_5:.1f}% → {change_5_to_2_5:.1f}% → {change_2_5_to_1:.1f}%)")
                print(f"  Final best-fit: RA={all_results['1.0_deg']['best_ra']}°, Dec={all_results['1.0_deg']['best_dec']}°")
                print(f"  Final correlation: r = {all_results['1.0_deg']['best_correlation']:.4f}")
        
        # Save comparison
        output_dir = Path("results/outputs/code_longspan")
        comparison_file = output_dir / "step_2_5_multi_resolution_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison saved: {comparison_file}")
        
        # Report total time
        total_elapsed = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"MULTI-RESOLUTION ANALYSIS COMPLETE")
        print(f"Total elapsed time: {total_elapsed/60:.1f} minutes")
        print(f"{'='*70}")
        
    else:
        # Single resolution run
        import __main__
        __main__.GRID_RESOLUTION = args.resolution
        print(f"\nRunning analysis with {args.resolution}° resolution\n")
        main()
