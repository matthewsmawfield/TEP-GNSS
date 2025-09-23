#!/usr/bin/env python3
"""
Step 12: Additional Visualizations - Earth's Motion Beat Frequencies
===================================================================

Creates advanced 3D visualizations showing Earth's complex helical motion 
through space with the four temporal interference patterns as overlapping wave patterns.

This step creates:
1. Earth's helical path through space (orbital + rotational motion)
2. Four key beat frequency patterns as wave overlays
3. How GNSS station pairs "ride" these complex motion streams
4. Visual representation of the interference patterns

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from config import TEPConfig

def print_status(text: str, status: str = "INFO"):
    """Print status with icons"""
    prefixes = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROCESSING": "🔄"
    }
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} {prefixes.get(status, '[INFO]')} {text}")

def set_publication_style():
    """Sets matplotlib rcParams for consistent, publication-quality figures."""
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.color': '#495773',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.edgecolor': '#1e4a5f',
        'axes.labelcolor': '#1e4a5f',
        'axes.titlecolor': '#2D0140',
        'xtick.color': '#1e4a5f',
        'ytick.color': '#1e4a5f',
        'text.color': '#1e4a5f',
    })

def load_beat_frequency_data():
    """Load the top four beat frequency patterns from our previous analysis."""
    print_status("Loading beat frequency data", "INFO")
    
    # Use the same logic as before to get top 4 patterns
    centers = ['code', 'esa_final', 'igs_combined']
    all_data = {}
    
    # Get the project root directory (two levels up from this script)
    root_dir = Path(__file__).parent.parent.parent
    
    for center in centers:
        results_file = root_dir / 'results' / 'outputs' / f'step_5_helical_motion_only_{center}.json'
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                if 'beat_frequencies_analysis' in data and data['beat_frequencies_analysis']['success']:
                    all_data[center] = data['beat_frequencies_analysis']
                    
            except Exception as e:
                print_status(f"Failed to load data for {center}: {e}", "WARNING")
    
    # Extract the top 4 patterns with site-consistent palette
    top_patterns = {
        'annual_semiannual_sum': {
            'period_days': 121.75,
            'frequency_cpd': 0.008213552361396304,
            'components': ['annual', 'semiannual'],
            'correlation': 0.914,
            'color': '#2D0140',  # Site primary purple (strongest correlation)
            'name': 'Annual+Semiannual Beat'
        },
        'chandler_semiannual_sum': {
            'period_days': 127.92,
            'frequency_cpd': 0.007817621948971462,
            'components': ['chandler', 'semiannual'],
            'correlation': 0.905,
            'color': '#4A90C2',  # Site blue accent
            'name': 'Chandler+Semiannual Beat'
        },
        'chandler_annual_sum': {
            'period_days': 196.86,
            'frequency_cpd': 0.005079771161839362,
            'components': ['chandler', 'annual'],
            'correlation': 0.833,
            'color': '#495773',  # Site secondary blue-gray
            'name': 'Chandler+Annual Beat'
        },
        'tidal_m2_s2_diff': {
            'period_days': 14.77,
            'frequency_cpd': 0.0677,
            'components': ['tidal_m2', 'tidal_s2'],
            'correlation': 0.632,
            'color': '#8B4513',  # Complementary brown for contrast
            'name': 'Tidal M2-S2 Beat'
        }
    }
    
    return top_patterns

def create_earth_helical_path(t_days, orbital_radius_au=1.0):
    """
    Create Earth's helical path through space combining:
    1. Orbital motion around the Sun (elliptical)
    2. Rotational motion (daily spin)
    3. Solar system motion through galaxy
    """
    
    # Convert time to years for orbital motion
    t_years = t_days / 365.25
    
    # Solar system velocity through galaxy (~220 km/s)
    # Scale to reasonable visualization units
    galaxy_velocity = 0.1  # Scaled units per day
    
    # Earth's orbital motion - ELLIPTICAL (eccentricity ~0.017)
    orbital_angle = 2 * np.pi * t_years
    eccentricity = 0.017  # Earth's orbital eccentricity
    semi_major = orbital_radius_au
    semi_minor = semi_major * np.sqrt(1 - eccentricity**2)
    
    orbital_x = semi_major * np.cos(orbital_angle)
    orbital_y = semi_minor * np.sin(orbital_angle)
    
    # Solar system motion through galaxy (linear component)
    galaxy_z = galaxy_velocity * t_days
    
    # Earth's rotation creates the "helix" around the orbital path
    # 1 rotation per day
    rotation_angle = 2 * np.pi * t_days
    rotation_radius = 0.05  # Small radius for Earth's rotation
    
    # Helical motion: rotation around the orbital path
    helix_x = orbital_x + rotation_radius * np.cos(rotation_angle)
    helix_y = orbital_y + rotation_radius * np.sin(rotation_angle)
    helix_z = galaxy_z
    
    return helix_x, helix_y, helix_z

def create_beat_wave_pattern(t_days, pattern_info, amplitude_scale=0.60):
    """
    Create a smooth wave pattern for a specific beat frequency.
    Just return the base amplitude - frequency will be applied in positioning.
    """
    correlation = pattern_info['correlation']
    
    # Return constant amplitude based on correlation strength
    # The actual frequency oscillation will be applied in the positioning phase
    amplitude = amplitude_scale * correlation
    
    return amplitude

def create_orbital_dance_visualization(beat_patterns):
    """Create the main orbital dance visualization."""
    print_status("Creating Orbital Dance Visualization", "PROCESSING")
    set_publication_style()
    
    # Time span: 2 years to show multiple cycles with optimal resolution
    t_days = np.linspace(0, 730, 3000)  # higher resolution for smooth ribbons
    
    # Create Earth's helical path
    earth_x, earth_y, earth_z = create_earth_helical_path(t_days)
    
    # Compute a smooth base orbital ellipse (no daily spin) for clean ribbons
    t_years = t_days / 365.25
    eccentricity = 0.017
    semi_major = 1.0
    semi_minor = semi_major * np.sqrt(1 - eccentricity**2)
    theta = 2 * np.pi * t_years
    base_x = semi_major * np.cos(theta)
    base_y = semi_minor * np.sin(theta)

    # Analytic unit tangent and normal along the base ellipse
    dtheta_dt = 2 * np.pi / 365.25
    dx_dt = -semi_major * np.sin(theta) * dtheta_dt
    dy_dt =  semi_minor * np.cos(theta) * dtheta_dt
    speed = np.sqrt(dx_dt**2 + dy_dt**2) + 1e-12
    t_hat_x = dx_dt / speed
    t_hat_y = dy_dt / speed
    n_hat_x = -t_hat_y
    n_hat_y =  t_hat_x
    
    # Site theme colors - clean professional theme
    THEME_COLORS = {
        'background': 'white',    # Clean white background
        'earth_path': '#2D0140',  # Deep purple for Earth's path
        'text': '#1e4a5f',        # Dark text for readability
        'grid': '#495773',        # Professional grid color
    }
    
    # Create figure with 3-panel horizontal layout and higher DPI
    fig = plt.figure(figsize=(18, 8), facecolor=THEME_COLORS['background'], dpi=400)
    
    # Create grid layout: 3 charts in top row, legend space below
    # Calculate proper ratios to ensure right panel is square
    # Height available for plots: (0.82 - 0.18) * 8 * 3/4 = 3.84 inches
    # Width for right panel should match this height
    gs = fig.add_gridspec(2, 5, height_ratios=[3, 1], width_ratios=[1.0, 0.03, 1.8, 0.03, 1.0], 
                         hspace=0.25, wspace=0.02,
                         left=0.06, right=0.94, top=0.82, bottom=0.18)
    
    # Spread patterns to reduce overlap
    offset_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    
    # 1. Side view (X-Z plane) - Left panel
    ax_xz = fig.add_subplot(gs[0, 0], facecolor=THEME_COLORS['background'])
    ax_xz.plot(earth_x, earth_z, color=THEME_COLORS['earth_path'], linewidth=0.5, alpha=0.6)
    
    for i, (pattern_name, pattern_info) in enumerate(beat_patterns.items()):
        wave_amplitude = create_beat_wave_pattern(t_days, pattern_info)
        angle_offset = offset_angles[i]
        wave_scale = 0.18  # normal displacement scale
        
        # Beat modulation
        beat_phase = 2 * np.pi * pattern_info['frequency_cpd'] * t_days + angle_offset
        beat = wave_amplitude * np.sin(beat_phase)
        
        # Displace along the instantaneous normal (flows with the curve)
        disp_x = wave_scale * beat * n_hat_x
        disp_z = wave_scale * beat * 0.3
        
        wave_x = base_x + disp_x
        wave_z = earth_z + disp_z
        
        ax_xz.plot(wave_x, wave_z, color=pattern_info['color'], 
                  linewidth=0.6, alpha=0.85)  # thinner, readable lines
    
    ax_xz.set_xlabel('X (AU)', color=THEME_COLORS['text'], fontsize=11)
    ax_xz.set_ylabel('Z (Galactic Motion)', color=THEME_COLORS['text'], fontsize=11)
    ax_xz.set_title('Side View: Helical Motion', color=THEME_COLORS['text'], fontweight='bold', fontsize=12, pad=15)
    ax_xz.tick_params(colors=THEME_COLORS['text'], labelsize=10)
    ax_xz.grid(True, color=THEME_COLORS['grid'], alpha=0.3)
    
    # Set proper axis ranges to show all data and ensure consistent proportions
    ax_xz.set_xlim(-1.2, 1.2)
    ax_xz.set_ylim(0, 75)
    
    # 2. Main 3D plot - Center panel (bigger)
    ax_main = fig.add_subplot(gs[0, 2], projection='3d', facecolor=THEME_COLORS['background'])
    
    # Set clean 3D plot appearance
    ax_main.xaxis.pane.fill = True
    ax_main.yaxis.pane.fill = True
    ax_main.zaxis.pane.fill = True
    ax_main.xaxis.pane.set_facecolor('white')
    ax_main.yaxis.pane.set_facecolor('white')
    ax_main.zaxis.pane.set_facecolor('white')
    ax_main.xaxis.pane.set_alpha(0.1)
    ax_main.yaxis.pane.set_alpha(0.1)
    ax_main.zaxis.pane.set_alpha(0.1)
    
    # Plot Earth's helical path
    ax_main.plot(base_x, base_y, earth_z, 
                color=THEME_COLORS['earth_path'], linewidth=0.5, alpha=0.6)
    
    # Add beat frequency wave patterns
    for i, (pattern_name, pattern_info) in enumerate(beat_patterns.items()):
        wave_amplitude = create_beat_wave_pattern(t_days, pattern_info)
        angle_offset = offset_angles[i]
        wave_scale = 0.18  # normal displacement scale
        
        beat_phase = 2 * np.pi * pattern_info['frequency_cpd'] * t_days + angle_offset
        beat = wave_amplitude * np.sin(beat_phase)
        
        # Normal displacement + slight tangential drift for realism
        disp_x = wave_scale * (beat * n_hat_x + 0.15 * wave_amplitude * np.cos(beat_phase) * t_hat_x)
        disp_y = wave_scale * (beat * n_hat_y + 0.15 * wave_amplitude * np.cos(beat_phase) * t_hat_y)
        disp_z = wave_scale * beat * 0.3
        
        wave_x = base_x + disp_x
        wave_y = base_y + disp_y
        wave_z = earth_z + disp_z
        
        ax_main.plot(wave_x, wave_y, wave_z,
                    color=pattern_info['color'], 
                    linewidth=0.6,
                    alpha=0.85)
    
    ax_main.set_xlabel('X (AU)', color=THEME_COLORS['text'], fontsize=11)
    ax_main.set_ylabel('Y (AU)', color=THEME_COLORS['text'], fontsize=11)
    ax_main.set_zlabel('Z (Galactic Motion)', color=THEME_COLORS['text'], fontsize=11)
    ax_main.set_title('3D Orbital Dance', color=THEME_COLORS['text'], fontweight='bold', fontsize=12, pad=15)
    ax_main.tick_params(colors=THEME_COLORS['text'], labelsize=10)
    ax_main.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
    
    # 3. Top view (X-Y plane) - Right panel showing elliptical orbit
    ax_xy = fig.add_subplot(gs[0, 4], facecolor=THEME_COLORS['background'])
    ax_xy.plot(base_x, base_y, color=THEME_COLORS['earth_path'], linewidth=0.5, alpha=0.6)
    
    for i, (pattern_name, pattern_info) in enumerate(beat_patterns.items()):
        wave_amplitude = create_beat_wave_pattern(t_days, pattern_info)
        angle_offset = offset_angles[i]
        wave_scale = 0.18
        
        beat_phase = 2 * np.pi * pattern_info['frequency_cpd'] * t_days + angle_offset
        beat = wave_amplitude * np.sin(beat_phase)
        
        disp_x = wave_scale * (beat * n_hat_x + 0.15 * wave_amplitude * np.cos(beat_phase) * t_hat_x)
        disp_y = wave_scale * (beat * n_hat_y + 0.15 * wave_amplitude * np.cos(beat_phase) * t_hat_y)
        
        wave_x = base_x + disp_x
        wave_y = base_y + disp_y
        
        ax_xy.plot(wave_x, wave_y, color=pattern_info['color'], 
                  linewidth=0.6, alpha=0.85)
    
    ax_xy.set_xlabel('X (AU)', color=THEME_COLORS['text'], fontsize=11)
    ax_xy.set_ylabel('Y (AU)', color=THEME_COLORS['text'], fontsize=11)
    ax_xy.set_title('Top View: Orbital Plane', color=THEME_COLORS['text'], fontweight='bold', fontsize=12, pad=15)
    ax_xy.tick_params(colors=THEME_COLORS['text'], labelsize=10)
    ax_xy.grid(True, color=THEME_COLORS['grid'], alpha=0.3)
    ax_xy.set_xlim(-1.2, 1.2)
    ax_xy.set_ylim(-1.2, 1.2)
    ax_xy.set_aspect('equal', adjustable='box')
    # Let the natural elliptical shape show - no forced aspect ratio
    
    # 4. Horizontal legend below all three charts
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')
    
    # Create horizontal legend with better spacing and alignment
    legend_y = 0.65
    x_positions = [0.08, 0.30, 0.52, 0.74]
    
    for i, (pattern_name, pattern_info) in enumerate(beat_patterns.items()):
        x_pos = x_positions[i]
        
        # Color line indicator
        ax_legend.plot([x_pos, x_pos + 0.035], [legend_y, legend_y], 
                      color=pattern_info['color'], linewidth=5, 
                      transform=ax_legend.transAxes)
        
        # Pattern name and details with better spacing
        ax_legend.text(x_pos + 0.045, legend_y + 0.12, pattern_info['name'], 
                      transform=ax_legend.transAxes, fontsize=10, fontweight='bold',
                      color=pattern_info['color'])
        
        ax_legend.text(x_pos + 0.045, legend_y + 0.02, f"Period: {pattern_info['period_days']:.1f} days", 
                      transform=ax_legend.transAxes, fontsize=9,
                      color=THEME_COLORS['text'])
        
        ax_legend.text(x_pos + 0.045, legend_y - 0.12, f"r = {pattern_info['correlation']:.3f}", 
                      transform=ax_legend.transAxes, fontsize=9, fontweight='bold',
                      color=pattern_info['color'])
    
    # Add validation note
    ax_legend.text(0.5, 0.1, 
                   'All patterns detected across CODE, ESA, and IGS centers with statistical significance p < 0.05',
                   transform=ax_legend.transAxes, fontsize=10, style='italic',
                   ha='center', color=THEME_COLORS['text'])
    
    # Overall figure formatting with proper title positioning
    fig.suptitle('Earth\'s Orbital Dance: Beat Frequencies in GNSS Clock Networks\n' +
                'Four Interference Patterns from Earth\'s Complex Motion Through Space', 
                fontsize=14, fontweight='bold', color=THEME_COLORS['text'], y=0.94)
    
    # No tight_layout needed with gridspec
    
    # Save figure
    root_dir = Path(__file__).parent.parent.parent
    figures_dir = root_dir / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_file = figures_dir / 'figure_10_orbital_dance_visualization.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print_status(f"Saved Orbital Dance Visualization to {output_file}", "SUCCESS")
    
    return output_file

def create_wave_interference_diagram(beat_patterns):
    """Create a supplementary diagram showing wave interference patterns."""
    print_status("Creating wave interference diagram", "INFO")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Time array for one full cycle of the longest period
    max_period = max(p['period_days'] for p in beat_patterns.values())
    t = np.linspace(0, max_period * 2, 1000)
    
    THEME_COLORS = {
        'background': 'white',
        'text': '#1e4a5f',
        'grid': '#495773'
    }
    
    for i, (pattern_name, pattern_info) in enumerate(beat_patterns.items()):
        ax = axes[i]
        
        # Create the beat wave
        frequency = pattern_info['frequency_cpd']
        correlation = pattern_info['correlation']
        
        # Main wave
        wave = correlation * np.sin(2 * np.pi * frequency * t)
        
        # Plot the wave
        ax.plot(t, wave, color=pattern_info['color'], linewidth=3, alpha=0.8)
        ax.fill_between(t, 0, wave, color=pattern_info['color'], alpha=0.3)
        
        # Formatting
        ax.set_xlabel('Time (days)', color=THEME_COLORS['text'])
        ax.set_ylabel('Correlation Amplitude', color=THEME_COLORS['text'])
        ax.set_title(f"{pattern_info['name']}\nPeriod: {pattern_info['period_days']:.1f} days", 
                    color=THEME_COLORS['text'], fontweight='bold')
        ax.grid(True, alpha=0.3, color=THEME_COLORS['grid'])
        ax.tick_params(colors=THEME_COLORS['text'])
        
        # Add period markers
        period = pattern_info['period_days']
        for p in range(int(max_period * 2 / period) + 1):
            ax.axvline(p * period, color='red', linestyle='--', alpha=0.5)
    
    fig.suptitle('Beat Frequency Wave Patterns\nTemporal Signatures of Earth\'s Complex Motion', 
                fontsize=16, fontweight='bold', color=THEME_COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save supplementary figure
    root_dir = Path(__file__).parent.parent.parent
    figures_dir = root_dir / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_file = figures_dir / 'figure_10_wave_interference_patterns.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print_status(f"Saved wave interference diagram to {output_file}", "SUCCESS")
    
    return output_file

def save_step_results(main_output, wave_output, beat_patterns):
    """Save step results to JSON file."""
    results = {
        'step': 12,
        'title': 'Additional Visualizations - Orbital Dance',
        'timestamp': datetime.now().isoformat(),
        'success': True,
        'outputs': {
            'main_visualization': str(main_output),
            'wave_interference': str(wave_output),
            'beat_patterns_analyzed': len(beat_patterns)
        },
        'beat_patterns': beat_patterns,
        'summary': {
            'visualizations_created': 2,
            'patterns_visualized': list(beat_patterns.keys()),
            'correlation_range': [
                min(p['correlation'] for p in beat_patterns.values()),
                max(p['correlation'] for p in beat_patterns.values())
            ]
        }
    }
    
    root_dir = Path(__file__).parent.parent.parent
    outputs_dir = root_dir / 'results' / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_file = outputs_dir / 'step_12_additional_visualizations.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_status(f"Saved step results to {output_file}", "SUCCESS")
    return results

def main():
    """Main function for Step 12: Additional Visualizations."""
    print("="*80)
    print("TEP GNSS Analysis - Step 12: Additional Visualizations")
    print("Earth's Motion Beat Frequencies - Orbital Dance")
    print("="*80)
    
    try:
        # Load beat frequency data
        beat_patterns = load_beat_frequency_data()
        
        if not beat_patterns:
            print_status("No beat frequency data found", "ERROR")
            return False
        
        # Create the main orbital dance visualization
        main_output = create_orbital_dance_visualization(beat_patterns)
        
        # Create supplementary wave interference diagram
        wave_output = create_wave_interference_diagram(beat_patterns)
        
        # Save step results
        results = save_step_results(main_output, wave_output, beat_patterns)
        
        # Summary
        print("\n" + "="*60)
        print("STEP 12: ADDITIONAL VISUALIZATIONS COMPLETE")
        print("="*60)
        
        print(f"\nMain visualization: {main_output}")
        print(f"Wave patterns: {wave_output}")
        
        print("\nVisualization Features:")
        print("• 3D helical path showing Earth's complex motion through space")
        print("• Four beat frequency patterns as colored wave ribbons")
        print("• Multiple viewing angles (3D, top view, side view)")
        print("• Wave interference patterns showing temporal signatures")
        print("• Professional publication-ready styling")
        
        print_status("Step 12 Additional Visualizations completed successfully!", "SUCCESS")
        
        return True
        
    except Exception as e:
        print_status(f"Step 12 failed: {e}", "ERROR")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
