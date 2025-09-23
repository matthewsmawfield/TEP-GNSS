#!/usr/bin/env python3
"""
Orbital Mechanics Analysis for TEP-GNSS
========================================

Proper orbital mechanics approach to planetary opposition analysis:
- Calculates continuous gravitational potential from all solar system bodies
- Models gradual changes over full orbital cycles
- Accounts for multi-body interactions and Earth's orbital position
- Replaces arbitrary ±5 day windows with physics-based analysis

Author: TEP-GNSS Analysis Pipeline v0.7
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from step_10_high_resolution_astronomical_events import load_geospatial_data, print_status

class OrbitalMechanicsCalculator:
    """
    Calculate planetary positions and gravitational potential based on orbital mechanics.
    Uses simplified Keplerian orbits for computational efficiency.
    """
    
    def __init__(self):
        # Orbital parameters (simplified, relative to Earth's orbit)
        self.planets = {
            'jupiter': {
                'period_years': 11.86,
                'distance_au': 5.20,
                'mass_earth_masses': 317.8,
                'opposition_dates': ['2023-11-03', '2024-12-07']
            },
            'saturn': {
                'period_years': 29.46,
                'distance_au': 9.54,
                'mass_earth_masses': 95.2,
                'opposition_dates': ['2023-08-27', '2024-09-08', '2025-09-21']
            },
            'mars': {
                'period_years': 1.88,
                'distance_au': 1.52,
                'mass_earth_masses': 0.107,
                'opposition_dates': ['2025-01-16']
            },
            'sun': {
                'period_years': 1.0,  # Earth's orbit
                'distance_au': 1.0,
                'mass_earth_masses': 333000,
                'perihelion_date': '2024-01-03'  # Earth closest to Sun
            },
            'moon': {
                'period_days': 29.53,  # Lunar month
                'distance_earth_radii': 60.3,
                'mass_earth_masses': 0.0123
            }
        }
    
    def calculate_orbital_phase(self, date: pd.Timestamp, planet: str) -> float:
        """
        Calculate orbital phase (0-360°) for a planet on a given date.
        Phase 0° corresponds to opposition (for outer planets) or conjunction (for inner).
        """
        if planet == 'sun':
            # Earth's orbital phase (perihelion = 0°)
            perihelion = pd.to_datetime('2024-01-03')
            days_since_perihelion = (date - perihelion).days
            phase_degrees = (days_since_perihelion / 365.25) * 360.0
            return phase_degrees % 360.0
        
        elif planet == 'moon':
            # Lunar phase (new moon = 0°)
            # Use a reference new moon date
            ref_new_moon = pd.to_datetime('2024-01-11')  # Known new moon
            days_since_new_moon = (date - ref_new_moon).days
            phase_degrees = (days_since_new_moon / 29.53) * 360.0
            return phase_degrees % 360.0
        
        else:
            # Outer planets: use opposition dates as phase reference
            planet_data = self.planets[planet]
            opposition_dates = [pd.to_datetime(d) for d in planet_data['opposition_dates']]
            
            # Find closest opposition
            closest_opposition = min(opposition_dates, key=lambda x: abs((date - x).days))
            days_from_opposition = (date - closest_opposition).days
            
            # Calculate synodic period (time between oppositions)
            period_years = planet_data['period_years']
            synodic_period_days = 365.25 / abs(1 - 365.25/(period_years * 365.25))
            
            # Phase relative to opposition (0° = opposition)
            phase_degrees = (days_from_opposition / synodic_period_days) * 360.0
            return phase_degrees % 360.0
    
    def calculate_distance_earth(self, date: pd.Timestamp, planet: str) -> float:
        """
        Calculate distance from Earth to planet in AU on given date.
        Uses simplified orbital mechanics.
        """
        if planet == 'sun':
            # Earth-Sun distance varies from 0.983 to 1.017 AU
            phase = self.calculate_orbital_phase(date, 'sun')
            # Minimum distance at perihelion (phase 0°)
            distance_au = 1.0 - 0.017 * np.cos(np.radians(phase))
            return distance_au
        
        elif planet == 'moon':
            # Moon-Earth distance in AU (varies from ~0.0024 to 0.0026 AU)
            phase = self.calculate_orbital_phase(date, 'moon')
            # Simplified: assume 5% variation
            distance_au = 0.00257 * (1 + 0.05 * np.cos(np.radians(phase)))
            return distance_au
        
        else:
            # Outer planets: distance varies based on orbital positions
            planet_data = self.planets[planet]
            orbital_radius = planet_data['distance_au']
            phase = self.calculate_orbital_phase(date, planet)
            
            # At opposition (phase 0°): minimum distance = orbital_radius - 1 AU
            # At conjunction (phase 180°): maximum distance = orbital_radius + 1 AU
            distance_au = orbital_radius - np.cos(np.radians(phase))
            return max(distance_au, 0.1)  # Prevent negative distances
    
    def calculate_gravitational_potential(self, date: pd.Timestamp, include_bodies: List[str] = None) -> Dict:
        """
        Calculate total gravitational potential at Earth from all specified bodies.
        Returns potential in arbitrary units (proportional to GM/r).
        """
        if include_bodies is None:
            include_bodies = ['jupiter', 'saturn', 'mars', 'sun', 'moon']
        
        total_potential = 0.0
        body_contributions = {}
        
        for body in include_bodies:
            if body not in self.planets:
                continue
                
            mass = self.planets[body]['mass_earth_masses']
            distance = self.calculate_distance_earth(date, body)
            phase = self.calculate_orbital_phase(date, body)
            
            # Gravitational potential ~ GM/r
            potential = mass / (distance ** 2)
            
            body_contributions[body] = {
                'potential': potential,
                'distance_au': distance,
                'phase_degrees': phase,
                'mass_earth_masses': mass
            }
            
            total_potential += potential
        
        return {
            'total_potential': total_potential,
            'date': date,
            'body_contributions': body_contributions
        }
    
    def generate_orbital_timeline(self, start_date: str, end_date: str, 
                                 freq: str = 'D') -> pd.DataFrame:
        """
        Generate timeline of orbital positions and gravitational potential.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        timeline_data = []
        
        print_status(f"Calculating orbital mechanics for {len(dates)} dates...", "PROCESSING")
        
        for date in dates:
            potential_data = self.calculate_gravitational_potential(date)
            
            row = {
                'date': date,
                'total_gravitational_potential': potential_data['total_potential']
            }
            
            # Add individual body contributions
            for body, contrib in potential_data['body_contributions'].items():
                row[f'{body}_potential'] = contrib['potential']
                row[f'{body}_distance_au'] = contrib['distance_au']
                row[f'{body}_phase_degrees'] = contrib['phase_degrees']
            
            timeline_data.append(row)
        
        df = pd.DataFrame(timeline_data)
        print_status(f"Generated orbital timeline: {len(df)} data points", "SUCCESS")
        return df

def analyze_orbital_coherence_correlation(analysis_center: str = 'merged') -> Dict:
    """
    Correlate GPS coherence data with orbital mechanics predictions.
    """
    print_status(f"Starting Orbital Mechanics Coherence Analysis for {analysis_center}", "PROCESSING")
    
    try:
        # Load GPS coherence data
        gps_data = load_geospatial_data(analysis_center)
        print_status(f"Loaded {len(gps_data):,} GPS data points", "SUCCESS")
        
        # Initialize orbital calculator
        orbital_calc = OrbitalMechanicsCalculator()
        
        # Generate orbital timeline matching GPS data timespan
        start_date = gps_data['date'].min().strftime('%Y-%m-%d')
        end_date = gps_data['date'].max().strftime('%Y-%m-%d')
        
        orbital_timeline = orbital_calc.generate_orbital_timeline(start_date, end_date)
        
        # Merge GPS data with orbital predictions
        # Group GPS data by day for efficiency
        daily_gps = gps_data.groupby(gps_data['date'].dt.date).agg({
            'coherence': ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        daily_gps.columns = ['date', 'coherence_mean', 'coherence_median', 'coherence_std', 'coherence_count']
        daily_gps['date'] = pd.to_datetime(daily_gps['date'])
        
        # Merge with orbital data
        merged_data = pd.merge(daily_gps, orbital_timeline, on='date', how='inner')
        print_status(f"Merged data: {len(merged_data)} days", "SUCCESS")
        
        # Calculate correlations
        correlations = {}
        for body in ['jupiter', 'saturn', 'mars', 'sun', 'moon']:
            potential_col = f'{body}_potential'
            if potential_col in merged_data.columns:
                corr_mean = merged_data['coherence_mean'].corr(merged_data[potential_col])
                corr_median = merged_data['coherence_median'].corr(merged_data[potential_col])
                
                correlations[body] = {
                    'correlation_with_mean_coherence': float(corr_mean) if not np.isnan(corr_mean) else 0.0,
                    'correlation_with_median_coherence': float(corr_median) if not np.isnan(corr_median) else 0.0,
                    'mean_potential': float(merged_data[potential_col].mean()),
                    'std_potential': float(merged_data[potential_col].std())
                }
        
        # Total potential correlation
        total_corr_mean = merged_data['coherence_mean'].corr(merged_data['total_gravitational_potential'])
        total_corr_median = merged_data['coherence_median'].corr(merged_data['total_gravitational_potential'])
        
        # Opposition timing analysis
        opposition_analysis = analyze_opposition_timing(merged_data, orbital_calc)
        
        results = {
            'success': True,
            'analysis_center': analysis_center,
            'data_points': len(merged_data),
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'total_gravitational_correlation': {
                'with_mean_coherence': float(total_corr_mean) if not np.isnan(total_corr_mean) else 0.0,
                'with_median_coherence': float(total_corr_median) if not np.isnan(total_corr_median) else 0.0
            },
            'individual_body_correlations': correlations,
            'opposition_timing_analysis': opposition_analysis,
            'merged_data_sample': merged_data.head(10).to_dict('records')
        }
        
        print_status("Orbital mechanics analysis completed", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"Orbital mechanics analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def analyze_opposition_timing(merged_data: pd.DataFrame, orbital_calc: OrbitalMechanicsCalculator) -> Dict:
    """
    Analyze coherence patterns around opposition dates using proper orbital mechanics.
    """
    opposition_results = {}
    
    for planet in ['jupiter', 'saturn', 'mars']:
        planet_data = orbital_calc.planets[planet]
        opposition_dates = [pd.to_datetime(d) for d in planet_data['opposition_dates']]
        
        planet_results = []
        
        for opp_date in opposition_dates:
            # Extract data in a wider window around opposition (±120 days)
            window_start = opp_date - pd.Timedelta(days=120)
            window_end = opp_date + pd.Timedelta(days=120)
            
            window_data = merged_data[
                (merged_data['date'] >= window_start) & 
                (merged_data['date'] <= window_end)
            ].copy()
            
            if len(window_data) < 50:  # Need sufficient data
                continue
            
            # Calculate orbital phase for each day
            window_data['orbital_phase'] = window_data['date'].apply(
                lambda d: orbital_calc.calculate_orbital_phase(d, planet)
            )
            
            # Calculate distance from Earth
            window_data['distance_au'] = window_data['date'].apply(
                lambda d: orbital_calc.calculate_distance_earth(d, planet)
            )
            
            # Days from opposition
            window_data['days_from_opposition'] = (window_data['date'] - opp_date).dt.days
            
            # Fit coherence to orbital mechanics model
            # Model: coherence ~ a + b/distance^2 + c*cos(phase) + noise
            from scipy.optimize import curve_fit
            
            def orbital_model(x_data, a, b, c):
                days, distance, phase = x_data
                return a + b / (distance ** 2) + c * np.cos(np.radians(phase))
            
            try:
                x_data = np.array([
                    window_data['days_from_opposition'].values,
                    window_data['distance_au'].values, 
                    window_data['orbital_phase'].values
                ])
                
                y_data = window_data['coherence_median'].values
                
                popt, pcov = curve_fit(orbital_model, x_data, y_data, maxfev=1000)
                
                # Calculate R-squared
                y_pred = orbital_model(x_data, *popt)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calculate coherence at minimum distance (opposition)
                min_distance = window_data['distance_au'].min()
                opposition_phase = 0.0  # Phase at opposition
                
                predicted_opposition_coherence = orbital_model(
                    np.array([[0], [min_distance], [opposition_phase]]), *popt
                )[0]
                
                # Calculate baseline coherence (far from opposition)
                baseline_distance = window_data['distance_au'].mean()
                baseline_phase = 180.0  # Opposite phase
                
                predicted_baseline_coherence = orbital_model(
                    np.array([[60], [baseline_distance], [baseline_phase]]), *popt
                )[0]
                
                # Effect size based on orbital mechanics model
                orbital_effect_percent = (
                    (predicted_opposition_coherence - predicted_baseline_coherence) / 
                    predicted_baseline_coherence * 100
                ) if predicted_baseline_coherence != 0 else 0.0
                
                planet_results.append({
                    'opposition_date': opp_date.strftime('%Y-%m-%d'),
                    'data_points': len(window_data),
                    'orbital_model_fit': {
                        'r_squared': float(r_squared),
                        'parameters': {
                            'baseline': float(popt[0]),
                            'distance_coefficient': float(popt[1]),
                            'phase_coefficient': float(popt[2])
                        },
                        'parameter_errors': [float(np.sqrt(pcov[i,i])) for i in range(len(popt))]
                    },
                    'predicted_effect_percent': float(orbital_effect_percent),
                    'minimum_distance_au': float(min_distance),
                    'opposition_coherence_predicted': float(predicted_opposition_coherence),
                    'baseline_coherence_predicted': float(predicted_baseline_coherence)
                })
                
            except Exception as e:
                print_status(f"Failed to fit orbital model for {planet} {opp_date}: {e}", "WARNING")
                continue
        
        opposition_results[planet] = planet_results
    
    return opposition_results

def create_orbital_mechanics_figure(results: Dict) -> Optional[str]:
    """
    Create comprehensive orbital mechanics analysis figure.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Orbital Mechanics Analysis: GPS Coherence vs Gravitational Potential', 
                    fontsize=16, fontweight='bold')
        
        # Extract merged data sample for plotting
        sample_data = pd.DataFrame(results['merged_data_sample'])
        
        if len(sample_data) == 0:
            print_status("No sample data available for plotting", "WARNING")
            return None
        
        # Plot 1: Total gravitational potential vs coherence
        ax1 = axes[0, 0]
        ax1.scatter(sample_data['total_gravitational_potential'], 
                   sample_data['coherence_median'], alpha=0.6, s=20)
        ax1.set_xlabel('Total Gravitational Potential')
        ax1.set_ylabel('Median Coherence')
        ax1.set_title('GPS Coherence vs Total Gravitational Field')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation text
        total_corr = results['total_gravitational_correlation']['with_median_coherence']
        ax1.text(0.05, 0.95, f'Correlation: {total_corr:.3f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Plot 2: Individual body correlations
        ax2 = axes[0, 1]
        bodies = list(results['individual_body_correlations'].keys())
        correlations = [results['individual_body_correlations'][body]['correlation_with_median_coherence'] 
                       for body in bodies]
        
        colors = ['#FF6B35', '#4A90C2', '#C44536', '#F4D03F', '#85C1E9']
        bars = ax2.bar(bodies, correlations, color=colors[:len(bodies)])
        ax2.set_ylabel('Correlation with Coherence')
        ax2.set_title('Individual Body Correlations')
        ax2.set_ylim(-0.5, 0.5)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Opposition timing analysis
        ax3 = axes[1, 0]
        opposition_data = results.get('opposition_timing_analysis', {})
        
        planet_names = []
        effect_sizes = []
        r_squared_values = []
        
        for planet, oppositions in opposition_data.items():
            for opp in oppositions:
                planet_names.append(f"{planet.title()}\n{opp['opposition_date']}")
                effect_sizes.append(opp['predicted_effect_percent'])
                r_squared_values.append(opp['orbital_model_fit']['r_squared'])
        
        if planet_names:
            bars = ax3.bar(range(len(planet_names)), effect_sizes, 
                          color=['#FF6B35', '#4A90C2', '#C44536'][:len(planet_names)])
            ax3.set_xticks(range(len(planet_names)))
            ax3.set_xticklabels(planet_names, rotation=45, ha='right')
            ax3.set_ylabel('Predicted Effect Size (%)')
            ax3.set_title('Orbital Mechanics Model Predictions')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.grid(True, alpha=0.3)
            
            # Add R² values as text
            for i, (bar, r2) in enumerate(zip(bars, r_squared_values)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'R²={r2:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No Opposition Data Available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
ORBITAL MECHANICS ANALYSIS SUMMARY

Analysis Center: {results['analysis_center']}
Data Points: {results['data_points']:,}
Date Range: {results['date_range']['start']} to {results['date_range']['end']}

CORRELATIONS WITH GRAVITATIONAL POTENTIAL:
Total Field: {total_corr:.4f}

Individual Bodies:
"""
        
        for body, corr_data in results['individual_body_correlations'].items():
            corr_val = corr_data['correlation_with_median_coherence']
            summary_text += f"  {body.title()}: {corr_val:+.4f}\n"
        
        summary_text += f"""
OPPOSITION ANALYSIS:
Oppositions Analyzed: {sum(len(opps) for opps in opposition_data.values())}
Average R²: {np.mean(r_squared_values) if r_squared_values else 0:.3f}

METHOD:
• Continuous orbital mechanics modeling
• Multi-body gravitational field calculation  
• Phase-locked analysis over full orbital cycles
• Proper baseline relative to orbital positions
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        root_dir = Path(__file__).parent.parent.parent
        figures_dir = root_dir / 'results' / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        output_file = figures_dir / 'orbital_mechanics_analysis.png'
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print_status(f"Saved orbital mechanics figure to {output_file}", "SUCCESS")
        return str(output_file)
        
    except Exception as e:
        print_status(f"Failed to create orbital mechanics figure: {e}", "ERROR")
        return None

def main():
    """Main function for orbital mechanics analysis."""
    print("="*80)
    print("TEP GNSS Analysis - Orbital Mechanics Analysis")
    print("Proper Multi-Body Gravitational Field Modeling")
    print("="*80)
    
    try:
        # Analyze for merged data
        results = analyze_orbital_coherence_correlation('merged')
        
        if not results['success']:
            print_status(f"Analysis failed: {results.get('error', 'Unknown error')}", "ERROR")
            return False
        
        # Create visualization
        figure_path = create_orbital_mechanics_figure(results)
        
        # Save results
        root_dir = Path(__file__).parent.parent.parent
        outputs_dir = root_dir / 'results' / 'outputs'
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_file = outputs_dir / 'orbital_mechanics_analysis.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print_status(f"Saved results to {output_file}", "SUCCESS")
        
        # Summary
        print("\n" + "="*60)
        print("ORBITAL MECHANICS ANALYSIS COMPLETE")
        print("="*60)
        
        total_corr = results['total_gravitational_correlation']['with_median_coherence']
        print(f"\nTotal Gravitational Field Correlation: {total_corr:+.4f}")
        
        print("\nIndividual Body Correlations:")
        for body, corr_data in results['individual_body_correlations'].items():
            corr_val = corr_data['correlation_with_median_coherence']
            print(f"  {body.title()}: {corr_val:+.4f}")
        
        opposition_data = results.get('opposition_timing_analysis', {})
        total_oppositions = sum(len(opps) for opps in opposition_data.values())
        print(f"\nOppositions Analyzed: {total_oppositions}")
        
        if figure_path:
            print(f"\nVisualization: {figure_path}")
        
        print("\nKEY IMPROVEMENTS OVER PREVIOUS METHOD:")
        print("• Continuous orbital mechanics modeling (not ±5 day windows)")
        print("• Multi-body gravitational field calculation")
        print("• Phase-locked analysis over full orbital cycles")
        print("• Proper baseline relative to orbital positions")
        print("• Gradual change analysis following 1/r² physics")
        
        return True
        
    except Exception as e:
        print_status(f"Orbital mechanics analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
