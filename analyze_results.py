#!/usr/bin/env python3
"""
Comprehensive analysis of CODE longspan results
Extracts all key findings and checks for manuscript consistency
"""

import json
import sys

def main():
    # Load main results
    with open('/Users/matthewsmawfield/www/TEP-GNSS/results/outputs/code_longspan/step_2_2_geospatial_temporal_analysis_code.json') as f:
        data = json.load(f)
    
    print('='*80)
    print('COMPREHENSIVE RESULTS ANALYSIS - CODE LONGSPAN (25.3 years)')
    print('='*80)
    
    # Data Summary
    print('\n### DATA SUMMARY')
    ds = data['data_summary']
    print(f"  Total pairs: {ds['total_pairs']:,}")
    print(f"  Unique stations: {ds['unique_stations']}")
    print(f"  Unique dates: {ds['unique_dates']}")
    print(f"  Distance range: {ds['distance_range_km'][0]:.1f} - {ds['distance_range_km'][1]:.1f} km")
    
    # Orbital Motion
    print('\n### 1. ORBITAL MOTION CORRELATION')
    om = data['orbital_motion_evidence']
    print(f"  Correlation (r): {om['correlation_coefficient']:.3f}")
    print(f"  P-value (autocorr-corrected): {om['p_value']:.2e}")
    print(f"  Monte Carlo p-value: {om['monte_carlo_p_value']:.2e}")
    print(f"  Sigma equivalent: {om['monte_carlo_sigma_equivalent']:.2f}σ")
    print(f"  N_eff: {om.get('n_samples', 'N/A')}")
    print(f"  Evidence: {om['monte_carlo_evidence_strength']}")
    
    # Anisotropy
    print('\n### 2. DIRECTIONAL ANISOTROPY (8 sectors)')
    anis = data['enhanced_anisotropy_analysis']
    sectors = anis['sector_results']
    
    lambdas = []
    for sector in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        vals = sectors[sector]
        lambdas.append(vals['lambda_km'])
        print(f"  {sector:3s}: λ={vals['lambda_km']:7.1f} km, R²={vals['r_squared']:.3f}, pairs={vals['n_pairs']:,}")
    
    stats = anis['anisotropy_statistics']
    print(f"\n  Mean λ: {stats['lambda_mean']:.1f} ± {stats['lambda_std']:.1f} km")
    print(f"  Coefficient of Variation: {stats['coefficient_of_variation']:.3f}")
    print(f"  Category: {stats['anisotropy_category']}")
    
    # Calculate ratios
    ew_avg = (sectors['E']['lambda_km'] + sectors['W']['lambda_km']) / 2
    ns_avg = (sectors['N']['lambda_km'] + sectors['S']['lambda_km']) / 2
    max_lambda = max(lambdas)
    min_lambda = min(lambdas)
    print(f"  EW:NS ratio: {ew_avg/ns_avg:.2f}")
    print(f"  Max/Min ratio: {max_lambda/min_lambda:.2f} ({max(sectors, key=lambda k: sectors[k]['lambda_km'])}/{min(sectors, key=lambda k: sectors[k]['lambda_km'])})")
    
    # Planetary Events
    print('\n### 3. PLANETARY EVENT DETECTIONS (Primary ±120-day window)')
    planets = {
        'jupiter_opposition_analysis': 'Jupiter',
        'saturn_opposition_analysis': 'Saturn', 
        'mars_opposition_analysis': 'Mars',
        'venus_conjunction_analysis': 'Venus',
        'mercury_conjunction_analysis': 'Mercury'
    }
    
    total_sig = 0
    total_events = 0
    total_bonf = 0
    total_fdr = 0
    
    for key, name in planets.items():
        if key in data and 'event_summary' in data[key]:
            es = data[key]['event_summary']
            sig = es['significant_events']
            tot = es['total_events']
            bonf = es.get('bonferroni_survivors', 0)
            fdr = es.get('by_fdr_survivors', 0)
            
            total_sig += sig
            total_events += tot
            total_bonf += bonf
            total_fdr += fdr
            
            print(f"  {name:8s}: {sig:2d}/{tot:2d} sig ({100*sig/tot:5.1f}%), Bonf={bonf:2d}, FDR={fdr:2d}")
    
    print(f"  {'TOTAL':8s}: {total_sig:2d}/{total_events:3d} sig ({100*total_sig/total_events:5.1f}%), Bonf={total_bonf:2d}, FDR={total_fdr:2d}")
    
    if total_sig > 0:
        print(f"\n  Bonferroni survival rate: {100*total_bonf/total_sig:.1f}%")
        print(f"  BY-FDR survival rate: {100*total_fdr/total_sig:.1f}%")
    
    # Nutation
    print('\n### 4. NUTATION SIGNATURES')
    nut = data['nutation_analysis']
    for period in ['18.6_year', 'semiannual', 'annual']:
        if period in nut and isinstance(nut[period], dict):
            r2 = nut[period].get('r_squared', 0)
            p = nut[period].get('p_value', 1)
            amp = nut[period].get('amplitude', 0)
            phase = nut[period].get('phase_deg', 0)
            print(f"  {period:12s}: R²={r2:.3f}, p={p:.2e}, amp={amp:.4f}, phase={phase:.1f}°")
    
    # Chandler Wobble
    print('\n### 5. CHANDLER WOBBLE')
    cw = data['chandler_wobble_analysis']
    print(f"  Period: {cw['period_days']:.1f} days ({cw['period_days']/30.44:.1f} months)")
    print(f"  R²: {cw['r_squared']:.3f}")
    print(f"  Amplitude: {cw.get('amplitude', 'N/A')}")
    print(f"  Phase: {cw.get('phase_deg', 'N/A')}°")
    if 'phase_coherence' in cw:
        print(f"  Phase coherence: {cw['phase_coherence']:.3f}")
        print(f"  Coherence p-value: {cw.get('coherence_p_value', 'N/A')}")
    if 'complete_cycles' in cw:
        print(f"  Complete cycles: {cw['complete_cycles']:.1f}")
    
    # Continuous Planetary
    print('\n### 6. CONTINUOUS PLANETARY CORRELATION')
    cp = data['continuous_planetary_analysis']
    print(f"  Best correlation (r): {cp['best_correlation']:.3f}")
    print(f"  P-value (autocorr-corrected): {cp['best_p_value_corrected']:.4f}")
    print(f"  Smoothing window: {cp['best_window_days']} days")
    print(f"  Detected: {cp.get('detected', False)}")
    
    # Spherical Harmonics
    print('\n### 7. 3D SPHERICAL HARMONICS ANISOTROPY')
    sh = data['spherical_harmonics_analysis']
    print(f"  Anisotropy strength: {sh['anisotropy_strength']:.3f}")
    print(f"  Detected: {sh['detected']}")
    if 'dominant_modes' in sh:
        print(f"  Dominant modes: {sh['dominant_modes']}")
    
    # Mesh Dance
    print('\n### 8. MESH NETWORK COHERENCE')
    md = data['mesh_dance_analysis']
    print(f"  Coherence score: {md['coherence_score']:.3f}")
    print(f"  Detected: {md['detected']}")
    if 'components' in md:
        print(f"  Components: {md['components']}")
    
    # Null Tests
    print('\n### 9. NULL TESTS (Expected Non-Detections)')
    if 'solar_rotation_analysis' in data:
        sr = data['solar_rotation_analysis']
        print(f"  Solar rotation (27-day):")
        print(f"    r = {sr.get('correlation', 0):.3f}")
        print(f"    p = {sr.get('p_value', 1):.3f}")
        print(f"    Detected: {sr.get('detected', False)}")
    
    if 'lunar_standstill_analysis' in data:
        ls = data['lunar_standstill_analysis']
        print(f"  Lunar standstill (2024-2025):")
        print(f"    Detected: {ls.get('detected', False)}")
        if 'interpretation' in ls:
            print(f"    Note: {ls['interpretation'][:80]}...")
    
    # Overall Assessment
    print('\n### OVERALL TEP EVIDENCE ASSESSMENT')
    if 'tep_assessment' in data:
        tep = data['tep_assessment']
        print(f"  Primary detections: {tep.get('primary_detections', 'N/A')}")
        print(f"  Secondary detections: {tep.get('secondary_detections', 'N/A')}")
        print(f"  Total score: {tep.get('total_score', 'N/A')}")
        print(f"  Conclusion: {tep.get('conclusion', 'N/A')}")
    
    print('\n' + '='*80)
    print('MANUSCRIPT CONSISTENCY CHECKS')
    print('='*80)
    
    # Check key values
    print('\n### KEY VALUES TO VERIFY IN MANUSCRIPT:')
    print(f"  ✓ Orbital correlation: r = {om['correlation_coefficient']:.3f}")
    print(f"  ✓ Planetary events: {total_sig}/{total_events} significant")
    print(f"  ✓ Bonferroni survivors: {total_bonf}")
    print(f"  ✓ BY-FDR survivors: {total_fdr}")
    print(f"  ✓ Chandler period: {cw['period_days']:.1f} days")
    print(f"  ✓ Chandler R²: {cw['r_squared']:.3f}")
    print(f"  ✓ Mean correlation length: {stats['lambda_mean']:.1f} ± {stats['lambda_std']:.1f} km")
    print(f"  ✓ EW:NS ratio: {ew_avg/ns_avg:.2f}")
    print(f"  ✓ Anisotropy CV: {stats['coefficient_of_variation']:.3f}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
