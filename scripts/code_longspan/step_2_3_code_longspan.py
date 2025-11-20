
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from scipy import stats
from jplephem.spk import SPK
from astropy.time import Time
import astropy.units as u

# Configure paths
BASE_DIR = "/Users/matthewsmawfield/www/TEP-GNSS"
RESULTS_DIR = os.path.join(BASE_DIR, "results/outputs/code_longspan")
FIGURES_DIR = os.path.join(BASE_DIR, "results/figures")
INPUT_FILE = os.path.join(RESULTS_DIR, "step_2_2_geospatial_temporal_analysis_code.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "step_2_3_physical_interpretation.json")
EPHEMERIS_FILE = os.path.join(BASE_DIR, "de432s.bsp")

def print_status(message, level="INFO"):
    print(f"[{level}] {message}")

def load_results():
    print_status(f"Loading results from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        return json.load(f)

def get_planet_mass_gm(planet_name):
    # GM values in km^3/s^2 (approximate)
    gm_values = {
        "mercury": 22032.09,
        "venus": 324858.592,
        "mars": 42828.37,
        "jupiter": 126686534.0,
        "saturn": 37931187.0,
        "sun": 132712440018.9
    }
    # Handle case sensitivity and variations
    key = planet_name.lower()
    if "mercury" in key: return gm_values["mercury"]
    if "venus" in key: return gm_values["venus"]
    if "mars" in key: return gm_values["mars"]
    if "jupiter" in key: return gm_values["jupiter"]
    if "saturn" in key: return gm_values["saturn"]
    return None

def get_planet_position(kernel, planet_name, jd):
    # Map names to SPK IDs
    # de432s usually has: 
    # 1=Mercury, 2=Venus, 3=Earth, 4=Mars, 5=Jupiter, 6=Saturn
    # IDs are often barycenters: 1, 2, 3, 4, 5, 6
    # Earth is 399 relative to 3 (Earth Barycenter), but de432s might be simpler.
    # Let's assume standard DE432s structure:
    # 0=SSB, 3=EarthBarycenter.
    # 199=Mercury, 299=Venus, 399=Earth, 499=Mars, etc in some kernels.
    # For de432s.bsp specifically:
    # It usually contains 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 399, 301
    
    # Target planet ID
    target_id = None
    if "mercury" in planet_name.lower(): target_id = 1
    elif "venus" in planet_name.lower(): target_id = 2
    elif "earth" in planet_name.lower(): target_id = 3
    elif "mars" in planet_name.lower(): target_id = 4
    elif "jupiter" in planet_name.lower(): target_id = 5
    elif "saturn" in planet_name.lower(): target_id = 6
    
    if target_id is None:
        return None
        
    # Calculate position relative to Earth (ID 3)
    # Note: This is approximate if using barycenters, but sufficient for M/r^3 vs M/r^2 scaling tests
    # The kernel likely gives positions w.r.t SSB (0).
    # Pos_Planet - Pos_Earth
    
    try:
        pos_planet = kernel[0, target_id].compute(jd)
        pos_earth = kernel[0, 3].compute(jd)
        
        # Distance vector
        r_vec = pos_planet - pos_earth
        distance_km = np.sqrt(np.sum(r_vec**2, axis=0))
        return distance_km
    except Exception as e:
        print_status(f"Ephemeris error for {planet_name}: {e}", "WARNING")
        return None

def analyze_gravitational_scaling(results):
    print_status("Analyzing Gravitational Scaling (M/r^3 vs M/r^2)...")
    
    # Load ephemeris
    if not os.path.exists(EPHEMERIS_FILE):
        print_status(f"Ephemeris file not found at {EPHEMERIS_FILE}", "ERROR")
        return None
        
    kernel = SPK.open(EPHEMERIS_FILE)
    
    events_data = []
    
    # Extract events
    planet_keys = [
        "jupiter_opposition_analysis", 
        "saturn_opposition_analysis", 
        "mars_opposition_analysis", 
        "venus_conjunction_analysis", 
        "mercury_conjunction_analysis"
    ]
    
    # Iterate through planet blocks
    for key in planet_keys:
        block = results.get(key, {})
        if not block or not block.get("success"):
            continue
            
        results_by_window = block.get("results_by_window_size", {})
        
        # Look for window_size = 120 (Primary)
        # Keys in results_by_window are strings like "120"
        primary_window_results = results_by_window.get("120")
        
        if not primary_window_results:
            print_status(f"No 120-day window results for {key}", "WARNING")
            continue
            
        # Determine planet name from key
        planet_name = key.split("_")[0]
        event_results = primary_window_results.get("event_results", {})
        
        for event_id, event_data in event_results.items():
            if not event_data.get("success"):
                continue
            
            gaussian_fit = event_data.get("gaussian_fit", {})
            if not gaussian_fit:
                continue
                
            event_date_str = event_data.get("event_date")
            amplitude = gaussian_fit.get("amplitude", 0)
            modulation_depth = gaussian_fit.get("amplitude_fraction_of_baseline", 0) # Note: key name might be different
            sigma = gaussian_fit.get("sigma_level", 0)
            
            # If amplitude is 0, check if it's just missing or actually 0
            if amplitude == 0 and "amplitude" not in gaussian_fit:
                # Try alternative keys if needed, but usually it's amplitude
                pass
            
            # Convert date to JD
            dt = datetime.fromisoformat(event_date_str)
            t = Time(dt)
            jd = t.jd
            
            # Get Distance
            distance_km = get_planet_position(kernel, planet_name, jd)
            if distance_km is None:
                continue
                
            # Get Mass
            gm = get_planet_mass_gm(planet_name)
            if gm is None:
                continue
                
            # Calculate Potentials
            # Tidal Potential V_tidal ~ GM / r^3 (approximate scaling for tidal force gradient)
            # Classical Potential V ~ GM / r
            # Gravity g ~ GM / r^2
            
            tidal_param = gm / (distance_km**3)
            gravity_param = gm / (distance_km**2)
            
            events_data.append({
                "planet": planet_name,
                "event_id": event_id,
                "amplitude": abs(amplitude),
                "modulation_depth": modulation_depth,
                "sigma": sigma,
                "distance_km": float(distance_km),
                "gm": gm,
                "tidal_param": float(tidal_param),   # ~ M/r^3
                "gravity_param": float(gravity_param) # ~ M/r^2
            })
            
    # Convert to DataFrame for analysis
    import pandas as pd
    df = pd.DataFrame(events_data)
    
    if len(df) == 0:
        print_status("No event data extracted", "WARNING")
        return None
        
    # Correlations
    r_tidal_amp, p_tidal_amp = stats.pearsonr(df["tidal_param"], df["amplitude"])
    r_gravity_amp, p_gravity_amp = stats.pearsonr(df["gravity_param"], df["amplitude"])
    
    r_tidal_mod, p_tidal_mod = stats.pearsonr(df["tidal_param"], df["modulation_depth"])
    r_gravity_mod, p_gravity_mod = stats.pearsonr(df["gravity_param"], df["modulation_depth"])
    
    print_status(f"Analyzed {len(df)} events")
    print_status(f"Correlation Amplitude vs Tidal (M/r^3): r={r_tidal_amp:.3f}, p={p_tidal_amp:.4f}")
    print_status(f"Correlation Amplitude vs Gravity (M/r^2): r={r_gravity_amp:.3f}, p={p_gravity_amp:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Amplitude vs Tidal
    axes[0].scatter(df["tidal_param"], df["amplitude"], alpha=0.6)
    axes[0].set_xlabel("Tidal Parameter (GM/r³)")
    axes[0].set_ylabel("Observed Amplitude")
    axes[0].set_title(f"Amplitude vs Tidal Potential\nr={r_tidal_amp:.3f}, p={p_tidal_amp:.3f}")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Amplitude vs Gravity
    axes[1].scatter(df["gravity_param"], df["amplitude"], alpha=0.6, color='orange')
    axes[1].set_xlabel("Gravity Parameter (GM/r²)")
    axes[1].set_ylabel("Observed Amplitude")
    axes[1].set_title(f"Amplitude vs Gravitational Field\nr={r_gravity_amp:.3f}, p={p_gravity_amp:.3f}")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "step_2_3_gravitational_scaling.png"))
    plt.close()
    
    return {
        "n_events": len(df),
        "correlations": {
            "amplitude_vs_tidal_mr3": {"r": r_tidal_amp, "p": p_tidal_amp},
            "amplitude_vs_gravity_mr2": {"r": r_gravity_amp, "p": p_gravity_amp},
            "depth_vs_tidal_mr3": {"r": r_tidal_mod, "p": p_tidal_mod},
            "depth_vs_gravity_mr2": {"r": r_gravity_mod, "p": p_gravity_mod}
        },
        "data_summary": df.to_dict(orient="records")
    }

def analyze_station_density(results):
    print_status("Analyzing Station Density Control...")
    
    anisotropy = results.get("enhanced_anisotropy_analysis", {})
    if not anisotropy.get("success"):
        print_status("No anisotropy data found", "WARNING")
        return None
        
    sectors = anisotropy.get("sector_results", {})
    
    data = []
    for sector, metrics in sectors.items():
        if not isinstance(metrics, dict): continue
        
        lambda_km = metrics.get("lambda_km")
        n_pairs = metrics.get("n_pairs")
        r_squared = metrics.get("r_squared")
        
        if lambda_km and n_pairs:
            data.append({
                "sector": sector,
                "lambda_km": lambda_km,
                "n_pairs": n_pairs,
                "r_squared": r_squared,
                "density_metric": n_pairs / 1e6 # Millions of pairs
            })
            
    import pandas as pd
    df = pd.DataFrame(data)
    
    if len(df) < 3:
        print_status("Insufficient sector data", "WARNING")
        return None
        
    # Correlation: Lambda vs Density
    r_dens, p_dens = stats.pearsonr(df["density_metric"], df["lambda_km"])
    
    print_status(f"Correlation Correlation_Length vs Pair_Density: r={r_dens:.3f}, p={p_dens:.4f}")
    
    # Normalize
    # Try to normalize lambda by density to see if structure persists
    # Hypothesis: Higher density -> shorter correlation length? (More noise sampling?)
    # Or Higher density -> longer? 
    # Let's calculate Lambda_norm = Lambda / Density^k
    # If r is significant, we try to remove it.
    
    # Simple normalization: Lambda / Density (assuming linear bias)
    df["lambda_norm"] = df["lambda_km"] / (df["density_metric"] / df["density_metric"].mean())
    
    # Check EW/NS ratio in normalized data
    # Find N and E/W sectors
    row_n = df[df["sector"] == "N"]
    row_e = df[df["sector"] == "E"]
    row_w = df[df["sector"] == "W"]
    
    if not row_n.empty and not row_e.empty:
        n_val = row_n.iloc[0]["lambda_norm"]
        e_val = row_e.iloc[0]["lambda_norm"]
        ratio_en = e_val / n_val
        print_status(f"Normalized EW/NS Ratio (E/N): {ratio_en:.2f}")
    
    if not row_n.empty and not row_w.empty:
        n_val = row_n.iloc[0]["lambda_norm"]
        w_val = row_w.iloc[0]["lambda_norm"]
        ratio_wn = w_val / n_val
        print_status(f"Normalized EW/NS Ratio (W/N): {ratio_wn:.2f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["density_metric"], df["lambda_km"], s=100)
    for i, row in df.iterrows():
        plt.annotate(row["sector"], (row["density_metric"], row["lambda_km"]), xytext=(5, 5), textcoords='offset points')
    
    # Regression line
    slope, intercept, _, _, _ = stats.linregress(df["density_metric"], df["lambda_km"])
    x = np.linspace(df["density_metric"].min(), df["density_metric"].max(), 100)
    plt.plot(x, slope*x + intercept, 'r--', label=f'r={r_dens:.3f}')
    
    plt.xlabel("Station Pairs (Millions)")
    plt.ylabel("Correlation Length (km)")
    plt.title("Station Density Bias Check")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "step_2_3_station_density.png"))
    plt.close()
    
    return {
        "correlation_lambda_density": {"r": r_dens, "p": p_dens},
        "sector_data": df.to_dict(orient="records")
    }

def main():
    print_status("Starting Step 2.3: Physical Interpretation & Controls", "TITLE")
    
    results = load_results()
    
    scaling_res = analyze_gravitational_scaling(results)
    density_res = analyze_station_density(results)
    
    output_data = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "description": "Post-hoc physical interpretation and control tests"
        },
        "gravitational_scaling": scaling_res,
        "station_density_control": density_res
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print_status(f"Analysis complete. Results saved to {OUTPUT_FILE}", "SUCCESS")

if __name__ == "__main__":
    main()
