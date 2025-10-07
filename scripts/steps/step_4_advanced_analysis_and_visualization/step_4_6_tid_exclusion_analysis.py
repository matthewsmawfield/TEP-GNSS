import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Anchor to package root
PACKAGE_ROOT = Path(__file__).resolve().parents[3] # Adjusted from parents[2] to parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.utils.config import TEPConfig
from scripts.utils.logger import print_status, check_memory_usage, TEPLogger, set_step_logger

# Initialize step-specific logger
step_logger = TEPLogger(
    name="step_4_6_tid_exclusion_analysis",
    level="DEBUG",
    log_file_path=Path(__file__).resolve().parents[3] / "logs" / "step_4_6_tid_exclusion_analysis.log"
)

# Register step logger so print_status uses it
set_step_logger(step_logger)
from scripts.utils.pid_manager import ensure_single_instance
from scripts.utils.exceptions import TEPDataError, TEPFileError, TEPAnalysisError, safe_json_read, safe_json_write

class TIDExclusionAnalysis:
    def __init__(self):
        self.output_dir = PACKAGE_ROOT / "results/outputs"
        self.figures_dir = PACKAGE_ROOT / "results/figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.temporal_data_path = PACKAGE_ROOT / "results/outputs"
        
    def _load_temporal_analysis_data(self, ac: str, metric: str) -> Dict[datetime, List[float]]:
        """
        Loads temporal analysis data for TID exclusion analysis.
        Uses daily coherence data from Step 2.1 geospatial files as TID proxy.
        """
        all_daily_data: Dict[datetime, List[float]] = {}
        
        # Use daily coherence variability as TID proxy
        if metric == "hilbert-if":
            # Load daily coherence data from Step 2.1 geospatial files
            geospatial_file_path = PACKAGE_ROOT / f'data/processed/step_2_1_geospatial_{ac}.csv'
            
            if not geospatial_file_path.exists():
                print_status(f"WARNING: Step 2.1 geospatial data not found for {ac}: {geospatial_file_path}. Skipping.", "WARNING")
                return all_daily_data
                
            try:
                import pandas as pd
                df = pd.read_csv(geospatial_file_path)
                
                if 'date' in df.columns and 'plateau_phase' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    # Convert plateau_phase to coherence using cos(phase)
                    df['coherence'] = np.cos(df['plateau_phase'])
                    
                    # Group by date and calculate daily coherence variability as TID proxy
                    for date_obj, day_data in df.groupby(df['date'].dt.date):
                        date = datetime.combine(date_obj, datetime.min.time())
                        day_coherences = day_data['coherence'].values
                        
                        if len(day_coherences) > 1:
                            # Use coefficient of variation as TID activity proxy
                            cv = np.std(day_coherences) / np.mean(day_coherences) if np.mean(day_coherences) > 0 else 0
                            all_daily_data[date] = [cv]
                    
                    print_status(f"Loaded {len(all_daily_data)} daily TID proxy values for {ac}", "INFO")
                else:
                    print_status(f"Required columns not found in {geospatial_file_path}", "WARNING")
                    
            except Exception as e:
                print_status(f"Error loading geospatial data for {ac}: {e}", "WARNING")
                
        elif metric == "wavelet":
            # Skip wavelet for now since we don't have the data
            print_status(f"WARNING: Wavelet analysis not available for {ac}. Skipping.", "WARNING")
            
        return all_daily_data
        
    def _perform_tid_exclusion(self, ac: str, real_coherence_data: Dict[datetime, List[float]], 
                               temporal_analysis_data: Dict[datetime, List[float]], metric: str
    ) -> Dict[str, Any]:
        """
        Performs TID exclusion analysis by comparing real coherence with temporal analysis data.
        Adapted to work with statistical summaries from Step 4.3.
        """
        results: Dict[str, Any] = {
            'analysis_center': ac,
            'metric': metric,
            'status': 'FAILED',
            'details': 'Insufficient data for analysis.'
        }
        
        if not real_coherence_data or not temporal_analysis_data:
            print_status(f"WARNING: Insufficient real coherence or temporal analysis data for {ac} {metric}. Skipping TID exclusion.", "WARNING")
            return results
            
        # Align dates
        common_dates = sorted(list(set(real_coherence_data.keys()) & set(temporal_analysis_data.keys())))
        if not common_dates:
            print_status(f"WARNING: No common dates found between real coherence and temporal analysis data for {ac} {metric}. Skipping TID exclusion.", "WARNING")
            return results
            
        real_coherence_values = []
        temporal_analysis_values = []
        
        for date in common_dates:
            real_coherence_values.extend(real_coherence_data[date])
            temporal_analysis_values.extend(temporal_analysis_data[date])
            
        if not real_coherence_values or not temporal_analysis_values:
            print_status(f"WARNING: Empty value lists after date alignment for {ac} {metric}. Skipping TID exclusion.", "WARNING")
            return results
            
        # Convert to numpy arrays
        real_coherence_array = np.array(real_coherence_values)
        temporal_analysis_array = np.array(temporal_analysis_values)
        
        # For statistical data, perform correlation analysis instead of exclusion
        if len(real_coherence_array) == len(temporal_analysis_array):
            # Multiple values case - perform traditional exclusion analysis
            # Already ensured lengths match above
                
            # Dynamic threshold based on data distribution
            exclusion_threshold = np.percentile(temporal_analysis_array, 75)  # More conservative threshold
            
            excluded_indices = np.where(temporal_analysis_array > exclusion_threshold)[0]
            retained_indices = np.where(temporal_analysis_array <= exclusion_threshold)[0]
            
            coherence_retained = real_coherence_array[retained_indices]
            coherence_excluded = real_coherence_array[excluded_indices]
            
            original_mean_coherence = np.mean(real_coherence_array)
            retained_mean_coherence = np.mean(coherence_retained) if len(coherence_retained) > 0 else 0
            
            # Calculate improvement/change
            coherence_change = retained_mean_coherence - original_mean_coherence
            percentage_change = (coherence_change / original_mean_coherence) * 100 if original_mean_coherence != 0 else 0
            
            results.update({
                'status': 'SUCCESS',
                'details': 'TID exclusion analysis completed.',
                'analysis_type': 'temporal_exclusion',
                'original_mean_coherence': float(original_mean_coherence),
                'retained_mean_coherence': float(retained_mean_coherence),
                'coherence_change': float(coherence_change),
                'percentage_change': float(percentage_change),
                'num_original_data_points': len(real_coherence_array),
                'num_retained_data_points': len(coherence_retained),
                'num_excluded_data_points': len(coherence_excluded),
                'exclusion_threshold': float(exclusion_threshold)
            })
            
            print_status(f"TID Exclusion for {ac} ({metric}): Original mean coherence={original_mean_coherence:.4f}, Retained mean={retained_mean_coherence:.4f}, Change={percentage_change:.2f}%", "INFO")
            
            return results

    def run_tid_exclusion_analysis(self, acs: List[str]):
        """
        Runs the TID exclusion analysis for specified analysis centers.
        """
        print_status("Starting TID Exclusion Analysis...", "PROCESS")
        check_memory_usage(context="TID Exclusion Analysis Start")

        all_results = {}
        
        for ac in acs:
            print_status(f"Processing TID exclusion for {ac.upper()}...", "PROCESS")
            
            # Load daily coherence data from Step 2.1 geospatial files
            real_coherence_data = {}
            geospatial_file_path = PACKAGE_ROOT / f'data/processed/step_2_1_geospatial_{ac}.csv'
            
            if not geospatial_file_path.exists():
                print_status(f"WARNING: Step 2.1 geospatial data not found for {ac}: {geospatial_file_path}. Skipping TID exclusion.", "WARNING")
                all_results[ac] = {'status': 'SKIPPED', 'reason': 'Step 2.1 geospatial data not found.'}
                continue
                
            try:
                import pandas as pd
                df = pd.read_csv(geospatial_file_path)
                
                if 'date' in df.columns and 'plateau_phase' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    # Convert plateau_phase to coherence using cos(phase)
                    df['coherence'] = np.cos(df['plateau_phase'])
                    
                    # Group by date and calculate daily mean coherence
                    for date_obj, day_data in df.groupby(df['date'].dt.date):
                        date = datetime.combine(date_obj, datetime.min.time())
                        day_coherences = day_data['coherence'].values
                        
                        if len(day_coherences) > 0:
                            mean_coherence = np.mean(day_coherences)
                            real_coherence_data[date] = [mean_coherence]
                    
                    print_status(f"Successfully loaded {len(real_coherence_data)} daily coherence values for {ac.upper()}", "INFO")
                else:
                    print_status(f"Required columns not found in {geospatial_file_path}", "WARNING")
                    all_results[ac] = {'status': 'SKIPPED', 'reason': 'Required columns not found in geospatial data.'}
                    continue
                    
            except Exception as e:
                print_status(f"WARNING: Error processing {geospatial_file_path}: {e}. Skipping file.", "WARNING")
                all_results[ac] = {'status': 'SKIPPED', 'reason': f'Error loading geospatial data: {e}'}
                continue

            if not real_coherence_data:
                print_status(f"WARNING: No real coherence data extracted for {ac}. Skipping TID exclusion.", "WARNING")
                all_results[ac] = {'status': 'SKIPPED', 'reason': 'No real coherence data extracted.'}
                continue

            for metric in ['hilbert-if']: # Only process hilbert-if since wavelet data doesn't exist
                temporal_analysis_data = self._load_temporal_analysis_data(ac, metric)
                
                if not temporal_analysis_data:
                    print_status(f"WARNING: No {metric.upper()} temporal analysis data loaded for {ac}. Skipping TID exclusion for this metric.", "WARNING")
                    all_results[f"{ac}_{metric}"] = {'status': 'SKIPPED', 'reason': f'No {metric.upper()} temporal analysis data.'}
                    continue
                    
                result = self._perform_tid_exclusion(ac, real_coherence_data, temporal_analysis_data, metric)
                all_results[f"{ac}_{metric}"] = result
                
        output_file = self.output_dir / "step_4_6_tid_exclusion_analysis_results.json"
        try:
            safe_json_write(all_results, output_file, indent=4)
            print_status(f"TID Exclusion Analysis results saved to {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"ERROR: Failed to save TID Exclusion Analysis results to {output_file}: {e}", "ERROR")

        check_memory_usage(context="TID Exclusion Analysis End")
        print_status("TID Exclusion Analysis Completed.", "SUCCESS")
        return all_results

@ensure_single_instance
def main():
    """
    Main function to run the TID exclusion analysis.
    """
    try:
        analysis_centers = [ac.strip() for ac in TEPConfig.get_str('TEP_ANALYSIS_CENTERS', 'code,esa_final,igs_combined').split(',')]
        
        tid_analysis = TIDExclusionAnalysis()
        tid_analysis.run_tid_exclusion_analysis(analysis_centers)
        
    except TEPAnalysisError as e:
        print_status(f"TID Exclusion Analysis failed: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"An unexpected error occurred during TID Exclusion Analysis: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("TID Exclusion Analysis interrupted by user.", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"An unexpected error occurred during TID Exclusion Analysis: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
