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
        Loads temporal analysis data (wavelet or Hilbert-IF) for a given analysis center and metric.
        Adapted to work with actual Step 4.3 data format which contains statistical summaries.
        """
        all_daily_data: Dict[datetime, List[float]] = {}
        
        file_patterns = {
            "wavelet": f"step_4_3_wavelet-analysis_high_res_merged.json",
            "hilbert-if": f"step_4_3_hilbert-if_high_res_merged.json"
        }
        
        if metric not in file_patterns:
            raise ValueError(f"Unknown metric for temporal analysis: {metric}")
            
        file_path = self.temporal_data_path / file_patterns[metric]
        
        if not file_path.exists():
            print_status(f"WARNING: No {metric.upper()} temporal analysis file found for {ac}: {file_path}. Skipping.", "WARNING")
            return all_daily_data
            
        try:
            temporal_data = safe_json_read(file_path)
            
            # Handle the actual Step 4.3 data format
            if 'center_results' in temporal_data and ac in temporal_data['center_results']:
                center_data = temporal_data['center_results'][ac]
                
                if metric == 'hilbert-if' and 'bands' in center_data:
                    # Extract temporal metrics from Hilbert-IF band analysis (require actual dated series)
                    bands = center_data['bands']
                    # Expect that upstream step provides dated arrays in future; for now, require merged daily key
                    if 'daily_metrics' in center_data and isinstance(center_data['daily_metrics'], dict):
                        for date_str, values in center_data['daily_metrics'].items():
                            try:
                                date = datetime.strptime(date_str, '%Y-%m-%d')
                                all_daily_data[date] = [float(v) for v in values]
                            except (ValueError, TypeError):
                                continue
                        print_status(f"Loaded {len(all_daily_data)} dated temporal entries from {metric.upper()} data for {ac}", "INFO")
                    else:
                        print_status(f"No dated temporal metrics found in {metric.upper()} data for {ac}", "WARNING")
                        
                elif metric == 'wavelet' and 'bands' in center_data:
                    # Require dated entries for wavelet data as well
                    if 'daily_metrics' in center_data and isinstance(center_data['daily_metrics'], dict):
                        for date_str, values in center_data['daily_metrics'].items():
                            try:
                                date = datetime.strptime(date_str, '%Y-%m-%d')
                                all_daily_data[date] = [float(v) for v in values]
                            except (ValueError, TypeError):
                                continue
                        print_status(f"Loaded {len(all_daily_data)} dated temporal entries from {metric.upper()} data for {ac}", "INFO")
                    else:
                        print_status(f"No dated temporal metrics found in {metric.upper()} data for {ac}", "WARNING")
                else:
                    print_status(f"No bands data found in {metric.upper()} data for {ac}", "WARNING")
            else:
                # Fallback to original format if available
                for date_str, values in temporal_data.get(ac, {}).items():
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                        all_daily_data[date] = [float(v) for v in values]
                    except (ValueError, TypeError) as e:
                        print_status(f"WARNING: Error parsing date or values for {date_str} in {file_path}: {e}. Skipping.", "WARNING")
                        continue
                        
            print_status(f"Successfully loaded {len(all_daily_data)} temporal data entries for {ac} from {file_path}", "INFO")
            
        except Exception as e:
            print_status(f"ERROR: Error loading {file_path} for {ac}: {e}. Skipping.", "ERROR")
            
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
        
        else:
            print_status(f"WARNING: Insufficient or non-aligned temporal vs coherence series for {ac} {metric}.", "WARNING")
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
            
            # Load real coherence data (e.g., r_squared from step 2.0)
            # This part needs to be adapted based on how real_coherence_data is generated/structured.
            # Assuming step_3_5 generates this, or it comes from step 2.0 directly.
            # For now, let's mock it or assume it's loaded similar to step_3_5's approach.
            
            # For demonstration, let's use the same logic as step 3.5's load_real_coherence_data
            real_coherence_data = {}
            # Construct the file path for the aggregated JSON output from Step 2.0
            file_path = PACKAGE_ROOT / f"results/outputs/step_2_0_correlation_{ac}.json"
            
            if not file_path.exists():
                print_status(f"WARNING: Aggregated correlation file not found for {ac}: {file_path}. Skipping TID exclusion for this AC.", "WARNING")
                all_results[ac] = {'status': 'SKIPPED', 'reason': 'Step 2.0 correlation file not found.'}
                continue
                
            try:
                correlation_data = safe_json_read(file_path)
                    
                if 'exponential_fit' in correlation_data and correlation_data['exponential_fit'] and 'r_squared' in correlation_data['exponential_fit']:
                    r_squared_value = correlation_data['exponential_fit']['r_squared']
                    if r_squared_value is not None:
                        # Assign a fixed date for aggregated results, similar to step_3_5
                        fixed_date = datetime(2023, 1, 1)
                        real_coherence_data[fixed_date] = [r_squared_value]
                        print_status(f"Successfully extracted r_squared from {file_path} for {ac.upper()}", "INFO")
                    else:
                        print_status(f"WARNING: 'r_squared' is None in exponential_fit for {file_path}. Skipping.", "WARNING")
                else:
                    print_status(f"WARNING: 'exponential_fit' or 'r_squared' not found in {file_path}. Skipping.", "WARNING")
                    
            except Exception as e:
                print_status(f"WARNING: Error processing {file_path}: {e}. Skipping file.", "WARNING")
                all_results[ac] = {'status': 'SKIPPED', 'reason': f'Error loading Step 2.0 correlation file: {e}'}
                continue

            if not real_coherence_data:
                print_status(f"WARNING: No real coherence data extracted for {ac}. Skipping TID exclusion.", "WARNING")
                all_results[ac] = {'status': 'SKIPPED', 'reason': 'No real coherence data extracted.'}
                continue

            for metric in ['wavelet', 'hilbert-if']: # Process for both metrics
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
