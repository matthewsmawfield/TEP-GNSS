#!/usr/bin/env python3
"""
Step 14 2022 Missing Dates Analysis
Identifies which specific dates are missing from 2022 data

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-27
"""

import os
import glob
import re
from datetime import datetime, timedelta
from typing import Set, List

def log(message: str) -> None:
    print(message, flush=True)

def get_existing_2022_dates() -> Set[str]:
    """Get all existing 2022 dates from filenames"""
    
    code_dir = "/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code"
    existing_files = glob.glob(f"{code_dir}/*2022*")
    
    date_pattern = re.compile(r'(\d{8})')
    existing_dates = set()
    
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1)
            existing_dates.add(date_str)
    
    return existing_dates

def get_all_2022_dates() -> Set[str]:
    """Generate all valid 2022 dates"""
    
    all_dates = set()
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    current_date = start_date
    while current_date <= end_date:
        all_dates.add(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
    
    return all_dates

def main():
    """Main execution function"""
    
    log("🔍 Analyzing missing 2022 dates...")
    log("=" * 60)
    
    # Get existing and expected dates
    existing_dates = get_existing_2022_dates()
    all_dates = get_all_2022_dates()
    
    # Find missing dates
    missing_dates = all_dates - existing_dates
    
    # Convert to sorted list for display
    missing_dates_list = sorted(list(missing_dates))
    
    log(f"📊 Coverage Analysis:")
    log(f"  Total 2022 days: {len(all_dates)}")
    log(f"  Existing dates: {len(existing_dates)}")
    log(f"  Missing dates: {len(missing_dates)}")
    log(f"  Coverage: {(len(existing_dates) / len(all_dates)) * 100:.1f}%")
    
    if missing_dates:
        log("")
        log("📅 Missing dates:")
        
        # Group by month for better readability
        months = {}
        for date_str in missing_dates_list:
            month = date_str[:6]  # YYYYMM
            if month not in months:
                months[month] = []
            months[month].append(date_str)
        
        for month in sorted(months.keys()):
            month_name = datetime.strptime(month, '%Y%m').strftime('%B %Y')
            log(f"  {month_name}: {len(months[month])} days")
            
            # Show first few missing dates in each month
            for i, date_str in enumerate(months[month][:5]):
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                log(f"    {date_str} ({date_obj.strftime('%A, %B %d')})")
            
            if len(months[month]) > 5:
                log(f"    ... and {len(months[month]) - 5} more")
        
        log("")
        log("📋 Complete missing dates list:")
        for date_str in missing_dates_list:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            log(f"  {date_str} - {date_obj.strftime('%A, %B %d, %Y')}")
    else:
        log("")
        log("✅ No missing dates! Complete coverage achieved.")
    
    log("=" * 60)

if __name__ == "__main__":
    main()
