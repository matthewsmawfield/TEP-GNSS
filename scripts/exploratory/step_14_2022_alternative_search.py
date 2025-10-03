#!/usr/bin/env python3
"""
Step 14 2022 Alternative Search
Search for missing 2022 dates in IGS and other GNSS archives

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-27
"""

import os
import sys
import requests
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional

def log(message: str) -> None:
    print(message, flush=True)

# Missing dates from previous analysis
MISSING_DATES = [
    '20220117', '20220127', '20220128', '20220129',  # January
    '20220218',  # February
    '20220328',  # March
    '20220417',  # April
    '20220517', '20220518',  # May
    '20220626',  # June
    '20220726', '20220727', '20220728',  # July
    '20220926',  # September
    '20221203', '20221204', '20221205', '20221213', '20221214', '20221215', '20221223', '20221224', '20221225'  # December
]

def convert_to_day_of_year(date_str: str) -> int:
    """Convert YYYYMMDD to day of year"""
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    return date_obj.timetuple().tm_yday

def search_igs_archive() -> Dict:
    """Search IGS (International GNSS Service) archives"""
    
    log("🔍 Searching IGS archives...")
    
    results = {
        'found_files': [],
        'downloadable_files': []
    }
    
    # IGS archive locations
    igs_locations = [
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/001",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/002",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/003",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/004",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/005",
    ]
    
    for location in igs_locations:
        log(f"  Checking: {location}")
        
        try:
            response = requests.get(location, timeout=30)
            response.raise_for_status()
            
            # Look for CODE files
            code_files = re.findall(r'href="([^"]*CODE[^"]*)"', response.text)
            
            for filename in code_files:
                # Extract date from filename
                date_match = re.search(r'(\d{8})', filename)
                if date_match:
                    date_str = date_match.group(1)
                    if date_str in MISSING_DATES:
                        file_url = f"{location}/{filename}"
                        log(f"    ✅ Found missing date: {date_str} - {filename}")
                        results['found_files'].append({
                            'filename': filename,
                            'url': file_url,
                            'date': date_str,
                            'source': 'IGS'
                        })
                        
                        # Check if downloadable
                        try:
                            head_response = requests.head(file_url, timeout=10)
                            if head_response.status_code == 200:
                                results['downloadable_files'].append({
                                    'filename': filename,
                                    'url': file_url,
                                    'date': date_str,
                                    'source': 'IGS'
                                })
                                log(f"      📥 Downloadable: {file_url}")
                        except:
                            log(f"      ❌ Not downloadable: {file_url}")
            
        except Exception as e:
            log(f"    ⚠️  Error accessing {location}: {e}")
        
        time.sleep(1)
    
    return results

def search_alternative_code_mirrors() -> Dict:
    """Search alternative CODE mirrors"""
    
    log("🔍 Searching alternative CODE mirrors...")
    
    results = {
        'found_files': [],
        'downloadable_files': []
    }
    
    # Alternative CODE mirrors
    mirrors = [
        "https://cddis.nasa.gov/archive/gnss/products/ionex",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/001",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/002",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/003",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/004",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/005",
    ]
    
    for mirror in mirrors:
        log(f"  Checking: {mirror}")
        
        try:
            response = requests.get(mirror, timeout=30)
            response.raise_for_status()
            
            # Look for CODE files
            code_files = re.findall(r'href="([^"]*CODE[^"]*)"', response.text)
            
            for filename in code_files:
                # Extract date from filename
                date_match = re.search(r'(\d{8})', filename)
                if date_match:
                    date_str = date_match.group(1)
                    if date_str in MISSING_DATES:
                        file_url = f"{mirror}/{filename}"
                        log(f"    ✅ Found missing date: {date_str} - {filename}")
                        results['found_files'].append({
                            'filename': filename,
                            'url': file_url,
                            'date': date_str,
                            'source': 'Alternative'
                        })
                        
                        # Check if downloadable
                        try:
                            head_response = requests.head(file_url, timeout=10)
                            if head_response.status_code == 200:
                                results['downloadable_files'].append({
                                    'filename': filename,
                                    'url': file_url,
                                    'date': date_str,
                                    'source': 'Alternative'
                                })
                                log(f"      📥 Downloadable: {file_url}")
                        except:
                            log(f"      ❌ Not downloadable: {file_url}")
            
        except Exception as e:
            log(f"    ⚠️  Error accessing {mirror}: {e}")
        
        time.sleep(1)
    
    return results

def search_by_day_of_year() -> Dict:
    """Search for missing dates using day of year format"""
    
    log("🔍 Searching by day of year format...")
    
    results = {
        'found_files': [],
        'downloadable_files': []
    }
    
    # Convert missing dates to day of year
    missing_days = {}
    for date_str in MISSING_DATES:
        day_of_year = convert_to_day_of_year(date_str)
        missing_days[day_of_year] = date_str
    
    log(f"  Missing days of year: {sorted(missing_days.keys())}")
    
    # Search CODE archive for day of year format
    try:
        response = requests.get("http://ftp.aiub.unibe.ch/CODE/", timeout=30)
        response.raise_for_status()
        
        # Look for files with day of year format
        for line in response.text.split('\n'):
            # Look for patterns like COD001.CLK.Z, COD002.CLK.Z, etc.
            day_match = re.search(r'href="COD(\d{3})\.CLK\.Z"', line)
            if day_match:
                day_of_year = int(day_match.group(1))
                if day_of_year in missing_days:
                    date_str = missing_days[day_of_year]
                    filename = f"COD{day_of_year:03d}.CLK.Z"
                    file_url = f"http://ftp.aiub.unibe.ch/CODE/{filename}"
                    
                    log(f"    ✅ Found missing date: {date_str} (day {day_of_year}) - {filename}")
                    results['found_files'].append({
                        'filename': filename,
                        'url': file_url,
                        'date': date_str,
                        'source': 'Day of Year'
                    })
                    
                    # Check if downloadable
                    try:
                        head_response = requests.head(file_url, timeout=10)
                        if head_response.status_code == 200:
                            results['downloadable_files'].append({
                                'filename': filename,
                                'url': file_url,
                                'date': date_str,
                                'source': 'Day of Year'
                            })
                            log(f"      📥 Downloadable: {file_url}")
                    except:
                        log(f"      ❌ Not downloadable: {file_url}")
    
    except Exception as e:
        log(f"  ⚠️  Error searching day of year format: {e}")
    
    return results

def main():
    """Main execution function"""
    
    log("🚀 Starting Alternative Search for Missing 2022 Dates")
    log("=" * 80)
    
    log(f"📅 Searching for {len(MISSING_DATES)} missing dates:")
    for date_str in MISSING_DATES:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        log(f"  {date_str} - {date_obj.strftime('%A, %B %d, %Y')}")
    
    # Search IGS archives
    igs_results = search_igs_archive()
    
    # Search alternative mirrors
    alt_results = search_alternative_code_mirrors()
    
    # Search by day of year
    doy_results = search_by_day_of_year()
    
    # Combine results
    all_found_files = igs_results['found_files'] + alt_results['found_files'] + doy_results['found_files']
    all_downloadable = igs_results['downloadable_files'] + alt_results['downloadable_files'] + doy_results['downloadable_files']
    
    # Generate summary
    log("")
    log("=" * 80)
    log("🎯 ALTERNATIVE SEARCH COMPLETE")
    log("=" * 80)
    
    log(f"📊 Search Results:")
    log(f"  Missing dates searched: {len(MISSING_DATES)}")
    log(f"  Files found: {len(all_found_files)}")
    log(f"  Downloadable files: {len(all_downloadable)}")
    
    if all_found_files:
        log("")
        log("✅ Found files:")
        for file_info in all_found_files:
            log(f"  {file_info['date']} - {file_info['filename']} ({file_info['source']})")
        
        if all_downloadable:
            log("")
            log("📥 Downloadable files:")
            for file_info in all_downloadable:
                log(f"  {file_info['date']} - {file_info['filename']} ({file_info['source']})")
                log(f"    URL: {file_info['url']}")
            
            # Generate download script
            script_content = f"""#!/usr/bin/env python3
'''
Auto-generated download script for missing 2022 dates
Generated by step_14_2022_alternative_search.py
'''

import os
import requests
import time
from pathlib import Path

def download_file(url: str, output_path: str) -> bool:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {{os.path.basename(output_path)}}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {{url}}: {{e}}")
        return False

def main():
    output_dir = "/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download candidates
    download_candidates = [
"""
            
            for file_info in all_downloadable:
                script_content += f'        ("{file_info["url"]}", "{output_dir}/{file_info["filename"]}"),\n'
            
            script_content += """    ]
    
    print(f"🚀 Starting download of {len(download_candidates)} files...")
    
    success_count = 0
    for url, output_path in download_candidates:
        if download_file(url, output_path):
            success_count += 1
        time.sleep(1)  # Be respectful to servers
    
    print(f"📊 Download complete: {success_count}/{len(download_candidates)} successful")

if __name__ == "__main__":
    main()
"""
            
            script_path = "/Users/matthewsmawfield/www/TEP-GNSS/scripts/experimental/step_14_2022_download_alternative.py"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            log("")
            log(f"📝 Download script generated: {script_path}")
            log("  Run: python scripts/experimental/step_14_2022_download_alternative.py")
        else:
            log("")
            log("⚠️  No downloadable files found")
    else:
        log("")
        log("❌ No missing dates found in alternative sources")
        log("  The missing dates may not be available in public archives")
    
    log("=" * 80)

if __name__ == "__main__":
    main()
