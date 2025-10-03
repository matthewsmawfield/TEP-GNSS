#!/usr/bin/env python3
"""
Step 14 2022 Targeted Search
Systematic search for specific missing 2022 dates

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

def search_2022_m_archive() -> Dict:
    """Search the 2022_M archive systematically"""
    
    log("🔍 Searching 2022_M archive systematically...")
    
    results = {
        'found_files': [],
        'downloadable_files': []
    }
    
    try:
        response = requests.get("http://ftp.aiub.unibe.ch/CODE/2022_M/", timeout=30)
        response.raise_for_status()
        
        # Parse all files and their timestamps
        files_with_dates = []
        for line in response.text.split('\n'):
            if 'CLK_M.Z' in line:
                # Extract filename and date
                filename_match = re.search(r'href="([^"]*\.CLK_M\.Z)"', line)
                date_match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', line)
                
                if filename_match and date_match:
                    filename = filename_match.group(1)
                    date_str = date_match.group(1)
                    
                    # Parse date (02-Jan-2022 -> 2022-01-02)
                    try:
                        date = datetime.strptime(date_str, '%d-%b-%Y')
                        date_formatted = date.strftime('%Y%m%d')
                        
                        files_with_dates.append({
                            'filename': filename,
                            'date': date_formatted,
                            'date_str': date_str,
                            'url': f"http://ftp.aiub.unibe.ch/CODE/2022_M/{filename}"
                        })
                    except ValueError:
                        continue
        
        log(f"  Found {len(files_with_dates)} files in 2022_M archive")
        
        # Check for missing dates
        for file_info in files_with_dates:
            if file_info['date'] in MISSING_DATES:
                log(f"    ✅ Found missing date: {file_info['date']} - {file_info['filename']}")
                results['found_files'].append(file_info)
                
                # Check if downloadable
                try:
                    head_response = requests.head(file_info['url'], timeout=10)
                    if head_response.status_code == 200:
                        results['downloadable_files'].append(file_info)
                        log(f"      📥 Downloadable: {file_info['url']}")
                except:
                    log(f"      ❌ Not downloadable: {file_info['url']}")
        
    except Exception as e:
        log(f"  ⚠️  Error searching 2022_M archive: {e}")
    
    return results

def search_main_code_archive() -> Dict:
    """Search the main CODE archive for missing dates"""
    
    log("🔍 Searching main CODE archive...")
    
    results = {
        'found_files': [],
        'downloadable_files': []
    }
    
    try:
        response = requests.get("http://ftp.aiub.unibe.ch/CODE/", timeout=30)
        response.raise_for_status()
        
        # Look for files with 2022 dates
        for line in response.text.split('\n'):
            if '2022' in line and 'CLK' in line:
                # Extract filename
                filename_match = re.search(r'href="([^"]*)"', line)
                if filename_match:
                    filename = filename_match.group(1)
                    
                    # Extract date from filename
                    date_match = re.search(r'(\d{8})', filename)
                    if date_match:
                        date_str = date_match.group(1)
                        if date_str in MISSING_DATES:
                            file_url = f"http://ftp.aiub.unibe.ch/CODE/{filename}"
                            log(f"    ✅ Found missing date: {date_str} - {filename}")
                            results['found_files'].append({
                                'filename': filename,
                                'date': date_str,
                                'url': file_url
                            })
                            
                            # Check if downloadable
                            try:
                                head_response = requests.head(file_url, timeout=10)
                                if head_response.status_code == 200:
                                    results['downloadable_files'].append({
                                        'filename': filename,
                                        'date': date_str,
                                        'url': file_url
                                    })
                                    log(f"      📥 Downloadable: {file_url}")
                            except:
                                log(f"      ❌ Not downloadable: {file_url}")
        
    except Exception as e:
        log(f"  ⚠️  Error searching main CODE archive: {e}")
    
    return results

def search_alternative_formats() -> Dict:
    """Search for missing dates in alternative file formats"""
    
    log("🔍 Searching alternative file formats...")
    
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
    
    # Search for day of year format files
    try:
        response = requests.get("http://ftp.aiub.unibe.ch/CODE/", timeout=30)
        response.raise_for_status()
        
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
                        'date': date_str,
                        'url': file_url
                    })
                    
                    # Check if downloadable
                    try:
                        head_response = requests.head(file_url, timeout=10)
                        if head_response.status_code == 200:
                            results['downloadable_files'].append({
                                'filename': filename,
                                'date': date_str,
                                'url': file_url
                            })
                            log(f"      📥 Downloadable: {file_url}")
                    except:
                        log(f"      ❌ Not downloadable: {file_url}")
        
    except Exception as e:
        log(f"  ⚠️  Error searching alternative formats: {e}")
    
    return results

def search_igs_archives() -> Dict:
    """Search IGS archives for missing dates"""
    
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

def generate_download_script(all_results: Dict) -> str:
    """Generate a download script for found files"""
    
    script_content = """#!/usr/bin/env python3
'''
Auto-generated download script for missing 2022 dates
Generated by step_14_2022_targeted_search.py
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
        
        print(f"✅ Downloaded: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def main():
    output_dir = "/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download candidates
    download_candidates = [
"""
    
    # Add download candidates
    for candidate in all_results.get('downloadable_files', []):
        filename = candidate['filename']
        url = candidate['url']
        script_content += f'        ("{url}", "{output_dir}/{filename}"),\n'
    
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
    
    return script_content

def main():
    """Main execution function"""
    
    log("🚀 Starting Targeted Search for Missing 2022 Dates")
    log("=" * 80)
    
    log(f"📅 Searching for {len(MISSING_DATES)} missing dates:")
    for date_str in MISSING_DATES:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        log(f"  {date_str} - {date_obj.strftime('%A, %B %d, %Y')}")
    
    # Search different sources
    m_archive_results = search_2022_m_archive()
    main_archive_results = search_main_code_archive()
    alt_format_results = search_alternative_formats()
    igs_results = search_igs_archives()
    
    # Combine results
    all_found_files = (m_archive_results['found_files'] + 
                      main_archive_results['found_files'] + 
                      alt_format_results['found_files'] + 
                      igs_results['found_files'])
    
    all_downloadable = (m_archive_results['downloadable_files'] + 
                       main_archive_results['downloadable_files'] + 
                       alt_format_results['downloadable_files'] + 
                       igs_results['downloadable_files'])
    
    # Generate summary
    log("")
    log("=" * 80)
    log("🎯 TARGETED SEARCH COMPLETE")
    log("=" * 80)
    
    log(f"📊 Search Results:")
    log(f"  Missing dates searched: {len(MISSING_DATES)}")
    log(f"  Files found: {len(all_found_files)}")
    log(f"  Downloadable files: {len(all_downloadable)}")
    
    if all_found_files:
        log("")
        log("✅ Found files:")
        for file_info in all_found_files:
            log(f"  {file_info['date']} - {file_info['filename']}")
        
        if all_downloadable:
            log("")
            log("📥 Downloadable files:")
            for file_info in all_downloadable:
                log(f"  {file_info['date']} - {file_info['filename']}")
                log(f"    URL: {file_info['url']}")
            
            # Generate download script
            script_content = generate_download_script({'downloadable_files': all_downloadable})
            script_path = "/Users/matthewsmawfield/www/TEP-GNSS/scripts/experimental/step_14_2022_download_targeted.py"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            log("")
            log(f"📝 Download script generated: {script_path}")
            log("  Run: python scripts/experimental/step_14_2022_download_targeted.py")
        else:
            log("")
            log("⚠️  No downloadable files found")
    else:
        log("")
        log("❌ No missing dates found in any searched archives")
        log("  The missing dates may not be available in public archives")
        log("  Consider proceeding with 93.7% coverage (342/365 days)")
    
    log("=" * 80)

if __name__ == "__main__":
    main()
