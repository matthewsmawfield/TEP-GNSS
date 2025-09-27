#!/usr/bin/env python3
"""
Step 14 2022 Missing Dates Search
Comprehensive search for missing 2022 dates across different CODE archives and formats

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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def search_archive_directory(base_url: str, year: str) -> List[Dict]:
    """Search a specific archive directory for files"""
    
    try:
        response = requests.get(f"{base_url}/{year}/", timeout=30)
        response.raise_for_status()
        
        files_found = []
        for line in response.text.split('\n'):
            # Look for various CODE file patterns
            patterns = [
                r'href="([^"]*\.CLK\.Z)"',  # Legacy format
                r'href="([^"]*\.CLK_M\.Z)"',  # Multi-GNSS format
                r'href="([^"]*_30S_CLK\.CLK\.gz)"',  # Modern format
                r'href="([^"]*_v3\.CLK\.Z)"',  # Version 3 format
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    files_found.append({
                        'filename': match,
                        'url': f"{base_url}/{year}/{match}",
                        'archive': base_url
                    })
        
        return files_found
        
    except Exception as e:
        log(f"  ⚠️  Error searching {base_url}/{year}/: {e}")
        return []

def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from various CODE filename formats"""
    
    # Try different date extraction patterns
    patterns = [
        r'(\d{8})',  # YYYYMMDD format
        r'COD(\d{2})(\d{3})',  # Legacy COD format
        r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD in various positions
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) == 1:
                # Single group - should be YYYYMMDD
                date_str = match.group(1)
                if len(date_str) == 8 and date_str.startswith('2022'):
                    return date_str
            elif len(match.groups()) == 2:
                # Two groups - likely year and day of year
                year_part = match.group(1)
                day_part = match.group(2)
                if year_part == '22':  # 2022
                    try:
                        # Convert day of year to date
                        day_of_year = int(day_part)
                        date = datetime(2022, 1, 1) + timedelta(days=day_of_year - 1)
                        return date.strftime('%Y%m%d')
                    except:
                        continue
            elif len(match.groups()) == 3:
                # Three groups - likely year, month, day
                year, month, day = match.groups()
                if year == '2022':
                    try:
                        date = datetime(int(year), int(month), int(day))
                        return date.strftime('%Y%m%d')
                    except:
                        continue
    
    return None

def check_file_availability(url: str) -> bool:
    """Check if a file is available for download"""
    
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def search_missing_dates() -> Dict:
    """Search for missing dates across different archives"""
    
    log("🔍 Searching for missing 2022 dates...")
    log("=" * 80)
    
    # Define search locations
    search_locations = [
        "http://ftp.aiub.unibe.ch/CODE",
        "http://ftp.aiub.unibe.ch/CODE/2022",
        "http://ftp.aiub.unibe.ch/CODE/2022_M",
        "https://cddis.nasa.gov/archive/gnss/products/ionex",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022",
    ]
    
    results = {
        'searched_archives': [],
        'files_found': [],
        'missing_dates_found': set(),
        'download_candidates': []
    }
    
    for base_url in search_locations:
        log(f"🔍 Searching: {base_url}")
        
        # Search main directory
        files = search_archive_directory(base_url, "")
        results['searched_archives'].append(base_url)
        
        # Search 2022 subdirectory
        files_2022 = search_archive_directory(base_url, "2022")
        
        all_files = files + files_2022
        
        log(f"  Found {len(all_files)} files")
        
        # Analyze found files
        for file_info in all_files:
            filename = file_info['filename']
            date_str = extract_date_from_filename(filename)
            
            if date_str and date_str in MISSING_DATES:
                log(f"  ✅ Found missing date: {date_str} - {filename}")
                results['missing_dates_found'].add(date_str)
                results['files_found'].append(file_info)
                
                # Check if file is downloadable
                if check_file_availability(file_info['url']):
                    results['download_candidates'].append(file_info)
                    log(f"    📥 Downloadable: {file_info['url']}")
                else:
                    log(f"    ❌ Not downloadable: {file_info['url']}")
        
        time.sleep(1)  # Be respectful to servers
    
    return results

def search_alternative_sources() -> Dict:
    """Search alternative CODE data sources"""
    
    log("")
    log("🔍 Searching alternative CODE sources...")
    log("=" * 80)
    
    alternative_sources = [
        "https://cddis.nasa.gov/archive/gnss/products/ionex",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/001",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/002",
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2022/003",
    ]
    
    results = {
        'alternative_files': [],
        'downloadable_alternatives': []
    }
    
    for source in alternative_sources:
        log(f"🔍 Checking: {source}")
        
        try:
            response = requests.get(source, timeout=30)
            response.raise_for_status()
            
            # Look for CODE files in the response
            code_files = re.findall(r'href="([^"]*CODE[^"]*)"', response.text)
            
            for filename in code_files:
                date_str = extract_date_from_filename(filename)
                if date_str and date_str in MISSING_DATES:
                    file_url = f"{source}/{filename}"
                    log(f"  ✅ Found: {date_str} - {filename}")
                    results['alternative_files'].append({
                        'filename': filename,
                        'url': file_url,
                        'source': source
                    })
                    
                    if check_file_availability(file_url):
                        results['downloadable_alternatives'].append({
                            'filename': filename,
                            'url': file_url,
                            'source': source
                        })
                        log(f"    📥 Downloadable: {file_url}")
            
        except Exception as e:
            log(f"  ⚠️  Error accessing {source}: {e}")
        
        time.sleep(1)
    
    return results

def generate_download_script(results: Dict) -> str:
    """Generate a download script for found files"""
    
    script_content = """#!/usr/bin/env python3
'''
Auto-generated download script for missing 2022 dates
Generated by step_14_2022_missing_dates_search.py
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
    for candidate in results.get('download_candidates', []):
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
    
    log("🚀 Starting Comprehensive Missing Dates Search")
    log("=" * 80)
    
    log(f"📅 Searching for {len(MISSING_DATES)} missing dates:")
    for date_str in MISSING_DATES:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        log(f"  {date_str} - {date_obj.strftime('%A, %B %d, %Y')}")
    
    # Search main archives
    main_results = search_missing_dates()
    
    # Search alternative sources
    alt_results = search_alternative_sources()
    
    # Combine results
    all_results = {
        'main_search': main_results,
        'alternative_search': alt_results,
        'total_missing_dates': len(MISSING_DATES),
        'dates_found': len(main_results['missing_dates_found']),
        'download_candidates': main_results['download_candidates'] + alt_results['downloadable_alternatives']
    }
    
    # Generate summary
    log("")
    log("=" * 80)
    log("🎯 SEARCH COMPLETE")
    log("=" * 80)
    
    log(f"📊 Search Results:")
    log(f"  Missing dates searched: {all_results['total_missing_dates']}")
    log(f"  Dates found: {all_results['dates_found']}")
    log(f"  Download candidates: {len(all_results['download_candidates'])}")
    log(f"  Archives searched: {len(main_results['searched_archives'])}")
    
    if all_results['dates_found'] > 0:
        log("")
        log("✅ Found missing dates:")
        for date_str in sorted(main_results['missing_dates_found']):
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            log(f"  {date_str} - {date_obj.strftime('%A, %B %d, %Y')}")
        
        if all_results['download_candidates']:
            log("")
            log("📥 Download candidates found:")
            for candidate in all_results['download_candidates']:
                log(f"  {candidate['filename']} - {candidate['url']}")
            
            # Generate download script
            script_content = generate_download_script(all_results)
            script_path = "/Users/matthewsmawfield/www/TEP-GNSS/scripts/experimental/step_14_2022_download_missing.py"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            log("")
            log(f"📝 Download script generated: {script_path}")
            log("  Run: python scripts/experimental/step_14_2022_download_missing.py")
        else:
            log("")
            log("⚠️  No downloadable files found")
    else:
        log("")
        log("❌ No missing dates found in searched archives")
        log("  The missing dates may not be available in public archives")
    
    log("=" * 80)

if __name__ == "__main__":
    main()
