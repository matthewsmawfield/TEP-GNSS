#!/usr/bin/env python3
"""
Version Utilities for TEP-GNSS
Provides dynamic version information to all Python scripts
"""

import json
from pathlib import Path

def get_version_info():
    """Get current version information from VERSION.json"""
    # Find project root by looking for VERSION.json
    current = Path(__file__).parent
    while current != current.parent:
        version_file = current / 'VERSION.json'
        if version_file.exists():
            with open(version_file, 'r') as f:
                version_data = json.load(f)
            return version_data
        current = current.parent
    
    # Fallback if VERSION.json not found
    return {
        'version': '0.15',
        'codename': 'Jaipur',
        'date': '2025-01-15',
        'description': 'Fallback version - VERSION.json not found'
    }

def get_version_string():
    """Get formatted version string (e.g., 'v0.15 (Jaipur)')"""
    version_data = get_version_info()
    return f"v{version_data['version']} ({version_data['codename']})"

def get_version_number():
    """Get version number only (e.g., '0.15')"""
    version_data = get_version_info()
    return version_data['version']

def get_codename():
    """Get codename only (e.g., 'Jaipur')"""
    version_data = get_version_info()
    return version_data['codename']

def get_date():
    """Get release date (e.g., '2025-01-15')"""
    version_data = get_version_info()
    return version_data['date']

def get_doi():
    """Get DOI (e.g., '10.5281/zenodo.17127229')"""
    version_data = get_version_info()
    return version_data.get('doi', '10.5281/zenodo.17127229')

def get_zenodo_record():
    """Get Zenodo record ID (e.g., '17216517')"""
    version_data = get_version_info()
    return version_data.get('zenodo_record', '17216517')

def get_pdf_filename():
    """Get PDF filename (e.g., 'Smawfield_2025_GlobalTimeEchoes_Preprint_v0.15_Jaipur.pdf')"""
    version_data = get_version_info()
    return version_data.get('pdf_filename', f'Smawfield_2025_GlobalTimeEchoes_Preprint_v{get_version_number()}_{get_codename()}.pdf')

def get_pdf_url():
    """Get full PDF URL"""
    zenodo_record = get_zenodo_record()
    pdf_filename = get_pdf_filename()
    return f"https://zenodo.org/records/{zenodo_record}/files/{pdf_filename}?download=1"

def get_doi_url():
    """Get DOI URL"""
    doi = get_doi()
    return f"https://doi.org/{doi}"

# Convenience variables for direct import
VERSION_INFO = get_version_info()
VERSION_STRING = get_version_string()
VERSION_NUMBER = get_version_number()
CODENAME = get_codename()
RELEASE_DATE = get_date()
DOI = get_doi()
ZENODO_RECORD = get_zenodo_record()
PDF_FILENAME = get_pdf_filename()
PDF_URL = get_pdf_url()
DOI_URL = get_doi_url()

if __name__ == "__main__":
    print(f"Version: {VERSION_STRING}")
    print(f"Date: {RELEASE_DATE}")
    print(f"DOI: {DOI}")
    print(f"PDF URL: {PDF_URL}")
    print(f"Description: {VERSION_INFO.get('description', 'N/A')}")
