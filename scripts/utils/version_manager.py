#!/usr/bin/env python3
"""
TEP-GNSS Version Management System
Single script for all version management tasks

Usage:
    python scripts/utils/version_manager.py              # Update all files
    python scripts/utils/version_manager.py --check      # Check consistency
    python scripts/utils/version_manager.py --info       # Show version info
    python scripts/utils/version_manager.py --core       # Update core files only
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

class VersionManager:
    def __init__(self, project_root: str = None):
        """Initialize version manager with project root path"""
        if project_root is None:
            # Find project root by looking for VERSION.json
            current = Path.cwd()
            while current != current.parent:
                if (current / 'VERSION.json').exists():
                    project_root = str(current)
                    break
                current = current.parent
            
            if project_root is None:
                raise FileNotFoundError("VERSION.json not found. Please run from project root.")
        
        self.project_root = Path(project_root)
        self.version_file = self.project_root / 'VERSION.json'
        
        # Load version data
        self._load_version_data()
        
        # Define update patterns for different file types
        self.patterns = {
            '*.py': [
                # Python version assignments - very specific patterns to avoid changing numerical values
                (r'version\s*=\s*["\']v?0\.1[0-5]["\']', f'version = "v{self.version}"'),
                (r'VERSION\s*=\s*["\']v?0\.1[0-5]["\']', f'VERSION = "v{self.version}"'),
                (r'["\']version["\']:\s*["\']v?0\.1[0-5]["\']', f'"version": "v{self.version}"'),
                # Python comments - more specific
                (r'# Version: v?0\.1[0-5](?![0-9])', f'# Version: v{self.version}'),
                (r'# v?0\.1[0-5](?![0-9])\s*\([^)]*\)', f'# v{self.version} ({self.codename})'),
                # Python print statements with hardcoded versions - more specific
                (r'print_status\("TEP GNSS Analysis Package v?0\.1[0-5](?![0-9])', f'print_status("TEP GNSS Analysis Package v{self.version}'),
                (r'"TEP GNSS Analysis Package v?0\.1[0-5](?![0-9])', f'"TEP GNSS Analysis Package v{self.version}'),
                (r'f"TEP GNSS Analysis Package v?0\.1[0-5](?![0-9])', f'f"TEP GNSS Analysis Package v{self.version}'),
            ],
            '*.html': [
                # HTML version references - more specific patterns
                (r'Version\s+v?0\.1[0-5](?![0-9])', f'Version v{self.version}'),
                (r'v?0\.1[0-5](?![0-9])\s*\([^)]*\)', f'v{self.version} ({self.codename})'),
                (r'<title>[^<]*v?0\.1[0-5](?![0-9])[^<]*</title>', f'<title>TEP-GNSS v{self.version} ({self.codename})</title>'),
                # HTML comments - more specific
                (r'<!-- Version: v?0\.1[0-5](?![0-9]) -->', f'<!-- Version: v{self.version} -->'),
                # PDF URLs and DOI links
                (r'https://zenodo\.org/records/\d+/files/[^"]*\.pdf(?:\?download=1)?', self._get_pdf_url()),
                (r'content="https://zenodo\.org/records/\d+/files/[^"]*\.pdf(?:\?download=1)?"', f'content="{self._get_pdf_url()}"'),
                (r'https://doi\.org/10\.5281/zenodo\.\d+', self._get_doi_url()),
                (r'<a href="https://doi\.org/10\.5281/zenodo\.\d+"', f'<a href="{self._get_doi_url()}"'),
                # Version update lists - replace hardcoded lists with dynamic ones
                (r'<h2>Version v?0\.1[0-5](?![0-9]) Updates</h2>', f'<h2>Version v{self.version} Updates</h2>'),
                (r'<strong>Version v?0\.1[0-5](?![0-9]) \([^)]*\)</strong>', f'<strong>Version v{self.version} ({self.codename})</strong>'),
                # Date modifications
                (r'"dateModified":\s*"[^"]*"', f'"dateModified": "{self.date}"'),
                (r'Last updated:\s*[^<]*', f'Last updated: {self._format_date_for_display()}'),
            ],
            '*.md': [
                # Markdown version references - more specific patterns
                (r'\*\*Version:\*\*\s*v?0\.1[0-5](?![0-9])', f'**Version:** v{self.version}'),
                (r'Version:\s*v?0\.1[0-5](?![0-9])', f'Version: v{self.version}'),
                (r'v?0\.1[0-5](?![0-9])\s*\([^)]*\)', f'v{self.version} ({self.codename})'),
                (r'# v?0\.1[0-5](?![0-9])\s*\([^)]*\)', f'# v{self.version} ({self.codename})'),
                # Date references
                (r'\*\*Date:\*\*\s*\d{1,2}\s+\w+\s+\d{4}', f'**Date:** {self._format_date(self.date)}'),
            ],
            '*.json': [
                # JSON version fields - more specific patterns
                (r'"version":\s*"0\.1[0-5](?![0-9])"', f'"version": "{self.version}"'),
                (r'"date-released":\s*"[^"]*"', f'"date-released": "{self.date}"'),
                (r'"date":\s*"[^"]*"', f'"date": "{self.date}"'),
            ],
            '*.cff': [
                # CFF version fields - more specific patterns
                (r'version:\s*0\.1[0-5](?![0-9])', f'version: {self.version}'),
                (r'date-released:\s*[^\n]*', f'date-released: {self.date}'),
            ],
            '*.toml': [
                # TOML version references - more specific patterns
                (r'version\s*=\s*"0\.1[0-5](?![0-9])"', f'version = "{self.version}"'),
                (r'date\s*=\s*"[^"]*"', f'date = "{self.date}"'),
            ],
            '*.txt': [
                # Text file version references - more specific patterns
                (r'Version\s+v?0\.1[0-5](?![0-9])', f'Version v{self.version}'),
                (r'v?0\.1[0-5](?![0-9])\s*\([^)]*\)', f'v{self.version} ({self.codename})'),
            ]
        }
        
        # Files to exclude from updates
        self.exclude_patterns = [
            '**/venv/**',
            '**/.venv/**',
            '**/node_modules/**',
            '**/.git/**',
            '**/__pycache__/**',
            '**/*.pyc',
            '**/VERSION.json',  # Don't update the version file itself
            '**/site-packages/**',  # Exclude all site-packages
            '**/theory/**',  # Exclude theoretical manuscripts
            '**/results/outputs/**',  # Exclude output data files
        ]
    
    def _load_version_data(self):
        """Load version data from VERSION.json"""
        with open(self.version_file, 'r') as f:
            self.version_data = json.load(f)
        
        self.version = self.version_data['version']
        self.codename = self.version_data['codename']
        self.date = self.version_data['date']
    
    def _format_date(self, date_str: str) -> str:
        """Format date string for display"""
        try:
            from datetime import datetime
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime('%d %B %Y')
        except:
            return date_str
    
    def _get_pdf_url(self) -> str:
        """Get PDF URL from version data"""
        zenodo_record = self.version_data.get('zenodo_record', '17216517')
        pdf_filename = self.version_data.get('pdf_filename', f'Smawfield_2025_GlobalTimeEchoes_Preprint_v{self.version}_{self.codename}.pdf')
        return f"https://zenodo.org/records/{zenodo_record}/files/{pdf_filename}?download=1"
    
    def _get_doi_url(self) -> str:
        """Get DOI URL from version data"""
        doi = self.version_data.get('doi', '10.5281/zenodo.17127229')
        return f"https://doi.org/{doi}"
    
    def _generate_version_update_list(self) -> str:
        """Generate HTML list of version changes from VERSION.json"""
        changes = self.version_data.get('changes', [])
        if not changes:
            return ""
        
        list_items = []
        for i, change in enumerate(changes, 1):
            list_items.append(f'                <li><strong>{change}</strong></li>')
        
        return '\n'.join(list_items)
    
    def _format_date_for_display(self) -> str:
        """Format date from YYYY-MM-DD to 'D Month YYYY' format"""
        try:
            from datetime import datetime
            date_obj = datetime.strptime(self.date, '%Y-%m-%d')
            return date_obj.strftime('%d %B %Y')
        except:
            return self.date  # Fallback to original format if parsing fails
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from updates"""
        file_str = str(file_path)
        
        # Check for exclusion patterns
        exclude_patterns = [
            '/venv/',
            '/.venv/',
            '/node_modules/',
            '/.git/',
            '/__pycache__/',
            'site-packages',
            'VERSION.json',
            '/theory/',  # Exclude theoretical manuscripts
            '/results/outputs/',  # Exclude output data files
        ]
        
        for pattern in exclude_patterns:
            if pattern in file_str:
                return True
        
        # Check for file extensions we don't want to update
        if file_path.suffix in ['.pyc', '.pyo', '.pyd']:
            return True
        
        # Exclude files with version numbers in their names (like manuscripts)
        if re.search(r'v\d+\.\d+', file_path.name):
            return True
            
        return False
    
    def update_file_version(self, file_path: Path) -> bool:
        """Update version in a single file"""
        if self._should_exclude_file(file_path):
            return False
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            updated = False
            
            # Special handling for references.html - replace hardcoded version list
            if file_path.name == 'references.html':
                content = self._update_references_version_list(content)
                if content != original_content:
                    updated = True
            
            # Apply patterns based on file extension
            patterns_to_apply = self.patterns.get('*' + file_path.suffix, [])
            
            # Skip DOI updates for references.html to preserve academic citations
            if file_path.name == 'references.html':
                # Filter out DOI-related patterns
                patterns_to_apply = [p for p in patterns_to_apply if not any(doi_term in str(p[0]) for doi_term in ['doi.org', 'zenodo'])]
            
            for pattern, replacement in patterns_to_apply:
                new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                if new_content != content:
                    content = new_content
                    updated = True
            
            # Write back if changed
            if updated:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
            return False
        
        return False
    
    def _update_references_version_list(self, content: str) -> str:
        """Update the version list in references.html with dynamic content"""
        # Find the version update section and replace the hardcoded list
        pattern = r'(<h2>Version v?0\.1[0-5] Updates</h2>.*?<p><strong>Version v?0\.1[0-5] \([^)]*\)</strong> represents[^<]*</p>\s*<ol>)(.*?)(</ol>)'
        
        def replace_version_section(match):
            header = match.group(1)
            footer = match.group(3)
            
            # Generate dynamic list from VERSION.json
            dynamic_list = self._generate_version_update_list()
            
            return f'{header}\n{dynamic_list}\n            {footer}'
        
        # Only update the version list, not DOI links
        content = re.sub(pattern, replace_version_section, content, flags=re.DOTALL)
        
        # Update version numbers in headers and titles
        content = re.sub(r'<h2>Version v?0\.1[0-5] Updates</h2>', f'<h2>Version v{self.version} Updates</h2>', content)
        content = re.sub(r'<strong>Version v?0\.1[0-5] \([^)]*\)</strong>', f'<strong>Version v{self.version} ({self.codename})</strong>', content)
        
        return content
    
    def update_core_files(self) -> Dict[str, List[str]]:
        """Update version in core project files only"""
        results = {
            'updated': [],
            'skipped': [],
            'errors': []
        }
        
        # Core files to update
        core_files = [
            'README.md',
            'document_map.md',
            'site/index.html',
            'site/components/section_3_results.html',
            'site/components/section_4_discussion.html',
            'site/components/references.html',
            'site/README.md',
            'site/manifest.json',
            'requirements/requirements.txt',
            'results/inconsistency_analysis.md',
            'results/gravitational_proportions_analysis.md',
        ]
        
        # Python scripts in steps directory
        steps_dir = self.project_root / 'scripts/steps'
        if steps_dir.exists():
            for py_file in steps_dir.rglob('*.py'):
                core_files.append(str(py_file))
        
        # Python scripts in exploratory directory
        exploratory_dir = self.project_root / 'scripts/exploratory'
        if exploratory_dir.exists():
            for py_file in exploratory_dir.rglob('*.py'):
                core_files.append(str(py_file))
        
        print(f"🔄 Updating version to v{self.version} ({self.codename}) in core files...")
        
        for file_path_str in core_files:
            file_path = self.project_root / file_path_str
            if not file_path.exists():
                results['skipped'].append(str(file_path))
                continue
                
            try:
                if self.update_file_version(file_path):
                    results['updated'].append(str(file_path))
                    print(f"✅ Updated: {file_path.relative_to(self.project_root)}")
                else:
                    results['skipped'].append(str(file_path))
            except Exception as e:
                results['errors'].append(f"{file_path}: {e}")
                print(f"❌ Error: {file_path} - {e}")
        
        return results

    def update_all_files(self) -> Dict[str, List[str]]:
        """Update version across all project files"""
        results = {
            'updated': [],
            'skipped': [],
            'errors': []
        }
        
        print(f"🔄 Updating version to v{self.version} ({self.codename}) across all files...")
        
        # Get all relevant files
        for pattern in self.patterns.keys():
            for file_path in self.project_root.rglob(pattern):
                if self._should_exclude_file(file_path):
                    results['skipped'].append(str(file_path))
                    continue
                
                try:
                    if self.update_file_version(file_path):
                        results['updated'].append(str(file_path))
                        print(f"✅ Updated: {file_path.relative_to(self.project_root)}")
                except Exception as e:
                    results['errors'].append(f"{file_path}: {e}")
                    print(f"❌ Error: {file_path} - {e}")
        
        return results
    
    def update_python_scripts(self) -> Dict[str, List[str]]:
        """Update Python scripts to use dynamic version loading"""
        results = {
            'updated': [],
            'skipped': [],
            'errors': []
        }
        
        print(f"🔄 Updating Python scripts to use dynamic version loading...")
        
        # Find all Python files in scripts directory
        for py_file in self.project_root.rglob('scripts/**/*.py'):
            if py_file.name in ['version_utils.py', 'version_manager.py']:
                results['skipped'].append(str(py_file))
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                updated = False
                
                # Pattern 1: Replace hardcoded version in print_status calls
                pattern1 = r'print_status\("TEP GNSS Analysis Package v?0\.\d+([^"]*)"'
                replacement1 = r'from scripts.utils.version_utils import VERSION_STRING\n    print_status(f"TEP GNSS Analysis Package {VERSION_STRING}\1"'
                
                if re.search(pattern1, content):
                    content = re.sub(pattern1, replacement1, content)
                    updated = True
                
                # Pattern 2: Replace other hardcoded version strings
                pattern2 = r'"TEP GNSS Analysis Package v?0\.\d+([^"]*)"'
                replacement2 = r'f"TEP GNSS Analysis Package {VERSION_STRING}\1"'
                
                if re.search(pattern2, content) and 'from scripts.utils.version_utils import VERSION_STRING' not in content:
                    # Add import if not already present
                    if 'import' in content:
                        # Find the last import statement
                        lines = content.split('\n')
                        last_import_idx = 0
                        for i, line in enumerate(lines):
                            if line.strip().startswith(('import ', 'from ')):
                                last_import_idx = i
                        
                        # Add our import after the last import
                        lines.insert(last_import_idx + 1, 'from scripts.utils.version_utils import VERSION_STRING')
                        content = '\n'.join(lines)
                    
                    content = re.sub(pattern2, replacement2, content)
                    updated = True
                
                # Write back if changed
                if updated:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    results['updated'].append(str(py_file))
                    print(f"✅ Updated: {py_file.relative_to(self.project_root)}")
                else:
                    results['skipped'].append(str(py_file))
                    
            except Exception as e:
                results['errors'].append(f"{py_file}: {e}")
                print(f"❌ Error: {py_file} - {e}")
        
        return results
    
    def check_version_consistency(self) -> Dict[str, List[str]]:
        """Check for version inconsistencies across files"""
        inconsistencies = {
            'old_versions': [],
            'missing_versions': [],
            'date_mismatches': []
        }
        
        print(f"🔍 Checking version consistency...")
        
        # Check for old version numbers
        old_version_pattern = r'v?0\.(?:1[0-3]|[0-9])'
        
        for file_path in self.project_root.rglob('*'):
            if (file_path.is_file() and 
                not self._should_exclude_file(file_path) and 
                file_path.suffix in ['.py', '.html', '.md', '.json', '.toml', '.txt']):
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for old versions
                    matches = re.findall(old_version_pattern, content)
                    if matches:
                        inconsistencies['old_versions'].append(f"{file_path}: {matches}")
                
                except Exception:
                    pass  # Skip files that can't be read as text
        
        return inconsistencies
    
    def display_version_info(self):
        """Display current version information"""
        print(f"\n📋 Current Version Information:")
        print(f"   Version: v{self.version}")
        print(f"   Codename: {self.codename}")
        print(f"   Date: {self.date}")
        print(f"   Changes: {len(self.version_data['changes'])} items")
        print(f"   Previous: v{self.version_data['previous_version']} ({self.version_data['previous_date']})")
        
        if self.version_data.get('description'):
            print(f"   Description: {self.version_data['description']}")
        
        print(f"\n📝 Changes in v{self.version} ({self.codename}):")
        for i, change in enumerate(self.version_data['changes'], 1):
            print(f"   {i:2d}. {change}")
    
    def display_system_info(self):
        """Display system information and usage"""
        print(f"\n🛠️  Available Commands:")
        print(f"   • python scripts/utils/version_manager.py --info     # Show version info")
        print(f"   • python scripts/utils/version_manager.py --check    # Check consistency")
        print(f"   • python scripts/utils/version_manager.py --core     # Update core files only")
        print(f"   • python scripts/utils/version_manager.py --python   # Update Python scripts to use dynamic loading")
        print(f"   • python scripts/utils/version_manager.py --update   # Update all files")
        
        print(f"\n📁 Key Files Updated:")
        key_files = [
            'README.md',
            'document_map.md', 
            'site/index.html',
            'site/CITATION.cff',
            'site/manifest.json',
            'site/components/*.html',
            'scripts/steps/**/*.py',
            'requirements/requirements.txt'
        ]
        
        for file_pattern in key_files:
            print(f"   • {file_pattern}")
        
        print(f"\n✅ Benefits of Centralized Version Management:")
        print(f"   • Single source of truth for version information")
        print(f"   • Automated updates across all file types")
        print(f"   • Consistent versioning across the entire project")
        print(f"   • Easy change tracking and documentation")
        print(f"   • Reduced human error in version updates")
        
        print(f"\n🔄 How to Update Version:")
        print(f"   1. Edit VERSION.json with new version details")
        print(f"   2. Run: python scripts/utils/version_manager.py --core")
        print(f"   3. Verify: python scripts/utils/version_manager.py --check")
        print(f"   4. Commit changes to git")
        
        print(f"\n🎉 System Status: ACTIVE")
        print(f"   All core files successfully updated to v{self.version} ({self.codename})")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TEP-GNSS Version Manager')
    parser.add_argument('--check', action='store_true', help='Check version consistency')
    parser.add_argument('--update', action='store_true', help='Update version across all files')
    parser.add_argument('--core', action='store_true', help='Update core files only')
    parser.add_argument('--info', action='store_true', help='Display version information')
    parser.add_argument('--system', action='store_true', help='Display system information')
    parser.add_argument('--python', action='store_true', help='Update Python scripts to use dynamic version loading')
    
    args = parser.parse_args()
    
    try:
        vm = VersionManager()
        
        if args.info or (not args.check and not args.update and not args.core and not args.system and not args.python):
            vm.display_version_info()
        
        if args.system:
            vm.display_version_info()
            vm.display_system_info()
        
        if args.python:
            results = vm.update_python_scripts()
            print(f"\n📊 Python Script Update Results:")
            print(f"   Updated: {len(results['updated'])} files")
            print(f"   Skipped: {len(results['skipped'])} files")
            print(f"   Errors: {len(results['errors'])} files")
        
        if args.core:
            results = vm.update_core_files()
            print(f"\n📊 Update Results:")
            print(f"   Updated: {len(results['updated'])} files")
            print(f"   Skipped: {len(results['skipped'])} files")
            print(f"   Errors: {len(results['errors'])} files")
        
        if args.update:
            results = vm.update_all_files()
            print(f"\n📊 Update Results:")
            print(f"   Updated: {len(results['updated'])} files")
            print(f"   Skipped: {len(results['skipped'])} files")
            print(f"   Errors: {len(results['errors'])} files")
        
        if args.check:
            inconsistencies = vm.check_version_consistency()
            if any(inconsistencies.values()):
                print(f"\n⚠️  Version Inconsistencies Found:")
                for category, items in inconsistencies.items():
                    if items:
                        print(f"   {category}: {len(items)} issues")
            else:
                print(f"\n✅ Version consistency check passed!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
